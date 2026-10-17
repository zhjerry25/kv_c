"""Single-GPU AdamW training with exact sampler/optimizer/RNG resume."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import time

import torch

from .config import ModelConfig, PRESETS, local_path, configure_runtime_cache
from .data import PackedTokens
from .model import Transformer
from .precision import Precision


DEFAULTS = dict(data="data/fineweb-edu", out="runs/train", seq_len=2048, batch_size=2,
                accumulation=8, steps=1000, lr=3e-4, min_lr_ratio=0.1, warmup=100,
                weight_decay=0.1, grad_clip=1.0, seed=42, fp8="off", dtype="auto",
                log_every=10, eval_every=100, eval_batches=10, save_every=100, loss_chunk_size=1024)


def lr_at(step, total, warmup, lr, min_lr_ratio):
    if warmup and step < warmup:
        return lr * (step + 1) / warmup
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup - 1)))
    return lr * (min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def amp_context(device, dtype):
    if dtype == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def atomic_save(payload, path):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def make_optimizer(model, settings, device):
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW([dict(params=decay, weight_decay=settings["weight_decay"]),
                              dict(params=no_decay, weight_decay=0.0)],
                             lr=settings["lr"], betas=(0.9, 0.95), eps=1e-8,
                             fused=device.type == "cuda")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=PRESETS, default="50m")
    p.add_argument("--config", help="Optional JSON ModelConfig inside the framework")
    p.add_argument("--attention", choices=("gqa", "mla", "tpa"))
    p.add_argument("--mode", choices=("standard", "cla", "cyclic"))
    p.add_argument("--window", type=int, help="0=full causal; positive=SWA window, including the current token")
    p.add_argument("--cla-group-size", type=int)
    p.add_argument("--write-fraction")
    p.add_argument("--share-rope", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--activation-checkpoint", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--dry-run", action="store_true", help="Print shapes/parameter count on meta device")
    p.add_argument("--resume")
    p.add_argument("--stop-after", type=int, help="Stop after this total step without changing the LR horizon")
    for key, default in DEFAULTS.items():
        choices = ("off", "delayed", "current", "mxfp8") if key == "fp8" else (("auto", "fp32", "bf16") if key == "dtype" else None)
        p.add_argument("--" + key.replace("_", "-"), type=type(default), choices=choices, default=None)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    saved = torch.load(local_path(args.resume), map_location="cpu", weights_only=True) if args.resume else None
    settings = dict(saved["settings"] if saved else DEFAULTS)
    adjustable = {"log_every", "eval_every", "save_every", "eval_batches"}
    for key in DEFAULTS:
        value = getattr(args, key)
        if value is not None:
            if saved and key not in adjustable and value != settings[key]:
                raise ValueError(f"Exact resume requires unchanged --{key.replace('_', '-')}; saved value: {settings[key]}")
            settings[key] = value
    for key in ("seq_len", "batch_size", "accumulation", "steps", "log_every", "eval_batches", "loss_chunk_size"):
        if settings[key] < 1:
            raise ValueError(f"{key} must be positive")
    if min(settings["eval_every"], settings["save_every"], settings["warmup"]) < 0:
        raise ValueError("eval_every/save_every/warmup must be nonnegative")
    if args.stop_after is not None and args.stop_after < 1:
        raise ValueError("stop_after must be positive")
    if settings["lr"] <= 0 or not 0 <= settings["min_lr_ratio"] <= 1 or settings["grad_clip"] <= 0 or settings["weight_decay"] < 0:
        raise ValueError("Invalid optimizer settings")
    if saved:
        cfg = ModelConfig(**saved["config"])
        for key in ("attention", "mode", "window", "cla_group_size", "write_fraction", "share_rope", "activation_checkpoint"):
            value = getattr(args, key)
            if value is not None and value != getattr(cfg, key):
                raise ValueError(f"Resume architecture differs at {key}")
        if args.config:
            raise ValueError("Resume already contains its model config")
    else:
        config = json.loads(local_path(args.config).read_text()) if args.config else dict(PRESETS[args.preset])
        for key in ("attention", "mode", "window", "cla_group_size", "write_fraction", "share_rope", "activation_checkpoint"):
            if getattr(args, key) is not None:
                config[key] = getattr(args, key)
        manifest = local_path(settings["data"]) / "manifest.json"
        if manifest.exists():
            vocab = json.loads(manifest.read_text())["tokenizer"]["vocab_size"]
            config["vocab_size"] = math.ceil(vocab / 64) * 64
        cfg = ModelConfig(**config)
    if args.dry_run:
        with torch.device("meta"):
            model = Transformer(cfg, Precision(device="meta"))
        print(json.dumps(dict(config=asdict(cfg), **model.summary()), indent=2))
        return
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    configure_runtime_cache()
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices: cpu, cuda")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        device = torch.device("cuda", torch.cuda.current_device())
    dtype = ("bf16" if device.type == "cuda" else "fp32") if settings["dtype"] == "auto" else settings["dtype"]
    if settings["fp8"] != "off" and dtype != "bf16":
        raise ValueError("FP8 training requires BF16 autocast (--dtype bf16 or auto on CUDA)")
    if args.compile and settings["fp8"] == "delayed":
        raise ValueError("TE delayed scaling does not support torch.compile; use --fp8 current or --no-compile")
    torch.manual_seed(settings["seed"])
    torch.set_float32_matmul_precision("high")
    train = PackedTokens(settings["data"], "train", settings["seq_len"], settings["seed"])
    val = PackedTokens(settings["data"], "val", settings["seq_len"], settings["seed"] + 1)
    precision = Precision(settings["fp8"], device)
    precision.validate_batch(settings["batch_size"] * settings["seq_len"])
    model = Transformer(cfg, precision).to(device)
    if settings["fp8"] != "off" and precision.converted == 0:
        raise ValueError("This model has no FP8-eligible large linear layers; use a larger preset or --fp8 off")
    model.kernel.prepare(settings["seq_len"], device)
    optimizer = make_optimizer(model, settings, device)
    completed = 0
    if saved:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        train.load_state_dict(saved["train_sampler"])
        val.load_state_dict(saved["val_sampler"])
        torch.set_rng_state(saved["rng"])
        if device.type == "cuda" and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        completed = saved["step"]
    out = local_path(settings["out"])
    out.mkdir(parents=True, exist_ok=True)
    if not saved and (out / "last.pt").exists():
        raise FileExistsError("Run already exists: use --resume or a new --out directory")
    (out / "config.json").write_text(json.dumps(dict(model=asdict(cfg), training=settings), indent=2) + "\n")
    print(json.dumps(dict(device=str(device), dtype=dtype, torch=torch.__version__,
                          effective_tokens_per_step=settings["batch_size"] * settings["seq_len"] * settings["accumulation"],
                          **model.summary())), flush=True)
    if args.compile:
        # Keep TE's quantization-aware checkpoint wrapper outside Dynamo. Compile
        # the block itself, retaining ordinary state_dict names for portable saves.
        for block in model.blocks:
            block.compile(dynamic=False)
    runner = model

    def save():
        atomic_save(dict(version=1, config=asdict(cfg), settings=settings, model=model.state_dict(),
                         optimizer=optimizer.state_dict(), step=completed,
                         train_sampler=train.state_dict(), val_sampler=val.state_dict(),
                         rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else None), out / "last.pt")

    def evaluate():
        model.eval()
        state = val.state_dict()
        values = []
        # Same held-out windows at every evaluation; no effects on train sampler.
        val.generator.manual_seed(settings["seed"] + 1)
        with torch.no_grad():
            for _ in range(settings["eval_batches"]):
                x, y = val.batch(settings["batch_size"], device)
                # Validate master weights without changing FP8 amax/scale history.
                with amp_context(device, dtype):
                    values.append(model(x, y, settings["loss_chunk_size"]).detach())
        result = torch.stack(values).mean().item()
        val.load_state_dict(state)
        model.train()
        return result

    stop = min(settings["steps"], args.stop_after if args.stop_after is not None else settings["steps"])
    model.train()
    interval_losses, interval_seconds, interval_tokens = [], 0.0, 0
    with (out / "metrics.jsonl").open("a", buffering=1) as log:
        while completed < stop:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            lr = lr_at(completed, settings["steps"], settings["warmup"], settings["lr"], settings["min_lr_ratio"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for _ in range(settings["accumulation"]):
                x, y = train.batch(settings["batch_size"], device)
                with amp_context(device, dtype), precision.context():
                    loss = runner(x, y, settings["loss_chunk_size"]) / settings["accumulation"]
                # TE backward must run after leaving its autocast region.
                loss.backward()
                losses.append(loss.detach())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            completed += 1
            interval_losses.append(torch.stack(losses).sum())
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            interval_seconds += time.perf_counter() - start
            interval_tokens += settings["batch_size"] * settings["seq_len"] * settings["accumulation"]
            if completed % settings["log_every"] == 0 or completed == stop:
                record = dict(step=completed, loss=torch.stack(interval_losses).mean().item(), lr=lr,
                              grad_norm=float(norm), tokens_per_second=interval_tokens / interval_seconds,
                              peak_cuda_gb=torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0)
                if not math.isfinite(record["loss"]):
                    raise FloatingPointError("Non-finite training loss")
                log.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
                interval_losses, interval_seconds, interval_tokens = [], 0.0, 0
            if settings["eval_every"] and completed % settings["eval_every"] == 0:
                record = dict(step=completed, val_loss=evaluate())
                log.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
            if settings["save_every"] and completed % settings["save_every"] == 0:
                save()
    if not (settings["save_every"] and completed % settings["save_every"] == 0):
        save()
    # Portable weights omit TE's quantizer state; load into fp8=off on any device.
    atomic_save(dict(config=asdict(cfg), model={k: v for k, v in model.state_dict().items() if not k.endswith("_extra_state")}), out / "model.pt")
    print(f"Saved {out / 'last.pt'} and {out / 'model.pt'}", flush=True)


if __name__ == "__main__":
    main()
