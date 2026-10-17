"""Single-seed, sequential write-fraction experiment matrix (50M by default)."""

import argparse
from dataclasses import asdict
from fractions import Fraction
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys

import torch

from .config import ModelConfig, PRESETS, ROOT, local_path
from .initialization import PAIRED_INIT_VERSION
from .model import Transformer
from .precision import Precision
from .train import DEFAULTS

FRACTIONS = ("1/16", "1/8", "1/4", "3/8", "1/2", "3/4", "1")


def build_plan(args):
    manifest_path = local_path(args.data) / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes()) if manifest_path.exists() else None
    if manifest is None and not args.dry_run:
        raise FileNotFoundError(f"Prepare FineWeb-Edu first: {manifest_path}")
    vocab = math.ceil(manifest["tokenizer"]["vocab_size"] / 64) * 64 if manifest else 50304
    settings = dict(DEFAULTS)
    for key in DEFAULTS:
        if hasattr(args, key):
            settings[key] = getattr(args, key)
    settings.update(data=str(local_path(args.data).relative_to(ROOT)), paired_init=True)
    settings.pop("out")
    tokens = args.steps * args.seq_len * args.batch_size * args.accumulation
    if min(args.steps, args.seq_len, args.batch_size, args.accumulation, args.eval_batches, args.eval_every, args.save_every, args.log_every) < 1:
        raise ValueError("Training and evaluation sizes/intervals must be positive")
    if not 0 <= args.warmup < args.steps:
        raise ValueError("warmup must be nonnegative and smaller than steps")
    if args.compile and args.fp8 == "delayed":
        raise ValueError("Use --fp8 current/off with --compile, or omit --compile")
    if manifest and not args.dry_run and not args.allow_data_repeat and manifest["splits"]["train"]["tokens"] < tokens:
        raise ValueError(f"Each run consumes {tokens:,} tokens, but data has only {manifest['splits']['train']['tokens']:,}. Prepare more data or explicitly use --allow-data-repeat.")
    if manifest and any(manifest["splits"][split]["tokens"] <= args.seq_len for split in ("train", "val")):
        raise ValueError("Both splits need at least seq_len + 1 tokens")
    fractions = [str(Fraction(f)) for f in args.fractions]
    if len(set(fractions)) != len(fractions) or len(set(args.attention)) != len(args.attention):
        raise ValueError("Duplicate fractions or attention types")
    rows = []
    for attention in args.attention:
        cases = [("standard", None)] + [("cyclic", f) for f in fractions]
        if args.include_cla:
            cases.append(("cla", None))
        for mode, fraction in cases:
            cfg = ModelConfig(**PRESETS[args.preset], vocab_size=vocab, attention=attention,
                              mode=mode, write_fraction=fraction or "1/4", window=args.window)
            with torch.device("meta"):
                model = Transformer(cfg, Precision(device="meta"))
            stats = model.summary()
            if args.preset == "50m" and not 40_000_000 <= stats["parameters"] <= 65_000_000:
                raise ValueError("50M experiment expects a GPT-2-sized vocabulary; this data gives a different model scale")
            name = f"{attention}-{mode}" + ("-w" + fraction.replace("/", "_") if fraction else "")
            rows.append(dict(name=name, attention=attention, mode=mode, write_fraction=fraction,
                             config=asdict(cfg), parameters=stats["parameters"],
                             nonembedding_parameters=stats["parameters"] - model.embed.weight.numel(),
                             C=model.layout.width, w=len(model.layout.indices[1]) if mode == "cyclic" else model.layout.width,
                             cache_scalars_per_token=stats["cache_scalars_per_token"],
                             cache_ratio=stats["cache_scalars_per_token"] / stats["standard_cache_scalars_per_token"]))
    # Establish both references for all architectures before scanning narrow writes.
    def order(row):
        stage = 0 if row["mode"] == "standard" else 1 if row["write_fraction"] == "1" else 3 if row["mode"] == "cla" else 2
        return stage, Fraction(row["write_fraction"] or "1"), args.attention.index(row["attention"])

    rows.sort(key=order)
    return dict(version=1, paired_init_version=PAIRED_INIT_VERSION, torch_version=str(torch.__version__), preset=args.preset,
                settings=settings, device=args.device, compile=args.compile,
                allow_data_repeat=args.allow_data_repeat, tokens_per_run=tokens,
                data_fingerprint=hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest else None,
                runs=rows)


def train_command(plan, row, out):
    command = [sys.executable, "-m", "kvtrain.train", "--device", plan["device"]]
    if plan["compile"]:
        command.append("--compile")
    if (out / "last.pt").exists():
        return command + ["--resume", str(out / "last.pt")]
    command += ["--preset", plan["preset"], "--attention", row["attention"], "--mode", row["mode"],
                "--window", str(row["config"]["window"]), "--out", str(out), "--paired-init"]
    if row["write_fraction"]:
        command += ["--write-fraction", row["write_fraction"]]
    for key, value in plan["settings"].items():
        if key != "paired_init":
            command += ["--" + key.replace("_", "-"), str(value)]
    return command


def collect_result(plan, row, out):
    records = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines() if line.strip()]
    final = [v for v in records if v["step"] == plan["settings"]["steps"] and "val_loss" in v]
    if not final or not math.isfinite(final[-1]["val_loss"]):
        raise ValueError(f"Missing finite final-step validation: {out}")
    train = [v for v in records if "loss" in v]
    loss = final[-1]["val_loss"]
    return dict(name=row["name"], attention=row["attention"], mode=row["mode"], write_fraction=row["write_fraction"],
                seed=plan["settings"]["seed"], parameters=row["parameters"], nonembedding_parameters=row["nonembedding_parameters"],
                C=row["C"], w=row["w"], cache_ratio=row["cache_ratio"],
                cache_scalars_per_token=row["cache_scalars_per_token"],
                steps=plan["settings"]["steps"], train_tokens=plan["tokens_per_run"],
                final_val_nll=loss, final_val_ppl=math.exp(loss) if loss < 700 else None,
                median_tokens_per_second=statistics.median(v["tokens_per_second"] for v in train[1:] or train),
                peak_cuda_gb=max(v["peak_cuda_gb"] for v in train))


def summarize(plan, out):
    rows = []
    for case in plan["runs"]:
        path = out / case["name"] / "result.json"
        if path.exists():
            rows.append(json.loads(path.read_text()))
    native = {v["attention"]: v["final_val_nll"] for v in rows if v["mode"] == "standard"}
    full_write = {v["attention"]: v["final_val_nll"] for v in rows if v["mode"] == "cyclic" and v["write_fraction"] == "1"}
    for row in rows:
        row["delta_nll_vs_standard"] = row["final_val_nll"] - native[row["attention"]] if row["attention"] in native else None
        row["delta_nll_vs_full_write"] = row["final_val_nll"] - full_write[row["attention"]] if row["attention"] in full_write else None
    (out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    if rows:
        with (out / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=PRESETS, default="50m")
    p.add_argument("--attention", nargs="+", choices=("gqa", "mla", "tpa"), default=["gqa", "mla", "tpa"])
    p.add_argument("--fractions", nargs="+", default=list(FRACTIONS))
    p.add_argument("--include-cla", action="store_true", help="Add one CLA group-size-2 control per attention")
    p.add_argument("--data", default="data/fineweb-edu-1b")
    p.add_argument("--out", default="runs/50m-write-sweep")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("auto", "fp32", "bf16"), default="auto")
    p.add_argument("--fp8", choices=("off", "delayed", "current"), default="off")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--window", type=int, default=0)
    p.add_argument("--allow-data-repeat", action="store_true", help="Allow a training corpus smaller than the per-run token budget")
    p.add_argument("--dry-run", action="store_true")
    for key, value in dict(seed=0, steps=32768, seq_len=2048, batch_size=4, accumulation=4,
                           warmup=512, lr=3e-4, eval_every=1024, eval_batches=64,
                           save_every=1024, log_every=50, loss_chunk_size=1024).items():
        p.add_argument("--" + key.replace("_", "-"), type=type(value), default=value)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    plan = build_plan(args)
    print(f"{len(plan['runs'])} runs, seed={args.seed}, {plan['tokens_per_run']:,} sampled tokens/run; FP8={args.fp8}", flush=True)
    print(f"{'run':28} {'params/M':>9} {'C':>5} {'w':>5} {'cache/%':>9}", flush=True)
    for row in plan["runs"]:
        print(f"{row['name']:28} {row['parameters'] / 1e6:9.3f} {row['C']:5} {row['w']:5} {100 * row['cache_ratio']:9.3f}", flush=True)
    if args.dry_run:
        return
    out = local_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "matrix.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != plan:
            raise ValueError("Experiment plan/data changed; use a new --out directory")
    elif any(out.iterdir()):
        raise ValueError("Experiment directory is nonempty without matrix.json; use a new --out")
    else:
        manifest.write_text(json.dumps(plan, indent=2) + "\n")
    summarize(plan, out)
    for index, row in enumerate(plan["runs"], 1):
        directory = out / row["name"]
        if (directory / "result.json").exists():
            print(f"[{index}/{len(plan['runs'])}] SKIP {row['name']} (complete)", flush=True)
            continue
        directory.mkdir(exist_ok=True)
        command = train_command(plan, row, directory)
        print(f"[{index}/{len(plan['runs'])}] {'RESUME' if '--resume' in command else 'TRAIN'} {row['name']}", flush=True)
        with (directory / "console.log").open("a", buffering=1) as log:
            with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                try:
                    for line in process.stdout:
                        log.write(line)
                        print(line, end="", flush=True)
                    code = process.wait()
                except KeyboardInterrupt:
                    process.terminate()
                    process.wait()
                    raise
        if code:
            raise SystemExit(f"Run failed ({code}): {row['name']}; rerun the same command to resume")
        result = collect_result(plan, row, directory)
        temporary = directory / "result.json.tmp"
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(directory / "result.json")
        summarize(plan, out)
    print(f"Complete: {out / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
