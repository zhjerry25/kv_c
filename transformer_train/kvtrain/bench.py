"""Cloud-only BF16/FP8 comparison: identical weights, real forward/backward/AdamW."""

import argparse
import gc
import json
import statistics

import torch

from .config import ModelConfig, PRESETS, configure_runtime_cache
from .model import Transformer
from .precision import Precision
from .train import amp_context, make_optimizer, DEFAULTS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=PRESETS, default="50m")
    p.add_argument("--attention", choices=("gqa", "mla", "tpa"), default="gqa")
    p.add_argument("--mode", choices=("standard", "cla", "cyclic"), default="cyclic")
    p.add_argument("--window", type=int, default=0)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--fp8", nargs="+", choices=("off", "delayed", "current", "mxfp8"), default=["off", "delayed", "current"])
    args = p.parse_args()
    configure_runtime_cache()
    if not torch.cuda.is_available():
        p.error("Benchmark requires a CUDA GPU; CPU smoke tests are in tests/test_smoke.py")
    if min(args.steps, args.warmup, args.batch_size, args.seq_len) < 1:
        p.error("steps, warmup, batch-size and seq-len must be positive")
    cfg = ModelConfig(**PRESETS[args.preset], attention=args.attention, mode=args.mode, window=args.window)
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    reference = Transformer(cfg)
    weights = reference.state_dict()
    device = torch.device("cuda")
    tokens = torch.randint(cfg.vocab_size, (args.batch_size, args.seq_len + 1), device=device)
    x, y = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
    baseline = None
    for fp8 in args.fp8:
        precision = Precision(fp8, device)
        precision.validate_batch(x.numel())
        model = Transformer(cfg, precision).to(device).train()
        result = model.load_state_dict(weights, strict=False)
        if result.unexpected_keys or any(not key.endswith("_extra_state") for key in result.missing_keys):
            raise RuntimeError(f"Benchmark weight mismatch: {result}")
        model.kernel.prepare(args.seq_len, device)
        optimizer = make_optimizer(model, DEFAULTS, device)
        elapsed = []
        for i in range(args.warmup + args.steps):
            if i == args.warmup:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, "bf16"), precision.context():
                loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            end.record()
            end.synchronize()
            if i >= args.warmup:
                elapsed.append(start.elapsed_time(end))
        median = statistics.median(elapsed)
        if fp8 == "off":
            baseline = median
        print(json.dumps(dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                              **model.summary(), median_step_ms=median,
                              tokens_per_second=x.numel() * 1000 / median,
                              speedup_vs_bf16=baseline / median if baseline else None,
                              peak_cuda_gb=torch.cuda.max_memory_allocated() / 1e9,
                              final_loss=loss.item())), flush=True)
        del optimizer, model, precision
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
