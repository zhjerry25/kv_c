"""Idle-CPU prefill: parallel exact-cache construction vs token append.

Both paths retain the same compact physical cache. No trained weights are fit.
"""
import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from public_memory import PublicMemory


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--lengths', default='128,512,1024')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--tag', default='clean_prefill_bench')
    args = ap.parse_args()
    assert args.repeats >= 0
    for path in (ROOT / 'outputs').glob('clean_*_queue.json'):
        assert json.loads(path.read_text())['status'] == 'completed', 'Timing requires completed training queues'
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    model = PublicMemory.load(args.checkpoint).eval().requires_grad_(False)
    alphabet = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    text = (ROOT / 'data/tinyshakespeare.txt').read_text()
    lookup = {ch: i for i, ch in enumerate(alphabet)}
    data = torch.tensor([lookup[ch] for ch in text[int(.9 * len(text)):]])
    metadata = dict(platform=platform.platform(), torch=torch.__version__, threads=args.threads,
        checkpoint_sha256=sha(args.checkpoint), repeats=args.repeats,
        code_sha256={p: sha(ROOT / p) for p in ('public_memory.py', 'narrow_cache.py', 'work/bench_clean_prefill.py')},
        note='CPU only, after all training queues complete; preallocated same compact cache, FP32 private/weights, '
             'paired alternating execution order, no inference from these numbers to GPU or standard-GQA speed. '
             'Parallel prefill omits unused past final-layer Query/FFN/output; all layer KV is still committed.')
    rows = []
    for n in map(int, args.lengths.split(',')):
        tokens = data[:n][None]
        for dtype in (torch.float32, torch.bfloat16):
            parallel = model.new_cache(n + 2, dtype=dtype)
            streaming = model.new_cache(n + 2, dtype=dtype)
            def run(mode):
                if mode == 0:
                    return parallel.prefill(tokens)
                streaming.reset()
                result = None
                for t in range(n):
                    result = streaming.append(tokens[:, t])
                return result
            outputs = [run(mode) for mode in (0, 1)]
            expected = model.forward_stored(tokens, dtype)[:, -1:]
            maximum = max(float((out - expected).abs().max()) for out in outputs)
            feedback_error = 0.
            for t in range(n, n + 2):
                a, b = parallel.append(data[t:t+1]), streaming.append(data[t:t+1])
                feedback_error = max(feedback_error, float((a - b).abs().max()))
            assert max(maximum, feedback_error) < (.03 if dtype == torch.bfloat16 else 1e-4)
            samples = [[], []]
            for repeat in range(args.repeats):
                for order in range(2):
                    mode = (repeat + order) % 2
                    started = time.perf_counter()
                    run(mode)
                    samples[mode].append(1000 * (time.perf_counter() - started))
            row = dict(length=n, dtype=str(dtype), max_prefill_logits_error=maximum,
                max_next_two_token_feedback_error=feedback_error, memory=parallel.memory_bytes(),
                milliseconds=[statistics.median(s) if s else None for s in samples], samples_ms=samples,
                modes=['parallel_layerwise_last_query', 'tokenwise_append'])
            rows.append(row)
            print(json.dumps(row), flush=True)
            (ROOT / f'outputs/{args.tag}.json').write_text(json.dumps(dict(status='running', metadata=metadata, rows=rows), indent=2) + '\n')
    (ROOT / f'outputs/{args.tag}.json').write_text(json.dumps(dict(status='completed', metadata=metadata, rows=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
