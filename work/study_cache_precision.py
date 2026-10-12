"""Paired local workspace-precision evaluation; never overwrites old results."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from train_public_memory import Model
from prefix_cache import PrefixCache, precision_forward

ROOT = Path(__file__).resolve().parents[1]


def load_model(path):
    checkpoint = torch.load(path, weights_only=False)
    cfg, row = checkpoint['args'], checkpoint['row']
    model = Model(len(checkpoint['alphabet']), row['mode'], row['rank'],
                  cfg['d'], cfg['depth'], cfg['source_norm']).eval()
    model.load_state_dict(checkpoint['state_dict'])
    return model, checkpoint


@torch.inference_mode()
def execution_checks():
    rows = []
    torch.manual_seed(314159)
    for norm in (False, True):
        for dtype in (torch.float64, torch.float32, torch.bfloat16):
            model = Model(65, 'moments_uniform', 16, 64, 4, norm).eval()
            if dtype == torch.float64:
                model = model.double()
            tokens = torch.randint(65, (2, 35))
            expected = precision_forward(model, tokens, dtype)
            for prefill in (0, 7, 8, 9, 16):
                predictions = {}
                for execution in ('material', 'raw', 'anchored'):
                    cache = PrefixCache(model, 35, 2, execution, 8, dtype)
                    cache.prefill(tokens[:, :prefill])
                    got = torch.cat([cache.append(tokens[:, i]) for i in range(prefill, 35)], 1)
                    error = float((got - expected[:, prefill:]).abs().max())
                    predictions[execution] = got
                    limit = 1e-9 if dtype == torch.float64 else 5e-5 if dtype == torch.float32 else 5e-3
                    assert torch.isfinite(got).all() and error < limit, (norm, dtype, execution, error)
                    rows.append(dict(source_norm=norm, dtype=str(dtype), prefill=prefill,
                                     execution=execution, max_error_vs_parallel=error,
                                     allocated_bytes=cache.memory_bytes()))
                for execution in ('raw', 'anchored'):
                    rows[-(1 if execution == 'anchored' else 2)]['max_error_vs_material'] = float(
                        (predictions[execution] - predictions['material']).abs().max())
    return rows


@torch.inference_mode()
def quality(batches, long_length):
    text = (ROOT / 'data/tinyshakespeare.txt').read_text()
    text = text[int(.9 * len(text)):]
    rows = []
    for group, stem in [('shallow', 'uniform_matched_screen'),
                        ('deep', 'deeper_uniform_matched_screen')]:
        for seed in range(3):
            path = ROOT / f'work/{stem}_moments_uniform_r16_s{seed}.pt'
            model, checkpoint = load_model(path)
            cfg = checkpoint['args']
            lookup = {v: i for i, v in enumerate(checkpoint['alphabet'])}
            data = torch.tensor([lookup[x] for x in text])
            for length, count, batch in [(cfg['length'], batches, cfg['batch']),
                                          (long_length, 2, 1)]:
                generator = torch.Generator().manual_seed(987654)
                loss, delta, kl, disagreement, maximum = [], [], [], [], 0.
                reference_error = 0.
                for _ in range(count):
                    starts = torch.randint(len(data) - length - 1, (batch,), generator=generator)
                    indices = starts[:, None] + torch.arange(length)
                    tokens, target = data[indices], data[indices + 1]
                    gold = precision_forward(model, tokens)
                    reference_error = max(reference_error, float((gold - model(tokens)).abs().max()))
                    got = precision_forward(model, tokens, torch.bfloat16)
                    gold_nll = F.cross_entropy(gold.flatten(0, 1), target.flatten(), reduction='none')
                    nll = F.cross_entropy(got.flatten(0, 1), target.flatten(), reduction='none')
                    loss.append(float(gold_nll.mean()))
                    delta.append(float((nll - gold_nll).mean()))
                    gold_log, got_log = gold.log_softmax(-1), got.log_softmax(-1)
                    kl.append(float((gold_log.exp() * (gold_log - got_log)).sum(-1).mean()))
                    disagreement.append(float((gold.argmax(-1) != got.argmax(-1)).float().mean()))
                    maximum = max(maximum, float((gold - got).abs().max()))
                assert reference_error < 2e-5, reference_error
                row = dict(group=group, seed=seed, depth=model.depth, length=length,
                           batches=count, batch=batch, tokens=count * batch * length,
                           fp32_nll=sum(loss) / count, bf16_delta_nll=sum(delta) / count,
                           bf16_kl=sum(kl) / count, argmax_disagreement=sum(disagreement) / count,
                           max_logits_error=maximum, fp32_error_vs_legacy=reference_error,
                           batch_delta_nll=delta,
                           checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                rows.append(row)
                print(json.dumps(row), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batches', type=int, default=20)
    ap.add_argument('--long-length', type=int, default=2048)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--out', default='outputs/cache_precision_local.json')
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    start = time.time()
    checks = execution_checks()
    print(json.dumps({'execution_checks': len(checks), 'status': 'passed'}), flush=True)
    rows = quality(args.batches, args.long_length)
    result = dict(manifest=dict(args=vars(args), platform=platform.platform(), torch=torch.__version__,
        device='CPU', cuda_available=torch.cuda.is_available(),
        mps_available=torch.backends.mps.is_available(),
        code_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in [Path(__file__).resolve(), ROOT / 'work/prefix_cache.py',
                                  ROOT / 'work/train_public_memory.py']},
        policy='FP32 weights/private arithmetic/base/controls; BF16 canonical wide source after every layer; FP32 prefix accumulation and attention',
        data_sha256=hashlib.sha256((ROOT / 'data/tinyshakespeare.txt').read_bytes()).hexdigest(),
        limitations='Frozen tiny character models. Long contexts extrapolate training length. BF16 paths are approximate; floating scan order can cross rounding thresholds. Not GPU evidence.',
        elapsed_seconds=time.time() - start), execution_checks=checks, quality=rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
