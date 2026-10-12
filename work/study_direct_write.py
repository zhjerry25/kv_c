"""Short paired screen: residual writes versus Z = N(H)W, from scratch."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cyclic_memory import Config, CyclicMemory
from verify_cyclic_memory import decode_check


@torch.inference_mode()
def evaluate(model, data, length):
    model.eval()
    total, count = 0., 0
    for start in range(0, len(data) - 1, length):
        n = min(length, len(data) - start - 1)
        total += float(F.cross_entropy(model(data[start:start+n][None])[0],
            data[start+1:start+n+1], reduction='sum'))
        count += n
    return dict(nll=total / count, targets=count)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--length', type=int, default=128)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--comparison', choices=('write', 'init'), default='write')
    ap.add_argument('--baseline', type=Path, help='Reuse a matched prior direct-write arm for the init screen')
    ap.add_argument('--output', type=Path, default=ROOT / 'outputs/direct_write_short.json')
    args = ap.parse_args()
    if min(args.steps, args.batch, args.length, args.threads) < 1:
        ap.error('steps, batch, length and threads must be positive')
    if args.output.exists():
        ap.error('output already exists; choose a new --output')
    if args.baseline and args.comparison != 'init':
        ap.error('--baseline is only valid with --comparison init')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    raw = (ROOT / 'data/tinyshakespeare.txt').read_bytes()
    alphabet = sorted(set(raw.decode()))
    lookup = {ch: i for i, ch in enumerate(alphabet)}
    data = torch.tensor([lookup[ch] for ch in raw.decode()])
    split = int(.9 * len(data))
    train, validation = data[:split], data[split:]
    cfg = Config(vocab=len(alphabet), pub_init_std=.0016)
    torch.manual_seed(args.seed)
    initial = CyclicMemory(cfg).state_dict()
    result = dict(config=asdict(cfg), args={k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()}, torch=torch.__version__,
        data_sha256=hashlib.sha256(raw).hexdigest(),
        code_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ('cyclic_memory.py', 'work/study_direct_write.py',
                         'work/verify_cyclic_memory.py', 'narrow_cache.py')},
        tokens_per_arm=args.steps * args.batch * args.length,
        note='Single-seed exploratory short screen, standalone random initialization; '
             'paired tensors/minibatches/AdamW/schedule, fixed final step. '
             'Full 90/10 validation split already exposed; no convergence claim.', arms={})
    arms = (('residual', True, .0016), ('direct', False, .0016)) if args.comparison == 'write' else (
        ('small_init', False, .0016), ('large_init', False, .02))
    for name, residual, std in arms:
        if name == 'small_init' and args.baseline:
            previous = json.loads(args.baseline.read_text())
            assert previous['data_sha256'] == result['data_sha256'] and previous['torch'] == result['torch']
            assert all(previous['args'][k] == getattr(args, k)
                       for k in ('seed', 'steps', 'batch', 'length', 'threads'))
            assert previous['config'] == {k: v for k, v in asdict(cfg).items() if k != 'pub_init_std'}
            result['arms'][name] = previous['arms']['direct']
            result['reused_baseline'] = dict(path=str(args.baseline),
                sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest())
            continue
        torch.manual_seed(args.seed)
        model = CyclicMemory(Config(vocab=len(alphabet), residual_write=residual, pub_init_std=std))
        if args.comparison == 'write':
            model.load_state_dict(initial)
        for k, v in model.state_dict().items():
            if k.endswith('.pub.weight'):
                torch.testing.assert_close(v, initial[k] * (std / .0016), rtol=2e-7, atol=1e-8)
            else:
                assert torch.equal(v, initial[k]), k
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4,
            betas=(.9, .95), weight_decay=.1)
        generator = torch.Generator().manual_seed(10000 + args.seed)
        started, curve = time.monotonic(), []
        for step in range(args.steps):
            model.train()
            starts = torch.randint(len(train) - args.length - 1, (args.batch,), generator=generator)
            index = starts[:, None] + torch.arange(args.length)
            loss = F.cross_entropy(model(train[index]).flatten(0, 1), train[index+1].flatten())
            assert torch.isfinite(loss), (name, step, 'loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
            assert math.isfinite(grad), (name, step, 'gradient')
            lr = 3e-4 * min((step+1)/30, 1) * (.1 + .9*.5*(1+math.cos(math.pi*step/args.steps)))
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.step()
            if step == 0 or (step+1) % 50 == 0 or step+1 == args.steps:
                point = dict(step=step+1, loss=float(loss.detach()), grad=grad,
                             seconds=time.monotonic()-started)
                curve.append(point)
                print(json.dumps(dict(arm=name, **point)), flush=True)
        training_seconds = time.monotonic() - started
        model.eval().requires_grad_(False)
        tokens = validation[:33][None]
        checks = [decode_check(model, tokens, dtype, 9, 7)
                  for dtype in (torch.float32, torch.bfloat16)]
        row = dict(parameters=sum(p.numel() for p in model.parameters()),
            training_seconds=training_seconds, curve=curve, checks=checks,
            evaluation={str(n): evaluate(model, validation, n) for n in (256, 1024)})
        result['arms'][name] = row
        print(json.dumps(dict(arm=name, evaluation=row['evaluation'])), flush=True)
    delta_key = 'delta_nll_direct_minus_residual' if args.comparison == 'write' else 'delta_nll_large_minus_small'
    result[delta_key] = {str(n):
        result['arms'][arms[1][0]]['evaluation'][str(n)]['nll'] -
        result['arms'][arms[0][0]]['evaluation'][str(n)]['nll'] for n in (256, 1024)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(output=str(args.output),
        delta_nll=result[delta_key])), flush=True)


if __name__ == '__main__':
    main()
