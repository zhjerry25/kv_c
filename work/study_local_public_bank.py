"""Matched local-transport screen; frozen ablation and fresh short training.

Old validation is already exposed. All outcomes are exploratory, and a short
training screen does not replace the existing 1600-step direct comparison.
No CPU time is used to infer GPU latency.
"""
import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from absorbed_public_bank import AbsorbedBank
from compare_tiny_scratch import batch, fresh_candidate
from local_public_bank import LocalBank
from study_cache_precision import load_model

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('global', 'local_anchor', 'local_prefix', 'none')


def variants(global_bank):
    return dict(global_=global_bank,
        local_anchor=LocalBank(global_bank, 'anchor'),
        local_prefix=LocalBank(global_bank, 'prefix'),
        none=LocalBank(global_bank, 'none'))


@torch.inference_mode()
def evaluate(model, data, length=256):
    model.eval()
    total, count = 0., 0
    for start in range(0, len(data) - 1, length):
        n = min(length, len(data) - 1 - start)
        logits = model(data[start:start + n][None])[0]
        total += float(F.cross_entropy(logits, data[start + 1:start + n + 1], reduction='sum'))
        count += n
    return dict(nll=total / count, ppl=math.exp(total / count), targets=count, length=length)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--asset-root', type=Path, default=ROOT)
    ap.add_argument('--mode', choices=('frozen', 'train'), required=True)
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--length', type=int, default=128)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--tag', default='local_bank')
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    raw = (args.asset_root / 'data/tinyshakespeare.txt').read_bytes()
    alphabet = sorted(set(raw.decode()))
    lookup = {v: i for i, v in enumerate(alphabet)}
    data = torch.tensor([lookup[ch] for ch in raw.decode()])
    split = int(.9 * len(data))
    train, validation = data[:split], data[split:]
    cfg = dict(d=128, depth=8, source_norm=False, steps=args.steps, batch=args.batch, length=args.length)
    metadata = dict(args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        torch=torch.__version__, data_sha256=hashlib.sha256(raw).hexdigest(),
        code_sha256={name: hashlib.sha256((ROOT / 'work' / name).read_bytes()).hexdigest()
            for name in ('local_public_bank.py', 'study_local_public_bank.py', 'absorbed_public_bank.py',
                         'train_public_memory.py', 'compare_tiny_scratch.py')},
        config=cfg, note='Exploratory exposed-validation screen; same selected random U@P initialization, '
        'minibatches, optimizer and schedule across all arms. Fixed final step, no checkpoint selection. '
        'Frozen changes measure sensitivity, not trained architecture quality.')
    rows = []
    output = ROOT / 'outputs' / f'{args.tag}_{args.mode}.json'
    if output.exists():
        previous = json.loads(output.read_text())
        if previous['metadata'] != metadata:
            raise ValueError('Existing result has different metadata; use a new tag')
        rows = previous['rows']
    for seed in map(int, args.seeds.split(',')):
        if args.mode == 'frozen':
            teacher, _ = load_model(args.asset_root / f'weights/pairing/deeper_uniform_matched_screen_moments_uniform_r16_s{seed}.pt')
            bank = AbsorbedBank(teacher, 24)
            saved = torch.load(ROOT / 'work' / f'direct_tiny_bank24_s{seed}.pt', weights_only=False)
            bank.load_state_dict(saved['state_dict'])
        else:
            bank = fresh_candidate(seed, len(alphabet), cfg)
        models = variants(bank)
        models['global'] = models.pop('global_')
        for name in VARIANTS:
            if any(row['seed'] == seed and row['variant'] == name for row in rows):
                continue
            model = models[name]
            started, curve = time.monotonic(), []
            if args.mode == 'train':
                optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(.9, .95), weight_decay=.1)
                generator = torch.Generator().manual_seed(10000 + seed)
                for step in range(args.steps):
                    model.train()
                    x, y = batch(train, generator, cfg)
                    loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
                    assert torch.isfinite(loss)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
                    assert math.isfinite(grad)
                    lr = 3e-4 * min((step + 1) / 30, 1) * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / args.steps)))
                    for group in optimizer.param_groups:
                        group['lr'] = lr
                    optimizer.step()
                    if step % 100 == 0 or step + 1 == args.steps:
                        point = dict(step=step + 1, loss=float(loss.detach()), grad=grad,
                                     seconds=time.monotonic() - started)
                        curve.append(point)
                        print(json.dumps(dict(seed=seed, variant=name, **point)), flush=True)
                torch.save(dict(state_dict=model.state_dict(), metadata=metadata, alphabet=alphabet,
                    variant=name, seed=seed, step=args.steps, optimizer=optimizer.state_dict(),
                    sampler_rng=generator.get_state(), torch_rng=torch.get_rng_state()),
                    ROOT / 'work' / f'{args.tag}_{name}_s{seed}.pt')
            row = dict(seed=seed, variant=name, parameters=sum(p.numel() for p in model.parameters()),
                       lam=bank.lam, short=evaluate(model, validation, 256),
                       long=evaluate(model, validation, 1024), curve=curve,
                       seconds=time.monotonic() - started)
            rows.append(row)
            output.write_text(json.dumps(dict(metadata=metadata, rows=rows), indent=2) + '\n')
            print(json.dumps({k: v for k, v in row.items() if k != 'curve'}), flush=True)


if __name__ == '__main__':
    main()
