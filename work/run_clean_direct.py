"""Full-budget, resumable comparison of support-preserving public writes.

Every run starts from the exact original random initialization, never from the
400-step screen or pretrained tensors. The old 1600-step baselines are reused.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from absorbed_public_bank import AbsorbedBank, write_indices
from compare_tiny_scratch import batch, fresh_candidate
from local_public_bank import LocalBank
from study_local_public_bank import evaluate
from train_public_memory import Model

ROOT = Path(__file__).resolve().parents[1]


def fresh_bank(seed, vocab, cfg, width):
    """General-width form of fresh_candidate; private tensors match exactly."""
    if width == 24:
        return fresh_candidate(seed, vocab, cfg)
    torch.manual_seed(seed)
    reference = Model(vocab, 'standard', 0, cfg['d'], cfg['depth'], cfg['source_norm'])
    torch.manual_seed(100000 + seed)
    original = Model(vocab, 'moments_uniform', 16, cfg['d'], cfg['depth'], cfg['source_norm'])
    own = original.state_dict()
    original.load_state_dict({k: v for k, v in reference.state_dict().items()
        if k in own and own[k].shape == v.shape}, strict=False)
    with torch.no_grad():
        original.initial.weight.copy_(reference.blocks[0].kv.weight)
        original.initial_norm.weight.copy_(reference.blocks[0].n1.weight)
    torch.manual_seed(100000 + seed)
    public = Model(vocab, 'linear', 16, cfg['d'], cfg['depth'], cfg['source_norm'])
    with torch.no_grad():
        for layer, ref in zip(original.blocks, public.blocks):
            if layer.pub is not None:
                layer.pub.weight.copy_(ref.pub.weight)
                layer.expand.weight.copy_(ref.expand.weight)
    return AbsorbedBank(original, width, 128, 8)


def atomic_save(value, path, binary=False):
    temporary = path.with_suffix('.partial')
    if binary:
        torch.save(value, temporary)
    else:
        temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--asset-root', type=Path, default=ROOT,
        help='Package root; pairing assets in weights/pairing, data in data/ (adapted for the package layout)')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--variant', choices=('anchor', 'prefix', 'none'), required=True)
    ap.add_argument('--width', type=int, default=24)
    ap.add_argument('--steps', type=int, default=1600)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--length', type=int, default=256)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--tag', default='clean_direct')
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    old_path = args.asset_root / f'weights/pairing/deeper_uniform_matched_screen_moments_uniform_r16_s{args.seed}.pt'
    old = torch.load(old_path, weights_only=False)
    cfg = dict(d=128, depth=8, source_norm=False, steps=args.steps, batch=args.batch, length=args.length)
    for field in ('d', 'depth', 'source_norm'):
        assert old['args'][field] == cfg[field]
    alphabet = old['alphabet']
    raw = (args.asset_root / 'data/tinyshakespeare.txt').read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == '86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed'
    lookup = {v: i for i, v in enumerate(alphabet)}
    data = torch.tensor([lookup[ch] for ch in raw.decode()])
    train, validation = data[:int(.9 * len(data))], data[int(.9 * len(data)):]
    bank = fresh_bank(args.seed, len(alphabet), cfg, args.width)
    counts = torch.zeros(bank.c, dtype=torch.long)
    for l in range(1, bank.depth):
        counts[write_indices(bank.c, args.width, l)] += 1
    lam = .25 / int(counts.max()) if args.variant != 'none' else 0.
    model = LocalBank(bank, args.variant, lam=lam)
    del bank, old
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(.9, .95), weight_decay=.1)
    generator = torch.Generator().manual_seed(10000 + args.seed)
    stem = f'{args.tag}_{args.variant}_w{args.width}_s{args.seed}'
    checkpoint_path = ROOT / 'work' / f'{stem}.pt'
    output = ROOT / 'outputs' / f'{stem}.json'
    metadata = dict(args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        config=cfg, seed=args.seed, variant=args.variant, width=args.width, lam=lam,
        coordinate_write_counts=counts.tolist(), tokens=args.steps * args.batch * args.length,
        data_sha256=digest, torch=torch.__version__,
        code_sha256={name: hashlib.sha256((ROOT / 'work' / name).read_bytes()).hexdigest()
            for name in ('run_clean_direct.py', 'local_public_bank.py', 'absorbed_public_bank.py',
                         'compare_tiny_scratch.py', 'train_public_memory.py')},
        note='Fixed final budget, paired random initialization/minibatches/optimizer; validation already exposed. '
             'No pretraining, continuation from a screen, distillation or best-checkpoint selection.')
    first, curve, elapsed = 0, [], 0.
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, weights_only=False)
        assert saved['metadata'] == metadata, 'Use a new tag for different settings or code'
        model.load_state_dict(saved['state_dict'])
        optimizer.load_state_dict(saved['optimizer'])
        generator.set_state(saved['sampler_rng'])
        torch.set_rng_state(saved['torch_rng'])
        first, curve, elapsed = saved['step'], saved['curve'], saved['elapsed_seconds']
    if output.exists():
        result = json.loads(output.read_text())
        assert result['metadata'] == metadata and first == args.steps
        print(json.dumps(dict(status='already_complete', output=str(output))), flush=True)
        return
    started = time.monotonic()
    for step in range(first, args.steps):
        model.train()
        x, y = batch(train, generator, cfg)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        assert torch.isfinite(loss), ('loss', stem, step)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
        assert math.isfinite(grad), ('gradient', stem, step)
        lr = 3e-4 * min((step + 1) / 30, 1) * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / args.steps)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.step()
        if step % 100 == 0 or step + 1 == args.steps:
            point = dict(step=step + 1, loss=float(loss.detach()), grad=grad, lr=lr,
                         elapsed_seconds=elapsed + time.monotonic() - started)
            curve.append(point)
            print(json.dumps(dict(run=stem, **point)), flush=True)
        if (step + 1) % 100 == 0 or step + 1 == args.steps:
            atomic_save(dict(metadata=metadata, step=step + 1, state_dict=model.state_dict(),
                alphabet=alphabet, optimizer=optimizer.state_dict(), sampler_rng=generator.get_state(),
                torch_rng=torch.get_rng_state(), curve=curve,
                elapsed_seconds=elapsed + time.monotonic() - started), checkpoint_path, binary=True)
    result = dict(metadata=metadata, parameters=sum(p.numel() for p in model.parameters()), curve=curve,
                  evaluation={str(length): evaluate(model, validation, length) for length in (256, 512, 1024)})
    atomic_save(result, output)
    print(json.dumps(dict(run=stem, status='complete', evaluation=result['evaluation'])), flush=True)


if __name__ == '__main__':
    main()
