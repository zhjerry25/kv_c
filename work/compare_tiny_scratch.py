"""Direct same-benchmark comparison. Train only the new candidate from scratch.

Reuse the original's EXISTING from-scratch checkpoints with exactly the same
data, initialization recipe, optimizer, minibatches and 1600-step token budget.
Never copy pretrained tensors into the new candidate or continue old weights.
CPU quality only, not a GPU latency test.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from train_public_memory import Model
from study_cache_precision import load_model
from absorbed_public_bank import AbsorbedBank

ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fresh_candidate(seed, vocab, cfg):
    # Exact original random initialization recipe, without any trained weights.
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
    # The selected random U@P rows initialize the candidate, not a trained U@P.
    candidate = AbsorbedBank(original, 24, 128, 8)
    assert sum(p.numel() for p in original.parameters()) == sum(p.numel() for p in candidate.parameters())
    return candidate


def batch(data, generator, cfg):
    starts = torch.randint(len(data) - cfg['length'] - 1, (cfg['batch'],), generator=generator)
    index = starts[:, None] + torch.arange(cfg['length'])
    return data[index], data[index + 1]


@torch.inference_mode()
def evaluate(model, validation, cfg):
    model.eval()
    generator = torch.Generator().manual_seed(987654)
    legacy = []
    for _ in range(20):
        x, y = batch(validation, generator, cfg)
        legacy.append(float(F.cross_entropy(model(x).flatten(0, 1), y.flatten())))
    # Full validation, every target scored once. Same context-reset rule for both.
    sums, counts = [], []
    for start in range(0, len(validation) - 1, cfg['length']):
        n = min(cfg['length'], len(validation) - 1 - start)
        x = validation[start:start + n][None]
        y = validation[start + 1:start + n + 1]
        loss = F.cross_entropy(model(x)[0], y, reduction='sum')
        sums.append(float(loss)); counts.append(n)
    assert sum(counts) == len(validation) - 1
    nll, legacy_nll = sum(sums) / sum(counts), sum(legacy) / len(legacy)
    return dict(nll=nll, ppl=math.exp(nll), targets=sum(counts),
                legacy_nll=legacy_nll, legacy_ppl=math.exp(legacy_nll),
                chunk_loss_sums=sums, chunk_targets=counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--asset-root', type=Path, default=ROOT)
    ap.add_argument('--seed', type=int, required=True, choices=(0, 1, 2))
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    old_path = args.asset_root / f'weights/pairing/deeper_uniform_matched_screen_moments_uniform_r16_s{args.seed}.pt'
    # Load metadata now; no checkpoint tensor enters fresh_candidate().
    old = torch.load(old_path, weights_only=False)
    cfg, alphabet = old['args'], old['alphabet']
    assert cfg['steps'] == 1600 and cfg['batch'] == 16 and cfg['length'] == 256
    assert cfg['d'] == 128 and cfg['depth'] == 8 and not cfg['source_norm']
    assert cfg['match_public_init'] and not cfg['zero_public'] and old['row']['seed'] == args.seed
    raw = (args.asset_root / 'data/tinyshakespeare.txt').read_bytes()
    assert sha(raw) == '86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed'
    lookup = {v: i for i, v in enumerate(alphabet)}
    data = torch.tensor([lookup[ch] for ch in raw.decode()])
    train, validation = data[:int(.9 * len(data))], data[int(.9 * len(data)):]
    candidate = fresh_candidate(args.seed, len(alphabet), cfg)
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=3e-4, betas=(.9, .95), weight_decay=.1)
    generator = torch.Generator().manual_seed(10000 + args.seed)
    metadata = dict(seed=args.seed, config=cfg, data_sha256=sha(raw),
        original_checkpoint_sha256=sha(old_path.read_bytes()), torch=torch.__version__,
        code_sha256={name: sha((ROOT / 'work' / name).read_bytes()) for name in
                     ('compare_tiny_scratch.py', 'train_public_memory.py', 'absorbed_public_bank.py')},
        tokens=cfg['steps'] * cfg['batch'] * cfg['length'],
        note='Same original benchmark and fixed budget; new candidate from scratch, old from-scratch baseline reused. '
             'Validation previously exposed. No independent test or convergence/equivalence claim.')
    path = ROOT / 'work' / f'direct_tiny_bank24_s{args.seed}.pt'
    first, curve = 0, []
    if path.exists():
        saved = torch.load(path, weights_only=False)
        assert saved['metadata'] == metadata
        candidate.load_state_dict(saved['state_dict']); optimizer.load_state_dict(saved['optimizer'])
        generator.set_state(saved['sampler_rng']); torch.set_rng_state(saved['torch_rng'])
        first, curve = saved['step'], saved['curve']
    started = time.monotonic()
    for step in range(first, cfg['steps']):
        candidate.train()
        x, y = batch(train, generator, cfg)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(candidate(x).flatten(0, 1), y.flatten())
        assert torch.isfinite(loss), ('loss', args.seed, step)
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.))
        assert math.isfinite(grad), ('gradient', args.seed, step)
        lr = 3e-4 * min((step + 1) / 30, 1) * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / cfg['steps'])))
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.step()
        if step % 100 == 0 or step + 1 == cfg['steps']:
            row = dict(step=step + 1, loss=float(loss.detach()), grad=grad, lr=lr,
                       elapsed_seconds=time.monotonic() - started)
            curve.append(row); print(json.dumps(dict(seed=args.seed, **row)), flush=True)
        if (step + 1) % 400 == 0:
            # Full state: interruption never turns this into a fresh optimizer continuation.
            temporary = path.with_suffix('.partial')
            torch.save(dict(metadata=metadata, step=step + 1, state_dict=candidate.state_dict(),
                optimizer=optimizer.state_dict(), sampler_rng=generator.get_state(),
                torch_rng=torch.get_rng_state(), curve=curve), temporary)
            temporary.replace(path)
    original, _ = load_model(old_path)
    result = dict(metadata=metadata, curve=curve, original=evaluate(original, validation, cfg),
                  bank24=evaluate(candidate, validation, cfg))
    result['legacy_original_reproduction_error'] = result['original']['legacy_nll'] - old['row']['validation_loss']
    result['delta_nll'] = result['bank24']['nll'] - result['original']['nll']
    result['ppl_relative_percent'] = math.expm1(result['delta_nll']) * 100
    output = ROOT / 'outputs' / f'direct_tiny_compare_s{args.seed}.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(seed=args.seed, output=str(output), delta_nll=result['delta_nll'],
                         ppl_relative_percent=result['ppl_relative_percent'])), flush=True)


if __name__ == '__main__':
    main()
