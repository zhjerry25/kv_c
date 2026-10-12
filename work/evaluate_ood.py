"""External-text evaluation for packaged standalone weights.

Trimmed from the locked evaluator: same corpora, normalization, frozen
exact-overlap exclusions and fixed chunk resets from
outputs/clean_ood_manifest.json; only the model loading path changed
(packaged standalone CyclicMemory / PublicMemory weights). Scores produced
here are exploratory: the locked selection used the original four groups;
none of these rows were part of that lock.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cyclic_memory import CyclicMemory
from public_memory import PublicMemory


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_weight(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    cls = PublicMemory if 'mix_budget' in saved['config'] else CyclicMemory
    return cls.load(path).eval().requires_grad_(False), saved['metadata']


@torch.inference_mode()
def score(model, data, length, excluded, stored=False):
    total, count = 0., 0
    for chunk, start in enumerate(range(0, len(data) - 1, length)):
        if chunk in excluded:
            continue
        n = min(length, len(data) - start - 1)
        tokens = data[start:start + n][None]
        logits = model.forward_stored(tokens) if stored else model(tokens)
        total += float(F.cross_entropy(logits[0], data[start + 1:start + n + 1], reduction='sum'))
        count += n
    assert count > 0
    return dict(nll=total / count, ppl=math.exp(total / count), targets=count, length=length,
                precision='BF16 physical KV, FP32 weights/private' if stored else 'FP32 canonical architecture')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, choices=range(3), required=True)
    ap.add_argument('--groups', nargs='*', default=['none_w24', 'none_w16', 'prefix_w24'])
    ap.add_argument('--modes', nargs='*', default=['fp32', 'bf16'])
    ap.add_argument('--check-recorded', action='store_true')
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--tag', default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    manifest = json.loads((ROOT / 'outputs/clean_ood_manifest.json').read_text())
    rows = []
    for group in args.groups:
        path = ROOT / f'weights/{group}_s{args.seed}.pt'
        model, meta = load_weight(path)
        alphabet = meta['alphabet']
        assert ''.join(alphabet) == manifest['alphabet']
        lookup = {ch: i for i, ch in enumerate(alphabet)}
        for corpus in manifest['corpora']:
            text_path = ROOT / 'data/ood' / Path(corpus['file']).name
            assert sha(text_path) == corpus['selected_sha256']
            data = torch.tensor([lookup[ch] for ch in text_path.read_text()])
            for length in (256, 1024):
                for stored in [m == 'bf16' for m in args.modes]:
                    evaluation = score(model, data, length, corpus['exclude_chunks'][str(length)], stored)
                    row = dict(group=group, seed=args.seed, corpus=corpus['name'], length=length,
                               stored=stored, weight_sha256=sha(path), evaluation=evaluation)
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    if args.check_recorded:
        recorded = json.loads((ROOT / f'outputs/clean_ood_s{args.seed}.json').read_text())['rows']
        worst = 0.
        for row in rows:
            if row['group'] != 'prefix_w24':
                continue
            ref = next(r for r in recorded if r['group'] == 'local_prefix24' and r['corpus'] == row['corpus']
                       and r['length'] == row['length'] and r['stored'] == row['stored'])
            diff = abs(ref['evaluation']['nll'] - row['evaluation']['nll'])
            worst = max(worst, diff)
            assert diff < 1e-4, (row['group'], row['corpus'], row['length'], row['stored'], diff)
        print(json.dumps(dict(check_recorded='prefix_w24 vs local_prefix24', max_abs_nll_diff=worst)), flush=True)
    tag = args.tag or f'ood_packaged_s{args.seed}'
    result = dict(status='completed', seed=args.seed, groups=args.groups, torch=torch.__version__,
        manifest_sha256=sha(ROOT / 'outputs/clean_ood_manifest.json'), rows=rows,
        code_sha256={p: sha(ROOT / p) for p in ('work/evaluate_ood.py', 'cyclic_memory.py',
            'public_memory.py', 'narrow_cache.py')},
        note='Same corpora/exclusions/chunking as the locked evaluation; only standalone weight '
             'loading changed. Packaged-weight scores are exploratory and were not part of the '
             'locked selection. The training/selection validation was already exposed.')
    (ROOT / f'outputs/{tag}.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
