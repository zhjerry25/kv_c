"""Complete MLA matrix: native and both cyclic RoPE modes, three paired seeds.

One command verifies, trains all 15 models, evaluates and writes a summary.
Workers block on subprocess completion; no process, log or result polling.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import math
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from factorized_memory import AttentionConfig, paired_model
from run_attention_adaptation import atomic_save, describe, initial_hashes, paths, run_one, sha

VARIANTS = {
    'native': ('native', True),
    'shared_f25': ('f25', True),
    'shared_f375': ('f375', True),
    'independent_f25': ('f25', False),
    'independent_f375': ('f375', False),
}


def run_args(args, variant, seed):
    arm, share_rope = VARIANTS[variant]
    return argparse.Namespace(attention='mla', arm=arm, seed=seed,
        tag=f'{args.tag}/{variant}', steps=1600, batch=16, length=256,
        threads=args.threads, skip_ood=False), share_rope


def result_path(args, variant, seed):
    worker, _ = run_args(args, variant, seed)
    return paths(worker)[0]


def check_pairing(args):
    torch.set_num_threads(args.threads)
    rows = []
    for seed in range(3):
        native = paired_model(AttentionConfig(vocab=65, cache_mode='native'), seed)
        reference = initial_hashes(native)
        for fraction in ('1/4', '3/8'):
            cfg = AttentionConfig(vocab=65, write_fraction=fraction)
            shared = paired_model(cfg, seed)
            independent = paired_model(replace(cfg, share_rope=False), seed)
            for model in (shared, independent):
                own = initial_hashes(model)
                assert all(own[k] == reference[k] for k in ('private', 'bottom'))
            assert torch.equal(shared.initial[0].weight, independent.initial[0].weight)
            for l, layer in enumerate(independent.blocks):
                assert torch.equal(layer.rope_key.weight, native.blocks[l].pub[1].weight)
                if l:
                    assert torch.equal(layer.pub[0].weight, shared.blocks[l].pub[0].weight)
                assert all(bank == l for bank, _, position, width in independent.layout.segments[l]
                           if position + width > cfg.kv_latent_dim)
            rows.append(dict(seed=seed, fraction=fraction, private_and_bottom_exact=True,
                             latent_publications_exact=True, rope_projections_match_native=True,
                             rope_latest_writer_is_current_layer=True))
    atomic_save(dict(status='passed', rows=rows), ROOT / 'outputs' / args.tag / 'pairing.json')


def summarize(args):
    rows = {(variant, seed): json.loads(result_path(args, variant, seed).read_text())
            for variant in VARIANTS for seed in range(3)}
    first = rows['native', 0]['metadata']
    for (variant, seed), row in rows.items():
        assert row['status'] == 'completed' and row['save_load_exact']
        assert sha(ROOT / row['weight']) == row['weight_sha256']
        metadata = row['metadata']
        for key in ('steps', 'batch', 'length', 'tokens', 'data_sha256', 'environment',
                    'optimizer', 'ood', 'code_sha256', 'training_loop_sha256'):
            assert metadata[key] == first[key], (variant, seed, key)
        for key in ('private', 'bottom'):
            assert metadata['initialization_sha256'][key] == rows['native', seed][
                'metadata']['initialization_sha256'][key]
    summary = {}
    for variant in VARIANTS:
        group = [rows[variant, seed] for seed in range(3)]
        entry = {k: group[0][k] for k in ('parameters', 'history_scalars_per_token',
                                         'history_ratio_to_native')}
        entry.update(evaluation={}, ood={})
        comparisons = {'native': 'native'}
        if variant.startswith('independent_'):
            comparisons['shared_rope'] = variant.replace('independent_', 'shared_')
        for length in ('256', '512', '1024'):
            values = [row['evaluation'][length]['nll'] for row in group]
            evaluation = dict(nll=describe(values), geometric_mean_ppl=math.exp(sum(values) / 3))
            for label, reference in comparisons.items():
                evaluation[f'paired_delta_vs_{label}'] = describe([
                    values[seed] - rows[reference, seed]['evaluation'][length]['nll'] for seed in range(3)])
            entry['evaluation'][length] = evaluation
        for item in group[0]['ood']:
            key = (item['corpus'], item['length'], item['precision'])
            def metric(row):
                return next(r['evaluation']['nll'] for r in row['ood']
                            if (r['corpus'], r['length'], r['precision']) == key)
            values = [metric(row) for row in group]
            evaluation = dict(nll=describe(values))
            for label, reference in comparisons.items():
                evaluation[f'paired_delta_vs_{label}'] = describe([
                    values[seed] - metric(rows[reference, seed]) for seed in range(3)])
            entry['ood']['/'.join(map(str, key))] = evaluation
        summary[variant] = entry
    record = dict(status='completed', runs=15, paired_initialization_verified=True,
        metadata=first, summary=summary,
        result_sha256={str(result_path(args, variant, seed).relative_to(ROOT)):
                       sha(result_path(args, variant, seed)) for variant, seed in rows},
        training_seconds=sum(row['training_seconds'] for row in rows.values()),
        evaluation_seconds=sum(row['evaluation_seconds'] for row in rows.values()),
        note='All 15 models freshly trained with Query/KV RMSNorm (eps=1e-6). '
             'Raw KV latents are cached and normalized in full on read before K/V up-projection. '
             'Three-seed descriptive t intervals, df=2; validation already exposed. '
             'CPU process time is not wall time or inference latency.')
    atomic_save(record, ROOT / 'outputs' / args.tag / 'summary.json')
    print(json.dumps(dict(status='all_complete', runs=15,
                          summary=f'outputs/{args.tag}/summary.json')), flush=True)


def run_all(args):
    output = ROOT / 'outputs' / args.tag
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'verification.log').open('w') as stream:
        subprocess.run([sys.executable, str(ROOT / 'work/verify_attention_adaptation.py'),
                        '--threads', str(args.threads), '--output', str(output / 'verification.json')],
                       cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    check_pairing(args)
    def worker(variant, seed):
        worker_args, _ = run_args(args, variant, seed)
        log = paths(worker_args)[3]
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), '--variant', variant,
                   '--seed', str(seed), '--tag', args.tag, '--threads', str(args.threads)]
        with log.open('a') as stream:
            completed = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f'{variant} seed {seed} failed:\n{log.read_text()[-5000:]}')
    print(json.dumps(dict(status='queued', runs=15, cpu_workers=args.jobs)), flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(worker, variant, seed) for seed in range(3) for variant in VARIANTS]
        for future in futures:
            future.result()
    summarize(args)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--summarize-only', action='store_true')
    ap.add_argument('--variant', choices=tuple(VARIANTS))
    ap.add_argument('--seed', type=int, choices=range(3))
    ap.add_argument('--jobs', type=int, choices=(1, 2), default=2)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--tag', default='mla_standard')
    args = ap.parse_args()
    if args.threads < 1 or Path(args.tag).name != args.tag or args.tag in ('', '.', '..'):
        ap.error('Positive threads and a directory-name tag required')
    if args.summarize_only:
        summarize(args)
    elif args.all:
        run_all(args)
    elif args.variant is not None and args.seed is not None:
        worker_args, share_rope = run_args(args, args.variant, args.seed)
        run_one(worker_args, share_rope=share_rope, extra_source_files=('work/run_mla_suite.py',))
    else:
        ap.error('Use --all, --summarize-only, or --variant/--seed')


if __name__ == '__main__':
    main()
