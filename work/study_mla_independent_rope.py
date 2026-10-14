"""Six fresh MLA runs: cyclic KV latent, ordinary per-layer full RoPE Key.

Reuse the previous native MLA and shared-RoPE results as fixed references.
Only the RoPE path changes; latent publications, private initialization,
minibatches, optimizer and the full 1600-step budget stay paired.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from factorized_memory import AttentionConfig, paired_model
from run_attention_adaptation import (ARMS, DATA_SHA256, TRAINING_LOOP_V1, atomic_save,
                                      describe, initial_hashes, paths, run_one, sha,
                                      tensor_digest, training_loop_sha256)


def references(args):
    result = {}
    for arm in ('native', 'f25', 'f375'):
        for seed in range(3):
            path = ROOT / 'outputs' / args.reference_tag / f'mla_{arm}_s{seed}.json'
            row = json.loads(path.read_text())
            metadata = row['metadata']
            assert row['status'] == 'completed' and metadata['data_sha256'] == DATA_SHA256
            assert metadata['steps'] == 1600 and metadata['batch'] == 16 and metadata['length'] == 256
            assert metadata.get('training_loop_sha256', TRAINING_LOOP_V1) == training_loop_sha256() == TRAINING_LOOP_V1
            assert sha(ROOT / row['weight']) == row['weight_sha256']
            result[arm, seed] = (path, row)
    return result


def pairing(args):
    torch.set_num_threads(args.threads)
    refs = references(args)
    rows = []
    for seed in range(3):
        base = AttentionConfig(vocab=65, attention='mla')
        native = paired_model(replace(base, cache_mode='native'), seed)
        assert initial_hashes(native) == refs['native', seed][1]['metadata']['initialization_sha256']
        for arm, fraction in (('f25', '1/4'), ('f375', '3/8')):
            shared = paired_model(replace(base, write_fraction=fraction), seed)
            independent = paired_model(replace(base, write_fraction=fraction, share_rope=False), seed)
            own = initial_hashes(independent)
            old = refs[arm, seed][1]['metadata']['initialization_sha256']
            assert initial_hashes(shared) == old
            assert own['private'] == old['private'] and own['bottom'] == old['bottom']
            assert torch.equal(independent.initial[0].weight, shared.initial[0].weight)
            for l, layer in enumerate(independent.blocks):
                assert torch.equal(layer.rope_key.weight, native.blocks[l].pub[1].weight)
                if l:
                    assert torch.equal(layer.pub[0].weight, shared.blocks[l].pub[0].weight)
                rope_start = independent.config.kv_latent_dim
                assert all(bank == l for bank, _, position, width in independent.layout.segments[l]
                           if position + width > rope_start)
            rows.append(dict(seed=seed, arm=arm, initialization_sha256=own,
                             private_and_bottom_match_previous=True, latent_publications_match_previous=True,
                             all_rope_weights_match_native=True, rope_latest_writer_is_current_layer=True,
                             private_rope_sha256=tensor_digest((str(l), layer.rope_key.weight)
                                 for l, layer in enumerate(independent.blocks)),
                             source_width=independent.config.source_width, write_width=independent.config.write_width))
    record = dict(status='passed', rows=rows, training_loop_sha256=training_loop_sha256(),
                  references={f'{arm}_s{seed}': dict(path=str(path.relative_to(ROOT)), sha256=sha(path),
                      weight=row['weight'], weight_sha256=row['weight_sha256'])
                      for (arm, seed), (path, row) in refs.items()})
    atomic_save(record, ROOT / 'outputs' / args.tag / 'pairing.json')
    return record


def summarize(args):
    refs = references(args)
    rows = {(arm, seed): json.loads(paths(args, 'mla', arm, seed)[0].read_text())
            for arm in ('f25', 'f375') for seed in range(3)}
    assert len(rows) == 6 and all(row['status'] == 'completed' for row in rows.values())
    summary = {}
    for arm in ('f25', 'f375'):
        group = [rows[arm, seed] for seed in range(3)]
        entry = dict(parameters=group[0]['parameters'], history_scalars_per_token=group[0]['history_scalars_per_token'],
                     history_ratio_to_native=group[0]['history_ratio_to_native'], evaluation={}, ood={})
        for row in group:
            metadata = row['metadata']
            assert metadata['config']['share_rope'] is False
            for key in ('steps', 'batch', 'length', 'tokens', 'data_sha256', 'optimizer', 'sampler_seed', 'environment'):
                assert metadata[key] == refs['native', metadata['seed']][1]['metadata'][key], key
            assert metadata['training_loop_sha256'] == TRAINING_LOOP_V1
            assert metadata['code_sha256'] == group[0]['metadata']['code_sha256']
            for key in ('private', 'bottom'):
                assert metadata['initialization_sha256'][key] == refs[arm, metadata['seed']][1][
                    'metadata']['initialization_sha256'][key]
        for length in ('256', '512', '1024'):
            values = [row['evaluation'][length]['nll'] for row in group]
            entry['evaluation'][length] = dict(nll=describe(values), geometric_mean_ppl=math.exp(statistics.mean(values)),
                paired_delta_vs_native=describe([values[seed] - refs['native', seed][1]['evaluation'][length]['nll']
                                                 for seed in range(3)]),
                paired_delta_vs_shared_rope=describe([values[seed] - refs[arm, seed][1]['evaluation'][length]['nll']
                                                      for seed in range(3)]))
        for key in [(r['corpus'], r['length'], r['precision']) for r in group[0]['ood']]:
            def metric(row):
                return next(r['evaluation']['nll'] for r in row['ood'] if (
                    r['corpus'], r['length'], r['precision']) == key)
            values = [metric(row) for row in group]
            entry['ood']['/'.join(map(str, key))] = dict(nll=describe(values),
                paired_delta_vs_native=describe([values[seed] - metric(refs['native', seed][1]) for seed in range(3)]),
                paired_delta_vs_shared_rope=describe([values[seed] - metric(refs[arm, seed][1]) for seed in range(3)]))
        summary[arm] = entry
    record = dict(status='completed', new_training_runs=6, reused_native_runs=3, summary=summary,
                  protocol=next(iter(rows.values()))['metadata'],
                  reference_tag=args.reference_tag, references={f'{arm}_s{seed}': dict(
                      path=str(path.relative_to(ROOT)), sha256=sha(path), weight=row['weight'],
                      weight_sha256=row['weight_sha256']) for (arm, seed), (path, row) in refs.items()},
                  result_sha256={str(paths(args, 'mla', arm, seed)[0].relative_to(ROOT)):
                      sha(paths(args, 'mla', arm, seed)[0]) for arm, seed in rows},
                  note='Only MLA RoPE changes: ordinary independent full-width per-layer projection/cache. '
                       'Fresh cyclic KV-latent training; identical latent publication tensors and shared private initialization. '
                       'Same optimizer/minibatches/final budget as previous runs. Reference scores are reused, not retrained. '
                       'n=3 descriptive t intervals, no significance claim; validation already exposed.')
    atomic_save(record, ROOT / 'outputs' / args.tag / 'summary.json')
    print(json.dumps(dict(status='all_complete', new_training_runs=6,
                          summary=str(ROOT / 'outputs' / args.tag / 'summary.json'))), flush=True)


def all_runs(args):
    pairing(args)
    def worker(arm, seed):
        _, _, _, log = paths(args, 'mla', arm, seed)
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), '--arm', arm, '--seed', str(seed),
                   '--tag', args.tag, '--reference-tag', args.reference_tag, '--threads', str(args.threads)]
        with log.open('a') as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{log.stem} failed:\n{log.read_text()[-5000:]}')
        return log.stem
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(worker, arm, seed) for seed in range(3) for arm in ('f25', 'f375')]
        print(json.dumps(dict(status='queued', runs=6, cpu_workers=args.jobs)), flush=True)
        for completed, future in enumerate(as_completed(futures), 1):
            print(json.dumps(dict(status='worker_complete', run=future.result(), completed=completed, total=6)), flush=True)
    summarize(args)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--summarize-only', action='store_true')
    ap.add_argument('--arm', choices=('f25', 'f375'))
    ap.add_argument('--seed', type=int, choices=range(3))
    ap.add_argument('--jobs', type=int, choices=(1, 2), default=2)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--tag', default='mla_independent_rope')
    ap.add_argument('--reference-tag', default='attention_adaptation')
    ap.set_defaults(attention='mla', steps=1600, batch=16, length=256, skip_ood=False)
    args = ap.parse_args()
    if args.threads < 1:
        ap.error('Positive thread count required')
    if any(Path(tag).name != tag or tag in ('', '.', '..') for tag in (args.tag, args.reference_tag)):
        ap.error('Tags must be directory names')
    if args.summarize_only:
        summarize(args)
    elif args.all:
        all_runs(args)
    elif args.arm is not None and args.seed is not None:
        run_one(args, share_rope=False, extra_source_files=('work/study_mla_independent_rope.py',))
    else:
        ap.error('Use --all, --summarize-only, or --arm/--seed')


if __name__ == '__main__':
    main()
