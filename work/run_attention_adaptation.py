"""Paired, resumable local standard experiments for native/cyclic MLA and TPA.

--all runs the locked 18-run matrix with at most two CPU worker processes.
Final weights and JSON results persist; optimizer checkpoints are removed only
after evaluation and verification succeed. Validation is already exposed.
"""
import argparse
import ast
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from factorized_memory import AttentionConfig, AttentionMemory, paired_model
from evaluate_ood import score
from verify_attention_adaptation import decode_check

DATA_SHA256 = '86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed'
ARMS = {'native': ('native', '3/8'), 'f25': ('cyclic', '1/4'), 'f375': ('cyclic', '3/8')}
SOURCE_FILES = ('factorized_memory.py', 'fragment_cache.py', 'cyclic_memory.py', 'narrow_cache.py',
                'work/run_attention_adaptation.py', 'work/verify_attention_adaptation.py',
                'work/evaluate_ood.py')
TRAINING_LOOP_V1 = 'dfc4ff595e24ae392abb67d241142287fc511146cfb4e6f741e6335b9b314658'
VERIFICATION_V1 = {
    'work/run_attention_adaptation.py': 'fbc41949e43cdce65b8245b914a971a74fcf6ccab0cf75e2d544ce4842473ca9',
    'work/verify_attention_adaptation.py': '934b6edaa3213cc3c8975cd73535ced68cac2e7917959142fbae368553b44e41',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_loop_sha256():
    tree = ast.parse(Path(__file__).read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_one')
    loop = next(n for n in function.body if isinstance(n, ast.For)
                and isinstance(n.target, ast.Name) and n.target.id == 'step')
    return hashlib.sha256(ast.dump(loop, include_attributes=False).encode()).hexdigest()


def verification_only_compatible(previous, current):
    """One audited migration: fix next-token targets, without changing training.

Every numerical training statement (the entire training-loop AST) and all
model/cache source hashes must still match the original run. Other changes
continue to require a new tag. Original source hashes are retained in results.
"""
    if previous == current:
        return True
    before, after = dict(previous), dict(current)
    old_codes, new_codes = before.pop('code_sha256'), after.pop('code_sha256')
    before.setdefault('training_loop_sha256', TRAINING_LOOP_V1)
    return (before == after and after['training_loop_sha256'] == TRAINING_LOOP_V1
            and all(old_codes[name] == (VERIFICATION_V1[name] if name in VERIFICATION_V1
                                       else new_codes[name]) for name in old_codes))


def trained_checks(model, validation):
    # Use true next-token targets, as in the existing trained-weight audit.
    # Keep the original adaptation audit's 33-token/9-prefill configuration.
    return [decode_check(model, validation[:33][None], dtype, 9, 7,
                         validation[1:34][None]) for dtype in (torch.float32, torch.bfloat16)]


def tensor_digest(items):
    digest = hashlib.sha256()
    for name, tensor in sorted(items):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def atomic_save(value, path, binary=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    if binary:
        torch.save(value, temporary)
    else:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def initial_hashes(model):
    private = [(name, tensor) for name, tensor in model.state_dict().items()
               if '.pub.' not in name and '.rope_key.' not in name
               and not name.startswith(('initial.', 'initial_norm.'))]
    if model.config.cache_mode == 'native':
        bottom = [('norm', model.blocks[0].n1.weight)] + [
            (str(i), p.weight) for i, p in enumerate(model.blocks[0].pub)]
    else:
        bottom = [('norm', model.initial_norm.weight)] + [
            (str(i), p.weight) for i, p in enumerate(model.initial)]
        if model.config.independent_rope:
            bottom.append(('1', model.blocks[0].rope_key.weight))
    return dict(private=tensor_digest(private), bottom=tensor_digest(bottom),
                complete=tensor_digest(model.state_dict().items()))


@torch.inference_mode()
def evaluate(model, validation, length):
    model.eval()
    total, count = 0., 0
    for start in range(0, len(validation) - 1, length):
        n = min(length, len(validation) - start - 1)
        logits = model(validation[start:start + n][None])[0]
        total += float(F.cross_entropy(logits, validation[start + 1:start + n + 1], reduction='sum'))
        count += n
    assert count == len(validation) - 1
    nll = total / count
    return dict(nll=nll, ppl=math.exp(nll), targets=count, length=length)


def paths(args, attention=None, arm=None, seed=None):
    attention, arm, seed = (attention or args.attention, arm or args.arm,
                             args.seed if seed is None else seed)
    stem = f'{attention}_{arm}_s{seed}'
    output_dir, weight_dir = ROOT / 'outputs' / args.tag, ROOT / 'weights' / args.tag
    return (output_dir / f'{stem}.json', weight_dir / f'{stem}.pt',
            weight_dir / f'{stem}.resume.pt', output_dir / 'logs' / f'{stem}.log')


def run_one(args, *, share_rope=True, extra_source_files=()):
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    raw = (ROOT / 'data/tinyshakespeare.txt').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == DATA_SHA256
    alphabet = sorted(set(raw.decode()))
    lookup = {ch: i for i, ch in enumerate(alphabet)}
    data = torch.tensor([lookup[ch] for ch in raw.decode()])
    split = int(.9 * len(data))
    train, validation = data[:split], data[split:]
    mode, fraction = ARMS[args.arm]
    cfg = AttentionConfig(vocab=len(alphabet), attention=args.attention,
                          cache_mode=mode, write_fraction=fraction, share_rope=share_rope)
    model = paired_model(cfg, args.seed)
    metadata = dict(seed=args.seed, attention=args.attention, arm=args.arm, config=asdict(cfg),
                    steps=args.steps, batch=args.batch, length=args.length, threads=args.threads,
                    tokens=args.steps * args.batch * args.length, data_sha256=DATA_SHA256,
                    train_characters=len(train), validation_targets=len(validation) - 1,
                    initialization_sha256=initial_hashes(model),
                    optimizer=dict(name='AdamW', lr=3e-4, betas=[.9, .95], weight_decay=.1,
                                   warmup_steps=30, final_lr_fraction=.1, clip=1.),
                    sampler_seed=10000 + args.seed, evaluation_lengths=[256, 512, 1024],
                    ood=not args.skip_ood, manifest_sha256=sha(ROOT / 'outputs/clean_ood_manifest.json'),
                    environment=dict(torch=str(torch.__version__), numpy=np.__version__,
                                     python=platform.python_version(), platform=platform.platform(),
                                     machine=platform.machine(), cpu_count=os.cpu_count(), device='cpu',
                                     training_dtype='float32', deterministic_algorithms=True),
                    code_sha256={name: sha(ROOT / name) for name in SOURCE_FILES + tuple(extra_source_files)},
                    training_loop_sha256=training_loop_sha256(),
                    note='Fresh random initialization; paired private/bottom tensors and minibatches within '
                         'each attention family. Fixed final step, no best-checkpoint selection or tuning. '
                         'Validation already exposed; OOD uses the existing frozen manifest. '
                         'MLA narrow std=.0016; TPA publications select native Xavier rows. '
                         'Parameter counts are reported, not matched by changing the backbone.')
    output, weight, checkpoint, _ = paths(args)
    if output.exists():
        previous = json.loads(output.read_text())
        assert verification_only_compatible(previous['metadata'], metadata), 'Existing result differs: choose another --tag'
        assert previous['weight_sha256'] == sha(weight)
        if previous['metadata'] != metadata and 'verification_revision' not in previous:
            restored = AttentionMemory.load(weight).eval().requires_grad_(False)
            checks = trained_checks(restored, validation)
            previous['verification_revision'] = dict(
                reason='Correct trained-audit next-token target alignment; shapes and thresholds unchanged.',
                previous_checks=previous['checks'], code_sha256=metadata['code_sha256'],
                training_loop_sha256=metadata['training_loop_sha256'])
            previous['checks'] = checks
            atomic_save(previous, output)
        print(json.dumps(dict(status='already_complete', run=output.stem)), flush=True)
        return
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(.9, .95), weight_decay=.1)
    generator = torch.Generator().manual_seed(10000 + args.seed)
    first, elapsed, curve = 0, 0., []
    provenance_path = checkpoint.with_suffix('.provenance.json')
    resume_provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else None
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert verification_only_compatible(saved['metadata'], metadata), 'Checkpoint differs: choose another --tag'
        if saved['metadata'] != metadata:
            resume_provenance = dict(checkpoint_step=saved['step'], previous_code_sha256=saved['metadata']['code_sha256'],
                execution_code_sha256=metadata['code_sha256'], training_loop_sha256=metadata['training_loop_sha256'],
                reason='Verification-only correction; the complete numerical training loop and model source are unchanged.')
            atomic_save(resume_provenance, provenance_path)
        model.load_state_dict(saved['state_dict'])
        optimizer.load_state_dict(saved['optimizer'])
        generator.set_state(saved['sampler_rng'])
        torch.set_rng_state(saved['torch_rng'])
        first, elapsed, curve = saved['step'], saved['elapsed_seconds'], saved['curve']
        print(json.dumps(dict(status='resumed', run=output.stem, step=first)), flush=True)
    started = time.monotonic()
    for step in range(first, args.steps):
        model.train()
        starts = torch.randint(len(train) - args.length - 1, (args.batch,), generator=generator)
        indices = starts[:, None] + torch.arange(args.length)
        logits = model(train[indices])
        loss = F.cross_entropy(logits.flatten(0, 1), train[indices + 1].flatten())
        assert torch.isfinite(loss), (output.stem, step, 'nonfinite loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
        assert math.isfinite(grad), (output.stem, step, 'nonfinite gradient')
        lr = 3e-4 * min((step + 1) / 30, 1) * (.1 + .9 * .5 * (
            1 + math.cos(math.pi * step / args.steps)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.step()
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == args.steps:
            point = dict(step=step + 1, loss=float(loss.detach()), grad=grad, lr=lr,
                         elapsed_seconds=elapsed + time.monotonic() - started)
            curve.append(point)
            print(json.dumps(dict(run=output.stem, **point)), flush=True)
        if (step + 1) % 100 == 0 or step + 1 == args.steps:
            atomic_save(dict(metadata=metadata, step=step + 1, state_dict=model.state_dict(),
                             optimizer=optimizer.state_dict(), sampler_rng=generator.get_state(),
                             torch_rng=torch.get_rng_state(), curve=curve,
                             elapsed_seconds=elapsed + time.monotonic() - started), checkpoint, binary=True)
    training_seconds = elapsed + time.monotonic() - started
    model.eval().requires_grad_(False)
    evaluation_started = time.monotonic()
    weight.parent.mkdir(parents=True, exist_ok=True)
    temporary = weight.with_suffix('.pt.partial')
    model.save(temporary, alphabet=alphabet, run=metadata, step=args.steps, resume_provenance=resume_provenance)
    temporary.replace(weight)
    restored = AttentionMemory.load(weight).eval().requires_grad_(False)
    with torch.inference_mode():
        assert torch.equal(model(validation[:33][None]), restored(validation[:33][None]))
    del restored, optimizer
    checks = trained_checks(model, validation)
    evaluations = {str(length): evaluate(model, validation, length) for length in (256, 512, 1024)}
    ood_rows = []
    if not args.skip_ood:
        manifest = json.loads((ROOT / 'outputs/clean_ood_manifest.json').read_text())
        assert ''.join(alphabet) == manifest['alphabet']
        assert manifest['source_train_sha256'] == DATA_SHA256
        for corpus in manifest['corpora']:
            path = ROOT / 'data/ood' / Path(corpus['file']).name
            assert sha(path) == corpus['selected_sha256']
            external = torch.tensor([lookup[ch] for ch in path.read_text()])
            for length in (256, 1024):
                for stored in (False, True):
                    row = dict(corpus=corpus['name'], length=length,
                               precision='bf16' if stored else 'fp32',
                               evaluation=score(model, external, length,
                                                corpus['exclude_chunks'][str(length)], stored))
                    ood_rows.append(row)
    memory = {str(dtype): model.new_cache(16384, dtype=dtype).memory_bytes()
              for dtype in (torch.float32, torch.bfloat16)}
    expected_scalars = cfg.depth * cfg.source_width if mode == 'native' else (
        cfg.source_width + (cfg.depth - 1) * cfg.write_width)
    result = dict(status='completed', metadata=metadata,
                  parameters=sum(p.numel() for p in model.parameters()), curve=curve,
                  training_seconds=training_seconds, evaluation_seconds=time.monotonic() - evaluation_started,
                  evaluation=evaluations, ood=ood_rows, checks=checks, memory=memory,
                  history_scalars_per_token=expected_scalars,
                  history_ratio_to_native=expected_scalars / (cfg.depth * cfg.source_width),
                  weight=str(weight.relative_to(ROOT)), weight_sha256=sha(weight), save_load_exact=True,
                  resume_provenance=resume_provenance)
    atomic_save(result, output)
    checkpoint.unlink(missing_ok=True)
    provenance_path.unlink(missing_ok=True)
    print(json.dumps(dict(status='complete', run=output.stem, evaluation=evaluations,
                          output=str(output))), flush=True)


def describe(values):
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    radius = 4.302652729911275 * sd / math.sqrt(3)
    return dict(seed_values=values, mean=mean, sample_sd=sd, descriptive_95_t_interval=[mean - radius, mean + radius])


def summarize(args):
    rows = [json.loads(paths(args, attention, arm, seed)[0].read_text())
            for attention in ('mla', 'tpa') for arm in ARMS for seed in range(3)]
    assert len(rows) == 18 and all(row['status'] == 'completed' for row in rows)
    for attention in ('mla', 'tpa'):
        for seed in range(3):
            matching = [row for row in rows if row['metadata']['attention'] == attention
                        and row['metadata']['seed'] == seed]
            for key in ('private', 'bottom'):
                assert len({row['metadata']['initialization_sha256'][key] for row in matching}) == 1
    for key in ('steps', 'batch', 'length', 'tokens', 'data_sha256', 'environment', 'ood'):
        assert all(row['metadata'][key] == rows[0]['metadata'][key] for row in rows), key
    for row in rows:
        assert row['metadata'].get('training_loop_sha256', TRAINING_LOOP_V1) == TRAINING_LOOP_V1
        assert all(row['metadata']['code_sha256'][name] == rows[0]['metadata']['code_sha256'][name]
                   for name in SOURCE_FILES if name not in VERIFICATION_V1)
    by_key = {(row['metadata']['attention'], row['metadata']['arm'], row['metadata']['seed']): row for row in rows}
    summary = {}
    for attention in ('mla', 'tpa'):
        for arm in ARMS:
            group = [by_key[attention, arm, seed] for seed in range(3)]
            entry = dict(parameters=group[0]['parameters'], history_scalars_per_token=group[0]['history_scalars_per_token'],
                         history_ratio_to_native=group[0]['history_ratio_to_native'],
                         training_seconds=sum(row['training_seconds'] for row in group), evaluation={}, ood={})
            for length in ('256', '512', '1024'):
                values = [row['evaluation'][length]['nll'] for row in group]
                entry['evaluation'][length] = dict(nll=describe(values), geometric_mean_ppl=math.exp(statistics.mean(values)),
                    paired_delta_vs_native=describe([values[seed] - by_key[attention, 'native', seed][
                        'evaluation'][length]['nll'] for seed in range(3)]))
            for key in [(r['corpus'], r['length'], r['precision']) for r in group[0]['ood']]:
                def metric(row):
                    return next(r['evaluation']['nll'] for r in row['ood'] if (
                        r['corpus'], r['length'], r['precision']) == key)
                values = [metric(row) for row in group]
                entry['ood']['/'.join(map(str, key))] = dict(nll=describe(values),
                    paired_delta_vs_native=describe([values[seed] - metric(by_key[attention, 'native', seed])
                                                    for seed in range(3)]))
            summary[f'{attention}_{arm}'] = entry
    record = dict(status='completed', runs=18, paired_initialization_verified=True,
                  metadata={k: rows[0]['metadata'][k] for k in (
                      'steps', 'batch', 'length', 'tokens', 'data_sha256', 'environment', 'ood')},
                  training_loop_sha256=TRAINING_LOOP_V1,
                  code_sha256_by_run={f"{row['metadata']['attention']}_{row['metadata']['arm']}_s{row['metadata']['seed']}":
                                      row['metadata']['code_sha256'] for row in rows},
                  note='n=3 descriptive t intervals, df=2; no multiple-comparison correction or significance claim. '
                       'Existing exposed validation. CPU time is a run record, not a GPU latency result.',
                  summary=summary, result_files={paths(args, attention, arm, seed)[0].name:
                      sha(paths(args, attention, arm, seed)[0]) for attention in ('mla', 'tpa')
                      for arm in ARMS for seed in range(3)})
    output = ROOT / 'outputs' / args.tag / 'summary.json'
    atomic_save(record, output)
    print(json.dumps(dict(status='all_complete', runs=18, summary=str(output))), flush=True)


def run_all(args):
    pending = [(attention, arm, seed) for seed in range(3) for attention in ('mla', 'tpa') for arm in ARMS]
    active, finished, last_report = [], 0, 0.
    try:
        while pending or active:
            while pending and len(active) < args.jobs:
                attention, arm, seed = pending.pop(0)
                _, _, _, log = paths(args, attention, arm, seed)
                log.parent.mkdir(parents=True, exist_ok=True)
                stream = log.open('a')
                command = [sys.executable, str(Path(__file__).resolve()), '--attention', attention,
                           '--arm', arm, '--seed', str(seed), '--steps', str(args.steps),
                           '--batch', str(args.batch), '--length', str(args.length),
                           '--threads', str(args.threads), '--tag', args.tag]
                if args.skip_ood:
                    command.append('--skip-ood')
                process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                active.append((process, stream, log))
                print(json.dumps(dict(status='started', run=log.stem, pid=process.pid)), flush=True)
            for process, stream, log in active[:]:
                code = process.poll()
                if code is not None:
                    stream.close()
                    active.remove((process, stream, log))
                    if code:
                        raise RuntimeError(f'{log.stem} failed ({code}):\n{log.read_text()[-4000:]}')
                    finished += 1
                    print(json.dumps(dict(status='worker_complete', run=log.stem, completed=finished, total=18)), flush=True)
            now = time.monotonic()
            if active and now - last_report >= 30:
                progress = []
                for _, _, log in active:
                    lines = log.read_text().splitlines()
                    progress.append(dict(run=log.stem, latest=lines[-1] if lines else 'initializing'))
                print(json.dumps(dict(status='progress', completed=finished, active=progress)), flush=True)
                last_report = now
            if active:
                time.sleep(1)
    finally:
        for process, stream, _ in active:
            process.terminate()
            process.wait()
            stream.close()
    summarize(args)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--summarize-only', action='store_true')
    ap.add_argument('--attention', choices=('mla', 'tpa'))
    ap.add_argument('--arm', choices=tuple(ARMS))
    ap.add_argument('--seed', type=int, choices=range(3))
    ap.add_argument('--steps', type=int, default=1600)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--length', type=int, default=256)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--jobs', type=int, choices=(1, 2), default=2)
    ap.add_argument('--tag', default='attention_adaptation')
    ap.add_argument('--skip-ood', action='store_true', help='For implementation smoke runs only')
    args = ap.parse_args()
    if min(args.steps, args.batch, args.length, args.threads) < 1:
        ap.error('Positive steps, batch, length and threads required')
    if Path(args.tag).name != args.tag or args.tag in ('', '.', '..'):
        ap.error('--tag must be a directory name')
    if args.summarize_only:
        summarize(args)
    elif args.all:
        run_all(args)
    elif args.attention is None or args.arm is None or args.seed is None:
        ap.error('Use --all, --summarize-only, or --attention/--arm/--seed')
    else:
        run_one(args)


if __name__ == '__main__':
    main()
