"""Independent full-width source recurrence for support-local prefix public memory."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from public_memory import Config, PublicMemory
from local_public_bank import LocalBank
from run_clean_direct import fresh_bank


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def oracle_rotate(x, positions, p, ids=None, inverse=False):
    coordinates = torch.arange(x.shape[-1], device=x.device) if ids is None else ids
    frequency = 10000. ** (-(coordinates[::2] % p).float() / p)
    phase = torch.as_tensor(positions, device=x.device).float()[..., None] * frequency
    cosine, sine = phase.cos().to(x.dtype), phase.sin().to(x.dtype)
    if inverse:
        norm = cosine.square() + sine.square()
        cosine, sine = cosine / norm, -sine / norm
    out = torch.empty_like(x)
    out[..., ::2] = x[..., ::2]*cosine - x[..., 1::2]*sine
    out[..., 1::2] = x[..., ::2]*sine + x[..., 1::2]*cosine
    return out


def dense_oracle(model, tokens, storage_dtype=None):
    cfg, b, n = model.config, tokens.shape[0], tokens.shape[1]
    p, c = cfg.head_dim, cfg.source_width//2
    positions = torch.arange(n, device=tokens.device)
    h = model.embed(tokens)
    source = model.initial(model.initial_norm(h))
    if storage_dtype is not None:
        key = oracle_rotate(source[..., :c], positions, p).to(storage_dtype).to(h.dtype)
        value = source[..., c:].to(storage_dtype).to(h.dtype)
        source = torch.cat((oracle_rotate(key, positions, p, inverse=True), value), -1)
    for l, layer in enumerate(model.blocks):
        if l:
            # Construct global logical coordinates directly; no Layout, cache
            # planes, static segments or per-head recurrence from implementation.
            key_ids = torch.tensor([head*p + ((l-1)*cfg.write_dim+j) % p
                for head in range(cfg.kv_heads) for j in range(cfg.write_dim)], device=h.device)
            joint = torch.cat((key_ids, key_ids+c))
            delta = source[..., joint] + layer.pub(layer.n1(h))
            accumulation_dtype = torch.float64 if storage_dtype is not None else delta.dtype
            count = torch.arange(1, n+1, dtype=accumulation_dtype, device=h.device)
            mean = (delta.to(accumulation_dtype).cumsum(1) / count[None, :, None]).to(delta.dtype)
            committed = (1-model.lam)*delta + model.lam*mean
            if storage_dtype is not None:
                raw_k, raw_v = committed.chunk(2, -1)
                k = oracle_rotate(raw_k, positions, p, key_ids).to(storage_dtype).to(h.dtype)
                v = raw_v.to(storage_dtype).to(h.dtype)
                key, value = key.index_copy(-1, key_ids, k), value.index_copy(-1, key_ids, v)
                committed = torch.cat((oracle_rotate(k, positions, p, key_ids, inverse=True), v), -1)
            source = source.index_copy(-1, joint, committed)
        if storage_dtype is None:
            key, value = oracle_rotate(source[..., :c], positions, p), source[..., c:]
        query = layer.q(layer.n1(h)).reshape(b, n, cfg.query_heads, p).transpose(1, 2)
        query = oracle_rotate(query, positions, p)
        keys = key.reshape(b, n, cfg.kv_heads, p).transpose(1, 2)
        values = value.reshape(b, n, cfg.kv_heads, p).transpose(1, 2)
        group = cfg.query_heads//cfg.kv_heads
        out = F.scaled_dot_product_attention(query, keys.repeat_interleave(group, 1),
            values.repeat_interleave(group, 1), is_causal=True)
        h = h + layer.o(out.transpose(1, 2).flatten(-2))
        gate, up = layer.up(layer.n2(h)).chunk(2, -1)
        h = h + layer.down(F.silu(gate)*up)
    return model.output(model.norm(h))


def difference(gold, actual, targets):
    a, b = gold.double().log_softmax(-1), actual.double().log_softmax(-1)
    return dict(max_logits_error=float((gold-actual).abs().max()),
        mean_kl=float((a.exp()*(a-b)).sum(-1).mean()),
        delta_nll=float(F.cross_entropy(actual.flatten(0, 1), targets.flatten()) -
                        F.cross_entropy(gold.flatten(0, 1), targets.flatten())))


@torch.inference_mode()
def decode_check(model, tokens, dtype, prefill, tile, targets=None):
    decoder = model.new_cache(tokens.shape[1], tokens.shape[0], tile, dtype)
    warm = decoder.prefill(tokens[:, :prefill])
    warm_error = None
    if prefill:
        warm_error = float((warm-dense_oracle(model, tokens[:, :prefill], dtype)[:, -1:]).abs().max())
    results = []
    for t in range(prefill, tokens.shape[1]):
        planes = decoder.cache.keys + decoder.cache.values
        old = [x[:, :, :t].clone() for x in planes]
        results.append(decoder.append(tokens[:, t]))
        assert all(torch.equal(a, x[:, :, :t]) for a, x in zip(old, planes))
    actual = torch.cat(results, 1)
    gold = dense_oracle(model, tokens, dtype)[:, prefill:]
    error = difference(gold, actual, (tokens if targets is None else targets)[:, prefill:])
    limit = 2e-11 if dtype == torch.float64 else 1e-4 if dtype == torch.float32 else .03
    assert max(error['max_logits_error'], warm_error or 0.) < limit, error
    assert abs(error['delta_nll']) < 2e-4 and error['mean_kl'] < 1e-5, error
    cfg = model.config
    expected = tokens.numel() * (cfg.source_width+(cfg.depth-1)*cfg.write_width) * torch.empty((), dtype=dtype).element_size()
    memory = decoder.memory_bytes()
    assert memory['history'] == expected
    assert memory['publication_state'] == 8 * (cfg.depth-1) * tokens.shape[0] * cfg.write_width
    decoder.prefill(tokens[:, :prefill])
    replay = torch.cat([decoder.append(tokens[:, t]) for t in range(prefill, tokens.shape[1])], 1)
    assert torch.equal(actual, replay)
    return dict(storage_dtype=str(dtype), length=tokens.shape[1], batch=tokens.shape[0], prefill=prefill,
        tile=tile, max_prefill_logits_error=warm_error, immutable_history=True, reset_replay_exact=True,
        memory=memory, **error)


def generic():
    torch.manual_seed(20261012)
    rows = []
    for shape in [dict(d=32, depth=1, query_heads=2, kv_heads=1, head_dim=8, write_dim=2, ffn_width=64),
        dict(d=48, depth=4, query_heads=4, kv_heads=1, head_dim=16, write_dim=4, ffn_width=96),
        dict(d=48, depth=5, query_heads=6, kv_heads=3, head_dim=8, write_dim=2, ffn_width=96),
        dict(d=64, depth=5, query_heads=4, kv_heads=2, head_dim=16, write_dim=6, ffn_width=128),
        dict(d=32, depth=3, query_heads=8, kv_heads=4, head_dim=8, write_dim=8, ffn_width=80)]:
        for dtype in (torch.float64, torch.float32, torch.bfloat16):
            model = PublicMemory(Config(vocab=17, **shape))
            if dtype == torch.float64:
                model = model.double()
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if parameter.ndim == 1:
                        parameter.copy_(.8 + .4*torch.rand_like(parameter))
            tokens = torch.randint(17, (2, 19))
            actual, expected = model(tokens), dense_oracle(model, tokens)
            parallel_error = float((actual-expected).detach().abs().max())
            assert parallel_error < (2e-12 if dtype == torch.float64 else 5e-5)
            gradients = torch.autograd.grad(actual.square().mean(), tuple(model.parameters()))
            assert all(torch.isfinite(g).all() for g in gradients)
            gradient_error = None
            if dtype == torch.float64:
                reference = torch.autograd.grad(expected.square().mean(), tuple(model.parameters()))
                gradient_error = max(float((g-r).detach().abs().max()) for g, r in zip(gradients, reference))
                assert gradient_error < 2e-12
            model.eval().requires_grad_(False)
            changed = tokens.clone(); changed[:, 11:] = (changed[:, 11:]+1) % 17
            assert torch.equal(model(tokens)[:, :11], model(changed)[:, :11])
            for prefill in (0, 9):
                for tile in (1, 7):
                    row = decode_check(model, tokens, dtype, prefill, tile)
                    row.update(config=shape, max_parallel_error_vs_dense=parallel_error,
                        max_fp64_gradient_error=gradient_error, gradients_finite=True, causal_prefix_exact=True)
                    rows.append(row)
    return rows


def trained(paths, full_validation):
    rows, quality = [], []
    for path in paths:
        saved = torch.load(path, weights_only=False)
        meta, alphabet = saved['metadata'], saved['alphabet']
        assert saved['step'] == 1600 and meta['variant'] == 'prefix' and meta['width'] == 24
        seed, steps = meta['seed'], 1600
        model = PublicMemory(Config(vocab=len(alphabet))).eval().requires_grad_(False)
        model.load_state_dict({k: v for k, v in saved['state_dict'].items() if k != 'address'})
        reference = LocalBank(fresh_bank(seed, len(alphabet), meta['config'], 24), 'prefix', lam=meta['lam']).eval().requires_grad_(False)
        state = dict(saved['state_dict']); state.setdefault('address', reference.address)
        reference.load_state_dict(state)
        text = (ROOT / 'data/tinyshakespeare.txt').read_text()
        lookup = {ch: i for i, ch in enumerate(alphabet)}
        data = torch.tensor([lookup[ch] for ch in text[int(.9*len(text)):]])
        tokens = data[:257][None]
        parallel_error = float((model(tokens)-reference(tokens)).abs().max())
        assert parallel_error < 5e-5
        core = ROOT / f'work/standalone_{path.stem}.pt'
        model.save(core, source_name=path.name, source_steps=steps, source_checkpoint_sha256=sha(path), alphabet=alphabet,
            note='Exact tensor conversion; no adaptation or continuation')
        assert torch.equal(model(tokens), PublicMemory.load(core).eval()(tokens))
        for dtype in (torch.float32, torch.bfloat16):
            row = decode_check(model, tokens, dtype, 127, 7, data[1:258][None])
            row.update(checkpoint=path.name, checkpoint_sha256=sha(path), training_steps=steps,
                standalone_checkpoint=core.name, standalone_checkpoint_sha256=sha(core),
                parameters=sum(p.numel() for p in model.parameters()), exact_tensor_conversion=True,
                max_parallel_error_vs_research_model=parallel_error)
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != 'memory'}), flush=True)
        if full_validation:
            for length in (256, 1024):
                total, low, count, maximum = 0., 0., 0, 0.
                for start in range(0, len(data)-1, length):
                    n = min(length, len(data)-start-1)
                    tokens, targets = data[start:start+n][None], data[start+1:start+n+1]
                    original, exact, stored = reference(tokens), model(tokens), model.forward_stored(tokens)
                    maximum = max(maximum, float((exact-original).abs().max()))
                    total += float(F.cross_entropy(exact[0], targets, reduction='sum'))
                    low += float(F.cross_entropy(stored[0], targets, reduction='sum'))
                    count += n
                quality.append(dict(checkpoint=path.name, seed=seed, length=length, targets=count,
                    fp32_nll=total/count, bf16_nll=low/count, bf16_delta_nll=(low-total)/count,
                    max_fp32_parallel_error_vs_research=maximum))
                print(json.dumps(quality[-1]), flush=True)
    return rows, quality


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoints', nargs='*', type=Path, default=[])
    ap.add_argument('--trained-only', action='store_true')
    ap.add_argument('--full-validation', action='store_true')
    ap.add_argument('--tag', default='clean_final_verification')
    args = ap.parse_args()
    torch.set_num_threads(1)
    cases = [] if args.trained_only else generic()
    rows, quality = trained(args.checkpoints, args.full_validation)
    result = dict(generic=cases, trained=rows, full_validation=quality, torch=torch.__version__,
        code_sha256={name: sha(ROOT / name) for name in ('public_memory.py', 'narrow_cache.py',
            'work/verify_clean_memory.py', 'work/local_public_bank.py')},
        note='Independent global full-width source oracle, actual private feedback, FP64 gradients, '
             'nontrivial affine RMS, causality, immutable history, reset/replay, balanced multiple KV heads. '
             'Low-precision reduction ties measured rather than asserted bitwise equivalent.')
    (ROOT / f'outputs/{args.tag}.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(generic_cases=len(cases), trained_cases=len(rows))), flush=True)


if __name__ == '__main__':
    main()
