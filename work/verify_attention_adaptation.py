"""Independent dense-source/manual-attention oracle for MLA and TPA.

The oracle does not use FragmentLayout, cache reads, query construction,
physical_fragment, parallel_attention, or either streaming contraction.
"""
import argparse
from dataclasses import asdict, replace
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from factorized_memory import AttentionConfig, AttentionMemory, paired_model
from fragment_cache import Component, FragmentCache, FragmentLayout


def oracle_rotate(x, positions, p, coordinates=None):
    coordinates = torch.arange(x.shape[-1], device=x.device) if coordinates is None else coordinates
    frequency = 10000. ** (-(coordinates[::2] % p).float() / p)
    phase = positions.float()[:, None] * frequency
    cosine, sine = phase.cos().to(x.dtype), phase.sin().to(x.dtype)
    a, b = x[..., ::2], x[..., 1::2]
    return torch.stack((a * cosine - b * sine, a * sine + b * cosine), -1).flatten(-2)


def dense_oracle(model, tokens, storage_dtype=None, return_states=False):
    cfg = model.config
    b, n = tokens.shape
    positions = torch.arange(n, device=tokens.device)
    groups = [(cfg.kv_latent_dim, 0), (cfg.rope_dim, cfg.rope_dim)] if cfg.attention == 'mla' else [
        (cfg.key_rank * cfg.heads, 0), (cfg.key_rank * cfg.head_dim, cfg.head_dim),
        (cfg.value_rank * cfg.heads, 0), (cfg.value_rank * cfg.head_dim, 0)]
    h, state, states = model.embed(tokens), None, []
    for l, layer in enumerate(model.blocks):
        normalized = layer.n1(h)
        full = cfg.cache_mode == 'native' or l == 0
        if state is None or cfg.cache_mode == 'native':
            state = h.new_zeros(b, n, sum(width for width, _ in groups))
        base = 0
        for index, (width, rotary) in enumerate(groups):
            private_rope = cfg.attention == 'mla' and cfg.cache_mode == 'cyclic' and not cfg.share_rope and index == 1
            projection = layer.rope_key if private_rope else (
                model.initial[index] if cfg.cache_mode == 'cyclic' and l == 0 else layer.pub[index])
            inputs = (model.initial_norm(h) if cfg.cache_mode == 'cyclic' and l == 0 and not private_rope
                      else normalized)
            full_component = full or private_rope
            write = width if full_component else int(width * Fraction(cfg.write_fraction))
            local = torch.arange(width, device=h.device) if full_component else (
                torch.arange(write, device=h.device) + (l - 1) * write) % width
            fragment = F.linear(inputs, projection.weight)
            if rotary:
                fragment = oracle_rotate(fragment, positions, rotary, local)
            if storage_dtype is not None:
                fragment = fragment.to(storage_dtype).to(h.dtype)
            state = state.index_copy(-1, local + base, fragment)
            base += width
        states.append(state)
        if cfg.attention == 'mla':
            q = F.linear(F.linear(normalized, layer.q_down.weight), layer.q_up.weight).reshape(
                b, n, cfg.heads, cfg.head_dim + cfg.rope_dim)
            qc, qr = q.split((cfg.head_dim, cfg.rope_dim), -1)
            qr = oracle_rotate(qr.flatten(-2), positions, cfg.rope_dim).reshape_as(qr)
            query = torch.cat((qc, qr), -1).transpose(1, 2)
            latent, positional = state.split((cfg.kv_latent_dim, cfg.rope_dim), -1)
            kc = F.linear(latent, layer.k_up.weight).reshape(b, n, cfg.heads, cfg.head_dim)
            key = torch.cat((kc, positional[:, :, None].expand(-1, -1, cfg.heads, -1)), -1).transpose(1, 2)
            value = F.linear(latent, layer.v_up.weight).reshape(b, n, cfg.heads, cfg.head_dim).transpose(1, 2)
            scale = math.sqrt(cfg.head_dim + cfg.rope_dim)
        else:
            a = F.linear(normalized, layer.q_a.weight).reshape(b, n, cfg.query_rank, cfg.heads)
            features = F.linear(normalized, layer.q_b.weight).reshape(b, n, cfg.query_rank, cfg.head_dim)
            # Rotate the reconstructed query, independently checking factor RoPE.
            q = (a.transpose(-1, -2) @ features) / cfg.query_rank
            query = oracle_rotate(q.flatten(-2), positions, cfg.head_dim).reshape_as(q).transpose(1, 2)
            ak, bk, av, bv = state.split([width for width, _ in groups], -1)
            key = (ak.reshape(b, n, cfg.key_rank, cfg.heads).transpose(-1, -2)
                   @ bk.reshape(b, n, cfg.key_rank, cfg.head_dim)).transpose(1, 2) / cfg.key_rank
            value = (av.reshape(b, n, cfg.value_rank, cfg.heads).transpose(-1, -2)
                     @ bv.reshape(b, n, cfg.value_rank, cfg.head_dim)).transpose(1, 2) / cfg.value_rank
            scale = math.sqrt(cfg.head_dim)
        scores = (query @ key.transpose(-1, -2)) / scale
        scores = scores.masked_fill(torch.ones(n, n, dtype=torch.bool, device=h.device).triu(1), -torch.inf)
        out = scores.softmax(-1) @ value
        h = h + F.linear(out.transpose(1, 2).flatten(-2), layer.o.weight)
        gate, up = F.linear(layer.n2(h), layer.up.weight).chunk(2, -1)
        h = h + F.linear(F.silu(gate) * up, layer.down.weight)
    logits = F.linear(model.norm(h), model.output.weight)
    return (logits, states) if return_states else logits


@torch.inference_mode()
def decode_check(model, tokens, dtype, prefill=7, tile=7, targets=None):
    decoder = model.new_cache(tokens.shape[1], tokens.shape[0], tile, dtype)
    warm = decoder.prefill(tokens[:, :prefill])
    warm_error = 0. if warm is None else float((warm - dense_oracle(
        model, tokens[:, :prefill], dtype)[:, -1:]).abs().max())
    results = []
    for t in range(prefill, tokens.shape[1]):
        before = [bank[:, :t].clone() for bank in decoder.cache.banks]
        results.append(decoder.append(tokens[:, t]))
        assert all(torch.equal(old, bank[:, :t]) for old, bank in zip(before, decoder.cache.banks))
    actual = torch.cat(results, 1)
    gold, states = dense_oracle(model, tokens, dtype, return_states=True)
    gold = gold[:, prefill:]
    error = float((gold - actual).abs().max())
    limit = 2e-11 if dtype == torch.float64 else 1e-4 if dtype == torch.float32 else .03
    log_a, log_b = gold.double().log_softmax(-1), actual.double().log_softmax(-1)
    kl = float((log_a.exp() * (log_a - log_b)).sum(-1).mean())
    targets = (tokens if targets is None else targets)[:, prefill:]
    delta = float(F.cross_entropy(actual.flatten(0, 1), targets.flatten()) -
                  F.cross_entropy(gold.flatten(0, 1), targets.flatten()))
    assert max(error, warm_error) < limit, (model.config, dtype, error, warm_error)
    assert abs(delta) < 2e-4 and kl < 1e-5, (delta, kl)
    read_error = max(float((decoder.cache.read_tile(l, 0, tokens.shape[1]) - state).abs().max())
                     for l, state in enumerate(states))
    assert read_error < (2e-11 if dtype == torch.float64 else 1e-4 if dtype == torch.float32 else .08), read_error
    cfg = model.config
    width = cfg.depth * cfg.source_width if cfg.cache_mode == 'native' else (
        cfg.source_width + (cfg.depth - 1) * cfg.write_width)
    memory = decoder.memory_bytes()
    assert memory['history'] == tokens.numel() * width * torch.empty((), dtype=dtype).element_size()
    assert memory['historical_workspace'] == memory['publication_state'] == 0
    assert memory['persistent_tensor_total'] == sum(
        x.numel() * x.element_size() for x in [*decoder.cache.banks, *model.buffers()])
    decoder.prefill(tokens[:, :prefill])
    replay = torch.cat([decoder.append(tokens[:, t]) for t in range(prefill, tokens.shape[1])], 1)
    assert torch.equal(actual, replay)
    try:
        decoder.append(tokens[:, 0])
    except ValueError:
        pass
    else:
        raise AssertionError('Capacity overflow accepted')
    assert decoder.cache.length == tokens.shape[1]
    return dict(storage_dtype=str(dtype), prefill=prefill, tile=tile, length=tokens.shape[1],
                batch=tokens.shape[0], max_logits_error=error, max_prefill_logits_error=warm_error,
                max_source_read_error=read_error, mean_kl=kl, delta_nll=delta,
                immutable_history=True, reset_replay_exact=True, memory=memory)


def pairing_check():
    rows = []
    for attention in ('mla', 'tpa'):
        base = AttentionConfig(vocab=17, attention=attention)
        native = paired_model(replace(base, cache_mode='native'), 0)
        for fraction in ('1/4', '3/8'):
            model = paired_model(replace(base, write_fraction=fraction), 0)
            for name, value in native.state_dict().items():
                if '.pub.' not in name:
                    assert torch.equal(value, model.state_dict()[name]), name
            for a, b in zip(native.blocks[0].pub, model.initial):
                assert torch.equal(a.weight, b.weight)
            assert torch.equal(model.initial_norm.weight, native.blocks[0].n1.weight)
            generator = torch.Generator().manual_seed(100000)
            for l in range(1, base.depth):
                for c, pub, ref in zip(model.config.components, model.blocks[l].pub, native.blocks[l].pub):
                    local = (torch.arange(c.write_width) + (l - 1) * c.write_width) % c.width
                    full = ref.weight if attention == 'tpa' else torch.randn(
                        c.width, base.d, generator=generator) * base.pub_init_std
                    assert torch.equal(pub.weight, full[local])
            rows.append(dict(attention=attention, fraction=fraction, paired_private_and_bottom=True,
                             publication_rows_exact=True))
    return rows


def invalid_checks():
    def rejects(fn):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError('Invalid input accepted')
    rejects(lambda: AttentionConfig(vocab=17, attention='other'))
    rejects(lambda: AttentionConfig(vocab=17, rope_dim=3))
    rejects(lambda: AttentionConfig(vocab=17, write_fraction='1/3'))
    rejects(lambda: AttentionConfig(vocab=17, attention='tpa', share_rope=False))
    rejects(lambda: FragmentLayout(2, [Component('key', 8, 3, 8)]))
    layout = FragmentLayout(3, [Component('coefficient', 3, 1), Component('key', 8, 2, 8)])
    cache = FragmentCache(layout, 2)
    rejects(lambda: cache.store(1, torch.zeros(1, 3)))
    rejects(lambda: cache.store(0, torch.zeros(1, 10)))
    cache.store(0, torch.zeros(1, 11))
    rejects(lambda: cache.read_tile(1, 0, 1))
    rejects(cache.finish_token)
    return dict(invalid_config_and_cache_inputs_rejected=True)


def generic():
    torch.manual_seed(20261013)
    shapes = [
        dict(attention='mla', d=32, depth=1, heads=2, head_dim=8, kv_latent_dim=16,
             query_latent_dim=16, rope_dim=8, ffn_width=64, write_fraction='1/4'),
        dict(attention='mla', d=48, depth=7, heads=3, head_dim=8, kv_latent_dim=32,
             query_latent_dim=24, rope_dim=16, ffn_width=96, write_fraction='3/8'),
        dict(attention='mla', d=32, depth=3, heads=2, head_dim=8, kv_latent_dim=16,
             query_latent_dim=16, rope_dim=8, ffn_width=64, write_fraction='1'),
        dict(attention='tpa', d=32, depth=1, heads=4, head_dim=8, query_rank=2,
             key_rank=2, value_rank=1, ffn_width=64, write_fraction='1/4'),
        dict(attention='tpa', d=48, depth=7, heads=4, head_dim=16, query_rank=3,
             key_rank=2, value_rank=2, ffn_width=96, write_fraction='3/8'),
        dict(attention='tpa', d=32, depth=3, heads=8, head_dim=8, query_rank=4,
             key_rank=1, value_rank=2, ffn_width=64, write_fraction='1'),
    ]
    shapes += [dict(shape, share_rope=False) for shape in shapes if shape['attention'] == 'mla']
    rows = []
    for shape in shapes:
        for mode in ('native', 'cyclic'):
            for dtype in (torch.float64, torch.float32, torch.bfloat16):
                model = AttentionMemory(AttentionConfig(vocab=17, cache_mode=mode, **shape))
                if dtype == torch.float64:
                    model.double()
                with torch.no_grad():
                    for parameter in model.parameters():
                        if parameter.ndim == 1:
                            parameter.copy_(.8 + .4 * torch.rand_like(parameter))
                tokens = torch.randint(17, (2, 19))
                actual, gold = model(tokens), dense_oracle(model, tokens)
                error = float((actual - gold).detach().abs().max())
                assert error < (2e-12 if dtype == torch.float64 else 5e-5), error
                gradients = torch.autograd.grad(actual.square().mean(), tuple(model.parameters()))
                assert all(torch.isfinite(g).all() for g in gradients)
                gradient_error = None
                if dtype == torch.float64:
                    reference = torch.autograd.grad(gold.square().mean(), tuple(model.parameters()))
                    gradient_error = max(float((a - b).abs().max()) for a, b in zip(gradients, reference))
                    assert gradient_error < 2e-12, gradient_error
                model.eval().requires_grad_(False)
                changed = tokens.clone()
                changed[:, 11:] = (changed[:, 11:] + 1) % 17
                assert torch.equal(model(tokens)[:, :11], model(changed)[:, :11])
                with tempfile.TemporaryDirectory(prefix='kv_c_attention_') as temporary:
                    path = Path(temporary) / 'roundtrip.pt'
                    model.save(path, test=True)
                    restored = AttentionMemory.load(path).eval().requires_grad_(False)
                    assert torch.equal(model(tokens), restored(tokens))
                complete = model.new_cache(19, 2, 7, dtype)
                warm = complete.prefill(tokens)
                torch.testing.assert_close(warm, dense_oracle(model, tokens, dtype)[:, -1:],
                                           atol=2e-11 if dtype == torch.float64 else .03,
                                           rtol=0)
                assert complete.cache.length == 19
                for prefill in (0, 7):
                    for tile in (1, 7):
                        row = decode_check(model, tokens, dtype, prefill, tile)
                        row.update(config=asdict(model.config), max_parallel_error=error,
                                   max_fp64_gradient_error=gradient_error, gradients_finite=True,
                                   causal=True, save_load_exact=True)
                        rows.append(row)
                print(json.dumps(dict(status='generic_case_ok', attention=shape['attention'],
                                      mode=mode, dtype=str(dtype), depth=shape['depth'])), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--weights', nargs='*', type=Path, default=[])
    ap.add_argument('--skip-generic', action='store_true')
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--output', type=Path, default=ROOT / 'outputs/attention_adaptation_verification.json')
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    result = dict(torch=str(torch.__version__), code_sha256={name: hashlib.sha256(
        (ROOT / name).read_bytes()).hexdigest() for name in (
            'factorized_memory.py', 'fragment_cache.py', 'work/verify_attention_adaptation.py')})
    if not args.skip_generic:
        result.update(pairing=pairing_check(), invalid_inputs=invalid_checks(), generic=generic())
    result['trained'] = []
    for path in args.weights:
        model = AttentionMemory.load(path).eval().requires_grad_(False)
        torch.manual_seed(123)
        tokens = torch.randint(model.config.vocab, (1, 33))
        checks = [decode_check(model, tokens, dtype, 9, 7)
                  for dtype in (torch.float32, torch.bfloat16)]
        result['trained'].append(dict(weight=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                      config=asdict(model.config), checks=checks))
        print(json.dumps(dict(status='trained_weight_ok', path=str(path))), flush=True)
    result['status'] = 'passed'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status='passed', generic=len(result.get('generic', [])),
                          trained=len(result['trained']), output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
