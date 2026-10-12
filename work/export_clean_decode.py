"""Export trained local-model weights and physically rounded prefill fixtures.

The dense and fragment histories are verification fixtures, never simultaneous
deployment caches. Only source fragments/KV and narrow prefix state are saved;
past private activations are not exported or replayed during native decoding.
"""
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fragment_cache import FragmentLayout, rotate_pairs
from local_public_bank import LocalBank
from fixed_public_bank import FixedBank
from increment_public_bank import IncrementBank
from run_clean_direct import fresh_bank


@torch.inference_mode()
def prefill(model, tokens, prefix_length, dtype):
    b, n = tokens.shape
    assert b == 1
    p, c = model.blocks[0].p, model.c // 2
    layout = FragmentLayout(model.depth, 1, p, model.write_width)
    positions = torch.arange(n)
    h = model.embed(tokens)
    raw = model.initial(model.initial_norm(h))
    key = rotate_pairs(raw[..., :c], positions, p).to(dtype).float()
    value = raw[..., c:].to(dtype).float()
    banks = [(key[:, :prefix_length].clone(), value[:, :prefix_length].clone())]
    current = torch.cat((rotate_pairs(key, positions, p, inverse=True), value), -1)
    dense, sums = [], []
    for l, layer in enumerate(model.blocks):
        if l:
            ids = torch.tensor(layout.indices[1 if isinstance(model, FixedBank) else l])
            joint = torch.cat((ids, ids + c))
            delta = layer.pub(layer.n1(h))
            z = current[..., joint] + delta
            if model.transport == 'increment_prefix':
                sums.append(delta[:, :prefix_length].double().sum(1))
                mean = (delta.double().cumsum(1) / torch.arange(1, n + 1).double()[None, :, None]).float()
                z = current[..., joint] + (1-model.lam)*delta + model.lam*mean
            elif model.transport == 'prefix':
                sums.append(z[:, :prefix_length].double().sum(1))
                mean = (z.double().cumsum(1) / torch.arange(1, n + 1).double()[None, :, None]).float()
                z = (1 - model.lam) * z + model.lam * mean
            elif model.transport == 'anchor':
                first_key, first_value = z.chunk(2, -1)
                rounded_key = rotate_pairs(first_key, positions, p, ids).to(dtype).float()
                rounded_value = first_value.to(dtype).float()
                rounded = torch.cat((rotate_pairs(rounded_key, positions, p, ids, inverse=True), rounded_value), -1)
                moved = z.clone()
                for start in range(0, n, model.block):
                    anchor_end = min(start + model.anchors, n)
                    end = min(start + model.block, n)
                    if anchor_end < end:
                        mean = rounded[:, start:anchor_end].double().mean(1).float()
                        moved[:, anchor_end:end] = (1 - model.lam) * z[:, anchor_end:end] + model.lam * mean[:, None]
                z = moved
            elif model.transport != 'none':
                raise ValueError('Unsupported source update')
            new_key, new_value = z.chunk(2, -1)
            committed_key = rotate_pairs(new_key, positions, p, ids).to(dtype).float()
            committed_value = new_value.to(dtype).float()
            saved = torch.cat((rotate_pairs(committed_key, positions, p, ids, inverse=True), committed_value), -1)
            if model.transport == 'anchor':
                start = prefix_length // model.block * model.block
                end = min(start + model.anchors, prefix_length)
                sums.append(saved[:, start:end].double().sum(1))
            elif model.transport == 'none':
                sums.append(torch.zeros(1, model.write_width, dtype=torch.float64))
            banks.append((committed_key[:, :prefix_length].clone(), committed_value[:, :prefix_length].clone()))
            current = current.index_copy(-1, joint, saved)
            key, value = key.index_copy(-1, ids, committed_key), value.index_copy(-1, ids, committed_value)
        dense.append((key[:, :prefix_length].clone(), value[:, :prefix_length].clone()))
        q = layer.q(layer.n1(h)).view(b, n, layer.heads, p).transpose(1, 2)
        q = rotate_pairs(q, positions, p)
        out = F.scaled_dot_product_attention(q, key[:, None].expand(-1, layer.heads, -1, -1),
                                             value[:, None].expand(-1, layer.heads, -1, -1), is_causal=True)
        h = h + layer.o(out.transpose(1, 2).flatten(-2))
        gate, up = layer.up(layer.n2(h)).chunk(2, -1)
        h = h + layer.down(F.silu(gate) * up)
    return model.output(model.norm(h)), banks, dense, sums


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--context', type=int, default=128)
    ap.add_argument('--steps', type=int, default=8)
    ap.add_argument('--tile', type=int, default=128)
    ap.add_argument('--dtype', choices=('fp32', 'bf16'), default='fp32')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    saved = torch.load(args.checkpoint, weights_only=False)
    meta = saved['metadata']
    alphabet, cfg = saved['alphabet'], meta['config']
    if saved.get('variant') == 'increment_prefix' and cfg['steps'] == 400:
        seed, width, variant, lam, steps = saved['seed'], 16, 'increment_prefix', .25, 400
        assert saved['curve'][-1]['step'] == steps
    elif saved.get('fixed') is True and cfg['steps'] == 400:
        # Explicit exploratory schema, never relabel a screen as full training.
        seed, width, variant, lam, steps = saved['seed'], 16, 'fixed_none', 0., 400
        assert saved['curve'][-1]['step'] == steps
    else:
        assert saved['step'] == meta['args']['steps']
        seed, width, variant, lam, steps = meta['seed'], meta['width'], meta['variant'], meta['lam'], saved['step']
    bank = fresh_bank(seed, len(alphabet), cfg, width)
    model = (FixedBank(bank) if variant == 'fixed_none' else
             IncrementBank(bank, lam=lam) if variant == 'increment_prefix' else LocalBank(bank, variant, lam=lam)).eval()
    state = dict(saved['state_dict'])
    if hasattr(model, 'address'):
        state.setdefault('address', model.address)
    model.load_state_dict(state)
    assert all(isinstance(layer.mn, torch.nn.Identity) for layer in model.blocks)
    text = (ROOT / 'data/tinyshakespeare.txt').read_text()
    validation = text[int(.9 * len(text)):]
    lookup = {ch: i for i, ch in enumerate(alphabet)}
    total = args.context + args.steps
    data = torch.tensor([lookup[ch] for ch in validation[:total + 1]])
    tokens = data[:total][None]
    assert data.numel() == total + 1
    dtype = torch.float32 if args.dtype == 'fp32' else torch.bfloat16
    expected, banks, dense, sums = prefill(model, tokens, args.context, dtype)
    d, depth, vocab = cfg['d'], cfg['depth'], len(alphabet)
    p, heads, kv = model.blocks[0].p, model.blocks[0].heads, 1
    frequency = 10000. ** (-torch.arange(0, p, 2).float() / p)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('wb') as stream:
        stream.write(struct.pack('<16i', 0x46434E32, 2, d, depth, vocab, p, heads, kv,
            model.write_width, args.context, args.steps, args.tile, int(dtype == torch.bfloat16),
            3 if isinstance(model, FixedBank) else {'none': 0, 'prefix': 1, 'anchor': 2, 'increment_prefix': 4}[model.transport], model.block, model.anchors))
        stream.write(struct.pack('<f', model.lam))
        def floats(value, kind='<f4'):
            value.detach().cpu().contiguous().numpy().astype(kind, copy=False).tofile(stream)
        floats(model.embed.weight)
        floats(model.initial.weight)
        floats(model.initial_norm.weight)
        for l, layer in enumerate(model.blocks):
            for weight in (layer.n1.weight, layer.n2.weight, layer.q.weight, layer.o.weight,
                           layer.up.weight, layer.down.weight):
                floats(weight)
            if l:
                floats(layer.pub.weight)
        floats(model.norm.weight)
        floats(model.output.weight)
        floats(frequency)
        for key, value in banks:
            floats(key)
            floats(value)
        for key, value in dense:
            floats(key)
            floats(value)
        for value in sums:
            floats(value, '<f8')
        floats(data[args.context:total], '<i4')
        floats(data[args.context + 1:total + 1], '<i4')
        floats(expected[:, args.context:])
    report = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        file=str(args.out.resolve()), file_sha256=hashlib.sha256(args.out.read_bytes()).hexdigest(),
        dtype=args.dtype, context=args.context, new_tokens=args.steps, tile=args.tile,
        transport=variant, width=model.write_width, lam=model.lam, training_steps=steps,
        code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), ROOT / 'fragment_cache.py')},
        note='Physically rounded trained-model histories; immutable prefill fixtures include both layouts only for '
             'verification. Native decoders release unused histories; no old private activations.')
    args.out.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
