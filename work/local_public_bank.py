"""Support-preserving public writes; immutable K/V fragments, ordinary RoPE.

Only the selected coordinates are ever written. Untouched coordinates
are exactly frozen. This deliberately changes the global-projector architecture.
The CPU tiled reader is a semantic implementation, not a GPU speed claim.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from absorbed_public_bank import projector, rotate_public, rotate_key, write_indices


def addresses(c, depth, width):
    address = torch.arange(c)
    rows = [address.clone()]
    for l in range(1, depth):
        ids = write_indices(c, width, l)
        address[ids] = c + (l - 1) * width + torch.arange(width)
        rows.append(address.clone())
    return torch.stack(rows)


class LocalBank(nn.Module):
    """Clone a coordinate bank's private network and publication parameters.

    transport='anchor' is idempotent; 'prefix' deliberately is not. Both obey
    the same support condition and therefore have the same direct cache reader.
    'none' is the necessary simpler control. No owner-dependent decay exists.
    """
    def __init__(self, bank, transport='anchor', lam=None,
                 storage_dtype=torch.float32, rotated_cache=False):
        super().__init__()
        if transport not in ('anchor', 'prefix', 'none'):
            raise ValueError('Unknown narrow transport')
        if bank.factorized or bank.rotated_transport:
            raise ValueError('Requires direct coordinate publication before RoPE')
        self.c, self.depth = bank.c, bank.depth
        self.write_width, self.block, self.anchors = bank.write_width, bank.block, bank.anchors
        self.transport = transport
        self.lam = bank.lam if lam is None else lam
        if not 0 <= self.lam < 1:
            raise ValueError('Transport must be invertible convex mixing')
        self.storage_dtype, self.rotated_cache = storage_dtype, rotated_cache
        for name in ('embed', 'initial_norm', 'initial', 'blocks', 'norm', 'output'):
            setattr(self, name, copy.deepcopy(getattr(bank, name)))
        self.register_buffer('address', addresses(self.c, self.depth, self.write_width))

    def transform(self, x, l, positions, inverse=False):
        if not self.rotated_cache:
            return x
        ids = write_indices(self.c, self.write_width, l).to(x.device) if l else None
        return rotate_public(x, positions, self.blocks[l].p, ids, inverse=inverse)

    def roundtrip(self, x, l, positions):
        dtype = x.dtype if x.dtype == torch.float64 else self.storage_dtype
        stored = self.transform(x, l, positions).to(dtype).to(x.dtype)
        return self.transform(stored, l, positions, inverse=True)

    def commit(self, z, l, positions):
        if self.transport == 'none' or self.lam == 0:
            return self.roundtrip(z, l, positions)
        if self.transport == 'anchor':
            # Actual stored anchors determine the summary, including rounding.
            pooled = projector(self.roundtrip(z, l, positions), self.block, self.anchors)
            anchor = positions % self.block < self.anchors
            moved = torch.where(anchor[None, :, None], z,
                                (1 - self.lam) * z + self.lam * pooled)
        else:
            count = torch.arange(1, z.shape[1] + 1, device=z.device, dtype=z.dtype)
            moved = (1 - self.lam) * z + self.lam * z.cumsum(1) / count[None, :, None]
        return self.roundtrip(moved, l, positions)

    def forward(self, tokens, states=False):
        h = self.embed(tokens)
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        source = self.roundtrip(self.initial(self.initial_norm(h)), 0, positions)
        sources, fragments = [], [source]
        for l, layer in enumerate(self.blocks):
            if l:
                ids = write_indices(self.c, self.write_width, l).to(h.device)
                z = source[..., ids] + layer.pub(layer.n1(h))
                committed = self.commit(z, l, positions)
                source = source.index_copy(-1, ids, committed)
                fragments.append(committed)
            sources.append(source)
            h = layer(h, source)
        logits = self.output(self.norm(h))
        return (logits, sources, fragments) if states else logits


class LocalCache:
    """Preallocated immutable fragments + O(L*w) streaming publication state.

    Reader gathers at most tile*c values, then runs ordinary online softmax.
    There is no n*c array, historical summary, attenuation, or score correction.
    Storage is pre-RoPE or rotated once at commit; main reader uses the latter.
    """
    def __init__(self, model, capacity, batch=1, tile=128):
        if capacity < 1 or tile < 1:
            raise ValueError('Positive capacity and tile required')
        self.model, self.capacity, self.batch, self.tile = model, capacity, batch, tile
        self.n = 0
        weight = model.initial.weight
        self.dtype = weight.dtype
        storage = self.dtype if self.dtype == torch.float64 else model.storage_dtype
        widths = [model.c] + [model.write_width] * (model.depth - 1)
        self.values = [torch.empty(batch, capacity, w, device=weight.device, dtype=storage)
                       for w in widths]
        accumulator = torch.float64 if self.dtype == torch.float64 else torch.float32
        self.sums = ([] if model.transport == 'none' or model.lam == 0 else
                     [torch.zeros(batch, model.write_width, device=weight.device,
                                  dtype=accumulator) for _ in range(model.depth - 1)])
        # Compile the static reader map once. Metadata is shared across sequences.
        offsets = [0]
        for w in widths:
            offsets.append(offsets[-1] + w)
        self.mapping = []
        for l in range(model.depth):
            row = []
            for bank in range(l + 1):
                mask = (model.address[l] >= offsets[bank]) & (model.address[l] < offsets[bank + 1])
                fields = mask.nonzero().flatten()
                if fields.numel():
                    row.append((bank, fields, model.address[l, fields] - offsets[bank]))
            self.mapping.append(row)

    def memory_bytes(self):
        size = lambda xs: sum(x.numel() * x.element_size() for x in xs)
        return dict(history=size(self.values), publication_state=size(self.sums),
                    historical_summaries=0, rolling_workspace=0,
                    persistent_total=size(self.values) + size(self.sums),
                    max_reader_values=self.batch * min(self.tile, self.capacity) * self.model.c,
                    note='Tensor payload; excludes static metadata, allocator, private activations and attention scratch')

    def _save(self, l, x):
        position = torch.tensor(self.n, device=x.device)
        self.values[l][:, self.n] = self.model.transform(x, l, position)
        return self.model.transform(self.values[l][:, self.n].to(self.dtype), l,
                                    position, inverse=True)

    def _read(self, l, start, end):
        out = torch.empty(self.batch, end - start, self.model.c,
                          dtype=self.dtype, device=self.values[0].device)
        for bank, fields, source in self.mapping[l]:
            out[..., fields] = self.values[bank][:, start:end, source].to(self.dtype)
        return out

    def _attention(self, l, q):
        m, layer = self.model, self.model.blocks[l]
        maximum = torch.full((self.batch, layer.heads), -torch.inf, dtype=self.dtype, device=q.device)
        denominator = torch.zeros_like(maximum)
        numerator = torch.zeros(*maximum.shape, layer.p, dtype=self.dtype, device=q.device)
        for start in range(0, self.n + 1, self.tile):
            end = min(start + self.tile, self.n + 1)
            positions = torch.arange(start, end, device=q.device)
            stored = self._read(l, start, end)
            if isinstance(layer.mn, nn.Identity):
                key, value = stored.split(layer.p, -1)
                if not m.rotated_cache:
                    key = rotate_key(key, positions, layer.p)
            else:
                # Correct reference for arbitrary source norm; no cheap-fold claim.
                source = rotate_public(stored, positions, layer.p, inverse=True) if m.rotated_cache else stored
                key, value = layer.mn(source).split(layer.p, -1)
                key = rotate_key(key, positions, layer.p)
            score = torch.einsum('bhp,bnp->bhn', q, key) / math.sqrt(layer.p)
            new_max = torch.maximum(maximum, score.amax(-1))
            correction = (maximum - new_max).exp()
            probability = (score - new_max[..., None]).exp()
            denominator = denominator * correction + probability.sum(-1)
            numerator = numerator * correction[..., None] + torch.einsum('bhn,bnp->bhp', probability, value)
            maximum = new_max
        return numerator / denominator[..., None]

    @torch.inference_mode()
    def append(self, token):
        if self.n >= self.capacity:
            raise ValueError('Capacity exceeded')
        m, t = self.model, self.n
        position = torch.tensor(t, device=token.device)
        h = m.embed(token)
        # A current-token work vector must not alias the immutable base bank.
        current = self._save(0, m.initial(m.initial_norm(h))).clone()
        if m.transport == 'anchor' and t % m.block == 0:
            for total in self.sums:
                total.zero_()
        for l, layer in enumerate(m.blocks):
            if l:
                ids = write_indices(m.c, m.write_width, l).to(h.device)
                z = current[:, ids] + layer.pub(layer.n1(h))
                if self.sums:
                    if m.transport == 'prefix':
                        self.sums[l - 1] += z.to(self.sums[l - 1].dtype)
                        z = (1 - m.lam) * z + m.lam * (self.sums[l - 1] / (t + 1)).to(z.dtype)
                    elif t % m.block >= m.anchors:
                        z = (1 - m.lam) * z + m.lam * (self.sums[l - 1] / m.anchors).to(z.dtype)
                saved = self._save(l, z)
                current[:, ids] = saved
                if self.sums and m.transport == 'anchor' and t % m.block < m.anchors:
                    self.sums[l - 1] += saved.to(self.sums[l - 1].dtype)
            q = layer.q(layer.n1(h)).view(self.batch, layer.heads, layer.p)
            q = rotate_key(q, position, layer.p)
            out = self._attention(l, q)
            h = h + layer.o(out.flatten(-2))
            a, b = layer.up(layer.n2(h)).chunk(2, -1)
            h = h + layer.down(F.silu(a) * b)
        self.n += 1
        return m.output(m.norm(h))[:, None]

    @torch.inference_mode()
    def prefill(self, tokens):
        self.n = 0
        for total in self.sums:
            total.zero_()
        for j in range(tokens.shape[1]):
            self.append(tokens[:, j])
