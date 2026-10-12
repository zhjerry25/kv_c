"""Experimental public feature bank with an idempotent causal transport.

Changes the architecture deliberately: joint K/V coordinate writes and a
block-anchor projection replace dense U and uniform prefix averaging.
The dense and compact readers below define the SAME candidate, not the legacy
model. CPU implementations verify semantics/quality, never GPU latency.
"""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from train_public_memory import rope


def projector(x, block, anchors):
    """First R rows per block are identity; subsequent rows read their mean.

    P is causal, nonnegative, row-stochastic and P^2=P. An incomplete anchor
    prefix has no non-anchor consumers, so padding cannot leak future tokens.
    """
    b, n, c = x.shape
    if not n:
        return x
    blocks = math.ceil(n / block)
    padded = F.pad(x, (0, 0, 0, blocks * block - n)).view(b, blocks, block, c)
    mean = padded[:, :, :anchors].mean(2)
    indices = torch.arange(n, device=x.device)
    return torch.where((indices % block < anchors)[None, :, None], x,
                       mean[:, indices // block])


def write_indices(c, width, stage):
    if c % 4 or width % 4 or not 0 < width <= c:
        raise ValueError('Joint writes require complete RoPE pairs in both K and V')
    half = c // 2
    key = (torch.arange(width // 2) + (stage - 1) * (width // 2)) % half
    return torch.cat((key, key + half))


def build_addresses(c, depth, width, lam):
    """Static latest-writer metadata, no data-dependent routing."""
    address = torch.arange(c)
    owner = torch.zeros(c, dtype=torch.long)
    addresses, coefficients = [address.clone()], [torch.ones(c, dtype=torch.float64)]
    for l in range(1, depth):
        ids = write_indices(c, width, l)
        address[ids] = c + (l - 1) * width + torch.arange(width)
        owner[ids] = l
        addresses.append(address.clone())
        coefficients.append((1 - lam) ** (l - owner).double())
    return torch.stack(addresses), torch.stack(coefficients)


def rotary_coefficients(positions, frequency, dtype, block=None):
    if block is None:
        phase = positions.float()[..., None] * frequency
        return phase.cos().to(dtype), phase.sin().to(dtype)
    start = (positions // block * block).float()[..., None] * frequency
    local = (positions % block).float()[..., None] * frequency
    cb, sb, ct, st = (x.to(dtype) for x in (start.cos(), start.sin(), local.cos(), local.sin()))
    return cb * ct - sb * st, sb * ct + cb * st


def rotate_key(x, positions, p, ids=None, inverse=False, block=None):
    key_ids = torch.arange(x.shape[-1], device=x.device) if ids is None else ids
    frequency = 10000. ** (-(key_ids[0::2] % p).float() / p)
    cosine, sine = rotary_coefficients(positions, frequency, x.dtype, block)
    if inverse:
        norm = cosine.square() + sine.square()
        cosine, sine = cosine / norm, -sine / norm
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a * cosine - b * sine, a * sine + b * cosine), -1).flatten(-2)


def rotate_public(x, positions, p, ids=None, inverse=False, block=None):
    """Rotate full/selected key pairs BEFORE their public transport/write."""
    half = x.shape[-1] // 2
    key_ids = None if ids is None else ids[:half]
    key = rotate_key(x[..., :half], positions, p, key_ids, inverse, block)
    return torch.cat((key, x[..., half:]), -1)


def private_block(layer, h, source, rotated, rope_block=None):
    if not rotated and rope_block is None:
        return layer(h, source)
    batch, n, _ = h.shape
    key, value = layer.mn(source).split(layer.p, -1)
    positions = torch.arange(n, device=h.device)
    q = layer.q(layer.n1(h)).view(batch, n, layer.heads, layer.p).transpose(1, 2)
    q = rotate_key(q, positions, layer.p, block=rope_block)
    if not rotated:
        key = rotate_key(key, positions, layer.p, block=rope_block)
    out = F.scaled_dot_product_attention(q, key[:, None].expand(-1, layer.heads, -1, -1),
                                        value[:, None].expand(-1, layer.heads, -1, -1), is_causal=True)
    h = h + layer.o(out.transpose(1, 2).flatten(-2))
    a, b = layer.up(layer.n2(h)).chunk(2, -1)
    return h + layer.down(F.silu(a) * b)


class AbsorbedBank(nn.Module):
    """Legacy backbone, full-width source, new public edit/transport rules.

    Source storage rounds only base and committed writes. Untouched coordinates
    are recovered analytically, avoiding repeated canonical rounding.
    """
    def __init__(self, teacher, write_width=16, block=128, anchors=8,
                 init_scale=1., storage_dtype=torch.float32, factorized=False,
                 mix_budget=.25, rotated_transport=False, rotated_cache=False,
                 factorized_rope=False):
        super().__init__()
        if teacher.mode != 'moments_uniform' or not 1 <= anchors <= block:
            raise ValueError('Unsupported teacher or anchor projector')
        self.c, self.depth = teacher.c, teacher.depth
        self.write_width, self.block, self.anchors = write_width, block, anchors
        self.storage_dtype, self.factorized = storage_dtype, factorized
        self.rotated_transport = rotated_transport
        self.rotated_cache = rotated_cache
        self.factorized_rope = factorized_rope
        if factorized_rope and not rotated_cache:
            raise ValueError('The factorized RoPE reader requires a rotated cache')
        if rotated_transport and rotated_cache:
            raise ValueError('Choose rotated transport OR a rotated cache for the original transport')
        self.mix_budget = mix_budget
        self.lam = mix_budget / (self.depth - 1)
        if not 0 <= self.lam < 1:
            raise ValueError('Transport must be a convex combination')
        self.embed = copy.deepcopy(teacher.embed)
        self.initial_norm = copy.deepcopy(teacher.initial_norm)
        self.initial = copy.deepcopy(teacher.initial)
        self.blocks = copy.deepcopy(teacher.blocks)
        self.norm, self.output = copy.deepcopy(teacher.norm), copy.deepcopy(teacher.output)
        for l, layer in enumerate(self.blocks):
            if not l:
                continue
            ids = write_indices(self.c, write_width, l)
            old_p, old_u = layer.pub.weight.detach(), layer.expand.weight.detach()
            if factorized:
                layer.expand = nn.Linear(old_p.shape[0], write_width, bias=False).to(old_p)
                layer.expand.weight.data.copy_(init_scale * old_u[ids])
            else:
                layer.pub = nn.Linear(old_p.shape[1], write_width, bias=False).to(old_p)
                layer.pub.weight.data.copy_(init_scale * (old_u[ids] @ old_p))
                layer.expand = None
        address, coefficient = build_addresses(self.c, self.depth, write_width, self.lam)
        self.register_buffer('address', address)
        self.register_buffer('coefficient', coefficient)

    def rounded(self, x):
        dtype = x.dtype if x.dtype == torch.float64 else self.storage_dtype
        return x.to(dtype).to(x.dtype)

    def delta(self, layer, h):
        value = layer.pub(layer.n1(h))
        return layer.expand(value) if self.factorized else value

    def stored_to_source(self, value, l, positions):
        if not self.rotated_cache:
            return value
        ids = write_indices(self.c, self.write_width, l).to(value.device) if l else None
        return rotate_public(value, positions, self.blocks[l].p, ids, inverse=True,
                             block=self.block if self.factorized_rope else None)

    def source_to_stored(self, value, l, positions):
        if not self.rotated_cache:
            return value
        ids = write_indices(self.c, self.write_width, l).to(value.device) if l else None
        return rotate_public(value, positions, self.blocks[l].p, ids,
                             block=self.block if self.factorized_rope else None)

    def assemble(self, banks, l):
        positions = torch.arange(banks[0].shape[1], device=banks[0].device)
        banks = [self.stored_to_source(value, s, positions) for s, value in enumerate(banks)]
        raw = torch.cat(banks, -1)[..., self.address[l]]
        mean = torch.cat([projector(x, self.block, self.anchors) for x in banks], -1)
        mean = mean[..., self.address[l]]
        a = self.coefficient[l].to(raw.dtype)
        return a * raw + (1 - a) * mean

    def forward(self, tokens, states=False):
        h = self.embed(tokens)
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        base = self.initial(self.initial_norm(h))
        if self.rotated_transport:
            base = rotate_public(base, positions, self.blocks[0].p)
        banks = [self.rounded(self.source_to_stored(base, 0, positions))]
        source, sources = self.stored_to_source(banks[0], 0, positions), []
        for l, layer in enumerate(self.blocks):
            if l:
                ids = write_indices(self.c, self.write_width, l).to(tokens.device)
                delta = self.delta(layer, h)
                if self.rotated_transport:
                    delta = rotate_public(delta, positions, layer.p, ids)
                z = source[..., ids] + delta
                # Only first R rows define P; use their stored precision.
                rounded_anchor = self.stored_to_source(self.rounded(self.source_to_stored(z, l, positions)), l, positions)
                mean = projector(rounded_anchor, self.block, self.anchors)
                local = torch.arange(tokens.shape[1], device=tokens.device) % self.block
                committed = torch.where((local < self.anchors)[None, :, None], z,
                                        (1 - self.lam) * z + self.lam * mean)
                banks.append(self.rounded(self.source_to_stored(committed, l, positions)))
                source = self.assemble(banks, l)
            sources.append(source)
            h = private_block(layer, h, source, self.rotated_transport,
                              self.block if self.factorized_rope else None)
        logits = self.output(self.norm(h))
        return (logits, sources, banks) if states else logits

    def dense_recurrence(self, tokens):
        """Independent, unrounded real/FP64 oracle; no latest-writer formula."""
        h = self.embed(tokens)
        source = self.initial(self.initial_norm(h))
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        if self.rotated_transport:
            source = rotate_public(source, positions, self.blocks[0].p)
        for l, layer in enumerate(self.blocks):
            if l:
                ids = write_indices(self.c, self.write_width, l).to(tokens.device)
                increment = torch.zeros_like(source)
                delta = self.delta(layer, h)
                if self.rotated_transport:
                    delta = rotate_public(delta, positions, layer.p, ids)
                increment[..., ids] = delta
                z = source + increment
                source = (1 - self.lam) * z + self.lam * projector(z, self.block, self.anchors)
            h = private_block(layer, h, source, self.rotated_transport,
                              self.block if self.factorized_rope else None)
        return self.output(self.norm(h))


class BankCache:
    """Preallocated compact append reference. No n*c rolling source allocated.

    _source materializes CPU attention input temporarily; a GPU fused gather
    reader would keep only a tile. This reference makes no GPU memory/latency
    claim about its temporary allocations.
    """
    def __init__(self, model, capacity, batch=1, absorbed=False):
        self.model, self.capacity, self.batch = model, capacity, batch
        self.absorbed = absorbed
        if absorbed and (not (model.rotated_transport or model.rotated_cache) or
                         any(not isinstance(b.mn, nn.Identity) for b in model.blocks)):
            raise ValueError('Absorbed oracle currently requires rotated identity-normalized source')
        self.n = 0
        device = model.initial.weight.device
        dtype = model.initial.weight.dtype
        storage = dtype if dtype == torch.float64 else model.storage_dtype
        widths = [model.c] + [model.write_width] * (model.depth - 1)
        self.values = [torch.empty(batch, capacity, w, device=device, dtype=storage) for w in widths]
        accumulator = torch.float64 if dtype == torch.float64 else torch.float32
        self.sums = [torch.zeros(batch, math.ceil(capacity / model.block), w,
                                 device=device, dtype=accumulator) for w in widths]

    def memory_bytes(self):
        size = lambda xs: sum(x.numel() * x.element_size() for x in xs)
        return dict(history=size(self.values), block_summaries=size(self.sums),
                    rolling_workspace=0, allocated_total=size(self.values) + size(self.sums))

    def _save(self, l, x):
        position = torch.tensor(self.n, device=x.device)
        self.values[l][:, self.n] = self.model.source_to_stored(x, l, position)
        if self.n % self.model.block < self.model.anchors:
            stored = self.values[l][:, self.n].to(self.sums[l].dtype)
            self.sums[l][:, self.n // self.model.block] += self.model.stored_to_source(stored, l, position)

    def _source(self, l):
        m, n = self.model, self.n + 1
        dtype = m.initial.weight.dtype
        indices = torch.arange(n, device=self.values[0].device)
        raw = torch.cat([m.stored_to_source(x[:, :n].to(dtype), s, indices)
                         for s, x in enumerate(self.values[:l + 1])], -1)[..., m.address[l]]
        means = torch.cat([x[:, indices // m.block] / m.anchors
                           for x in self.sums[:l + 1]], -1)[..., m.address[l]]
        means = torch.where((indices % m.block < m.anchors)[None, :, None], raw, means)
        a = m.coefficient[l].to(dtype)
        return a * raw + (1 - a) * means

    def _current(self, l):
        m, t = self.model, self.n
        dtype = m.initial.weight.dtype
        position = torch.tensor(t, device=self.values[0].device)
        raw = torch.cat([m.stored_to_source(x[:, t].to(dtype), s, position)
                         for s, x in enumerate(self.values[:l + 1])], -1)[..., m.address[l]]
        if t % m.block < m.anchors:
            return raw
        mean = torch.cat([x[:, t // m.block] / m.anchors for x in self.sums[:l + 1]], -1)[..., m.address[l]]
        a = m.coefficient[l].to(dtype)
        return a * raw + (1 - a) * mean

    def _absorbed_attention(self, l, q):
        """Algebra oracle: block score bias + block probability mass.

        CPU concatenates raw banks for clarity; a GPU reader must gather only
        addressed tiles. NO reconstructed source is used by this calculation.
        """
        m, n = self.model, self.n + 1
        layer = m.blocks[l]
        raw = torch.cat([x[:, :n].to(q.dtype) for x in self.values[:l + 1]], -1)[..., m.address[l]]
        blocks = math.ceil(n / m.block)
        mean = torch.cat([x[:, :blocks] / m.anchors for x in self.sums[:l + 1]], -1)[..., m.address[l]]
        ak, av = m.coefficient[l].to(q.dtype).split(layer.p)
        key, value = raw.split(layer.p, -1)
        mk, mv = mean.split(layer.p, -1)
        indices = torch.arange(n, device=q.device)
        anchor = indices % m.block < m.anchors
        qa = q * ak
        score = torch.einsum('bhp,bnp->bhn', qa, key)
        if m.rotated_cache:
            # Ordinary pre-RoPE transport is conjugated in the stored frame.
            # Its score correction is a block-specific sinusoidal bias, not a
            # reconstruction of historical keys. alpha is constant per pair.
            assert torch.equal(ak[0::2], ak[1::2])
            qb = q * (1 - ak)
            qe, qo = qb[..., 0::2], qb[..., 1::2]
            me, mo = mk[..., 0::2], mk[..., 1::2]
            freq = 10000. ** (-torch.arange(0, layer.p, 2, device=q.device).float() / layer.p)
            if m.factorized_rope:
                start = torch.arange(blocks, device=q.device) * m.block
                cb, sb = rotary_coefficients(start, freq, q.dtype)
                me, mo = me * cb - mo * sb, me * sb + mo * cb
            a = qe[:, :, None] * me[:, None] + qo[:, :, None] * mo[:, None]
            b = qo[:, :, None] * me[:, None] - qe[:, :, None] * mo[:, None]
            if m.factorized_rope:
                local = torch.arange(m.block, device=q.device)
                cos, sin = rotary_coefficients(local, freq, q.dtype)
                bias = (torch.einsum('bhkf,jf->bhkj', a, cos) +
                        torch.einsum('bhkf,jf->bhkj', b, sin)).flatten(-2)[..., :n]
            else:
                phase = torch.arange(blocks * m.block, device=q.device).float()[:, None] * freq
                cos = phase.cos().to(q.dtype).view(blocks, m.block, -1)
                sin = phase.sin().to(q.dtype).view(blocks, m.block, -1)
                bias = (torch.einsum('bhkf,kjf->bhkj', a, cos) +
                        torch.einsum('bhkf,kjf->bhkj', b, sin)).flatten(-2)[..., :n]
            score = score + bias
        else:
            bias = torch.einsum('bhp,bkp->bhk', q * (1 - ak), mk)
            score = score + bias[:, :, indices // m.block]
        anchor_score = torch.einsum('bhp,bnp->bhn', q, key)
        score = torch.where(anchor[None, None], anchor_score, score) / math.sqrt(layer.p)
        probability = score.softmax(-1)
        nonanchor = probability * (~anchor)[None, None]
        out = torch.einsum('bhn,bnp->bhp', nonanchor, value) * av
        out = out + torch.einsum('bhn,bnp->bhp', probability * anchor[None, None], value)
        mass = F.pad(nonanchor, (0, blocks * m.block - n)).view(self.batch, layer.heads, blocks, m.block).sum(-1)
        out = out + torch.einsum('bhk,bkp->bhp', mass, mv) * (1 - av)
        return out

    @torch.inference_mode()
    def append(self, token):
        if self.n >= self.capacity:
            raise ValueError('Capacity exceeded')
        m, t = self.model, self.n
        h = m.embed(token)
        position = torch.tensor(t, device=h.device)
        base = m.initial(m.initial_norm(h))
        if m.rotated_transport:
            base = rotate_public(base, position, m.blocks[0].p)
        self._save(0, base)
        current = m.stored_to_source(self.values[0][:, t].to(h.dtype), 0, position)
        for l, layer in enumerate(m.blocks):
            if l:
                ids = write_indices(m.c, m.write_width, l).to(h.device)
                delta = m.delta(layer, h)
                if m.rotated_transport:
                    delta = rotate_public(delta, position, layer.p, ids)
                z = current[:, ids] + delta
                if t % m.block >= m.anchors:
                    z = (1 - m.lam) * z + m.lam * self.sums[l][:, t // m.block] / m.anchors
                self._save(l, z)
            if self.absorbed:
                current = self._current(l)
            else:
                source = self._source(l)
                current = source[:, -1]
                key, value = layer.mn(source).split(layer.p, -1)
                if not m.rotated_transport:
                    if m.factorized_rope:
                        key = rotate_key(key, torch.arange(t + 1, device=h.device), layer.p, block=m.block)
                    else:
                        key = rope(key[:, None])[:, 0]
            phase = t * 10000. ** (-torch.arange(0, layer.p, 2, device=h.device).float() / layer.p)
            q = layer.q(layer.n1(h)).view(self.batch, layer.heads, layer.p)
            a, b = q[..., 0::2], q[..., 1::2]
            q = torch.stack((a * phase.cos() - b * phase.sin(),
                             a * phase.sin() + b * phase.cos()), -1).flatten(-2)
            if m.factorized_rope:
                raw_q = layer.q(layer.n1(h)).view(self.batch, layer.heads, layer.p)
                q = rotate_key(raw_q, position, layer.p, block=m.block)
            if self.absorbed:
                out = self._absorbed_attention(l, q)
            else:
                score = torch.einsum('bhp,bnp->bhn', q, key) / math.sqrt(layer.p)
                out = torch.einsum('bhn,bnp->bhp', score.softmax(-1), value)
            h = h + layer.o(out.flatten(-2))
            a, b = layer.up(layer.n2(h)).chunk(2, -1)
            h = h + layer.down(F.silu(a) * b)
        self.n += 1
        return m.output(m.norm(h))[:, None]

    @torch.inference_mode()
    def prefill(self, tokens):
        self.n = 0
        for sums in self.sums:
            sums.zero_()
        for j in range(tokens.shape[1]):
            self.append(tokens[:, j])
