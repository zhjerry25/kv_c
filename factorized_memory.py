"""Native and cross-layer cyclic MLA / fully contextual TPA.

Training uses materialized attention for the small CPU benchmark. Streaming
MLA absorbs the key up-projection into the query and applies the value
up-projection after latent aggregation. TPA contracts tile-local factors.
Neither decoder stores or materializes expanded historical K/V.
"""
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
import math

import torch
from torch import nn
from torch.nn import functional as F

from cyclic_memory import RMSNorm
from fragment_cache import Component, FragmentCache, FragmentLayout
from narrow_cache import rotary_coefficients, rotate


@dataclass(frozen=True)
class AttentionConfig:
    vocab: int
    attention: str = 'mla'
    cache_mode: str = 'cyclic'
    write_fraction: str = '3/8'
    d: int = 128
    depth: int = 8
    heads: int = 4
    head_dim: int = 32
    ffn_width: int = 512
    kv_latent_dim: int = 64
    query_latent_dim: int = 64
    rope_dim: int = 16
    query_rank: int = 6
    key_rank: int = 2
    value_rank: int = 2
    pub_init_std: float = .0016
    share_rope: bool = True

    def __post_init__(self):
        if self.attention not in ('mla', 'tpa') or self.cache_mode not in ('native', 'cyclic'):
            raise ValueError('MLA/TPA and native/cyclic required')
        if not isinstance(self.share_rope, bool) or (self.attention != 'mla' and not self.share_rope):
            raise ValueError('share_rope=False is an MLA option')
        dimensions = (self.vocab, self.d, self.depth, self.heads, self.head_dim, self.ffn_width,
                      self.kv_latent_dim, self.query_latent_dim, self.rope_dim,
                      self.query_rank, self.key_rank, self.value_rank)
        if min(dimensions) < 1 or self.rope_dim % 2 or self.head_dim % 2:
            raise ValueError('Positive dimensions and complete rotary pairs required')
        if not math.isfinite(self.pub_init_std) or self.pub_init_std <= 0:
            raise ValueError('Positive finite publication initialization required')
        if not 0 < Fraction(self.write_fraction) <= 1:
            raise ValueError('Write fraction must be in (0, 1]')
        FragmentLayout(self.depth, self.components, self.cache_mode)

    @property
    def components(self):
        groups = [('latent', self.kv_latent_dim, 0), ('rope', self.rope_dim, self.rope_dim)]
        if self.attention == 'tpa':
            groups = [('a_key', self.key_rank * self.heads, 0),
                      ('b_key', self.key_rank * self.head_dim, self.head_dim),
                      ('a_value', self.value_rank * self.heads, 0),
                      ('b_value', self.value_rank * self.head_dim, 0)]
        fraction = Fraction(1) if self.cache_mode == 'native' else Fraction(self.write_fraction)
        result = []
        for name, width, rotary in groups:
            write = Fraction(width) if self.independent_rope and name == 'rope' else width * fraction
            if write.denominator != 1:
                raise ValueError(f'Nonintegral write width for {name}')
            result.append(Component(name, width, int(write), rotary))
        return tuple(result)

    @property
    def independent_rope(self):
        return self.attention == 'mla' and self.cache_mode == 'cyclic' and not self.share_rope

    @property
    def source_width(self):
        return sum(c.width for c in self.components)

    @property
    def write_width(self):
        return sum(c.write_width for c in self.components)


class AttentionBlock(nn.Module):
    def __init__(self, cfg, layer):
        super().__init__()
        self.n1, self.n2 = RMSNorm(cfg.d), RMSNorm(cfg.d)
        if cfg.attention == 'mla':
            self.q_down = nn.Linear(cfg.d, cfg.query_latent_dim, bias=False)
            self.q_up = nn.Linear(cfg.query_latent_dim, cfg.heads * (cfg.head_dim + cfg.rope_dim), bias=False)
            self.k_up = nn.Linear(cfg.kv_latent_dim, cfg.heads * cfg.head_dim, bias=False)
            self.v_up = nn.Linear(cfg.kv_latent_dim, cfg.heads * cfg.head_dim, bias=False)
        else:
            self.q_a = nn.Linear(cfg.d, cfg.query_rank * cfg.heads, bias=False)
            self.q_b = nn.Linear(cfg.d, cfg.query_rank * cfg.head_dim, bias=False)
        self.o = nn.Linear(cfg.heads * cfg.head_dim, cfg.d, bias=False)
        self.up = nn.Linear(cfg.d, 2 * cfg.ffn_width, bias=False)
        self.down = nn.Linear(cfg.ffn_width, cfg.d, bias=False)
        self.rope_key = nn.Linear(cfg.d, cfg.rope_dim, bias=False) if cfg.independent_rope else None
        self.pub = nn.ModuleList()
        if cfg.cache_mode == 'native' or layer:
            self.pub.extend(nn.Linear(cfg.d, c.write_width, bias=False) for c in cfg.components
                            if not (cfg.independent_rope and c.name == 'rope'))

    def publication(self, normalized):
        values = [p(normalized) for p in self.pub]
        if self.rope_key is not None:
            values.append(self.rope_key(normalized))
        return torch.cat(values, -1)

    def feedforward(self, h):
        gate, up = self.up(self.n2(h)).chunk(2, -1)
        return h + self.down(F.silu(gate) * up)


class AttentionMemory(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = cfg = config
        self.layout = FragmentLayout(cfg.depth, cfg.components, cfg.cache_mode)
        self.embed = nn.Embedding(cfg.vocab, cfg.d)
        self.initial_norm = RMSNorm(cfg.d) if cfg.cache_mode == 'cyclic' else None
        self.initial = nn.ModuleList(nn.Linear(cfg.d, c.width, bias=False)
                                    for c in cfg.components if not (cfg.independent_rope and c.name == 'rope')
                                    ) if cfg.cache_mode == 'cyclic' else nn.ModuleList()
        self.blocks = nn.ModuleList(AttentionBlock(cfg, l) for l in range(cfg.depth))
        self.norm, self.output = RMSNorm(cfg.d), nn.Linear(cfg.d, cfg.vocab, bias=False)
        for l in range(cfg.depth):
            self.register_buffer(f'write_ids_{l}', torch.tensor(self.layout.indices[l]), persistent=False)
            self.register_buffer(f'rotary_ids_{l}', torch.tensor(self.layout.rotary_ids[l], dtype=torch.long), persistent=False)
            self.register_buffer(f'rotary_frequency_{l}', torch.tensor(self.layout.rotary_frequencies[l], dtype=torch.float32), persistent=False)
        self.apply(self._initialize)
        for layer in self.blocks:
            nn.init.normal_(layer.o.weight, std=.02 / math.sqrt(2 * cfg.depth))
            nn.init.normal_(layer.down.weight, std=.02 / math.sqrt(2 * cfg.depth))
            if cfg.attention == 'tpa':
                for projection in (layer.q_a, layer.q_b):
                    nn.init.xavier_uniform_(projection.weight)
                for component, projection in zip(cfg.components, layer.pub):
                    # A narrow projection has the variance of selected rows from
                    # the full native factor map, rather than a larger fan-out gain.
                    bound = math.sqrt(6 / (cfg.d + component.width))
                    nn.init.uniform_(projection.weight, -bound, bound)
            elif cfg.cache_mode == 'cyclic':
                for projection in layer.pub:
                    nn.init.normal_(projection.weight, std=cfg.pub_init_std)
        if cfg.attention == 'tpa':
            for projection in self.initial:
                nn.init.xavier_uniform_(projection.weight)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
            nn.init.normal_(module.weight, std=.02)

    def physical_fragment(self, source, layer, positions, storage_dtype=None):
        ids = getattr(self, f'rotary_ids_{layer}')
        if ids.numel():
            phase = positions.float()[..., None] * getattr(self, f'rotary_frequency_{layer}').float()
            rotated = rotate(source.index_select(-1, ids), phase.cos().to(source.dtype), phase.sin().to(source.dtype))
            source = source.index_copy(-1, ids, rotated)
        return source if storage_dtype is None else source.to(storage_dtype).to(source.dtype)

    def initial_publication(self, h):
        normalized = self.initial_norm(h)
        return torch.cat([p(normalized) for p in self.initial], -1)

    def source_publication(self, h, normalized, layer):
        if self.config.cache_mode == 'cyclic' and layer == 0:
            source = self.initial_publication(h)
            if self.config.independent_rope:
                source = torch.cat((source, self.blocks[0].rope_key(normalized)), -1)
            return source
        return self.blocks[layer].publication(normalized)

    def query(self, layer, normalized, positions):
        cfg = self.config
        b, n = normalized.shape[:2]
        if cfg.attention == 'mla':
            q = layer.q_up(layer.q_down(normalized)).reshape(
                b, n, cfg.heads, cfg.head_dim + cfg.rope_dim).transpose(1, 2)
            content, positional = q.split((cfg.head_dim, cfg.rope_dim), -1)
            cosine, sine = rotary_coefficients(positions, cfg.rope_dim, q.dtype, q.device)
            return torch.cat((content, rotate(positional, cosine, sine)), -1)
        a = layer.q_a(normalized).reshape(b, n, cfg.query_rank, cfg.heads)
        features = layer.q_b(normalized).reshape(b, n, cfg.query_rank, cfg.head_dim)
        cosine, sine = rotary_coefficients(positions, cfg.head_dim, features.dtype, features.device)
        features = rotate(features, cosine[:, None], sine[:, None])
        return torch.einsum('btrh,btrd->bhtd', a, features) / cfg.query_rank

    def tpa_factors(self, source):
        cfg = self.config
        ak, bk, av, bv = source.split([c.width for c in cfg.components], -1)
        shape = source.shape[:-1]
        return (ak.reshape(*shape, cfg.key_rank, cfg.heads),
                bk.reshape(*shape, cfg.key_rank, cfg.head_dim),
                av.reshape(*shape, cfg.value_rank, cfg.heads),
                bv.reshape(*shape, cfg.value_rank, cfg.head_dim))

    def parallel_attention(self, layer, query, source, causal=True):
        cfg = self.config
        b, n = source.shape[:2]
        if cfg.attention == 'mla':
            latent, positional = source.split((cfg.kv_latent_dim, cfg.rope_dim), -1)
            content = layer.k_up(latent).reshape(b, n, cfg.heads, cfg.head_dim).transpose(1, 2)
            key = torch.cat((content, positional[:, None].expand(-1, cfg.heads, -1, -1)), -1)
            value = layer.v_up(latent).reshape(b, n, cfg.heads, cfg.head_dim).transpose(1, 2)
            # Equal Q/K/V widths enable CPU flash SDPA; padded value channels are zero.
            value = F.pad(value, (0, cfg.rope_dim))
            return F.scaled_dot_product_attention(query, key, value, is_causal=causal)[..., :cfg.head_dim]
        ak, bk, av, bv = self.tpa_factors(source)
        key = torch.einsum('btrh,btrd->bhtd', ak, bk) / cfg.key_rank
        value = torch.einsum('btrh,btrd->bhtd', av, bv) / cfg.value_rank
        return F.scaled_dot_product_attention(query, key, value, is_causal=causal)

    def _forward(self, tokens, storage_dtype=None, decoder=None):
        if tokens.ndim != 2 or tokens.shape[1] < 1:
            raise ValueError('Nonempty batch/time tokens required')
        cfg = self.config
        h = self.embed(tokens)
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        state = None
        for l, layer in enumerate(self.blocks):
            normalized = layer.n1(h)
            raw = self.source_publication(h, normalized, l)
            fragment = self.physical_fragment(raw, l, positions, storage_dtype)
            if cfg.cache_mode == 'native' or l == 0:
                state = fragment
            else:
                state = state.index_copy(-1, getattr(self, f'write_ids_{l}'), fragment)
            if decoder is not None:
                decoder.cache.store_prefill_layer(l, fragment)
            last_query = decoder is not None and l == cfg.depth - 1
            if last_query:
                h, normalized = h[:, -1:], normalized[:, -1:]
            query = self.query(layer, normalized, positions[-1:] if last_query else positions)
            out = self.parallel_attention(layer, query, state, causal=not last_query)
            h = layer.feedforward(h + layer.o(out.transpose(1, 2).flatten(-2)))
        if decoder is not None:
            decoder.cache.finish_prefill(tokens.shape[1])
        return self.output(self.norm(h))

    def forward(self, tokens):
        return self._forward(tokens)

    def forward_stored(self, tokens, dtype=torch.bfloat16):
        return self._forward(tokens, dtype)

    def new_cache(self, capacity, batch=1, tile=128, dtype=torch.bfloat16):
        return AttentionDecoder(self, capacity, batch, tile, dtype)

    def save(self, path, **metadata):
        torch.save(dict(config=asdict(self.config), state_dict=self.state_dict(), metadata=metadata), path)

    @classmethod
    def load(cls, path):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        model = cls(AttentionConfig(**saved['config'])).to(dtype=saved['state_dict']['embed.weight'].dtype)
        model.load_state_dict(saved['state_dict'])
        return model


def paired_model(config, seed):
    """Share all private tensors and the bottom projection, never trained weights.

TPA publications select native Xavier rows. MLA narrow publications select
rows from a second, seed-paired full Gaussian bank with the project's .0016 std.
"""
    torch.manual_seed(seed)
    reference = AttentionMemory(replace(config, cache_mode='native'))
    if config.cache_mode == 'native':
        return reference
    torch.manual_seed(100000 + seed)
    model = AttentionMemory(config)
    common = {k: v for k, v in reference.state_dict().items() if '.pub.' not in k}
    model.load_state_dict(common, strict=False)
    generator = torch.Generator().manual_seed(100000 + seed)
    with torch.no_grad():
        model.initial_norm.load_state_dict(reference.blocks[0].n1.state_dict())
        for dst, src in zip(model.initial, reference.blocks[0].pub):
            dst.weight.copy_(src.weight)
        if config.independent_rope:
            for layer, ref in zip(model.blocks, reference.blocks):
                layer.rope_key.weight.copy_(ref.pub[1].weight)
        for l in range(1, config.depth):
            for index, (c, src) in enumerate(zip(config.components, reference.blocks[l].pub)):
                # Per-component local coordinates, independent of packed offsets.
                local = torch.tensor([((l - 1) * c.write_width + j) % c.width
                                      for j in range(c.write_width)])
                full = src.weight if config.attention == 'tpa' else (
                    torch.randn(c.width, config.d, generator=generator) * config.pub_init_std)
                # Consume the original full Gaussian bank even for independent
                # RoPE, keeping every latent publication paired with the old arm.
                if not (config.independent_rope and c.name == 'rope'):
                    model.blocks[l].pub[index].weight.copy_(full.index_select(0, local))
    return model


class AttentionDecoder:
    def __init__(self, model, capacity, batch, tile, dtype):
        self.model = model
        weight = model.embed.weight
        self.cache = FragmentCache(model.layout, capacity, batch, tile, dtype=dtype,
                                   compute_dtype=weight.dtype, device=weight.device,
                                   shared_buffers=tuple(model.buffers()))

    def memory_bytes(self):
        return self.cache.memory_bytes()

    def reset(self):
        self.cache.reset()

    @torch.inference_mode()
    def attention(self, l, query):
        model, cfg = self.model, self.model.config
        layer = model.blocks[l]
        q = query[:, :, 0]
        if cfg.attention == 'mla':
            qc, qr = q.split((cfg.head_dim, cfg.rope_dim), -1)
            qc = torch.einsum('bhd,hdc->bhc', qc, layer.k_up.weight.reshape(
                cfg.heads, cfg.head_dim, cfg.kv_latent_dim))
            output_width, scale = cfg.kv_latent_dim, math.sqrt(cfg.head_dim + cfg.rope_dim)
        else:
            output_width, scale = cfg.head_dim, math.sqrt(cfg.head_dim)
        maximum = torch.full(q.shape[:2], -torch.inf, dtype=q.dtype, device=q.device)
        mass = torch.zeros_like(maximum)
        numerator = torch.zeros(*q.shape[:2], output_width, dtype=q.dtype, device=q.device)
        for start in range(0, self.cache.length, self.cache.tile):
            source = self.cache.read_tile(l, start, min(start + self.cache.tile, self.cache.length))
            if cfg.attention == 'mla':
                latent, positional = source.split((cfg.kv_latent_dim, cfg.rope_dim), -1)
                scores = (torch.einsum('bhc,btc->bht', qc, latent)
                          + torch.einsum('bhr,btr->bht', qr, positional)) / scale
            else:
                ak, bk, av, bv = model.tpa_factors(source)
                dots = torch.einsum('bhd,btrd->bhtr', q, bk)
                scores = (dots * ak.permute(0, 3, 1, 2)).sum(-1) / (cfg.key_rank * scale)
            next_maximum = torch.maximum(maximum, scores.amax(-1))
            correction = (maximum - next_maximum).exp()
            probability = (scores - next_maximum[..., None]).exp()
            if cfg.attention == 'mla':
                weighted = torch.einsum('bht,btc->bhc', probability, latent)
            else:
                weights = probability[..., None] * av.permute(0, 3, 1, 2)
                weighted = torch.einsum('bhtr,btrd->bhd', weights, bv) / cfg.value_rank
            numerator = numerator * correction[..., None] + weighted
            mass = mass * correction + probability.sum(-1)
            maximum = next_maximum
        out = numerator / mass[..., None]
        if cfg.attention == 'mla':
            out = torch.einsum('bhc,hdc->bhd', out, layer.v_up.weight.reshape(
                cfg.heads, cfg.head_dim, cfg.kv_latent_dim))
        return out

    @torch.inference_mode()
    def append(self, token):
        if token.ndim != 1 or token.shape[0] != self.cache.batch:
            raise ValueError('Incorrect append batch')
        if self.cache.n >= self.cache.capacity or self.cache.written != -1:
            raise ValueError('Full cache or incomplete previous token')
        model, cfg = self.model, self.model.config
        h = model.embed(token[:, None])
        positions = torch.tensor([self.cache.n], device=h.device)
        for l, layer in enumerate(model.blocks):
            normalized = layer.n1(h)
            raw = model.source_publication(h, normalized, l)
            physical = model.physical_fragment(raw, l, positions, self.cache.dtype)
            self.cache.store(l, physical[:, 0])
            query = model.query(layer, normalized, positions)
            out = self.attention(l, query)
            h = layer.feedforward(h + layer.o(out.flatten(-2))[:, None])
        self.cache.finish_token()
        return model.output(model.norm(h))

    @torch.inference_mode()
    def prefill(self, tokens):
        if tokens.ndim != 2 or tokens.shape[0] != self.cache.batch or tokens.shape[1] > self.cache.capacity:
            raise ValueError('Incorrect prefill batch or capacity')
        self.reset()
        if tokens.shape[1] == 0:
            return None
        return self.model._forward(tokens, self.cache.dtype, self)
