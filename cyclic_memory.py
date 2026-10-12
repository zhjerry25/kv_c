"""Cross-layer cyclic residual public K/V with immutable narrow fragments.

One narrow residual publication per layer, static complete-pair cyclic writes,
no prefix sums, no anchors, no mixing budget, no per-layer temporal state.
Cache closure comes from the write support alone: coordinates outside the
layer's write set are never touched, so every committed fragment stays
immutable and is read back directly by static latest-writer addresses.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from narrow_cache import Layout, NarrowCache, rotary_coefficients, rotate


@dataclass(frozen=True)
class Config:
    vocab: int
    d: int = 128
    depth: int = 8
    query_heads: int = 4
    kv_heads: int = 1
    head_dim: int = 32
    write_dim: int | None = None
    ffn_width: int = 512

    def __post_init__(self):
        if self.write_dim is None:
            object.__setattr__(self, 'write_dim', 3 * self.head_dim // 8)
        if min(self.vocab, self.d, self.query_heads, self.kv_heads, self.ffn_width) < 1:
            raise ValueError('Positive dimensions required')
        if self.query_heads % self.kv_heads:
            raise ValueError('GQA divisibility required')
        Layout(self.depth, self.head_dim, self.write_dim)

    @property
    def write_width(self):
        return 2 * self.kv_heads * self.write_dim

    @property
    def source_width(self):
        return 2 * self.kv_heads * self.head_dim


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) * self.weight


class Block(nn.Module):
    def __init__(self, cfg, layer):
        super().__init__()
        self.n1, self.n2 = RMSNorm(cfg.d), RMSNorm(cfg.d)
        self.q = nn.Linear(cfg.d, cfg.query_heads * cfg.head_dim, bias=False)
        self.o = nn.Linear(cfg.query_heads * cfg.head_dim, cfg.d, bias=False)
        self.up = nn.Linear(cfg.d, 2 * cfg.ffn_width, bias=False)
        self.down = nn.Linear(cfg.ffn_width, cfg.d, bias=False)
        self.pub = nn.Linear(cfg.d, cfg.write_width, bias=False) if layer else None

    def feedforward(self, h):
        gate, up = self.up(self.n2(h)).chunk(2, -1)
        return h + self.down(F.silu(gate) * up)


class CyclicMemory(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.layout = Layout(cfg.depth, cfg.head_dim, cfg.write_dim)
        self.embed = nn.Embedding(cfg.vocab, cfg.d)
        self.initial_norm = RMSNorm(cfg.d)
        self.initial = nn.Linear(cfg.d, cfg.source_width, bias=False)
        self.blocks = nn.ModuleList(Block(cfg, l) for l in range(cfg.depth))
        self.norm, self.output = RMSNorm(cfg.d), nn.Linear(cfg.d, cfg.vocab, bias=False)
        for l in range(1, cfg.depth):
            self.register_buffer(f'write_ids_{l}', torch.tensor(self.layout.indices[l]), persistent=False)
        self.apply(self._initialize)
        for layer in self.blocks:
            nn.init.normal_(layer.o.weight, std=.02 / math.sqrt(2 * cfg.depth))
            nn.init.normal_(layer.down.weight, std=.02 / math.sqrt(2 * cfg.depth))
            if layer.pub is not None:
                nn.init.normal_(layer.pub.weight, std=.0016)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Embedding, nn.Linear)):
            nn.init.normal_(module.weight, std=.02)

    def _forward(self, tokens, storage_dtype=None, decoder=None):
        cfg, b, n = self.config, tokens.shape[0], tokens.shape[1]
        h = self.embed(tokens)
        cosine, sine = rotary_coefficients(torch.arange(n, device=tokens.device), cfg.head_dim, h.dtype, h.device)
        reshape = lambda x, dim: x.reshape(b, n, cfg.kv_heads, dim).transpose(1, 2)
        raw_key, raw_value = self.initial(self.initial_norm(h)).chunk(2, -1)
        key, value = reshape(raw_key, cfg.head_dim), reshape(raw_value, cfg.head_dim)
        physical_key = rotate(key, cosine, sine)
        if storage_dtype is not None:
            physical_key, value = physical_key.to(storage_dtype).to(h.dtype), value.to(storage_dtype).to(h.dtype)
            key = rotate(physical_key, cosine, sine, inverse=True)
        for l, layer in enumerate(self.blocks):
            normalized = layer.n1(h)
            if l:
                ids = getattr(self, f'write_ids_{l}')
                previous = torch.cat((key.index_select(-1, ids).transpose(1, 2).flatten(-2),
                                      value.index_select(-1, ids).transpose(1, 2).flatten(-2)), -1)
                delta = previous + layer.pub(normalized)
                committed_key, committed_value = [reshape(x, cfg.write_dim) for x in delta.chunk(2, -1)]
                pairs = ids[::2] // 2
                k = rotate(committed_key, cosine[:, pairs], sine[:, pairs])
                if storage_dtype is not None:
                    k, committed_value = k.to(storage_dtype).to(h.dtype), committed_value.to(storage_dtype).to(h.dtype)
                    committed_key = rotate(k, cosine[:, pairs], sine[:, pairs], inverse=True)
                key, value = key.index_copy(-1, ids, committed_key), value.index_copy(-1, ids, committed_value)
                physical_key = physical_key.index_copy(-1, ids, k)
            if decoder is not None:
                decoder.cache.store_prefill_layer(l, physical_key, value)
            last_query = decoder is not None and l == cfg.depth-1
            if last_query:
                h, normalized = h[:, -1:], normalized[:, -1:]
            query = layer.q(normalized).reshape(b, h.shape[1], cfg.query_heads, cfg.head_dim).transpose(1, 2)
            query = rotate(query, cosine[-1:] if last_query else cosine, sine[-1:] if last_query else sine)
            group = cfg.query_heads // cfg.kv_heads
            out = F.scaled_dot_product_attention(query, physical_key.repeat_interleave(group, 1),
                value.repeat_interleave(group, 1), is_causal=not last_query)
            h = layer.feedforward(h + layer.o(out.transpose(1, 2).flatten(-2)))
        if decoder is not None:
            decoder.cache.finish_prefill(n)
        return self.output(self.norm(h))

    def forward(self, tokens):
        return self._forward(tokens)

    def forward_stored(self, tokens, dtype=torch.bfloat16):
        return self._forward(tokens, dtype)

    def new_cache(self, capacity, batch=1, tile=128, dtype=torch.bfloat16):
        return Decoder(self, capacity, batch, tile, dtype)

    def save(self, path, **metadata):
        torch.save(dict(config=asdict(self.config), state_dict=self.state_dict(), metadata=metadata), path)

    @classmethod
    def load(cls, path):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        model = cls(Config(**saved['config'])).to(dtype=saved['state_dict']['embed.weight'].dtype)
        model.load_state_dict(saved['state_dict'])
        return model


class Decoder:
    def __init__(self, model, capacity, batch=1, tile=128, dtype=torch.bfloat16):
        self.model = model
        cfg, weight = model.config, model.initial.weight
        ids = [None] + [getattr(model, f'write_ids_{l}') for l in range(1, cfg.depth)]
        self.cache = NarrowCache(model.layout, ids, cfg.kv_heads, capacity, batch, tile,
            dtype=dtype, compute_dtype=weight.dtype, device=weight.device)

    def memory_bytes(self):
        return self.cache.memory_bytes()

    def reset(self):
        self.cache.reset()

    @torch.inference_mode()
    def append(self, token):
        m, cfg = self.model, self.model.config
        h = m.embed(token)
        key, value = [x.reshape(token.shape[0], cfg.kv_heads, cfg.head_dim)
                      for x in self.cache.store(0, m.initial(m.initial_norm(h))).chunk(2, -1)]
        for l, layer in enumerate(m.blocks):
            normalized = layer.n1(h)
            if l:
                ids = self.cache.ids[l]
                previous = torch.cat((key.index_select(-1, ids).flatten(1), value.index_select(-1, ids).flatten(1)), -1)
                committed = previous + layer.pub(normalized)
                k, v = [x.reshape(token.shape[0], cfg.kv_heads, cfg.write_dim)
                        for x in self.cache.store(l, committed).chunk(2, -1)]
                key.index_copy_(-1, ids, k); value.index_copy_(-1, ids, v)
            query = layer.q(normalized).reshape(token.shape[0], cfg.query_heads, cfg.head_dim)
            query = self.cache.rotate_query(query)
            h = layer.feedforward(h + layer.o(self.cache.attention(l, query).flatten(-2)))
        self.cache.finish_token()
        return m.output(m.norm(h))[:, None]

    @torch.inference_mode()
    def prefill(self, tokens):
        self.reset()
        if tokens.ndim != 2 or tokens.shape[0] != self.cache.batch or tokens.shape[1] > self.cache.capacity:
            raise ValueError('Incorrect prefill batch or capacity')
        if tokens.shape[1] == 0:
            return None
        return self.model._forward(tokens, self.cache.dtype, self)
