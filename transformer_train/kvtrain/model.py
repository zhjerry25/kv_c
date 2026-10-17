import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig, configure_runtime_cache
from .memory import Layout, FragmentCache
from .precision import Precision


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        source = x if x.dtype == torch.float64 else x.float()
        normalized = source * torch.rsqrt(source.square().mean(-1, keepdim=True) + 1e-6)
        return (normalized.to(x.dtype) * self.weight.to(x.dtype)).to(x.dtype)


def rotate(x, cosine, sine):
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    a, b = pairs.unbind(-1)
    return torch.stack((a * cosine - b * sine, a * sine + b * cosine), -1).flatten(-2)


class AttentionKernel:
    def __init__(self, window=0):
        self.window = window
        self.block_mask, self.flex = None, None
        self.prepared = None

    def prepare(self, length, device):
        if device.type != "cuda" or not self.window or self.prepared == (length, str(device)):
            return
        configure_runtime_cache()
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
        window = self.window

        def mask(batch, head, q, k):
            return (q >= k) & (q - k < window)

        self.block_mask = create_block_mask(mask, B=None, H=None, Q_LEN=length, KV_LEN=length, device=str(device))
        if self.flex is None:
            self.flex = torch.compile(flex_attention, dynamic=False)
        self.prepared = (length, str(device))

    def __call__(self, q, k, v, *, decode=False):
        scale = q.shape[-1] ** -0.5
        if self.window and q.device.type == "cuda" and not decode:
            # Flex requires power-of-two head width. Preserve the original scale.
            width = 1 << (q.shape[-1] - 1).bit_length()
            output_width = v.shape[-1]
            q, k, v = (F.pad(t, (0, width - t.shape[-1])) for t in (q, k, v))
            return self.flex(q, k, v, block_mask=self.block_mask, scale=scale,
                             enable_gqa=q.shape[1] != k.shape[1])[..., :output_width]
        mask = None
        if self.window:
            nq, nk = q.shape[-2], k.shape[-2]
            qi = torch.arange(nk - nq, nk, device=q.device)[:, None]
            ki = torch.arange(nk, device=q.device)[None, :]
            mask = (qi >= ki) & (qi - ki < self.window)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                              is_causal=not decode and mask is None,
                                              scale=scale, enable_gqa=q.shape[1] != k.shape[1])


class Block(nn.Module):
    def __init__(self, cfg, layer, layout, precision):
        super().__init__()
        self.cfg, self.layer = cfg, layer
        linear = precision.linear
        self.n1, self.n2 = RMSNorm(cfg.dim), RMSNorm(cfg.dim)
        self.register_buffer("write_ids", torch.tensor(layout.indices[layer], dtype=torch.long), persistent=False)
        self.register_buffer("rotary_ids", torch.tensor(layout.rotary[layer], dtype=torch.long), persistent=False)
        self.register_buffer("commit_freq", torch.tensor(layout.frequencies[layer]), persistent=False)
        q_rope = cfg.rope_dim if cfg.attention == "mla" else cfg.head_dim
        self.register_buffer("query_freq", cfg.rope_theta ** (-torch.arange(0, q_rope, 2).float() / q_rope), persistent=False)
        width = len(layout.indices[layer])
        self.pub = linear(cfg.dim, width) if width and not (cfg.mode == "cyclic" and layer == 0) else None
        self.rope_key = linear(cfg.dim, cfg.rope_dim) if cfg.mode == "cyclic" and layer == 0 and not cfg.share_rope else None
        if cfg.attention == "gqa":
            self.q = linear(cfg.dim, cfg.heads * cfg.head_dim)
        elif cfg.attention == "mla":
            self.q_down = linear(cfg.dim, cfg.q_latent_dim)
            self.q_norm = RMSNorm(cfg.q_latent_dim)
            self.q_up = linear(cfg.q_latent_dim, cfg.heads * (cfg.head_dim + cfg.rope_dim))
            self.kv_norm = RMSNorm(cfg.kv_latent_dim)
            self.kv_up = linear(cfg.kv_latent_dim, 2 * cfg.heads * cfg.head_dim)
        else:
            self.q = linear(cfg.dim, cfg.q_rank * (cfg.heads + cfg.head_dim))
        self.o = linear(cfg.heads * cfg.head_dim, cfg.dim)
        self.up = linear(cfg.dim, 2 * cfg.ffn_dim)
        self.down = linear(cfg.ffn_dim, cfg.dim)

    def physical(self, fragment, positions):
        if self.rotary_ids.numel():
            phase = positions.float()[:, None] * self.commit_freq.float()
            rotated = rotate(fragment.index_select(-1, self.rotary_ids),
                             phase.cos().to(fragment.dtype), phase.sin().to(fragment.dtype))
            fragment = fragment.index_copy(-1, self.rotary_ids, rotated)
        return fragment

    def query(self, normalized, positions):
        cfg = self.cfg
        batch, time = normalized.shape[:2]
        phase = positions.float()[:, None] * self.query_freq.float()
        if cfg.attention == "mla":
            query = self.q_up(self.q_norm(self.q_down(normalized)))
            query = query.view(batch, time, cfg.heads, cfg.head_dim + cfg.rope_dim).transpose(1, 2)
            content, positional = query.split((cfg.head_dim, cfg.rope_dim), -1)
            return torch.cat((content, rotate(positional, phase.cos().to(query.dtype), phase.sin().to(query.dtype))), -1)
        if cfg.attention == "tpa":
            a, b = self.q(normalized).split((cfg.q_rank * cfg.heads, cfg.q_rank * cfg.head_dim), -1)
            a = a.view(batch, time, cfg.q_rank, cfg.heads)
            b = b.view(batch, time, cfg.q_rank, cfg.head_dim).permute(0, 2, 1, 3)
            b = rotate(b, phase.cos().to(b.dtype), phase.sin().to(b.dtype))
            return torch.einsum("btrh,brtd->bhtd", a, b) / cfg.q_rank
        query = self.q(normalized).view(batch, time, cfg.heads, cfg.head_dim).transpose(1, 2)
        return rotate(query, phase.cos().to(query.dtype), phase.sin().to(query.dtype))

    def keys_values(self, state):
        cfg = self.cfg
        batch, time = state.shape[:2]
        if cfg.attention == "gqa":
            key, value = state.chunk(2, -1)
            return tuple(x.reshape(batch, time, cfg.kv_heads, cfg.head_dim).transpose(1, 2) for x in (key, value))
        if cfg.attention == "mla":
            latent, positional = state.split((cfg.kv_latent_dim, cfg.rope_dim), -1)
            # Normalize the complete raw latent at the reader, never each fragment.
            key, value = self.kv_up(self.kv_norm(latent)).chunk(2, -1)
            key = key.reshape(batch, time, cfg.heads, cfg.head_dim).transpose(1, 2)
            key = torch.cat((key, positional[:, None].expand(-1, cfg.heads, -1, -1)), -1)
            value = value.reshape(batch, time, cfg.heads, cfg.head_dim).transpose(1, 2)
            return key, F.pad(value, (0, cfg.rope_dim))
        ak, bk, av, bv = state.split([c.width for c in cfg.components], -1)
        ak, av = ak.view(batch, time, cfg.k_rank, cfg.heads), av.view(batch, time, cfg.v_rank, cfg.heads)
        bk, bv = bk.view(batch, time, cfg.k_rank, cfg.head_dim), bv.view(batch, time, cfg.v_rank, cfg.head_dim)
        return (torch.einsum("btrh,btrd->bhtd", ak, bk) / cfg.k_rank,
                torch.einsum("btrh,btrd->bhtd", av, bv) / cfg.v_rank)

    def finish(self, h, normalized, state, positions, kernel, decode=False):
        query = self.query(normalized, positions)
        key, value = self.keys_values(state)
        output = kernel(query, key, value, decode=decode)[..., :self.cfg.head_dim]
        h = h + self.o(output.transpose(1, 2).flatten(-2))
        gate, up = self.up(self.n2(h)).chunk(2, -1)
        return h + self.down(F.silu(gate) * up)

    def forward(self, h, state, positions, kernel):
        normalized = self.n1(h)
        if self.pub is not None:
            fragment = self.physical(self.pub(normalized), positions)
            if self.write_ids.numel() == state.shape[-1] or state.shape[-1] == 0:
                state = fragment
            else:
                # Functional update preserves autograd into all previous writers.
                state = state.index_copy(-1, self.write_ids, fragment)
        return self.finish(h, normalized, state, positions, kernel), state


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig, precision=None):
        super().__init__()
        self.config, self.precision = cfg, precision or Precision()
        self.layout = Layout(cfg)
        self.kernel = AttentionKernel(cfg.window)
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.initial_norm = RMSNorm(cfg.dim) if cfg.mode == "cyclic" else None
        initial_width = self.layout.width - (cfg.rope_dim if not cfg.share_rope else 0)
        self.initial = self.precision.linear(cfg.dim, initial_width) if cfg.mode == "cyclic" else None
        self.blocks = nn.ModuleList(Block(cfg, i, self.layout, self.precision) for i in range(cfg.layers))
        self.norm = RMSNorm(cfg.dim)
        self.reset_parameters()

    def reset_parameters(self):
        cfg = self.config
        for module in self.modules():
            weight = getattr(module, "weight", None)
            if weight is not None and weight.ndim == 2:
                nn.init.normal_(weight, std=0.02)
        for block in self.blocks:
            nn.init.normal_(block.o.weight, std=0.02 / math.sqrt(2 * cfg.layers))
            nn.init.normal_(block.down.weight, std=0.02 / math.sqrt(2 * cfg.layers))
            if cfg.attention == "tpa":
                offset = 0
                for width in (cfg.q_rank * cfg.heads, cfg.q_rank * cfg.head_dim):
                    nn.init.uniform_(block.q.weight[offset:offset + width], -math.sqrt(6 / (cfg.dim + width)), math.sqrt(6 / (cfg.dim + width)))
                    offset += width
            if block.pub is not None:
                offset, base = 0, 0
                ids = self.layout.indices[block.layer]
                for component in cfg.components:
                    width = sum(base <= j < base + component.width for j in ids)
                    weights = block.pub.weight[offset:offset + width]
                    if cfg.attention == "tpa":
                        bound = math.sqrt(6 / (cfg.dim + component.width))
                        nn.init.uniform_(weights, -bound, bound)
                    elif cfg.mode == "cyclic" and not (component.name == "rope" and not cfg.share_rope):
                        nn.init.normal_(weights, std=cfg.publication_std)
                    offset += width
                    base += component.width
        if cfg.attention == "tpa" and self.initial is not None:
            offset = 0
            for component in cfg.components:
                bound = math.sqrt(6 / (cfg.dim + component.width))
                nn.init.uniform_(self.initial.weight[offset:offset + component.width], -bound, bound)
                offset += component.width

    def initial_state(self, h, positions):
        if self.initial is None:
            return h.new_empty(*h.shape[:2], 0)
        fragment = self.initial(self.initial_norm(h))
        if not self.config.share_rope:
            fragment = torch.cat((fragment, self.blocks[0].rope_key(self.blocks[0].n1(h))), -1)
        return self.blocks[0].physical(fragment, positions)

    def forward(self, tokens, targets=None, loss_chunk_size=1024):
        if tokens.ndim != 2 or min(tokens.shape) < 1:
            raise ValueError("Expected nonempty [batch, time] token IDs")
        self.precision.validate_batch(tokens.numel())
        self.kernel.prepare(tokens.shape[1], tokens.device)
        h = self.embed(tokens)
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        state = self.initial_state(h, positions)
        for block in self.blocks:
            if self.config.activation_checkpoint and self.training and torch.is_grad_enabled():
                h, state = self.precision.checkpoint(block, h, state, positions, self.kernel)
            else:
                h, state = block(h, state, positions, self.kernel)
        h = self.norm(h)
        if targets is None:
            return F.linear(h, self.embed.weight)
        if targets.shape != tokens.shape or loss_chunk_size < 1:
            raise ValueError("Targets must match tokens and loss_chunk_size must be positive")
        flat_h, flat_y = h.flatten(0, 1), targets.flatten()

        def chunk_loss(hidden, labels):
            return F.cross_entropy(F.linear(hidden, self.embed.weight).float(), labels, reduction="sum")

        # Recompute chunk logits in backward instead of retaining B*T*vocab.
        loss = h.new_zeros((), dtype=torch.float32)
        for start in range(0, flat_y.numel(), loss_chunk_size):
            args = (flat_h[start:start + loss_chunk_size], flat_y[start:start + loss_chunk_size])
            if self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                loss = loss + checkpoint(chunk_loss, *args, use_reentrant=False)
            else:
                loss = loss + chunk_loss(*args)
        return loss / flat_y.numel()

    def new_cache(self, batch, capacity, dtype=torch.float32):
        return FragmentCache(self.layout, batch, capacity, self.embed.weight.device, dtype)

    @torch.inference_mode()
    def step(self, tokens, cache):
        """One-token reference decode; training uses fused sequence attention."""
        if tokens.ndim != 2 or tokens.shape[1] != 1 or tokens.shape[0] != cache.banks[0].shape[0]:
            raise ValueError("step expects [cache_batch, 1]")
        if cache.length >= cache.capacity or cache.layout is not self.layout:
            raise ValueError("Cache is full or belongs to another model")
        h = self.embed(tokens)
        positions = torch.tensor([cache.length], device=tokens.device)
        initial = self.initial_state(h, positions)
        for block in self.blocks:
            normalized = block.n1(h)
            if block.pub is not None:
                fragment = block.physical(block.pub(normalized), positions)
            elif block.layer == 0:
                fragment = initial
            else:
                fragment = h.new_empty(h.shape[0], 1, 0)
            cache.write(block.layer, fragment)
            state = cache.read(block.layer).to(h.dtype)
            h = block.finish(h, normalized, state, positions, self.kernel, decode=True)
        cache.length += 1
        return F.linear(self.norm(h), self.embed.weight)

    def summary(self):
        cfg = self.config
        return dict(parameters=sum(p.numel() for p in self.parameters()),
                    attention=cfg.attention, mode=cfg.mode, window=cfg.window,
                    cache_scalars_per_token=self.layout.scalars_per_token,
                    standard_cache_scalars_per_token=cfg.layers * self.layout.width,
                    fp8=self.precision.fp8, fp8_linear_count=self.precision.converted,
                    bf16_or_fp32_linear_count=self.precision.high_precision)
