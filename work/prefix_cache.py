"""Incremental CPU correctness reference for raw/block-summary public caches.

Private/model arithmetic and base/control storage stay FP32. Only the canonical
wide source is rounded to workspace_dtype, before norm/RoPE, at every layer.
Prefix accumulation stays FP32 (FP64 when the model is FP64 for oracle tests).
This module is an execution prototype for the legacy single-KV-head models.
"""
import math
import torch
from torch.nn import functional as F
from train_public_memory import rope


def rounded(x, dtype):
    return x.to(dtype).to(x.dtype)


@torch.inference_mode()
def precision_forward(model, tokens, workspace_dtype=torch.float32, states=False):
    """Parallel causal reference, with explicit canonical-source rounding."""
    if model.mode != 'moments_uniform':
        raise ValueError('Only moments_uniform is supported')
    h = model.embed(tokens)
    base = model.initial(model.initial_norm(h))
    m = rounded(base, workspace_dtype)
    sources, controls, prefixes = [], [None], [None]
    for l, block in enumerate(model.blocks):
        if l:
            a = block.pub(block.n1(h))
            z = m + block.expand(a)
            prefix = z.cumsum(1)
            count = torch.arange(1, tokens.shape[1] + 1, device=z.device,
                                 dtype=z.dtype)[None, :, None]
            lam = getattr(block, 'mix_budget', .25) / (model.depth - 1)
            m = rounded((1 - lam) * z + lam * (prefix / count), workspace_dtype)
            controls.append(a)
            prefixes.append(prefix)
        sources.append(m)
        h = block(h, m)
    logits = model.output(model.norm(h))
    return (logits, base, sources, controls, prefixes) if states else logits


class PrefixCache:
    """Preallocated real rolling cache; no historical private states.

    execution=raw: full prefix scan on every replay.
    execution=anchored: independent block scans using immutable start summaries.
    execution=material: same rounded function, storing each layer's rotated KV.
    Model weights and this reference cache are CPU tensors; no performance claim.
    """
    def __init__(self, model, capacity, batch=1, execution='anchored', block_size=128,
                 workspace_dtype=torch.float32):
        if execution not in ('raw', 'anchored', 'material'):
            raise ValueError(execution)
        if model.mode != 'moments_uniform' or capacity <= 0 or block_size <= 0 or batch <= 0:
            raise ValueError('Unsupported model or capacity/block size')
        if model.embed.weight.device.type != 'cpu' or workspace_dtype not in (
                torch.float32, torch.float64, torch.bfloat16):
            raise ValueError('This local reference requires CPU and FP32/FP64/BF16 workspace')
        self.model, self.capacity, self.batch = model, capacity, batch
        self.execution, self.block_size = execution, block_size
        self.dtype = model.embed.weight.dtype
        self.workspace_dtype = workspace_dtype
        self.n = 0
        L, c, r = model.depth, model.c, model.rank
        self.base = self.controls = self.wide = self.anchors = self.kv = None
        self.prefix = torch.zeros(L - 1, batch, c, dtype=self.dtype)
        if execution == 'material':
            self.kv = torch.empty(L, batch, capacity, c, dtype=self.dtype)
        else:
            self.base = torch.empty(batch, capacity, c, dtype=self.dtype)
            self.controls = torch.empty(L - 1, batch, capacity, r, dtype=self.dtype)
            self.wide = torch.empty(batch, capacity, c, dtype=workspace_dtype)
            if execution == 'anchored':
                self.anchors = torch.empty(L - 1, batch,
                    math.ceil(capacity / block_size), c, dtype=self.dtype)

    def memory_bytes(self):
        size = lambda t: 0 if t is None else t.numel() * t.element_size()
        return dict(history=size(self.base) + size(self.controls) + size(self.kv),
                    workspace=size(self.wide), summaries=size(self.anchors),
                    append_prefix=size(self.prefix))

    def _rope_at(self, x, position):
        p = x.shape[-1]
        phase = position * 10000. ** (-torch.arange(0, p, 2).float() / p)
        a, b = x[..., 0::2], x[..., 1::2]
        return torch.stack((a * phase.cos() - b * phase.sin(),
                            a * phase.sin() + b * phase.cos()), -1).flatten(-2)

    def _read(self, block, source, start=0):
        normalized = block.mn(source)
        k, v = normalized.split(block.p, -1)
        if k.shape[1] == 1:
            k = self._rope_at(k, start)
        else:
            # The public reference uses the same FP32 RoPE tables as Model.
            p = block.p
            phase = torch.arange(start, start + k.shape[1]).float()[:, None] * (
                10000. ** (-torch.arange(0, p, 2).float() / p))[None, :]
            a, b = k[..., 0::2], k[..., 1::2]
            k = torch.stack((a * phase.cos() - b * phase.sin(),
                             a * phase.sin() + b * phase.cos()), -1).flatten(-2)
        return torch.cat((k, v), -1)

    @torch.inference_mode()
    def prefill(self, tokens):
        """Build summaries from the actual rounded prefill, without decode replay."""
        n = tokens.shape[1]
        if self.n or n > self.capacity or tokens.shape[0] != self.batch:
            raise ValueError('prefill requires empty compatible cache')
        if not n:
            return None
        logits, base, sources, controls, prefixes = precision_forward(
            self.model, tokens, self.workspace_dtype, True)
        if self.execution == 'material':
            for l, block in enumerate(self.model.blocks):
                self.kv[l, :, :n] = self._read(block, sources[l])
        else:
            self.base[:, :n] = base
            for l in range(1, self.model.depth):
                self.controls[l - 1, :, :n] = controls[l]
        for l in range(1, self.model.depth):
            self.prefix[l - 1] = prefixes[l][:, -1]
            if self.anchors is not None:
                self.anchors[l - 1, :, 0].zero_()
                for start in range(self.block_size, n, self.block_size):
                    self.anchors[l - 1, :, start // self.block_size] = prefixes[l][:, start - 1]
        self.n = n
        return logits

    @torch.inference_mode()
    def append(self, token):
        if self.n >= self.capacity:
            raise ValueError('cache capacity exceeded')
        token = token.reshape(self.batch, 1)
        h = self.model.embed(token)
        pos, n, L = self.n, self.n + 1, self.model.depth
        m = self.model.initial(self.model.initial_norm(h))
        if self.base is not None:
            self.base[:, pos:pos + 1] = m
            self.wide[:, :n] = self.base[:, :n]
        m = rounded(m, self.workspace_dtype)
        for l, block in enumerate(self.model.blocks):
            hn = block.n1(h)
            if l:
                a = block.pub(hn)
                prev = m if self.wide is None else self.wide[:, pos:pos + 1].to(self.dtype)
                z_last = prev + block.expand(a)
                if self.anchors is not None and pos % self.block_size == 0:
                    self.anchors[l - 1, :, pos // self.block_size] = self.prefix[l - 1]
                self.prefix[l - 1] += z_last[:, 0]
                lam = getattr(block, 'mix_budget', .25) / (L - 1)
                if self.wide is None:
                    m = rounded((1 - lam) * z_last + lam * (self.prefix[l - 1, :, None] / n),
                                self.workspace_dtype)
                else:
                    self.controls[l - 1, :, pos:pos + 1] = a
                    starts = range(0, n, self.block_size) if self.anchors is not None else [0]
                    for start in starts:
                        end = min(n, start + self.block_size) if self.anchors is not None else n
                        z = self.wide[:, start:end].to(self.dtype) + block.expand(
                            self.controls[l - 1, :, start:end])
                        sums = z.cumsum(1)
                        if self.anchors is not None:
                            sums += self.anchors[l - 1, :, start // self.block_size, None]
                        count = torch.arange(start + 1, end + 1, dtype=self.dtype)[None, :, None]
                        self.wide[:, start:end] = (1 - lam) * z + lam * (sums / count)
            if self.wide is None:
                self.kv[l, :, pos:pos + 1] = self._read(block, m, pos)
                read = self.kv[l, :, :n]
            else:
                read = self._read(block, self.wide[:, :n].to(self.dtype))
            q = self._rope_at(block.q(hn).view(self.batch, 1, block.heads, block.p)
                             .transpose(1, 2), pos)
            k, v = read.split(block.p, -1)
            attn = F.scaled_dot_product_attention(q, k[:, None].expand(-1, block.heads, -1, -1),
                                                 v[:, None].expand(-1, block.heads, -1, -1))
            h = h + block.o(attn.transpose(1, 2).reshape(self.batch, 1, -1))
            x, y = block.up(block.n2(h)).chunk(2, -1)
            h = h + block.down(F.silu(x) * y)
        self.n = n
        return self.model.output(self.model.norm(h))
