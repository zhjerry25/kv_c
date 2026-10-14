"""Immutable heterogeneous fragments with static latest-writer tile reads.

Coordinates are grouped into components. Only components carrying RoPE must
write complete pairs; head coefficients and latent coordinates need no RoPE.
The cache stores physical (already rotated and rounded) fragments only.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Component:
    name: str
    width: int
    write_width: int
    rotary_dim: int = 0


class FragmentLayout:
    def __init__(self, depth, components, mode='cyclic'):
        if depth < 1 or mode not in ('native', 'cyclic') or not components:
            raise ValueError('Positive depth, components and native/cyclic mode required')
        self.depth, self.components, self.mode = depth, tuple(components), mode
        self.width = sum(c.width for c in components)
        self.write_width = sum(c.write_width for c in components)
        if len({c.name for c in components}) != len(components):
            raise ValueError('Unique component names required')
        for c in components:
            if not 0 < c.write_width <= c.width:
                raise ValueError('Invalid component write width')
            if c.rotary_dim and (c.rotary_dim < 2 or c.rotary_dim % 2 or
                                 c.width % c.rotary_dim or c.write_width % 2):
                raise ValueError('Complete rotary pairs required')
        self.indices, self.rotary_ids, self.rotary_frequencies = [], [], []
        self.segments = []
        owner, offset = [0] * self.width, list(range(self.width))
        for layer in range(depth):
            full = mode == 'native' or layer == 0
            row, rotation, frequencies, base = [], [], [], 0
            for c in components:
                ids = list(range(c.width)) if full else [
                    ((layer - 1) * c.write_width + j) % c.width
                    for j in range(c.write_width)]
                if c.rotary_dim:
                    rotation.extend(range(len(row), len(row) + len(ids)))
                    frequency = 10000. ** (-torch.arange(0, c.rotary_dim, 2).float() / c.rotary_dim)
                    frequencies.extend(float(frequency[(j % c.rotary_dim) // 2]) for j in ids[::2])
                row.extend(base + j for j in ids)
                base += c.width
            self.indices.append(tuple(row))
            self.rotary_ids.append(tuple(rotation))
            self.rotary_frequencies.append(tuple(frequencies))
            for j, coordinate in enumerate(row):
                owner[coordinate], offset[coordinate] = layer, j
            segments, start = [], 0
            while start < self.width:
                end = start + 1
                while (end < self.width and owner[end] == owner[start]
                       and offset[end] == offset[end - 1] + 1):
                    end += 1
                segments.append((owner[start], offset[start], start, end - start))
                start = end
            self.segments.append(tuple(segments))


class FragmentCache:
    def __init__(self, layout, capacity, batch=1, tile=128, *, dtype=torch.bfloat16,
                 compute_dtype=torch.float32, device='cpu', shared_buffers=()):
        if min(capacity, batch, tile) < 1:
            raise ValueError('Positive cache dimensions required')
        if dtype not in (torch.float64, torch.float32, torch.bfloat16):
            raise ValueError('Storage FP64/FP32/BF16 required')
        if compute_dtype not in (torch.float64, torch.float32):
            raise ValueError('Computation FP64/FP32 required')
        self.layout, self.capacity, self.batch, self.tile = layout, capacity, batch, tile
        self.dtype, self.compute_dtype = dtype, compute_dtype
        self.banks = [torch.empty(batch, capacity, len(ids), dtype=dtype, device=device)
                      for ids in layout.indices]
        self.shared_buffers = tuple(shared_buffers)
        self.n, self.written = 0, -1

    @property
    def length(self):
        return self.n + int(self.written >= 0)

    def reset(self):
        self.n, self.written = 0, -1

    def memory_bytes(self):
        size = lambda xs: sum(x.numel() * x.element_size() for x in xs)
        history = size(self.banks)
        indices = size(x for x in self.shared_buffers if not x.is_floating_point())
        rotary = size(x for x in self.shared_buffers if x.is_floating_point())
        return dict(history=history, publication_state=0, historical_workspace=0,
                    historical_summaries=0, shared_index_state=indices,
                    rotary_state=rotary, persistent_tensor_total=history + indices + rotary,
                    max_reader_tile=self.batch * min(self.tile, self.capacity) * self.layout.width
                    * torch.empty((), dtype=self.compute_dtype).element_size(),
                    note='Tensor payload: history plus shared model indices/frequencies. Reader tile '
                         'counts packed source only; excludes tile-local attention scratch, current '
                         'activations, weights, Python layout and allocator. No persistent expanded KV.')

    @torch.inference_mode()
    def store(self, layer, physical_fragment):
        if self.n >= self.capacity or not 0 <= layer < self.layout.depth or layer != self.written + 1:
            raise ValueError('Capacity or layer order')
        if tuple(physical_fragment.shape) != (self.batch, len(self.layout.indices[layer])):
            raise ValueError('Incorrect fragment shape')
        self.banks[layer][:, self.n].copy_(physical_fragment)
        self.written = layer

    def finish_token(self):
        if self.written != self.layout.depth - 1:
            raise ValueError('Incomplete token')
        self.n += 1
        self.written = -1

    @torch.inference_mode()
    def store_prefill_layer(self, layer, physical_fragment):
        if self.n or not 0 <= layer < self.layout.depth or layer != self.written + 1:
            raise ValueError('Prefill requires empty cache and ordered layers')
        if physical_fragment.ndim != 3:
            raise ValueError('Incorrect prefill fragment shape')
        n = physical_fragment.shape[1]
        if not 0 < n <= self.capacity or tuple(physical_fragment.shape) != (
                self.batch, n, len(self.layout.indices[layer])):
            raise ValueError('Incorrect prefill fragment shape')
        self.banks[layer][:, :n].copy_(physical_fragment)
        self.written = layer

    def finish_prefill(self, n):
        if self.n or self.written != self.layout.depth - 1 or not 0 < n <= self.capacity:
            raise ValueError('Incomplete prefill')
        self.n, self.written = n, -1

    def read_tile(self, layer, start, end):
        if not 0 <= layer < self.layout.depth or not 0 <= start < end <= self.length:
            raise ValueError('Invalid tile')
        if self.written >= 0 and layer > self.written and end > self.n:
            raise ValueError('Current token not yet committed')
        source = torch.empty(self.batch, end - start, self.layout.width,
                             dtype=self.compute_dtype, device=self.banks[0].device)
        for bank, offset, position, width in self.layout.segments[layer]:
            source[..., position:position + width] = self.banks[bank][
                :, start:end, offset:offset + width]
        return source
