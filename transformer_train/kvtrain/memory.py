"""Coordinate layout shared by differentiable training and immutable KV storage."""

import torch


class Layout:
    def __init__(self, cfg):
        self.width = sum(c.width for c in cfg.components)
        self.indices, self.rotary, self.frequencies, self.addresses, self.segments = [], [], [], [], []
        owner, offset = [0] * self.width, list(range(self.width))
        for layer in range(cfg.layers):
            ids, rotary, frequency, base = [], [], [], 0
            for c in cfg.components:
                independent = c.name == "rope" and not cfg.share_rope
                if cfg.mode == "standard" or layer == 0 or independent:
                    local = list(range(c.width))
                elif cfg.mode == "cla":
                    local = list(range(c.width)) if layer % cfg.cla_group_size == 0 else []
                else:
                    w = cfg.narrow_width(c)
                    local = [((layer - 1) * w + j) % c.width for j in range(w)]
                if c.rotary_dim:
                    rotary.extend(range(len(ids), len(ids) + len(local)))
                    frequency.extend(cfg.rope_theta ** (-(j % c.rotary_dim) / c.rotary_dim) for j in local[::2])
                ids.extend(base + j for j in local)
                base += c.width
            for j, coordinate in enumerate(ids):
                owner[coordinate], offset[coordinate] = layer, j
            self.indices.append(ids)
            self.rotary.append(rotary)
            self.frequencies.append(frequency)
            self.addresses.append(list(zip(owner, offset)))
            segments, start = [], 0
            while start < self.width:
                end = start + 1
                while end < self.width and owner[end] == owner[start] and offset[end] == offset[end - 1] + 1:
                    end += 1
                segments.append((owner[start], offset[start], end - start))
                start = end
            self.segments.append(segments)

    @property
    def scalars_per_token(self):
        return sum(map(len, self.indices))


class FragmentCache:
    """Evaluation cache: only committed fragments, no persistent full layer K/V."""

    def __init__(self, layout, batch, capacity, device, dtype):
        if min(batch, capacity) < 1:
            raise ValueError("Positive batch and capacity required")
        self.layout, self.capacity, self.length = layout, capacity, 0
        self.banks = [torch.empty(batch, capacity, len(ids), device=device, dtype=dtype) for ids in layout.indices]

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.banks)

    def write(self, layer, fragment):
        self.banks[layer][:, self.length:self.length + 1].copy_(fragment)

    def read(self, layer):
        # Only the necessary latest-writer slices are assembled; no replay.
        parts = [self.banks[bank][:, :self.length + 1, col:col + width]
                 for bank, col, width in self.layout.segments[layer]]
        return torch.cat(parts, dim=-1)
