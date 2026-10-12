"""Immutable per-layer KV fragments, with static latest-writer tile reads.

Each KV head writes the same cyclic complete-pair range. Head-major banks,
current-position RoPE, no historical reconstruction or per-layer reader state.
"""
import math

import torch


def rotary_coefficients(positions, p, dtype, device):
    frequency = 10000. ** (-torch.arange(0, p, 2, device=device).float() / p)
    phase = torch.as_tensor(positions, device=device).float()[..., None] * frequency
    return phase.cos().to(dtype), phase.sin().to(dtype)


def rotate(x, cosine, sine, inverse=False):
    if inverse:
        norm = cosine.square() + sine.square()
        cosine, sine = cosine / norm, -sine / norm
    a, b = x[..., ::2], x[..., 1::2]
    return torch.stack((a*cosine - b*sine, a*sine + b*cosine), -1).flatten(-2)


class Layout:
    def __init__(self, depth, p, q):
        if min(depth, p, q) < 1 or p % 2 or q % 2 or q > p:
            raise ValueError('Positive depth and complete per-head rotary pairs required')
        self.depth, self.p, self.q = depth, p, q
        self.indices = [tuple(range(p))] + [tuple(((l-1)*q+j) % p for j in range(q)) for l in range(1, depth)]
        owner, offset = [0]*p, list(range(p))
        self.segments = []
        for l in range(depth):
            if l:
                for j, coordinate in enumerate(self.indices[l]):
                    owner[coordinate], offset[coordinate] = l, j
            row, start = [], 0
            while start < p:
                end = start + 1
                while end < p and owner[end] == owner[start] and offset[end] == offset[end-1] + 1:
                    end += 1
                row.append((owner[start], offset[start], start, end-start))
                start = end
            self.segments.append(tuple(row))


class NarrowCache:
    def __init__(self, layout, ids, kv_heads, capacity, batch=1, tile=128, *,
                 dtype=torch.bfloat16, compute_dtype=torch.float32, device='cpu'):
        if min(kv_heads, capacity, batch, tile) < 1:
            raise ValueError('Positive cache dimensions required')
        if dtype not in (torch.float64, torch.float32, torch.bfloat16) or compute_dtype not in (torch.float64, torch.float32):
            raise ValueError('Storage FP64/FP32/BF16; computation FP64/FP32')
        self.layout, self.ids, self.kv_heads = layout, ids, kv_heads
        self.capacity, self.batch, self.tile = capacity, batch, tile
        self.dtype, self.compute_dtype = dtype, compute_dtype
        self.keys = [torch.empty(batch, kv_heads, capacity, layout.p if l == 0 else layout.q,
                    dtype=dtype, device=device) for l in range(layout.depth)]
        self.values = [torch.empty_like(k) for k in self.keys]
        self.frequency = 10000. ** (-torch.arange(0, layout.p, 2, device=device).float() / layout.p)
        self.cosine = torch.empty(layout.p//2, dtype=compute_dtype, device=device)
        self.sine = torch.empty_like(self.cosine)
        self.n, self.written = 0, -1

    @property
    def length(self):
        return self.n + int(self.written >= 0)

    def reset(self):
        self.n, self.written = 0, -1

    def memory_bytes(self):
        size = lambda tensors: sum(x.numel()*x.element_size() for x in tensors)
        history = size(self.keys+self.values)
        rotary = size([self.frequency, self.cosine, self.sine])
        indices = size(self.ids[1:])
        return dict(history=history, publication_state=0, historical_workspace=0, historical_summaries=0,
            rotary_state=rotary, shared_index_state=indices, persistent_tensor_total=history+rotary+indices,
            max_reader_tile=2*self.batch*self.kv_heads*min(self.tile,self.capacity)*self.layout.p
                *torch.empty((),dtype=self.compute_dtype).element_size(),
            note='Tensor payload, including shared model indices and current/static RoPE; prefix state charged '
                 'by Decoder. Excludes Python layout objects, allocator, weights, current source/private activations '
                 'and online-softmax scratch. No history-sized auxiliary tensor.')

    @torch.inference_mode()
    def store(self, layer, source, *, return_source=True):
        if self.n >= self.capacity or not 0 <= layer < self.layout.depth or layer != self.written+1:
            raise ValueError('Capacity or layer order')
        dim = self.layout.p if layer == 0 else self.layout.q
        if tuple(source.shape) != (self.batch, 2*self.kv_heads*dim):
            raise ValueError('Incorrect source fragment shape')
        key, value = [v.reshape(self.batch,self.kv_heads,dim) for v in source.chunk(2,-1)]
        if layer == 0:
            phase = self.n*self.frequency
            self.cosine.copy_(phase.cos()); self.sine.copy_(phase.sin())
            cosine, sine = self.cosine, self.sine
        else:
            pairs = self.ids[layer][::2]//2
            cosine, sine = self.cosine[pairs], self.sine[pairs]
        self.keys[layer][:,:,self.n].copy_(rotate(key,cosine,sine))
        self.values[layer][:,:,self.n].copy_(value)
        self.written = layer
        if not return_source:
            return None
        current = rotate(self.keys[layer][:,:,self.n].to(self.compute_dtype),cosine,sine,inverse=True)
        return torch.cat((current.flatten(1),self.values[layer][:,:,self.n].to(self.compute_dtype).flatten(1)),-1)

    def finish_token(self):
        if self.written != self.layout.depth-1:
            raise ValueError('Incomplete token')
        self.n += 1; self.written = -1

    @torch.inference_mode()
    def store_prefill_layer(self, layer, key, value):
        n = key.shape[-2]
        if self.n or layer != self.written+1 or not 0 <= layer < self.layout.depth:
            raise ValueError('Prefill requires empty cache and ordered layers')
        if not 0 < n <= self.capacity or tuple(key.shape) != (self.batch,self.kv_heads,n,self.layout.p) or value.shape != key.shape:
            raise ValueError('Incorrect physical prefill shape')
        if layer:
            key, value = key.index_select(-1,self.ids[layer]), value.index_select(-1,self.ids[layer])
        self.keys[layer][:,:,:n].copy_(key); self.values[layer][:,:,:n].copy_(value)
        self.written = layer

    def finish_prefill(self, n):
        if self.n or self.written != self.layout.depth-1 or not 0 < n <= self.capacity:
            raise ValueError('Incomplete prefill')
        phase = (n-1)*self.frequency
        self.cosine.copy_(phase.cos()); self.sine.copy_(phase.sin())
        self.n, self.written = n, -1

    def rotate_query(self, query):
        return rotate(query,self.cosine,self.sine)

    def read_tile(self, layer, start, end):
        if not 0 <= layer < self.layout.depth or not 0 <= start < end <= self.length:
            raise ValueError('Invalid tile')
        if self.written >= 0 and layer > self.written and end > self.n:
            raise ValueError('Current token not yet committed')
        shape = (self.batch,self.kv_heads,end-start,self.layout.p)
        key = torch.empty(shape,dtype=self.compute_dtype,device=self.keys[0].device)
        value = torch.empty_like(key)
        for bank, offset, position, width in self.layout.segments[layer]:
            key[...,position:position+width] = self.keys[bank][:,:,start:end,offset:offset+width]
            value[...,position:position+width] = self.values[bank][:,:,start:end,offset:offset+width]
        return key,value

    @torch.inference_mode()
    def attention(self, layer, query):
        if query.ndim != 3 or query.shape[0] != self.batch or query.shape[-1] != self.layout.p or query.shape[1] % self.kv_heads or not self.length:
            raise ValueError('Invalid GQA query or empty cache')
        q = query.reshape(self.batch,self.kv_heads,query.shape[1]//self.kv_heads,self.layout.p)
        maximum = torch.full(q.shape[:-1],-torch.inf,dtype=q.dtype,device=q.device)
        mass, numerator = torch.zeros_like(maximum),torch.zeros_like(q)
        for start in range(0,self.length,self.tile):
            key,value = self.read_tile(layer,start,min(start+self.tile,self.length))
            scores = torch.einsum('bhgp,bhtp->bhgt',q,key)/math.sqrt(self.layout.p)
            next_maximum = torch.maximum(maximum,scores.amax(-1))
            correction = (maximum-next_maximum).exp()
            probability = (scores-next_maximum[...,None]).exp()
            numerator = numerator*correction[...,None]+torch.einsum('bhgt,bhtp->bhgp',probability,value)
            mass = mass*correction+probability.sum(-1)
            maximum = next_maximum
        return (numerator/mass[...,None]).reshape_as(query)
