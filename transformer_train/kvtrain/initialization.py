"""Paired random initialization for controlled write-fraction experiments."""

import hashlib
import math

import torch

PAIRED_INIT_VERSION = 1


@torch.no_grad()
def paired_initialize(model, seed):
    cfg = model.config

    def draw(rows, columns, key, *, std=None, bound=None):
        digest = hashlib.blake2b(f"{seed}:{key}".encode(), digest_size=8).digest()
        generator = torch.Generator().manual_seed(int.from_bytes(digest, "big"))
        values = torch.empty(rows, columns, dtype=torch.float32, device="cpu")
        if bound is None:
            return values.normal_(std=std, generator=generator)
        return values.uniform_(-bound, bound, generator=generator)

    for name, parameter in model.named_parameters():
        if name == "initial.weight" or name.endswith((".pub.weight", ".rope_key.weight")):
            continue
        if parameter.ndim == 1:
            parameter.fill_(1)
        elif cfg.attention == "tpa" and name.endswith(".q.weight"):
            offset = 0
            for label, width in (("a", cfg.q_rank * cfg.heads), ("b", cfg.q_rank * cfg.head_dim)):
                bound = math.sqrt(6 / (cfg.dim + width))
                parameter[offset:offset + width].copy_(draw(width, cfg.dim, name + "." + label, bound=bound))
                offset += width
        else:
            std = 0.02 / math.sqrt(2 * cfg.layers) if name.endswith((".o.weight", ".down.weight")) else 0.02
            parameter.copy_(draw(*parameter.shape, name, std=std))

    for layer, block in enumerate(model.blocks):
        if block.pub is None and not (layer == 0 and model.initial is not None):
            continue
        banks = []
        for component in cfg.components:
            key = f"blocks.{layer}.pub.{component.name}"
            if cfg.attention == "tpa":
                bank = draw(component.width, cfg.dim, key, bound=math.sqrt(6 / (cfg.dim + component.width)))
            else:
                independent_rope = component.name == "rope" and not cfg.share_rope
                std = cfg.publication_std if cfg.mode == "cyclic" and layer > 0 and not independent_rope else 0.02
                bank = draw(component.width, cfg.dim, key, std=std)
            banks.append(bank)
        full = torch.cat(banks)
        if layer == 0 and model.initial is not None:
            model.initial.weight.copy_(full[:model.initial.weight.shape[0]])
            if block.rope_key is not None:
                block.rope_key.weight.copy_(full[-cfg.rope_dim:])
        elif block.pub is not None:
            ids = torch.tensor(model.layout.indices[layer], dtype=torch.long)
            block.pub.weight.copy_(full.index_select(0, ids))
