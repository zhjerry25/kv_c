from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import math
import os

ROOT = Path(__file__).resolve().parents[1]


def local_path(path):
    """All datasets, caches and run artifacts stay inside this standalone tree."""
    result = (ROOT / path).resolve()
    if not result.is_relative_to(ROOT):
        raise ValueError(f"Path must be inside the standalone framework: {path}")
    return result


def configure_runtime_cache():
    for variable, directory in (("TORCHINDUCTOR_CACHE_DIR", "inductor"),
                                ("TRITON_CACHE_DIR", "triton"), ("CUDA_CACHE_PATH", "cuda")):
        path = local_path(".cache/" + directory)
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)


@dataclass(frozen=True)
class Component:
    name: str
    width: int
    rotary_dim: int = 0


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 50304
    dim: int = 512
    layers: int = 10
    heads: int = 8
    kv_heads: int = 2
    head_dim: int = 64
    ffn_dim: int = 1408
    attention: str = "gqa"
    mode: str = "standard"
    cla_group_size: int = 2
    write_fraction: str = "1/4"
    kv_latent_dim: int = 256
    q_latent_dim: int = 256
    rope_dim: int = 32
    share_rope: bool = True
    q_rank: int = 6
    k_rank: int = 2
    v_rank: int = 2
    window: int = 0
    rope_theta: float = 10000.0
    publication_std: float = 0.0016
    activation_checkpoint: bool = True

    def __post_init__(self):
        dims = (self.vocab_size, self.dim, self.layers, self.heads, self.kv_heads,
                self.head_dim, self.ffn_dim, self.cla_group_size, self.kv_latent_dim,
                self.q_latent_dim, self.rope_dim, self.q_rank, self.k_rank, self.v_rank)
        if min(dims) < 1 or self.heads % self.kv_heads:
            raise ValueError("Positive dimensions and heads divisible by kv_heads required")
        if self.head_dim % 2 or self.rope_dim % 2:
            raise ValueError("RoPE dimensions must contain complete pairs")
        if self.attention not in ("gqa", "mla", "tpa"):
            raise ValueError("attention must be gqa, mla or tpa; SWA is the window option")
        if self.mode not in ("standard", "cla", "cyclic"):
            raise ValueError("mode must be standard, cla or cyclic")
        if self.window < 0 or not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError("window >= 0 and positive finite rope_theta required")
        if not math.isfinite(self.publication_std) or self.publication_std <= 0:
            raise ValueError("publication_std must be positive and finite")
        if not self.share_rope and self.attention != "mla":
            raise ValueError("Independent RoPE is an MLA option")
        fraction = Fraction(self.write_fraction)
        if not 0 < fraction <= 1:
            raise ValueError("write_fraction must be in (0, 1]")
        if self.mode == "cyclic":
            for c in self.components:
                width = self.narrow_width(c)
                if width < 1 or (c.rotary_dim and width % 2):
                    raise ValueError(f"Invalid narrow width for {c.name}: complete RoPE pairs required")

    @property
    def components(self):
        if self.attention == "gqa":
            width = self.kv_heads * self.head_dim
            return (Component("key", width, self.head_dim), Component("value", width))
        if self.attention == "mla":
            return (Component("latent", self.kv_latent_dim), Component("rope", self.rope_dim, self.rope_dim))
        return (Component("ak", self.k_rank * self.heads),
                Component("bk", self.k_rank * self.head_dim, self.head_dim),
                Component("av", self.v_rank * self.heads),
                Component("bv", self.v_rank * self.head_dim))

    def narrow_width(self, component):
        if component.name == "rope" and not self.share_rope:
            return component.width
        width = Fraction(self.write_fraction) * component.width
        if width.denominator != 1:
            raise ValueError(f"write_fraction does not divide component {component.name}")
        return int(width)


PRESETS = {
    "tiny": dict(dim=64, layers=3, heads=4, kv_heads=2, head_dim=16,
                 ffn_dim=128, kv_latent_dim=32, q_latent_dim=32, rope_dim=8),
    "50m": dict(dim=512, layers=10, heads=8, kv_heads=2, ffn_dim=1408),
    "125m": dict(dim=768, layers=14, heads=12, kv_heads=4, ffn_dim=2048),
    "350m": dict(dim=1024, layers=26, heads=16, kv_heads=4, ffn_dim=2816,
                  kv_latent_dim=384, q_latent_dim=384),
    "1b": dict(dim=2048, layers=20, heads=32, kv_heads=8, ffn_dim=5504,
               kv_latent_dim=512, q_latent_dim=512),
}
