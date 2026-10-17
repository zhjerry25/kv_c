"""Self-contained training package; no imports or assets from its parent tree."""

from .config import ModelConfig
from .model import Transformer

__all__ = ["ModelConfig", "Transformer"]
