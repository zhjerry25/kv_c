from contextlib import nullcontext

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class Precision:
    """Real TE FP8 GEMMs, with FP32 master weights and BF16 surrounding ops."""

    def __init__(self, fp8="off", device="cpu"):
        self.fp8, self.device = fp8, torch.device(device)
        self.te, self.recipe = None, None
        self.converted, self.high_precision = 0, 0
        if fp8 == "off":
            return
        if fp8 not in ("delayed", "current", "mxfp8"):
            raise ValueError("fp8 must be off, delayed, current or mxfp8")
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("FP8 requires an NVIDIA CUDA GPU; use --fp8 off on CPU")
        try:
            import transformer_engine.pytorch as te
            from transformer_engine.common.recipe import DelayedScaling, Float8CurrentScaling, MXFP8BlockScaling, Format
        except ImportError as exc:
            raise RuntimeError("Install transformer_engine[pytorch]>=2.14, or use the NGC PyTorch container") from exc
        if fp8 == "mxfp8" and not (10 <= torch.cuda.get_device_capability(self.device)[0] < 12):
            raise RuntimeError("TE MXFP8 requires SM100/SM103; RTX PRO 6000 SM120: use --fp8 delayed or current")
        self.te = te
        if fp8 == "delayed":
            self.recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max", reduce_amax=False)
        elif fp8 == "current":
            self.recipe = Float8CurrentScaling(fp8_format=Format.HYBRID)
        else:
            self.recipe = MXFP8BlockScaling(fp8_format=Format.E4M3)

    def linear(self, in_features, out_features):
        # Small/unaligned publications stay in BF16: padding them wastes work.
        alignment = 32 if self.fp8 == "mxfp8" else 16
        if self.te is not None and min(in_features, out_features) >= 128 and in_features % alignment == out_features % alignment == 0:
            self.converted += 1
            return self.te.Linear(in_features, out_features, bias=False, device=self.device, params_dtype=torch.float32)
        self.high_precision += 1
        return nn.Linear(in_features, out_features, bias=False, device=self.device)

    def context(self):
        if self.te is None:
            return nullcontext()
        return self.te.autocast(enabled=True, recipe=self.recipe)

    def checkpoint(self, function, *args):
        if self.te is not None:
            # TE preserves quantization state during recomputation.
            return self.te.checkpoint(function, *args, use_reentrant=False)
        return checkpoint(function, *args, use_reentrant=False)

    def validate_batch(self, tokens):
        alignment = 32 if self.fp8 == "mxfp8" else 16
        if self.te is not None and tokens % alignment:
            raise ValueError(f"micro_batch * seq_len must be divisible by {alignment} for FP8")
