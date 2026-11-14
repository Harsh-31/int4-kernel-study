"""cuBLAS BF16 baseline: no quantization at run time, BF16 compute.

The precision-matched counterpart of impls/cublas for BF16-only kernels
(TorchAO's tinygemm). prepare() dequantizes the INT4 weights once to a dense
BF16 [K, N] matrix; prepare_input() casts the FP16 activations to BF16 outside
the timed region; run() is a plain BF16 torch.matmul (a cuBLAS GEMM on GPU).

Its latency is the speedup_vs_bf16 reference, and its error is the correctness
baseline for COMPUTE_DTYPE = bfloat16 implementations.
"""

import torch

from common.quantize import dequantize

WEIGHT_BITS = 16              # read by the harness to count bytes moved
COMPUTE_DTYPE = torch.bfloat16

_w: torch.Tensor | None = None   # [K, N] BF16, set by prepare()


def prepare(weights: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor,
            group_size: int) -> None:
    """Dequantize once to BF16 (not timed).

    Our scales are exact in BF16 (common/quantize.py), but the dequantized
    weights (q - z) * s can need up to 12 significant bits, so this cast rounds
    some of them, exactly as any BF16 kernel must.
    """
    global _w
    _w = None   # release the previous layer's weights before allocating new ones
    _w = dequantize(weights, scales, zeros, group_size, dtype=torch.bfloat16).contiguous()


def prepare_input(x: torch.Tensor) -> torch.Tensor:
    """FP16 activations -> contiguous BF16 (not timed)."""
    return x.to(torch.bfloat16).contiguous()


def run(x: torch.Tensor) -> torch.Tensor:
    """y [M, N] = x [M, K] @ W [K, N] in BF16 (cuBLAS GEMM on the GPU)."""
    return torch.matmul(x, _w)


def reference_weights() -> torch.Tensor:
    """The BF16 weights cuBLAS actually multiplies by."""
    return _w
