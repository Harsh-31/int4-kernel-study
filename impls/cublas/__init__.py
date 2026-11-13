"""cuBLAS FP16 baseline: no quantization at run time.

prepare() dequantizes the INT4 weights once to a dense FP16 [K, N] matrix, so
the baseline multiplies by exactly the same values the INT4 kernels represent
(only the storage format differs). run() is a plain FP16 torch.matmul, which
PyTorch dispatches to a cuBLAS GEMM on the GPU.

This is the "do nothing clever" reference point: it reads 16 bits per weight,
so in the memory-bound decode regime the INT4 kernels can be at most ~4x
faster than it.
"""

import torch

from common.quantize import dequantize

WEIGHT_BITS = 16   # read by the harness to count bytes moved

_w: torch.Tensor | None = None   # [K, N] FP16, set by prepare()


def prepare(weights: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor,
            group_size: int) -> None:
    """Dequantize once to FP16 (not timed).

    .contiguous() makes sure the matrix is a plain row-major buffer, so cuBLAS
    gets a standard layout and no hidden copy happens inside run().
    """
    global _w
    _w = None   # release the previous layer's weights before allocating new ones
    _w = dequantize(weights, scales, zeros, group_size, dtype=torch.float16).contiguous()


def run(x: torch.Tensor) -> torch.Tensor:
    """y [M, N] = x [M, K] @ W [K, N] in FP16 (cuBLAS GEMM on the GPU)."""
    return torch.matmul(x, _w)


def reference_weights() -> torch.Tensor:
    """The FP16 weights cuBLAS actually multiplies by."""
    return _w
