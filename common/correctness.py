"""FP32 reference and the precision-matched pass/fail rule from CLAUDE.md.

Rule
----
1. Reference: dequantize the SAME INT4 weights the implementation uses and
   compute x @ w_ref in FP32. Each library packs weights in its own layout, so
   each implementation produces w_ref from ITS OWN packed weights/scales/zeros
   (unpacked into our [K, N] layout). A bug in a library's packing therefore
   shows up as a failure instead of being hidden by a shared reference.
2. Baseline, precision-matched: every implementation declares its compute
   dtype (COMPUTE_DTYPE, FP16 by default, or BF16). The baseline is cuBLAS in
   THAT dtype on the same dequantized weights: torch.matmul with x and w_ref
   cast to that dtype (dispatches to cuBLAS on GPU). Its error against the
   reference is the "normal" rounding error for that precision.
3. Pass if BOTH the max-abs error and the relative-L2 error of the
   implementation are <= 2x the baseline's corresponding error.

Why precision-matched: BF16 has 8 significant bits to FP16's 11, so a BF16-only
kernel (TorchAO's tinygemm) is ~8x less precise than FP16 by construction.
Comparing it to cuBLAS FP16 would fail it for its dtype, not for a bug.
Comparing each kernel to cuBLAS in its own dtype still catches real errors
(wrong layout, wrong scales, bad accumulation) while letting both precisions be
judged fairly.

The rule is applied strictly. In practice the baseline error is never exactly
zero (its output rounding alone is nonzero), so the 2x threshold is well defined.
"""

from dataclasses import dataclass

import torch

from common.quantize import dequantize

TOLERANCE_FACTOR = 2.0  # "<= 2x cuBLAS's error in the same dtype" from CLAUDE.md
COMPUTE_DTYPES = (torch.float16, torch.bfloat16)


@dataclass
class Errors:
    max_abs: float
    rel_l2: float


@dataclass
class CorrectnessResult:
    impl: Errors       # implementation vs FP32 reference
    baseline: Errors   # cuBLAS in the implementation's compute dtype vs FP32 reference
    passed: bool
    compute_dtype: torch.dtype = torch.float16


def reference_output(x: torch.Tensor, w_ref: torch.Tensor) -> torch.Tensor:
    """FP32 reference: x [M, K] @ w_ref [K, N], both upcast to FP32."""
    return x.to(torch.float32) @ w_ref.to(torch.float32)


def reference_from_int4(x: torch.Tensor, q: torch.Tensor, scales: torch.Tensor,
                        zeros: torch.Tensor, group_size: int) -> torch.Tensor:
    """FP32 reference from unpacked INT4 weights in our [K, N] layout."""
    return reference_output(x, dequantize(q, scales, zeros, group_size))


def cublas_baseline_output(x: torch.Tensor, w_ref: torch.Tensor,
                           dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """cuBLAS baseline in `dtype` (FP16 or BF16) on the same dequantized weights."""
    if dtype not in COMPUTE_DTYPES:
        raise ValueError(f"compute dtype must be one of {COMPUTE_DTYPES}, got {dtype}")
    return torch.matmul(x.to(dtype), w_ref.to(dtype))


def compute_errors(out: torch.Tensor, ref: torch.Tensor) -> Errors:
    """max |out - ref| and ||out - ref||_2 / ||ref||_2, computed in FP32."""
    if out.shape != ref.shape:
        raise ValueError(f"output shape {tuple(out.shape)} != reference {tuple(ref.shape)}")
    diff = out.to(torch.float32) - ref
    return Errors(
        max_abs=diff.abs().max().item(),
        rel_l2=(torch.linalg.norm(diff) / torch.linalg.norm(ref)).item(),
    )


def check(out: torch.Tensor, x: torch.Tensor, w_ref: torch.Tensor,
          compute_dtype: torch.dtype = torch.float16) -> CorrectnessResult:
    """Apply the precision-matched CLAUDE.md rule to an implementation's output.

    out:           the implementation's result for input x, shape [M, N]
    x:             the FP16 input activations, shape [M, K] (the reference uses
                   these; a BF16 implementation and its BF16 baseline both see
                   them rounded to BF16)
    w_ref:         [K, N] weights dequantized from the implementation's own packed data
    compute_dtype: the implementation's COMPUTE_DTYPE; selects the cuBLAS baseline
    """
    ref = reference_output(x, w_ref)
    impl_err = compute_errors(out, ref)
    base_err = compute_errors(cublas_baseline_output(x, w_ref, compute_dtype), ref)
    passed = (
        impl_err.max_abs <= TOLERANCE_FACTOR * base_err.max_abs
        and impl_err.rel_l2 <= TOLERANCE_FACTOR * base_err.rel_l2
    )
    return CorrectnessResult(impl=impl_err, baseline=base_err, passed=passed,
                             compute_dtype=compute_dtype)
