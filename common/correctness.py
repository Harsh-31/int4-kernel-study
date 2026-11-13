"""FP32 reference and the pass/fail rule from CLAUDE.md.

Rule
----
1. Reference: dequantize the SAME INT4 weights the implementation uses,
   w_ref = (q - z) * s, and compute x @ w_ref in FP32.
   Each library packs weights in its own layout, so each implementation is
   responsible for producing w_ref from ITS OWN packed weights/scales/zeros
   (unpacked into our [K, N] layout). That way a bug in a library's packing
   shows up as a failure instead of being hidden by a shared reference.
2. Baseline: cuBLAS FP16, i.e. the same dequantized weights cast to FP16 and
   multiplied with torch.matmul in FP16 (dispatches to cuBLAS on GPU). Its error
   against the reference is the "normal" FP16 rounding error for this problem.
3. Pass if BOTH the max-abs error and the relative-L2 error of the
   implementation are <= 2x the baseline's corresponding error.

The rule is applied strictly. In practice the FP16 baseline error is never
exactly zero (its FP16 output rounding alone is nonzero), so the 2x threshold
is well defined.
"""

from dataclasses import dataclass

import torch

from common.quantize import dequantize

TOLERANCE_FACTOR = 2.0  # "<= 2x cuBLAS FP16's error" from CLAUDE.md


@dataclass
class Errors:
    max_abs: float
    rel_l2: float


@dataclass
class CorrectnessResult:
    impl: Errors       # implementation vs FP32 reference
    baseline: Errors   # cuBLAS FP16 vs FP32 reference
    passed: bool


def reference_output(x: torch.Tensor, w_ref: torch.Tensor) -> torch.Tensor:
    """FP32 reference: x [M, K] @ w_ref [K, N], both upcast to FP32."""
    return x.to(torch.float32) @ w_ref.to(torch.float32)


def reference_from_int4(x: torch.Tensor, q: torch.Tensor, scales: torch.Tensor,
                        zeros: torch.Tensor, group_size: int) -> torch.Tensor:
    """FP32 reference from unpacked INT4 weights in our [K, N] layout."""
    return reference_output(x, dequantize(q, scales, zeros, group_size))


def fp16_baseline_output(x: torch.Tensor, w_ref: torch.Tensor) -> torch.Tensor:
    """cuBLAS FP16 baseline on the same dequantized weights."""
    return torch.matmul(x.to(torch.float16), w_ref.to(torch.float16))


def compute_errors(out: torch.Tensor, ref: torch.Tensor) -> Errors:
    """max |out - ref| and ||out - ref||_2 / ||ref||_2, computed in FP32."""
    if out.shape != ref.shape:
        raise ValueError(f"output shape {tuple(out.shape)} != reference {tuple(ref.shape)}")
    diff = out.to(torch.float32) - ref
    return Errors(
        max_abs=diff.abs().max().item(),
        rel_l2=(torch.linalg.norm(diff) / torch.linalg.norm(ref)).item(),
    )


def check(out: torch.Tensor, x: torch.Tensor, w_ref: torch.Tensor) -> CorrectnessResult:
    """Apply the CLAUDE.md rule to an implementation's output.

    out:   the implementation's result for input x, shape [M, N]
    x:     the input activations, shape [M, K]
    w_ref: [K, N] weights dequantized from the implementation's own packed data
    """
    ref = reference_output(x, w_ref)
    impl_err = compute_errors(out, ref)
    base_err = compute_errors(fp16_baseline_output(x, w_ref), ref)
    passed = (
        impl_err.max_abs <= TOLERANCE_FACTOR * base_err.max_abs
        and impl_err.rel_l2 <= TOLERANCE_FACTOR * base_err.rel_l2
    )
    return CorrectnessResult(impl=impl_err, baseline=base_err, passed=passed)
