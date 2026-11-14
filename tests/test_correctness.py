"""Tests for the precision-matched pass/fail rule in common/correctness.py
(run for both quantization schemes and both compute dtypes)."""

import pytest
import torch

from common.config import QUANT_SCHEMES, batch_sizes, group_sizes, load_config
from common.correctness import check, cublas_baseline_output, reference_output
from common.quantize import random_quantized_weights

DTYPES = pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])


@pytest.fixture(params=QUANT_SCHEMES)
def problem(request):
    """(x, w_ref): random FP16 activations and dequantized INT4 weights."""
    G = group_sizes("early")[0]
    M = max(batch_sizes("early"))
    qw = random_quantized_weights(4 * G, 64, G, request.param)
    gen = torch.Generator().manual_seed(load_config()["seed"])
    x = torch.randn(M, 4 * G, generator=gen).to(torch.float16)
    return x, qw.dequantize()


@DTYPES
def test_cublas_baseline_passes_against_itself(problem, dtype):
    x, w_ref = problem
    assert check(cublas_baseline_output(x, w_ref, dtype), x, w_ref, dtype).passed


@DTYPES
def test_exact_reference_passes(problem, dtype):
    x, w_ref = problem
    result = check(reference_output(x, w_ref), x, w_ref, dtype)
    assert result.passed and result.impl.max_abs == 0.0


@DTYPES
def test_wrong_output_fails(problem, dtype):
    x, w_ref = problem
    out = cublas_baseline_output(x, w_ref, dtype).float().clone()
    out[0, 0] += 1.0   # one corrupted element
    assert not check(out, x, w_ref, dtype).passed


@DTYPES
def test_wrong_weights_fail(problem, dtype):
    """Using slightly different weights (e.g. a packing bug) must fail."""
    x, w_ref = problem
    w_bad = w_ref.clone()
    w_bad[0] = 0.0     # one row of weights lost
    assert not check(cublas_baseline_output(x, w_bad, dtype), x, w_ref, dtype).passed


def test_bf16_result_fails_the_fp16_rule(problem):
    """Why the rule is precision-matched: an honest BF16 computation is judged
    a failure against the FP16 baseline, but passes against the BF16 one."""
    x, w_ref = problem
    out_bf16 = cublas_baseline_output(x, w_ref, torch.bfloat16)
    assert not check(out_bf16, x, w_ref, torch.float16).passed
    assert check(out_bf16, x, w_ref, torch.bfloat16).passed


def test_unknown_compute_dtype_raises(problem):
    x, w_ref = problem
    with pytest.raises(ValueError):
        check(reference_output(x, w_ref), x, w_ref, torch.float32)
