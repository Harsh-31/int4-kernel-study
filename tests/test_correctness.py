"""Tests for the pass/fail rule in common/correctness.py (run for both schemes)."""

import pytest
import torch

from common.config import QUANT_SCHEMES, batch_sizes, group_sizes, load_config
from common.correctness import check, fp16_baseline_output, reference_output
from common.quantize import random_quantized_weights


@pytest.fixture(params=QUANT_SCHEMES)
def problem(request):
    """(x, w_ref): random activations and dequantized INT4 weights."""
    G = group_sizes("early")[0]
    M = max(batch_sizes("early"))
    qw = random_quantized_weights(4 * G, 64, G, request.param)
    gen = torch.Generator().manual_seed(load_config()["seed"])
    x = torch.randn(M, 4 * G, generator=gen).to(torch.float16)
    return x, qw.dequantize()


def test_fp16_baseline_passes_against_itself(problem):
    x, w_ref = problem
    assert check(fp16_baseline_output(x, w_ref), x, w_ref).passed


def test_exact_reference_passes(problem):
    x, w_ref = problem
    result = check(reference_output(x, w_ref), x, w_ref)
    assert result.passed and result.impl.max_abs == 0.0


def test_wrong_output_fails(problem):
    x, w_ref = problem
    out = fp16_baseline_output(x, w_ref).clone()
    out[0, 0] += 1.0   # one corrupted element
    assert not check(out, x, w_ref).passed


def test_wrong_weights_fail(problem):
    """Using slightly different weights (e.g. a packing bug) must fail."""
    x, w_ref = problem
    w_bad = w_ref.clone()
    w_bad[0] = 0.0     # one row of weights lost
    assert not check(fp16_baseline_output(x, w_bad), x, w_ref).passed
