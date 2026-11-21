"""Correctness of Triton v2 with tuned tile settings (impls/triton_kernel_v2_tuned).

Needs a GPU and Triton, so it runs only through Slurm.

  1. Passes the precision-matched rule (FP16, <= 2x cuBLAS FP16's error) on all
     four layers and the early batch sizes.
  2. Uses its own state, so preparing it never changes what v2 computes.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU (run via Slurm)", allow_module_level=True)
try:
    from impls import triton_kernel_v2 as v2
    from impls import triton_kernel_v2_tuned as tuned
except (ImportError, OSError) as e:
    pytest.skip(f"triton unavailable: {e}", allow_module_level=True)

from common.config import batch_sizes, group_sizes, load_config  # noqa: E402
from common.correctness import check  # noqa: E402
from common.quantize import random_activations, random_quantized_weights  # noqa: E402

LAYERS = list(load_config()["layers"])
G = group_sizes("early")[0]


def _weights(layer):
    shape = load_config()["layers"][layer]
    return random_quantized_weights(shape["K"], shape["N"], G, "symmetric", device="cuda")


def test_declares_fp16_compute():
    assert tuned.COMPUTE_DTYPE == torch.float16


@pytest.mark.parametrize("layer", LAYERS)
def test_tuned_passes_correctness(layer):
    qw = _weights(layer)
    tuned.prepare(qw.q, qw.scales, qw.zeros, G)
    w_ref = tuned.reference_weights()
    assert torch.equal(w_ref, qw.dequantize())
    for M in batch_sizes("early"):
        x = random_activations(M, qw.q.shape[0], device="cuda")
        out = tuned.run(x)
        assert out.shape == (M, qw.q.shape[1]) and out.dtype == torch.float16
        result = check(out, x, w_ref, tuned.COMPUTE_DTYPE)
        assert result.passed, f"M={M}: tuned {result.impl} vs cuBLAS FP16 {result.baseline}"


def test_tuned_does_not_disturb_v2():
    a, b = _weights("attn_out"), _weights("mlp_down")
    v2.prepare(a.q, a.scales, a.zeros, G)
    x = random_activations(16, a.q.shape[0], device="cuda")
    before = v2.run(x).clone()
    tuned.prepare(b.q, b.scales, b.zeros, G)       # different layer in tuned's state
    tuned.run(random_activations(16, b.q.shape[0], device="cuda"))
    assert v2._state["N"] == a.q.shape[1]
    # Same inputs, same kernel: v2's result is unchanged (up to split-K atomic order).
    torch.testing.assert_close(v2.run(x), before, rtol=1e-3, atol=1e-3)
