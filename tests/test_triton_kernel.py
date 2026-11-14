"""Correctness of our Triton INT4 kernel, version 1 (impls/triton_kernel).

Needs a GPU and Triton, so it runs only through Slurm.

Version 1 goals: correct on all four layers, symmetric scheme, the early group
size (128) and the early batch sizes (1, 16, 64).

  1. Layout: reference_weights(), decoded from the packed buffer the kernel
     reads, equals our canonical dequantization exactly.
  2. Kernel: passes the precision-matched rule (COMPUTE_DTYPE = FP16, so
     <= 2x cuBLAS FP16's error).
  3. Scope: the asymmetric scheme is refused (NotImplementedError), so the
     benchmark records a skip instead of a wrong result.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU (run via Slurm)", allow_module_level=True)
try:
    from impls import triton_kernel as tk
except (ImportError, OSError) as e:
    pytest.skip(f"triton unavailable: {e}", allow_module_level=True)

from common.config import batch_sizes, group_sizes, load_config  # noqa: E402
from common.correctness import check  # noqa: E402
from common.quantize import random_activations, random_quantized_weights  # noqa: E402

LAYERS = list(load_config()["layers"])
G = group_sizes("early")[0]
SCHEME = "symmetric"


def _prepare(layer):
    shape = load_config()["layers"][layer]
    K, N = shape["K"], shape["N"]
    qw = random_quantized_weights(K, N, G, SCHEME, device="cuda")
    tk.prepare(qw.q, qw.scales, qw.zeros, G)
    return qw, K, N


def test_declares_fp16_compute():
    assert tk.COMPUTE_DTYPE == torch.float16


@pytest.mark.parametrize("layer", LAYERS)
def test_packed_weights_decode_exactly(layer):
    qw, K, N = _prepare(layer)
    assert tk._state["packed"].shape == (K // 2, N)
    assert tk._state["packed"].dtype == torch.uint8
    assert torch.equal(tk.unpack(tk._state["packed"]), qw.q)
    assert torch.equal(tk.reference_weights(), qw.dequantize())


@pytest.mark.parametrize("layer", LAYERS)
def test_triton_kernel_passes_correctness(layer):
    _, K, N = _prepare(layer)
    w_ref = tk.reference_weights()
    for M in batch_sizes("early"):
        x = random_activations(M, K, device="cuda")
        out = tk.run(x)
        assert out.shape == (M, N) and out.dtype == torch.float16
        result = check(out, x, w_ref, tk.COMPUTE_DTYPE)
        assert result.passed, f"M={M}: triton {result.impl} vs cuBLAS FP16 {result.baseline}"


def test_asymmetric_is_refused():
    qw = random_quantized_weights(4 * G, 64, G, "asymmetric", device="cuda")
    with pytest.raises(NotImplementedError):
        tk.prepare(qw.q, qw.scales, qw.zeros, G)
