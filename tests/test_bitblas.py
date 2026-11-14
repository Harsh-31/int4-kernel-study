"""Correctness of the BitBLAS implementation (impls/bitblas).

EXCLUDED: BitBLAS is not part of the study (see CLAUDE.md and
common.harness.EXCLUDED_IMPLS). bitblas 0.1.0's bundled TVM needs the CUDA 12
runtime (libcudart.so.12 / libnvrtc.so.12) and our environment is CUDA 13
only. This whole file is skipped before anything is imported; the tests below
are kept for reference.

Same two checks as tests/test_marlin.py:
  1. Layout: reference_weights(), decoded from BitBLAS's own packed buffer,
     scales and zeros, equals our canonical dequantization exactly.
  2. Kernel: BitBLAS's output passes the CLAUDE.md rule against that reference.

Only the early group size is tested: every (layer, group size) is a separate
BitBLAS tuning run (minutes each, the first time). Both schemes share one
tuned operator, since they differ only in the zeros tensor. Tuned operators
are cached on disk, so the benchmark reuses what the tests tuned.
"""

import pytest
import torch

from common.harness import EXCLUDED_IMPLS

# Skip first, before importing impls.bitblas (which would try to load the
# CUDA 12 libraries). Reason text: "excluded: requires CUDA 12".
if "bitblas" in EXCLUDED_IMPLS:
    pytest.skip(f"excluded: {EXCLUDED_IMPLS['bitblas']}", allow_module_level=True)
if not torch.cuda.is_available():
    pytest.skip("needs a GPU (run via Slurm)", allow_module_level=True)
try:
    from impls import bitblas as bb
except (ImportError, OSError) as e:   # OSError: bundled native libraries failed to load
    pytest.skip(f"bitblas unavailable: {e}", allow_module_level=True)

from common.config import QUANT_SCHEMES, batch_sizes, group_sizes, load_config  # noqa: E402
from common.correctness import check  # noqa: E402
from common.quantize import random_activations, random_quantized_weights  # noqa: E402

LAYERS = list(load_config()["layers"])
G = group_sizes("early")[0]


def _prepare(layer, scheme):
    shape = load_config()["layers"][layer]
    K, N = shape["K"], shape["N"]
    qw = random_quantized_weights(K, N, G, scheme, device="cuda")
    bb.prepare(qw.q, qw.scales, qw.zeros, G)
    return qw, K, N


@pytest.mark.parametrize("scheme", QUANT_SCHEMES)
def test_zeros_tensor_matches_scheme(scheme):
    _prepare(LAYERS[0], scheme)
    z = bb._state["zeros"]
    assert z.dtype == torch.float16
    assert bool(torch.all(z == 8)) == (scheme == "symmetric")


@pytest.mark.parametrize("scheme", QUANT_SCHEMES)
@pytest.mark.parametrize("layer", LAYERS)
def test_packed_weights_decode_exactly(layer, scheme):
    qw, _, _ = _prepare(layer, scheme)
    assert torch.equal(bb.reference_weights(), qw.dequantize().to("cuda"))


@pytest.mark.parametrize("scheme", QUANT_SCHEMES)
@pytest.mark.parametrize("layer", LAYERS)
def test_bitblas_passes_correctness(layer, scheme):
    _, K, N = _prepare(layer, scheme)
    w_ref = bb.reference_weights()
    for M in batch_sizes("full"):
        x = random_activations(M, K, device="cuda")
        out = bb.run(x)
        assert out.shape == (M, N) and out.dtype == torch.float16
        result = check(out, x, w_ref)
        assert result.passed, f"M={M}: bitblas {result.impl} vs fp16 {result.baseline}"
