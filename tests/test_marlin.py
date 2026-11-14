"""Correctness of the Marlin implementation (impls/marlin, via vLLM's ops).

Needs a GPU and vLLM, so it runs only through Slurm; elsewhere it is skipped.

Two separate checks:
  1. Layout: reference_weights(), decoded from Marlin's own packed buffers,
     equals our canonical dequantization exactly. This proves prepare() put
     the right values in the right places (and that the decoder is right).
  2. Kernel: Marlin's output passes the CLAUDE.md rule against the FP32
     reference built from those packed buffers.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU (run via Slurm)", allow_module_level=True)
pytest.importorskip("vllm")

from common.config import QUANT_SCHEMES, batch_sizes, group_sizes, load_config  # noqa: E402
from common.correctness import check  # noqa: E402
from common.quantize import random_activations, random_quantized_weights  # noqa: E402
from impls import marlin  # noqa: E402
from vllm.scalar_type import scalar_types  # noqa: E402

LAYERS = list(load_config()["layers"])


def _prepare(layer, G, scheme):
    shape = load_config()["layers"][layer]
    K, N = shape["K"], shape["N"]
    qw = random_quantized_weights(K, N, G, scheme, device="cuda")
    marlin.prepare(qw.q, qw.scales, qw.zeros, G)
    return qw, K, N


@pytest.mark.parametrize("scheme", QUANT_SCHEMES)
def test_scheme_selects_marlin_type(scheme):
    _prepare(LAYERS[0], group_sizes("early")[0], scheme)
    expected = scalar_types.uint4b8 if scheme == "symmetric" else scalar_types.uint4
    assert marlin._state["wtype"] == expected
    assert (marlin._state["zp"].numel() == 0) == (scheme == "symmetric")


@pytest.mark.parametrize("scheme", QUANT_SCHEMES)
@pytest.mark.parametrize("group_size", group_sizes("full"))
@pytest.mark.parametrize("layer", LAYERS)
def test_packed_weights_decode_exactly(layer, group_size, scheme):
    qw, _, _ = _prepare(layer, group_size, scheme)
    assert torch.equal(marlin.reference_weights(), qw.dequantize())


@pytest.mark.parametrize("scheme", QUANT_SCHEMES)
@pytest.mark.parametrize("group_size", group_sizes("full"))
@pytest.mark.parametrize("layer", LAYERS)
def test_marlin_passes_correctness(layer, group_size, scheme):
    _, K, N = _prepare(layer, group_size, scheme)
    w_ref = marlin.reference_weights()
    for M in batch_sizes("full"):
        x = random_activations(M, K, device="cuda")
        out = marlin.run(x)
        assert out.shape == (M, N) and out.dtype == torch.float16
        result = check(out, x, w_ref)
        assert result.passed, f"M={M}: marlin {result.impl} vs fp16 {result.baseline}"


def test_unsupported_shape_is_skipped_not_crashed():
    """N not divisible by 64 is outside Marlin's supported shapes."""
    G = group_sizes("early")[0]
    qw = random_quantized_weights(2 * G, 96, G, "symmetric", device="cuda")
    with pytest.raises(NotImplementedError):
        marlin.prepare(qw.q, qw.scales, qw.zeros, G)
