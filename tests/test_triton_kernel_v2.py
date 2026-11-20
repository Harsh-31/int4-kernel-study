"""Correctness of our Triton INT4 kernel, version 2 = v1 + split-K (impls/triton_kernel_v2).

Needs a GPU and Triton, so it runs only through Slurm.

  1. Layout: the packed data equals v1's, and reference_weights() decodes it exactly.
  2. Kernel: passes the precision-matched rule (FP16, <= 2x cuBLAS FP16's error)
     on all four layers and the early batch sizes, both with split-K forced to
     1 (no split, the plain v1 path), 2 and 8, and with the automatic choice.
  3. The automatic SPLIT_K fills the GPU on the narrow layers and leaves
     mlp_up (already enough programs) unsplit.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU (run via Slurm)", allow_module_level=True)
try:
    from impls import triton_kernel as v1
    from impls import triton_kernel_v2 as v2
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
    v2.prepare(qw.q, qw.scales, qw.zeros, G)
    return qw, K, N


@pytest.fixture(autouse=True)
def _reset_forced_split():
    yield
    v2.FORCE_SPLIT_K = v2.FORCE_BLOCK_M = v2.FORCE_BLOCK_N = None
    v2.NUM_WARPS = v2.NUM_STAGES = None


def test_declares_fp16_compute():
    assert v2.COMPUTE_DTYPE == torch.float16


@pytest.mark.parametrize("layer", LAYERS)
def test_packed_weights_match_v1_and_decode_exactly(layer):
    qw, K, N = _prepare(layer)
    assert torch.equal(v2._state["packed"], v1.pack(qw.q, qw.scales, qw.zeros, G)["packed"])
    assert torch.equal(v2.reference_weights(), qw.dequantize())


@pytest.mark.parametrize("split", [1, 2, 8, None])
@pytest.mark.parametrize("layer", LAYERS)
def test_triton_v2_passes_correctness(layer, split):
    _, K, N = _prepare(layer)
    v2.FORCE_SPLIT_K = split
    w_ref = v2.reference_weights()
    for M in batch_sizes("early"):
        x = random_activations(M, K, device="cuda")
        out = v2.run(x)
        assert out.shape == (M, N) and out.dtype == torch.float16
        result = check(out, x, w_ref, v2.COMPUTE_DTYPE)
        assert result.passed, (f"split={split} M={M}: triton_v2 {result.impl} "
                               f"vs cuBLAS FP16 {result.baseline}")


def test_overrides_stay_correct_and_expose_the_compiled_kernel():
    """The knobs used by analysis/triton_inspect.py must not break correctness."""
    _, K, N = _prepare("attn_out")
    w_ref = v2.reference_weights()
    v2.FORCE_BLOCK_M, v2.FORCE_BLOCK_N, v2.NUM_WARPS, v2.NUM_STAGES = 32, 128, 8, 2
    for M in batch_sizes("early"):
        x = random_activations(M, K, device="cuda")
        result = check(v2.run(x), x, w_ref, v2.COMPUTE_DTYPE)
        assert result.passed, f"M={M}: {result.impl} vs {result.baseline}"
    assert v2._state["last_kernel"] is not None


def test_auto_split_fills_the_gpu():
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    bk = v1.block_k_for(G)
    target = v2.TARGET_PROGRAMS_PER_SM * sms
    for layer, shape in load_config()["layers"].items():
        K, N = shape["K"], shape["N"]
        for M in batch_sizes("early"):
            split = v2.split_k_for(M, N, K, bk, sms)
            programs = -(-N // v2.BLOCK_N) * -(-M // v1.block_m_for(M))
            assert (K // bk) % split == 0, (layer, M, split)
            if programs >= target:
                assert split == 1, (layer, M, split)      # e.g. mlp_up: no split needed
            else:
                assert split > 1, (layer, M, split)       # narrow layers get split


def test_asymmetric_is_refused():
    qw = random_quantized_weights(4 * G, 64, G, "asymmetric", device="cuda")
    with pytest.raises(NotImplementedError):
        v2.prepare(qw.q, qw.scales, qw.zeros, G)
