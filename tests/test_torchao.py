"""Correctness of the TorchAO implementation (impls/torchao, tinygemm path).

Needs a GPU (sm80+) and torchao, so it runs only through Slurm.

  1. Layout: the q unpacked from TorchAO's packed qdata equals our q exactly,
     and reference_weights() equals tinygemm's formula (q - 8)*scale + zero
     computed independently from our q, s, z. For the symmetric scheme (the
     project default) that is exactly our canonical dequantization, because our
     scales are BF16-exact and the zero is 0.
  2. Kernel: passes the precision-matched project rule, i.e. error <= 2x
     cuBLAS BF16's error on the same weights (TorchAO's COMPUTE_DTYPE is BF16).
  3. Direct call: run() (aten._weight_int4pack_mm on a BF16 input) gives the
     same result as TorchAO's own linear() on an Int4TilePackedTo4dTensor built
     from the same packed data, i.e. timing the op directly does not change
     what is computed.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU (run via Slurm)", allow_module_level=True)
try:
    from impls import torchao as tao
    from torchao.quantization import Int4TilePackedTo4dTensor
except (ImportError, OSError) as e:
    pytest.skip(f"torchao unavailable: {e}", allow_module_level=True)

from common.config import QUANT_SCHEMES, batch_sizes, group_sizes, load_config  # noqa: E402
from common.correctness import check  # noqa: E402
from common.quantize import random_activations, random_quantized_weights  # noqa: E402

LAYERS = list(load_config()["layers"])


def _prepare(layer, G, scheme):
    shape = load_config()["layers"][layer]
    K, N = shape["K"], shape["N"]
    qw = random_quantized_weights(K, N, G, scheme, device="cuda")
    tao.prepare(qw.q, qw.scales, qw.zeros, G)
    return qw, K, N


def _tinygemm_form(qw):
    """(q - 8) * bf16(s) + bf16((8 - z) * s), in FP32, from OUR tensors."""
    G = qw.group_size
    s = qw.scales.float()
    s_b = s.to(torch.bfloat16).float().repeat_interleave(G, 0)
    z_b = ((8 - qw.zeros.float()) * s).to(torch.bfloat16).float().repeat_interleave(G, 0)
    return (qw.q.float() - 8) * s_b + z_b


def test_declares_bf16_compute():
    assert tao.COMPUTE_DTYPE == torch.bfloat16


@pytest.mark.parametrize("scheme", QUANT_SCHEMES)
@pytest.mark.parametrize("group_size", group_sizes("full"))
@pytest.mark.parametrize("layer", LAYERS)
def test_packed_weights_decode_exactly(layer, group_size, scheme):
    qw, K, N = _prepare(layer, group_size, scheme)
    assert torch.equal(tao.unpack_tinygemm_weight(tao._state["qdata"], N, K), qw.q.t())
    w_ref = tao.reference_weights()
    assert torch.equal(w_ref, _tinygemm_form(qw))
    if scheme == "symmetric":
        # BF16-exact scales + zero 0: TorchAO stores exactly our weights.
        assert torch.equal(w_ref, qw.dequantize())


def test_symmetric_zero_is_exactly_zero():
    _prepare(LAYERS[0], group_sizes("early")[0], "symmetric")
    _, zero = tao.unpack_tinygemm_scales_and_zeros(tao._state["scale_and_zero"])
    assert torch.all(zero == 0)


@pytest.mark.parametrize("scheme", QUANT_SCHEMES)
@pytest.mark.parametrize("layer", LAYERS)
def test_torchao_passes_correctness(layer, scheme):
    """Precision-matched project rule: <= 2x cuBLAS BF16 error."""
    _, K, N = _prepare(layer, group_sizes("early")[0], scheme)
    w_ref = tao.reference_weights()
    for M in batch_sizes("full"):
        x = random_activations(M, K, device="cuda")
        out = tao.run(tao.prepare_input(x))
        assert out.shape == (M, N) and out.dtype == torch.bfloat16
        result = check(out, x, w_ref, tao.COMPUTE_DTYPE)
        assert result.passed, f"M={M}: torchao {result.impl} vs cuBLAS BF16 {result.baseline}"


@pytest.mark.parametrize("layer", LAYERS)
def test_direct_call_matches_torchao_linear(layer):
    """run() computes exactly what TorchAO's own linear() computes."""
    G = group_sizes("early")[0]
    _, K, N = _prepare(layer, G, "symmetric")
    weight = Int4TilePackedTo4dTensor(
        qdata=tao._state["qdata"], scale_and_zero=tao._state["scale_and_zero"],
        block_size=[1, G], shape=torch.Size([N, K]))
    for M in batch_sizes("early"):
        x = random_activations(M, K, device="cuda")
        via_torchao = torch.nn.functional.linear(x, weight)          # FP16 in, FP16 out
        direct = tao.run(tao.prepare_input(x))                       # BF16 in, BF16 out
        assert torch.equal(via_torchao, direct.to(torch.float16))


def test_unsupported_shape_is_skipped_not_crashed():
    """K not a multiple of 1024 would need TorchAO's zero padding."""
    G = group_sizes("early")[0]
    qw = random_quantized_weights(2 * G, 64, G, "symmetric", device="cuda")
    with pytest.raises(NotImplementedError):
        tao.prepare(qw.q, qw.scales, qw.zeros, G)
