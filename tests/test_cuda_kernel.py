"""Tests for our CUDA INT4 kernel, version 1 (impls/cuda_kernel).

CPU (run anywhere):
  * pack() / unpack() are exact inverses, and the bit layout is what the kernel
    assumes: nibble j of word w in column n holds q[8w + j, n].
GPU (run via Slurm; the first test compiles the kernel, about a minute):
  * reference_weights() decodes the packed data exactly;
  * the kernel passes the precision-matched rule (FP16: <= 2x cuBLAS FP16's
    error) on all four layers at the early batch sizes, at every group size,
    and at batch sizes that do not fill a row tile (3, 9, 256);
  * the asymmetric scheme is refused.
"""

import pytest
import torch

from common.config import batch_sizes, group_sizes, load_config
from common.correctness import check
from common.quantize import random_activations, random_quantized_weights
from impls import cuda_kernel as ck

LAYERS = list(load_config()["layers"])
G = group_sizes("early")[0]
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (run via Slurm)")


# --------------------------------------------------------------------- CPU

def test_pack_unpack_round_trip():
    q = torch.randint(0, 16, (256, 40), dtype=torch.uint8)
    packed = ck.pack(q)
    assert packed.dtype == torch.int32 and packed.shape == (40, 256 // 8)
    assert torch.equal(ck.unpack(packed), q)


def test_bit_layout_matches_the_kernel():
    """Decode exactly the way int4_kernel.cu does: (word >> 4j) & 0xF is k = 8*word_index + j."""
    q = torch.randint(0, 16, (64, 5), dtype=torch.uint8)
    words = ck.pack(q).to(torch.int64) & 0xFFFFFFFF           # unsigned 32-bit values
    for n in range(5):
        for w in range(64 // 8):
            for j in range(8):
                assert (int(words[n, w]) >> (4 * j)) & 0xF == int(q[8 * w + j, n])


# --------------------------------------------------------------------- GPU

def _prepare(layer, group_size=G):
    shape = load_config()["layers"][layer]
    qw = random_quantized_weights(shape["K"], shape["N"], group_size, "symmetric", device="cuda")
    ck.prepare(qw.q, qw.scales, qw.zeros, group_size)
    return qw


def _assert_correct(K, Ms):
    w_ref = ck.reference_weights()
    for M in Ms:
        x = random_activations(M, K, device="cuda")
        out = ck.run(x)
        assert out.shape == (M, w_ref.shape[1]) and out.dtype == torch.float16
        result = check(out, x, w_ref, ck.COMPUTE_DTYPE)
        assert result.passed, f"M={M}: cuda {result.impl} vs cuBLAS FP16 {result.baseline}"


@gpu
@pytest.mark.parametrize("layer", LAYERS)
def test_packed_weights_decode_exactly(layer):
    qw = _prepare(layer)
    assert torch.equal(ck.reference_weights(), qw.dequantize())


@gpu
@pytest.mark.parametrize("layer", LAYERS)
def test_cuda_kernel_passes_correctness(layer):
    qw = _prepare(layer)
    _assert_correct(qw.q.shape[0], batch_sizes("early"))


@gpu
@pytest.mark.parametrize("group_size", group_sizes("full"))
def test_every_group_size(group_size):
    qw = _prepare("attn_out", group_size)
    _assert_correct(qw.q.shape[0], [1, 16])


@gpu
def test_batch_sizes_that_do_not_fill_a_row_tile():
    qw = _prepare("attn_qkv")
    _assert_correct(qw.q.shape[0], [3, 9, max(batch_sizes("full"))])


@gpu
def test_asymmetric_is_refused():
    qw = random_quantized_weights(4 * G, 64, G, "asymmetric", device="cuda")
    with pytest.raises(NotImplementedError):
        ck.prepare(qw.q, qw.scales, qw.zeros, G)
