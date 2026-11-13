"""Correctness of the cuBLAS FP16 baseline (impls/cublas).

The reference here is built from the canonical (q, s, z), NOT from
cublas.reference_weights(), so a bug in prepare() (wrong dequantization,
transposed weights, ...) would make the test fail.

The small-shape test also runs on CPU (torch.matmul in FP16 without cuBLAS),
which checks the prepare()/run() logic on a laptop. The real-shape test needs a
GPU, where torch.matmul actually calls cuBLAS.
"""

import pytest
import torch

from common.config import QUANT_SCHEMES, batch_sizes, group_sizes, load_config
from common.correctness import check
from common.quantize import random_activations, random_quantized_weights
from impls import cublas

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
gpu_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (run via Slurm)")


def _check_cublas(K, N, G, scheme, device):
    qw = random_quantized_weights(K, N, G, scheme, device=device)
    cublas.prepare(qw.q, qw.scales, qw.zeros, G)
    w_canonical = qw.dequantize()
    for M in batch_sizes("early"):
        x = random_activations(M, K, device=device)
        out = cublas.run(x)
        assert out.shape == (M, N) and out.dtype == torch.float16
        result = check(out, x, w_canonical)
        assert result.passed, f"M={M}: impl {result.impl} vs fp16 {result.baseline}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("scheme", QUANT_SCHEMES)
def test_cublas_correct_small(scheme, device):
    G = group_sizes("early")[0]
    _check_cublas(4 * G, 96, G, scheme, device)


@gpu_only
@pytest.mark.parametrize("scheme", QUANT_SCHEMES)
@pytest.mark.parametrize("layer", list(load_config()["layers"]))
def test_cublas_correct_real_shapes(layer, scheme):
    shape = load_config()["layers"][layer]
    _check_cublas(shape["K"], shape["N"], group_sizes("early")[0], scheme, "cuda")


def test_cublas_declares_fp16_weights():
    assert cublas.WEIGHT_BITS == 16


def test_reference_weights_are_what_run_uses():
    G = group_sizes("early")[0]
    qw = random_quantized_weights(2 * G, 16, G, QUANT_SCHEMES[0])
    cublas.prepare(qw.q, qw.scales, qw.zeros, G)
    w = cublas.reference_weights()
    assert w.dtype == torch.float16 and w.shape == (2 * G, 16)
    x = random_activations(4, 2 * G)
    assert torch.equal(cublas.run(x), x @ w)
