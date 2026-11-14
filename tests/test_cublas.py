"""Correctness of the cuBLAS baselines (impls/cublas FP16, impls/cublas_bf16 BF16).

The reference here is built from the canonical (q, s, z), NOT from
reference_weights(), so a bug in prepare() (wrong dequantization, transposed
weights, ...) would make the test fail. Each baseline is checked against cuBLAS
in its own compute dtype (precision-matched rule).

The small-shape tests also run on CPU (torch.matmul without cuBLAS), which
checks the prepare()/run() logic on a laptop. The real-shape tests need a GPU,
where torch.matmul actually calls cuBLAS.
"""

import pytest
import torch

from common.config import QUANT_SCHEMES, batch_sizes, group_sizes, load_config
from common.correctness import check
from common.harness import compute_dtype, prepare_input
from common.quantize import random_activations, random_quantized_weights
from impls import cublas, cublas_bf16

BASELINES = pytest.mark.parametrize("impl", [cublas, cublas_bf16], ids=["fp16", "bf16"])
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
gpu_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (run via Slurm)")


def _check_baseline(impl, K, N, G, scheme, device):
    qw = random_quantized_weights(K, N, G, scheme, device=device)
    impl.prepare(qw.q, qw.scales, qw.zeros, G)
    w_canonical = qw.dequantize()
    dtype = compute_dtype(impl)
    for M in batch_sizes("early"):
        x = random_activations(M, K, device=device)
        out = impl.run(prepare_input(impl, x))
        assert out.shape == (M, N) and out.dtype == dtype
        result = check(out, x, w_canonical, dtype)
        assert result.passed, f"M={M}: impl {result.impl} vs cuBLAS {dtype} {result.baseline}"


@BASELINES
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("scheme", QUANT_SCHEMES)
def test_baseline_correct_small(impl, scheme, device):
    G = group_sizes("early")[0]
    _check_baseline(impl, 4 * G, 96, G, scheme, device)


@gpu_only
@BASELINES
@pytest.mark.parametrize("scheme", QUANT_SCHEMES)
@pytest.mark.parametrize("layer", list(load_config()["layers"]))
def test_baseline_correct_real_shapes(impl, layer, scheme):
    shape = load_config()["layers"][layer]
    _check_baseline(impl, shape["K"], shape["N"], group_sizes("early")[0], scheme, "cuda")


def test_baselines_declare_dtypes_and_weight_bits():
    assert cublas.WEIGHT_BITS == 16 and cublas.COMPUTE_DTYPE == torch.float16
    assert cublas_bf16.WEIGHT_BITS == 16 and cublas_bf16.COMPUTE_DTYPE == torch.bfloat16


@BASELINES
def test_reference_weights_are_what_run_uses(impl):
    G = group_sizes("early")[0]
    qw = random_quantized_weights(2 * G, 16, G, QUANT_SCHEMES[0])
    impl.prepare(qw.q, qw.scales, qw.zeros, G)
    w = impl.reference_weights()
    assert w.dtype == compute_dtype(impl) and w.shape == (2 * G, 16)
    x_in = prepare_input(impl, random_activations(4, 2 * G))
    assert x_in.dtype == compute_dtype(impl)
    assert torch.equal(impl.run(x_in), x_in @ w)
