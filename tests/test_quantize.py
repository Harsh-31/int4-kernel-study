"""Round-trip tests for common/quantize.py, for both quantization schemes.

Unit tests use small matrices so they run on a laptop CPU. The group sizes come
from config/workloads.yaml. A separate GPU-only test runs the real Llama-3-8B
layer shapes from the config.

Tests pass the scheme explicitly, so they do not depend on which scheme the
config currently selects.
"""

import pytest
import torch

import common.config
from common.config import QUANT_SCHEMES, group_sizes, load_config
from common.quantize import (
    INT4_MAX,
    SYMMETRIC_ZERO,
    dequantize,
    quantize,
    random_quantized_weights,
    random_weights,
)

# Worst-case |w - w_hat| in units of the group's scale s. Rounding alone gives
# 0.5; FP16 rounding of s can add a tiny clamp error at the ends of the range
# (see the module docstring in common/quantize.py), so allow a hair more.
MAX_ERR_IN_STEPS = 0.51

schemes = pytest.mark.parametrize("scheme", QUANT_SCHEMES)
groups = pytest.mark.parametrize("group_size", group_sizes("full"))


def small_shape() -> tuple[int, int]:
    """A few groups along K and a modest N: enough to exercise every code path."""
    return 4 * max(group_sizes("full")), 96


def per_element_scale(qw) -> torch.Tensor:
    return qw.scales.to(torch.float32).repeat_interleave(qw.group_size, dim=0)


# ---------------------------------------------------------------- both schemes

@schemes
@groups
def test_shapes_and_dtypes(scheme, group_size):
    K, N = small_shape()
    qw = random_quantized_weights(K, N, group_size, scheme)
    assert qw.scheme == scheme
    assert qw.w.shape == (K, N) and qw.w.dtype == torch.float16
    assert qw.q.shape == (K, N) and qw.q.dtype == torch.uint8
    assert qw.scales.shape == (K // group_size, N) and qw.scales.dtype == torch.float16
    assert qw.zeros.shape == (K // group_size, N) and qw.zeros.dtype == torch.uint8
    assert int(qw.q.max()) <= INT4_MAX and int(qw.zeros.max()) <= INT4_MAX


@schemes
@groups
def test_round_trip_within_half_step(scheme, group_size):
    """Every weight comes back within ~half a quantization step of the original."""
    K, N = small_shape()
    qw = random_quantized_weights(K, N, group_size, scheme)
    err = (qw.w.to(torch.float32) - qw.dequantize()).abs()
    assert torch.all(err <= MAX_ERR_IN_STEPS * per_element_scale(qw))


@schemes
@groups
def test_round_trip_relative_error_is_small(scheme, group_size):
    """Sanity check on the overall error: 4-bit quantization of Gaussian weights
    gives a relative L2 error of roughly 0.1; anything far above that means the
    math is wrong."""
    K, N = small_shape()
    qw = random_quantized_weights(K, N, group_size, scheme)
    w = qw.w.to(torch.float32)
    rel = torch.linalg.norm(w - qw.dequantize()) / torch.linalg.norm(w)
    assert rel < 0.15


@schemes
@groups
def test_dequantize_matches_formula(scheme, group_size):
    """dequantize() is exactly (q - z) * s, computed element by element."""
    K, N = small_shape()
    qw = random_quantized_weights(K, N, group_size, scheme)
    w_hat = qw.dequantize()
    for k in (0, group_size - 1, group_size, K - 1):   # group edges
        g = k // group_size
        expected = (qw.q[k].float() - qw.zeros[g].float()) * qw.scales[g].float()
        assert torch.equal(w_hat[k], expected)


@schemes
def test_zero_is_exact_and_constant_groups_are_handled(scheme):
    """Zeros quantize to exactly 0; an all-zero group and a same-sign group work."""
    G = group_sizes("early")[0]
    w = torch.randn(2 * G, 8).to(torch.float16)
    w[:G, 0] = 0.0                  # all-zero group
    w[G:, 1] = w[G:, 1].abs() + 1   # all-positive group
    w[5, 2] = 0.0                   # a single zero inside a mixed group
    qw = quantize(w, G, scheme)
    w_hat = qw.dequantize()
    assert torch.all(w_hat[:G, 0] == 0.0)
    assert w_hat[5, 2] == 0.0
    err = (w.to(torch.float32) - w_hat).abs()
    assert torch.all(err <= MAX_ERR_IN_STEPS * per_element_scale(qw))


@schemes
def test_standalone_dequantize_matches_method(scheme):
    G = group_sizes("early")[0]
    qw = random_quantized_weights(2 * G, 16, G, scheme)
    assert torch.equal(qw.dequantize(), dequantize(qw.q, qw.scales, qw.zeros, G))


@schemes
def test_bad_group_size_raises(scheme):
    G = group_sizes("early")[0]
    with pytest.raises(ValueError):
        quantize(torch.zeros(G + 1, 4, dtype=torch.float16), G, scheme)


# ------------------------------------------------------------ symmetric only

@groups
def test_symmetric_zero_point_is_fixed(group_size):
    K, N = small_shape()
    qw = random_quantized_weights(K, N, group_size, "symmetric")
    assert torch.all(qw.zeros == SYMMETRIC_ZERO)


@groups
def test_symmetric_scale_is_max_abs_over_7_5(group_size):
    """s = max|w| / 7.5 per group (up to the FP16 rounding of s)."""
    K, N = small_shape()
    qw = random_quantized_weights(K, N, group_size, "symmetric")
    amax = qw.w.float().reshape(K // group_size, group_size, N).abs().amax(dim=1)
    expected = (amax / 7.5).to(torch.float16)
    assert torch.equal(qw.scales, expected)


def test_symmetric_grid_ends():
    """The largest negative weight maps to level 0 (-8s), the largest positive
    one clamps to level 15 (+7s): the grid reaches -8s but only +7s."""
    G = group_sizes("early")[0]
    w = torch.zeros(G, 2, dtype=torch.float16)
    w[0, 0], w[1, 0] = -3.0, 1.0    # negative end is the max magnitude
    w[0, 1], w[1, 1] = 3.0, -1.0    # positive end is the max magnitude
    qw = quantize(w, G, "symmetric")
    assert qw.q[0, 0] == 0
    assert qw.q[0, 1] == INT4_MAX


def test_symmetric_and_asymmetric_differ():
    G = group_sizes("early")[0]
    a = random_quantized_weights(2 * G, 16, G, "asymmetric")
    s = random_quantized_weights(2 * G, 16, G, "symmetric")
    assert torch.equal(a.w, s.w)                # same seed, same weights
    assert not torch.equal(a.scales, s.scales)


# ------------------------------------------------------------ config / seed

def test_seed_is_reproducible():
    G = group_sizes("early")[0]
    a = random_weights(2 * G, 16)
    b = random_weights(2 * G, 16)
    c = random_weights(2 * G, 16, seed=load_config()["seed"] + 1)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_unknown_scheme_raises():
    G = group_sizes("early")[0]
    with pytest.raises(ValueError):
        quantize(torch.zeros(G, 4, dtype=torch.float16), G, "nf4")


def _config_with_scheme(monkeypatch, scheme):
    cfg = dict(load_config(), quant_scheme=scheme)
    monkeypatch.setattr(common.config, "load_config", lambda *a, **k: cfg)


def test_default_scheme_comes_from_config(monkeypatch):
    G = group_sizes("early")[0]
    for scheme in QUANT_SCHEMES:
        _config_with_scheme(monkeypatch, scheme)
        assert random_quantized_weights(G, 4, G).scheme == scheme


def test_unset_config_scheme_raises(monkeypatch):
    G = group_sizes("early")[0]
    _config_with_scheme(monkeypatch, None)
    with pytest.raises(ValueError, match="quant_scheme is not set"):
        random_quantized_weights(G, 4, G)


# ------------------------------------------------------------ real shapes (GPU)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (run via Slurm)")
@schemes
@pytest.mark.parametrize("layer", list(load_config()["layers"]))
def test_round_trip_real_layer_shapes(scheme, layer):
    shape = load_config()["layers"][layer]
    G = group_sizes("early")[0]
    qw = random_quantized_weights(shape["K"], shape["N"], G, scheme, device="cuda")
    err = (qw.w.to(torch.float32) - qw.dequantize()).abs()
    assert torch.all(err <= MAX_ERR_IN_STEPS * per_element_scale(qw))
