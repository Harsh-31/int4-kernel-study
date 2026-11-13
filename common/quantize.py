"""Random FP16 weights and their INT4 (W4A16) group-wise quantization.

Layout conventions (shared by every implementation and test)
-----------------------------------------------------------
* The weight matrix W has shape [K, N]: K input features, N output features,
  so a layer computes  y[M, N] = x[M, K] @ W[K, N].
* Groups run along K: each column n is split into K / group_size groups of
  consecutive rows, and every (group, column) pair has its own scale and zero.
  So scales and zeros have shape [K / group_size, N].
* q is returned UNPACKED: one INT4 value (0..15) per uint8 element, shape [K, N].
  Packing two (or eight) values per byte/word is layout-specific, so each
  implementation does it in its own `prepare()`.
* Both schemes return a zeros tensor (all 8s in symmetric mode), so
  dequantization is always  w_hat = (q - z) * s  and the reference code does
  not need to know which scheme was used.

Which scheme is used is set by `quant_scheme` in config/workloads.yaml.

Asymmetric scheme (scale + zero point, per group)
-------------------------------------------------
For one group of weights g = {w_1, ..., w_G}:

  1. Range. lo = min(min(g), 0) and hi = max(max(g), 0).
     Including 0 makes sure 0.0 is exactly representable, and guarantees the
     zero point below lands in [0, 15] even if every weight in the group has
     the same sign.

  2. Scale. The 16 INT4 levels 0..15 span 15 steps, so
         s = (hi - lo) / 15
     is the step size between neighbouring levels. s is stored in FP16 (that is
     what kernels read), and all later steps use the FP16-rounded s so that the
     stored (q, s, z) triple is exactly what dequantization sees.

  3. Zero point. z is the integer level that represents 0.0:
         z = round(-lo / s),  clamped to [0, 15].
     Because lo <= 0 <= hi, -lo / s lies in [0, 15] (up to FP16 rounding of s).

  4. Quantize each weight:
         q = clamp(round(w / s + z), 0, 15)

  5. Dequantize:
         w_hat = (q - z) * s

Error bound: without clamping, rounding moves each weight by at most half a
step, so |w - w_hat| <= s / 2. Clamping at 15 can only bite by a hair, because
the FP16 rounding of s (relative error <= 2^-11) can push hi / s + z just past
15.5. The worst case is then about 0.5073 * s. The tests check this bound.

Symmetric scheme (scale only, zero point fixed at 8)
----------------------------------------------------
The grid is centred on 0, so only a scale is stored. This is the GPTQ-style
"sym" convention used by e.g. Marlin.

  1. Range. a = max(|g|), the largest magnitude in the group.

  2. Scale. Levels 0..15 minus the fixed zero 8 give signed values -8..7.
     We map [-a, a] onto [-7.5, 7.5] steps, which uses all 16 levels:
         s = a / 7.5 = 2a / 15        (stored in FP16, as above)

  3. Quantize:
         q = clamp(round(w / s) + 8, 0, 15)
     w = -a gives round(-7.5) + 8 = 0. w = +a gives round(7.5) + 8 = 16,
     which clamps to 15 (value 7s, so the error is 0.5s). The grid is
     therefore slightly lopsided: it reaches -8s but only +7s.

  4. Dequantize (z = 8 for every group):
         w_hat = (q - 8) * s

Error bound: as above, |w - w_hat| <= s / 2 from rounding. FP16 rounding of s
can push a / s to about 7.504, giving at most ~0.504 * s at either end.
Compared with the asymmetric scheme, the step is larger whenever the group's
weights are not balanced around 0 (s depends on max |w|, not max - min), so
the average error is a little higher.
"""

from dataclasses import dataclass

import torch

from common.config import QUANT_SCHEMES, load_config, quant_scheme

INT4_MAX = 15        # unsigned 4-bit levels are 0..15
SYMMETRIC_ZERO = 8   # fixed zero point in symmetric mode (middle of 0..15)


def random_weights(K: int, N: int, seed: int | None = None, device="cpu") -> torch.Tensor:
    """Random FP16 weights of shape [K, N], N(0, 1), reproducible from `seed`.

    The numbers are drawn on the CPU with a dedicated generator so the same seed
    gives the same weights on any machine or GPU, and then moved to `device`.
    `seed` defaults to the seed in config/workloads.yaml.
    """
    if seed is None:
        seed = load_config()["seed"]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn(K, N, generator=gen, dtype=torch.float32)
    return w.to(device=device, dtype=torch.float16)


def random_activations(M: int, K: int, seed: int | None = None, device="cpu") -> torch.Tensor:
    """Random FP16 activations x of shape [M, K], N(0, 1), reproducible from `seed`."""
    if seed is None:
        seed = load_config()["seed"]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(M, K, generator=gen).to(device=device, dtype=torch.float16)


def dequantize(q: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor,
               group_size: int, dtype=torch.float32) -> torch.Tensor:
    """w_hat = (q - z) * s, with s and z broadcast over each group of rows.

    q: [K, N] uint8 in 0..15, scales: [K/G, N] FP16, zeros: [K/G, N] uint8.
    Works for both schemes (symmetric zeros are all 8).
    Returns [K, N] in `dtype` (FP32 by default, as the reference needs).
    """
    # repeat_interleave expands each group row G times, giving [K, N] tensors
    # that line up element-by-element with q.
    s = scales.to(torch.float32).repeat_interleave(group_size, dim=0)
    z = zeros.to(torch.float32).repeat_interleave(group_size, dim=0)
    return ((q.to(torch.float32) - z) * s).to(dtype)


@dataclass
class QuantizedWeights:
    w: torch.Tensor        # [K, N] FP16, original weights (before quantization)
    q: torch.Tensor        # [K, N] uint8, INT4 values 0..15, unpacked
    scales: torch.Tensor   # [K/G, N] FP16
    zeros: torch.Tensor    # [K/G, N] uint8, integer zero points 0..15 (all 8 if symmetric)
    group_size: int
    scheme: str            # "asymmetric" or "symmetric"

    def dequantize(self, dtype=torch.float32) -> torch.Tensor:
        return dequantize(self.q, self.scales, self.zeros, self.group_size, dtype)


def quantize(w: torch.Tensor, group_size: int, scheme: str | None = None) -> QuantizedWeights:
    """Quantize FP16 weights [K, N] to INT4 with per-group scales (and zeros).

    `scheme` is "asymmetric" or "symmetric"; it defaults to `quant_scheme` in
    config/workloads.yaml (and raises if that has not been set yet).
    """
    if scheme is None:
        scheme = quant_scheme()
    if scheme not in QUANT_SCHEMES:
        raise ValueError(f"scheme must be one of {QUANT_SCHEMES}, got '{scheme}'")

    K, N = w.shape
    if K % group_size != 0:
        raise ValueError(f"K={K} is not divisible by group_size={group_size}")
    n_groups = K // group_size

    # View as [n_groups, G, N] so each group is a slice along dim 1.
    # Do the math in FP32 to avoid FP16 overflow/rounding in intermediates.
    wg = w.to(torch.float32).reshape(n_groups, group_size, N)

    if scheme == "asymmetric":
        # Step 1: per-group range, widened to include 0.
        lo = wg.amin(dim=1).clamp(max=0.0)   # [n_groups, N]
        hi = wg.amax(dim=1).clamp(min=0.0)
        step_range = hi - lo
    else:
        # Step 1: largest magnitude; the range is [-a, a] = 2a wide.
        step_range = 2.0 * wg.abs().amax(dim=1)

    # Step 2: scale = range / 15 in both schemes, rounded to FP16 because that
    # is what gets stored. An all-zero group would give s = 0; use a tiny scale
    # so we never divide by zero (its weights quantize to z and dequantize to
    # exactly 0).
    scales = (step_range / INT4_MAX).clamp(min=1e-8).to(torch.float16)
    s = scales.to(torch.float32)

    # Step 3: zero point (the level that represents 0.0).
    if scheme == "asymmetric":
        z = torch.round(-lo / s).clamp(0, INT4_MAX)
    else:
        z = torch.full_like(s, SYMMETRIC_ZERO)

    # Step 4: quantize. s and z are [n_groups, N]; unsqueeze to broadcast over G.
    # For symmetric, z is the integer 8, so round(w/s + 8) == round(w/s) + 8.
    q = torch.round(wg / s.unsqueeze(1) + z.unsqueeze(1)).clamp(0, INT4_MAX)

    return QuantizedWeights(
        w=w,
        q=q.reshape(K, N).to(torch.uint8),
        scales=scales,
        zeros=z.to(torch.uint8),
        group_size=group_size,
        scheme=scheme,
    )


def random_quantized_weights(K: int, N: int, group_size: int, scheme: str | None = None,
                             seed: int | None = None, device="cpu") -> QuantizedWeights:
    """Convenience: random FP16 weights (fixed seed) quantized to INT4."""
    return quantize(random_weights(K, N, seed=seed, device=device), group_size, scheme)
