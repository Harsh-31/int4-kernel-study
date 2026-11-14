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

Which scheme is used is set by `quant_scheme` in config/workloads.yaml
(project default: symmetric).

Scales are exactly representable in BOTH FP16 and BF16
-------------------------------------------------------
Every implementation must receive identical weights and scales. FP16 kernels
(cuBLAS FP16, Marlin) read FP16 scales; BF16 kernels (TorchAO's tinygemm) read
BF16 scales. A scale is stored losslessly in both only if it has at most 8
significant bits (BF16's precision) and lies in FP16's normal range
[2^-14, 65504]. So after computing s = range / 15 in FP32 we ROUND IT UP to the
next BF16 value (round_scale_to_bf16_and_fp16) and store it as FP16; converting
that FP16 value to BF16 is then exact. Rounding up (not to nearest) keeps every
weight inside the representable grid, so the half-step error bound below still
holds; the price is a step up to 2^-7 (0.8%) larger than the minimum.

Asymmetric scheme (scale + zero point, per group)
-------------------------------------------------
For one group of weights g = {w_1, ..., w_G}:

  1. Range. lo = min(min(g), 0) and hi = max(max(g), 0).
     Including 0 makes sure 0.0 is exactly representable, and guarantees the
     zero point below lands in [0, 15] even if every weight in the group has
     the same sign.

  2. Scale. The 16 INT4 levels 0..15 span 15 steps, so
         s = (hi - lo) / 15, rounded UP to a BF16-and-FP16-exact value
     is the step size between neighbouring levels. All later steps use this
     stored s, so the stored (q, s, z) triple is exactly what dequantization
     sees.

  3. Zero point. z is the integer level that represents 0.0:
         z = round(-lo / s),  clamped to [0, 15].
     Because lo <= 0 <= hi and s >= (hi - lo) / 15, -lo / s lies in [0, 15].

  4. Quantize each weight:
         q = clamp(round(w / s + z), 0, 15)

  5. Dequantize:
         w_hat = (q - z) * s

Error bound: rounding moves each weight by at most half a step, so
|w - w_hat| <= s / 2. Because s was rounded UP, (hi - lo) / s <= 15, so
hi / s + z <= 15.5 and clamping at 15 costs at most that same half step.
The tests check this bound.

Symmetric scheme (scale only, zero point fixed at 8)
----------------------------------------------------
The grid is centred on 0, so only a scale is stored. This is the GPTQ-style
"sym" convention used by e.g. Marlin.

  1. Range. a = max(|g|), the largest magnitude in the group.

  2. Scale. Levels 0..15 minus the fixed zero 8 give signed values -8..7.
     We map [-a, a] onto [-7.5, 7.5] steps, which uses all 16 levels:
         s = a / 7.5 = 2a / 15        (rounded up to BF16/FP16-exact, as above)

  3. Quantize:
         q = clamp(round(w / s) + 8, 0, 15)
     w = -a gives round(-7.5) + 8 = 0. w = +a gives round(7.5) + 8 = 16,
     which clamps to 15 (value 7s, so the error is 0.5s). The grid is
     therefore slightly lopsided: it reaches -8s but only +7s.

  4. Dequantize (z = 8 for every group):
         w_hat = (q - 8) * s

Error bound: as above, |w - w_hat| <= s / 2. With s rounded up, a / s <= 7.5,
so even the clamped top end (+a -> level 15 = +7s) is within half a step.
Compared with the asymmetric scheme, the step is larger whenever the group's
weights are not balanced around 0 (s depends on max |w|, not max - min), so
the average error is a little higher.
"""

from dataclasses import dataclass

import torch

from common.config import QUANT_SCHEMES, load_config, quant_scheme

INT4_MAX = 15        # unsigned 4-bit levels are 0..15
SYMMETRIC_ZERO = 8   # fixed zero point in symmetric mode (middle of 0..15)

FP16_MIN_NORMAL = 2.0 ** -14   # below this FP16 is subnormal and loses BF16 values
BF16_MAX_IN_FP16 = 65280.0     # largest BF16 value <= FP16 max (65504)


def round_scale_to_bf16_and_fp16(s: torch.Tensor) -> torch.Tensor:
    """Round positive FP32 scales UP to values exact in both BF16 and FP16.

    BF16 has 8 significant bits, FP16 has 11, so any BF16 value inside FP16's
    normal range is also an FP16 value. We (1) clamp to that range, (2) round
    to the nearest BF16, (3) step one BF16 ulp up wherever that rounded down,
    and return FP16. An all-zero group (s = 0) gets the smallest normal scale;
    its weights quantize to z and dequantize to exactly 0.
    """
    s = s.to(torch.float32).clamp(min=FP16_MIN_NORMAL)
    if bool((s > BF16_MAX_IN_FP16).any()):
        raise ValueError("a group's range is too large for an FP16 scale")
    b = s.to(torch.bfloat16)
    # For positive floats, the next representable value up is bit pattern + 1.
    bits = b.view(torch.int16) + (b.to(torch.float32) < s).to(torch.int16)
    out = bits.view(torch.bfloat16).to(torch.float16)
    assert torch.equal(out.to(torch.bfloat16).to(torch.float16), out)   # exact in both
    return out


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

    # Step 2: scale = range / 15 in both schemes, rounded UP to a value exact
    # in both BF16 and FP16 (see module docstring), stored as FP16.
    scales = round_scale_to_bf16_and_fp16(step_range / INT4_MAX)
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
