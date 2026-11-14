"""Marlin INT4 x FP16 GEMM, through vLLM's ops (checked against vLLM 0.21.0 source).

What vLLM's Marlin accepts (vllm/model_executor/layers/quantization/utils/marlin_utils.py,
query_marlin_supported_quant_types):

  * scalar_types.uint4b8  -- GPTQ-style SYMMETRIC, no zero-point tensor. The type
    has a built-in bias of 8 ("stored_value = value + 8", vllm/scalar_type.py),
    so the kernel computes  w = (q - 8) * s.  This is exactly our "symmetric"
    scheme: q is passed through unchanged.
  * scalar_types.uint4    -- AWQ-style ASYMMETRIC, with integer zero points:
    w = (q - zp) * s  (quantize_weights(..., zero_points=True) in quant_utils.py).
    This is exactly our "asymmetric" scheme.

  Group sizes: -1 (per-channel), 32, 64, 128. Shapes: N % 64 == 0, K % 128 == 0.
  GPU: compute capability >= 7.5 (A40 = 8.6, A100 = 8.0).

prepare() picks the type from the zero points it receives: all zeros == 8 means
the symmetric scheme (uint4b8, no zero-point tensor to read); anything else
uses uint4 + zero points. Both give w = (q - z) * s, so the choice never changes
the math, only whether the kernel reads a zero-point tensor.

The steps follow vLLM's production loading path
(vllm/model_executor/kernels/linear/mixed_precision/marlin.py,
MarlinLinearKernel.process_weights_after_loading) and its matmul call
(apply_gptq_marlin_linear in marlin_utils.py).
"""

import numpy
import torch
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    USE_FP32_REDUCE_DEFAULT,
    get_scale_perms,
    marlin_make_empty_g_idx,
    marlin_make_workspace_new,
    marlin_permute_scales,
    marlin_zero_points,
    verify_marlin_supported,
    verify_marlin_supports_shape,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import get_weight_perm
from vllm.model_executor.layers.quantization.utils.quant_utils import gptq_pack, unpack_cols
from vllm.scalar_type import scalar_types

from common.quantize import SYMMETRIC_ZERO, dequantize

NUM_BITS = 4
MARLIN_TILE = 16
COMPUTE_DTYPE = torch.float16   # FP16 activations/output, FP32 accumulation
# Column interleave marlin_zero_points() applies to 4-bit zero points
# (hard-coded inside that function in marlin_utils.py; needed to invert it).
ZP_INTERLEAVE_4BIT = numpy.array([0, 2, 4, 6, 1, 3, 5, 7])

_state: dict = {}


# ------------------------------------------------------------------ prepare

def prepare(weights: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor,
            group_size: int) -> None:
    """Repack our canonical INT4 weights into Marlin's layout (not timed).

    weights: [K, N] uint8 in 0..15 (one value per byte), scales: [K/G, N] FP16,
    zeros: [K/G, N] uint8. All on the GPU.
    """
    _state.clear()   # release the previous layer's buffers first
    K, N = weights.shape
    device = weights.device
    symmetric = bool(torch.all(zeros == SYMMETRIC_ZERO))
    wtype = scalar_types.uint4b8 if symmetric else scalar_types.uint4

    # vLLM's own capability checks; raise NotImplementedError so the
    # benchmark records a skip instead of crashing.
    try:
        verify_marlin_supported(wtype, group_size, has_zp=not symmetric)
        verify_marlin_supports_shape(N, K, K, group_size)
    except ValueError as e:
        raise NotImplementedError(f"Marlin: {e}") from e

    # 1. GPTQ packing: 8 consecutive K-rows of 4-bit values into one int32,
    #    giving [K/8, N]. This is the on-disk format of GPTQ checkpoints and the
    #    input format gptq_marlin_repack expects. (gptq_pack runs on the CPU in
    #    NumPy; fine here because prepare() is not timed.)
    q_gptq = gptq_pack(weights.to(torch.int32), NUM_BITS, K, N).to(device)

    # 2. Marlin repack (GPU op). Rearranges the weights into 16x64 tiles in the
    #    order the kernel's tensor-core MMA instructions consume them, so each
    #    warp loads its fragment with wide, contiguous reads. Output [K/16, N*16/8]
    #    int32. perm is empty because we use no activation reordering (act_order).
    empty = marlin_make_empty_g_idx(device)
    q_marlin = ops.gptq_marlin_repack(q_gptq.contiguous(), perm=empty,
                                      size_k=K, size_n=N, num_bits=NUM_BITS)

    # 3. Scales permuted to match the tile layout (same shape [K/G, N]).
    s_marlin = marlin_permute_scales(scales.contiguous(), size_k=K, size_n=N,
                                     group_size=group_size)

    # 4. Zero points: none for uint4b8 (the bias of 8 is part of the type);
    #    otherwise permuted like the scales, interleaved, and packed 8 per int32
    #    along N, giving [K/G, N/8].
    num_groups = K // group_size
    if symmetric:
        zp_marlin = empty
    else:
        zp_marlin = marlin_zero_points(zeros.to(torch.int32), size_k=num_groups,
                                       size_n=N, num_bits=NUM_BITS).to(device)

    _state.update(
        K=K, N=N, group_size=group_size, wtype=wtype, symmetric=symmetric,
        q=q_marlin, s=s_marlin, zp=zp_marlin,
        g_idx=empty, g_idx_sort=empty,          # no act_order
        # Scratch space for the kernel's cross-threadblock reduction locks;
        # one int per threadblock, allocated once (as vLLM does per layer).
        workspace=marlin_make_workspace_new(device),
    )


# ---------------------------------------------------------------------- run

def run(x: torch.Tensor) -> torch.Tensor:
    """y [M, N] FP16 = x [M, K] FP16 @ dequant(W). The only timed call.

    Same marlin_gemm call and defaults as vLLM's apply_gptq_marlin_linear:
    no bias, no activation quantization, is_k_full (no act_order and no tensor
    parallelism), atomic-add reduction off (vLLM's default unless
    VLLM_MARLIN_USE_ATOMIC_ADD=1), FP32 reduction on. We call the op directly
    to skip that wrapper's per-call Python checks.
    """
    st = _state
    return ops.marlin_gemm(
        x, None, st["q"], None, st["s"], None, None, st["zp"],
        st["g_idx"], st["g_idx_sort"], st["workspace"], st["wtype"],
        size_m=x.shape[0], size_n=st["N"], size_k=st["K"],
        is_k_full=True, use_atomic_add=False,
        use_fp32_reduce=USE_FP32_REDUCE_DEFAULT, is_zp_float=False,
    )


# ------------------------------------------------- reference (from Marlin data)

def reference_weights() -> torch.Tensor:
    """[K, N] FP32 weights decoded from Marlin's OWN packed buffers.

    Undoes every layout step of prepare() on the tensors the kernel actually
    reads (q, s, zp) and then applies the type's formula w = (q - z) * s.
    A mistake in prepare() therefore shows up as a correctness failure.
    """
    st = _state
    K, N, G = st["K"], st["N"], st["group_size"]
    q = unpack_marlin_weights(st["q"], K, N)
    s = unpermute_marlin_scales(st["s"], K, N, G)
    if st["symmetric"]:
        z = torch.full_like(s, SYMMETRIC_ZERO, dtype=torch.uint8)
    else:
        z = unpack_marlin_zero_points(st["zp"], K // G, N)
    return dequantize(q, s, z, G)


def unpack_marlin_weights(q_marlin: torch.Tensor, K: int, N: int) -> torch.Tensor:
    """Inverse of gptq_marlin_repack: [K/16, N*16/8] int32 -> [K, N] uint8.

    vLLM's test_gptq_marlin_repack asserts that the repack equals the Python
    reference marlin_weights(q, get_weight_perm(4)) in marlin_utils_test.py,
    which is: (a) tile permute, (b) column permutation `perm` in chunks of
    len(perm), (c) pack 8 neighbouring columns per int32. Undo in reverse.
    """
    t = MARLIN_TILE
    # (c) unpack: int32 column j holds columns 8j..8j+7, value i at bits 4i.
    p = q_marlin.to(torch.int64) & 0xFFFFFFFF          # reinterpret as unsigned
    vals = [(p >> (NUM_BITS * i)) & 0xF for i in range(8)]
    q = torch.stack(vals, dim=-1).reshape(K // t, N * t)
    # (b) undo the column permutation: forward did q2[:, c] = q1[:, perm[c]].
    perm = get_weight_perm(NUM_BITS).to(q.device)
    q = q.reshape(-1, perm.numel())[:, torch.argsort(perm)].reshape(K // t, N * t)
    # (a) undo the 16x16 tile permute.
    q = q.reshape(K // t, N // t, t, t).permute(0, 2, 1, 3).reshape(K, N)
    return q.to(torch.uint8)


def unpermute_marlin_scales(s_marlin: torch.Tensor, K: int, N: int, G: int) -> torch.Tensor:
    """Inverse of marlin_permute_scales (same branch condition as vLLM)."""
    scale_perm, scale_perm_single = get_scale_perms()
    perm = scale_perm if (G < K and G != -1) else scale_perm_single
    inv = torch.from_numpy(numpy.argsort(perm)).to(s_marlin.device)
    return s_marlin.reshape(-1, len(perm))[:, inv].reshape(-1, N)


def unpack_marlin_zero_points(zp_marlin: torch.Tensor, num_groups: int, N: int) -> torch.Tensor:
    """Inverse of marlin_zero_points: [K/G, N/8] int32 -> [K/G, N] uint8.

    Forward: permute like scales -> interleave columns in groups of 8 ->
    pack 8 per int32 along N (pack_cols). Undo in reverse order.
    """
    scale_perm, _ = get_scale_perms()
    zp = unpack_cols(zp_marlin, NUM_BITS, num_groups, N)
    inv_interleave = torch.from_numpy(numpy.argsort(ZP_INTERLEAVE_4BIT)).to(zp.device)
    inv_scale_perm = torch.from_numpy(numpy.argsort(scale_perm)).to(zp.device)
    zp = zp.reshape(-1, len(ZP_INTERLEAVE_4BIT))[:, inv_interleave]
    zp = zp.reshape(-1, len(scale_perm))[:, inv_scale_perm]
    return zp.reshape(num_groups, N).to(device=zp_marlin.device, dtype=torch.uint8)
