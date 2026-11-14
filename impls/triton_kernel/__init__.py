"""Our INT4 x FP16 matmul in Triton -- version 1: simple and readable, NOT fast.

Scope of this version
---------------------
  * Symmetric scheme only (w = (q - 8) * s); prepare() refuses zero points
    other than 8.
  * Group sizes 32, 64, 128 (the goal is 128).
  * FP16 activations and output, FP32 accumulation  -> COMPUTE_DTYPE = float16.
  * No split-K, no autotuning, fixed block sizes. Those come later.

Weight format (built in prepare)
--------------------------------
Our canonical q is [K, N], one 4-bit value per uint8. We pack PAIRS OF
NEIGHBOURING K-ROWS of the same column into one byte:

    packed[i, n] = q[2i, n]  |  (q[2i+1, n] << 4)        packed: [K/2, N] uint8
                   low nibble   high nibble

Why this format, for a first kernel:
  1. Same [K, N] orientation as our canonical layout, so no transposes and the
     indexing in the kernel reads like the math (row = k, column = n).
  2. N is the fastest-moving (contiguous) dimension. A program that handles
     BLOCK_N neighbouring output columns loads BLOCK_N neighbouring bytes from
     each packed row: one contiguous, coalesced memory access per row.
  3. The two values in a byte share the same column n and (because the group
     size is even) the same group, so they use the same scale.
  4. Unpacking is one AND (& 0xF) for the low value and one SHIFT (>> 4) for the
     high value; uint8 has no sign bit to worry about.
  The cost: byte-sized loads and no layout tuned for tensor cores. Later
  versions can pack 8 values per int32 and reorder for wider loads.

Scales stay as they are: FP16 [K/G, N].

Why the dequantized weights are EXACT in FP16
---------------------------------------------
common/quantize.py makes every scale exactly representable in BF16, i.e. it
has at most 8 significant bits. (q - 8) is an integer in -8..7 (at most 3
significant bits). Their product needs at most 8 + 3 = 11 bits, which is
exactly FP16's precision, so (q - 8) * s computed in FP16 has no rounding error.
The only rounding in this kernel is the final FP32 -> FP16 store of the output.
"""

import torch
import triton
import triton.language as tl

from common.quantize import SYMMETRIC_ZERO, dequantize

COMPUTE_DTYPE = torch.float16

# Fixed tile sizes for version 1 (no autotuning yet).
BLOCK_N = 64        # output columns per program
MAX_BLOCK_K = 64    # K values consumed per loop iteration (capped by the group size)
MIN_BLOCK = 16      # tl.dot needs every tile dimension >= 16
MAX_BLOCK_M = 64    # rows of x per program; larger M uses more programs

_state: dict = {}


# --------------------------------------------------------------------- kernel

# @triton.jit compiles this Python function into a GPU kernel the first time it
# is called with a new combination of `tl.constexpr` values and dtypes. The
# body is NOT run as normal Python: each name is a block of values that the
# whole program (a group of GPU threads) works on together.
@triton.jit
def int4_matmul_kernel(
    # Pointers to the first element of each tensor (Triton receives tensors as
    # raw GPU addresses plus the strides below).
    x_ptr,          # activations  [M, K]   fp16
    wq_ptr,         # packed q     [K/2, N] uint8
    s_ptr,          # scales       [K/G, N] fp16
    out_ptr,        # output       [M, N]   fp16
    # Sizes (ordinary runtime integers).
    M, N, K,
    # Strides: how many ELEMENTS to step to move by one row / one column.
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_sg, stride_sn,
    stride_om, stride_on,
    # tl.constexpr values are known at compile time, so Triton can use them for
    # tile shapes and unroll loops. A new value triggers a new compilation.
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # ---- Which output tile does this program compute? ----------------------
    # The launch grid is 2-D: axis 0 walks over column blocks, axis 1 over row
    # blocks. program_id tells this program its position in that grid.
    pid_n = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    # tl.arange(0, B) is the vector [0, 1, ..., B-1]. Adding the block start
    # gives the actual row / column indices this program is responsible for.
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)          # [BLOCK_M] rows of x / out
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)          # [BLOCK_N] columns of out

    # One loop iteration covers BLOCK_K values of k, i.e. BLOCK_K / 2 packed rows.
    offs_kp = tl.arange(0, BLOCK_K // 2)                      # [BLOCK_K/2] packed-row offsets

    # Masks switch off loads/stores that would fall outside the matrices
    # (e.g. M = 1 with BLOCK_M = 16: rows 1..15 do not exist).
    mask_m = offs_m < M
    mask_n = offs_n < N

    # The accumulator for this output tile, kept in FP32 so that summing
    # thousands of products does not lose precision.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ---- Walk along K, one BLOCK_K-wide slice at a time -------------------
    # prepare() guarantees K % BLOCK_K == 0, so no K mask is needed.
    for k0 in range(0, K, BLOCK_K):
        # Each packed byte holds an EVEN k (low nibble) and the next ODD k (high
        # nibble). Instead of interleaving the two halves back into one tile,
        # we load the matching even and odd COLUMNS of x separately and do two
        # smaller matrix products. Mathematically:
        #   x[:, k-slice] @ W[k-slice, :] = x_even @ W_even + x_odd @ W_odd
        k_even = k0 + 2 * offs_kp                             # [BLOCK_K/2] = k0, k0+2, ...
        k_odd = k_even + 1                                    # [BLOCK_K/2] = k0+1, k0+3, ...

        # Address arithmetic: offs_m[:, None] is a column vector and
        # k_even[None, :] a row vector, so their combination broadcasts to a
        # [BLOCK_M, BLOCK_K/2] grid of addresses, one per element to load.
        x_even = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k_even[None, :] * stride_xk,
            mask=mask_m[:, None], other=0.0)                  # [BLOCK_M, BLOCK_K/2] fp16
        x_odd = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k_odd[None, :] * stride_xk,
            mask=mask_m[:, None], other=0.0)                  # [BLOCK_M, BLOCK_K/2] fp16

        # Packed weights for the same slice: packed rows k0/2 ... k0/2 + BLOCK_K/2 - 1,
        # and this program's BLOCK_N columns. Neighbouring columns are neighbouring
        # bytes in memory, so each row is one contiguous load.
        packed = tl.load(
            wq_ptr + (k0 // 2 + offs_kp)[:, None] * stride_wk + offs_n[None, :] * stride_wn,
            mask=mask_n[None, :], other=0)                    # [BLOCK_K/2, BLOCK_N] uint8

        # Unpack: low nibble = even k, high nibble = odd k. Convert to int32
        # before subtracting 8 so the result can be negative (uint8 cannot).
        q_even = (packed & 0xF).to(tl.int32) - 8              # values in -8..7
        q_odd = (packed >> 4).to(tl.int32) - 8

        # All BLOCK_K values of this slice lie in ONE quantization group,
        # because BLOCK_K divides GROUP_SIZE and k0 is a multiple of BLOCK_K.
        # So one row of scales (one per column) covers the whole slice.
        group = k0 // GROUP_SIZE
        scale = tl.load(s_ptr + group * stride_sg + offs_n * stride_sn,
                        mask=mask_n, other=0.0)               # [BLOCK_N] fp16

        # Dequantize: w = (q - 8) * s. scale[None, :] broadcasts the per-column
        # scale down every row. Exact in FP16 (see module docstring).
        w_even = q_even.to(tl.float16) * scale[None, :]       # [BLOCK_K/2, BLOCK_N] fp16
        w_odd = q_odd.to(tl.float16) * scale[None, :]

        # tl.dot multiplies two FP16 tiles on the tensor cores and adds into
        # the FP32 accumulator: acc += x_even @ w_even, then acc += x_odd @ w_odd.
        acc = tl.dot(x_even, w_even, acc)
        acc = tl.dot(x_odd, w_odd, acc)

    # ---- Write the finished tile ------------------------------------------
    # Round the FP32 sums to FP16 once, at the very end, and store only the
    # elements that exist (both masks).
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc.to(tl.float16), mask=mask_m[:, None] & mask_n[None, :])


# ------------------------------------------------------------------ prepare

def block_k_for(group_size: int) -> int:
    """K slice per loop iteration: must divide the group size (one scale per
    slice) and give tl.dot at least 16 K-values in each of the even/odd halves."""
    return min(MAX_BLOCK_K, group_size)


def prepare(weights: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor,
            group_size: int) -> None:
    """Pack q into [K/2, N] uint8 (two K-neighbours per byte). Not timed."""
    _state.clear()
    K, N = weights.shape
    if not bool(torch.all(zeros == SYMMETRIC_ZERO)):
        raise NotImplementedError("triton_kernel v1 supports the symmetric scheme only")
    block_k = block_k_for(group_size)
    if block_k // 2 < MIN_BLOCK or group_size % block_k:
        raise NotImplementedError(f"triton_kernel v1 needs group size >= {2 * MIN_BLOCK}, "
                                  f"a multiple of {block_k}")
    if K % block_k:
        raise NotImplementedError(f"triton_kernel v1 needs K % {block_k} == 0")

    q = weights.to(torch.uint8)
    packed = (q[0::2] | (q[1::2] << 4)).contiguous()      # even k low, odd k high
    _state.update(K=K, N=N, group_size=group_size, block_k=block_k,
                  packed=packed, scales=scales.contiguous())


# ---------------------------------------------------------------------- run

def block_m_for(M: int) -> int:
    """Rows of x per program: covers all of M up to 64 (so the weights are read
    once), at least 16 for tl.dot. Powers of two keep the compiled variants few."""
    return min(MAX_BLOCK_M, max(MIN_BLOCK, triton.next_power_of_2(M)))


def run(x: torch.Tensor) -> torch.Tensor:
    """y [M, N] FP16 = x [M, K] FP16 @ dequant(W). The only timed call."""
    st = _state
    M, K = x.shape
    N = st["N"]
    out = torch.empty((M, N), dtype=torch.float16, device=x.device)
    block_m = block_m_for(M)
    # One program per (column block, row block).
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(M, block_m))
    int4_matmul_kernel[grid](
        x, st["packed"], st["scales"], out,
        M, N, K,
        x.stride(0), x.stride(1),
        st["packed"].stride(0), st["packed"].stride(1),
        st["scales"].stride(0), st["scales"].stride(1),
        out.stride(0), out.stride(1),
        GROUP_SIZE=st["group_size"],
        BLOCK_M=block_m, BLOCK_N=BLOCK_N, BLOCK_K=st["block_k"],
    )
    return out


# ------------------------------------------------- reference (from our packed data)

def unpack(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of the packing in prepare(): [K/2, N] uint8 -> [K, N] uint8."""
    K2, N = packed.shape
    q = torch.empty((2 * K2, N), dtype=torch.uint8, device=packed.device)
    q[0::2] = packed & 0xF
    q[1::2] = packed >> 4
    return q


def reference_weights() -> torch.Tensor:
    """[K, N] FP32 weights decoded from the packed buffer and scales the kernel reads."""
    st = _state
    q = unpack(st["packed"])
    z = torch.full_like(st["scales"], SYMMETRIC_ZERO, dtype=torch.uint8)
    return dequantize(q, st["scales"], z, st["group_size"])
