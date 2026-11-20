"""Our INT4 x FP16 matmul in Triton -- version 2: v1 + split-K.

What changed from v1 (impls/triton_kernel), and ONLY this
---------------------------------------------------------
v1 gives each program one strip of BLOCK_N output columns and lets it walk the
WHOLE K dimension alone. With N = 4096 that is only 64 programs for an A100's
108 SMs: about 40% of the GPU sits idle, and on mlp_down (K = 14336) each busy
program has a very long loop. v1 was fast only on mlp_up (N = 28672, 448
programs), where the GPU is full.

Split-K: cut K into SPLIT_K equal pieces and give each piece its own program.

    v1:   program (n)      sums k = 0 .. K-1
    v2:   program (n, s)   sums k = s*K/SPLIT_K .. (s+1)*K/SPLIT_K - 1

Each program now produces a PARTIAL sum for its output tile, so the partial
sums must be added together:

  * SPLIT_K == 1: nothing to add; the program stores FP16 directly (exactly v1).
  * SPLIT_K  > 1: every program adds its FP32 partial sum into a shared FP32
    buffer with an atomic add (the hardware makes concurrent adds to the same
    address safe), and one cast turns the buffer into the FP16 output.

The cost of split-K: zeroing the FP32 buffer and the final FP32 -> FP16 cast
are two extra small GPU operations, timed like everything else in run(). The
buffer is M x N x 4 bytes, tiny next to the weights for decode-sized M.

split_k_for() picks SPLIT_K automatically: the smallest power of two that gives
the GPU enough programs (TARGET_PROGRAMS_PER_SM per SM), and no more, because
every extra split adds atomic traffic. For mlp_up (already 448 programs) it
picks 1, i.e. v2 runs exactly v1's kernel there.

Everything else is identical to v1, so any speed difference comes from
split-K alone: the same packed [K/2, N] uint8 weights (prepare() reuses v1's
packing), the same tile sizes, the same dequantization, the same FP32
accumulation. See impls/triton_kernel/__init__.py for the line-by-line
explanation of the shared parts; comments here focus on what is new.

Note on reproducibility: with SPLIT_K > 1 the order in which partial sums are
added depends on which program finishes first, so results can differ between
runs in the last FP32 bits. That is far below the FP16 rounding of the output
and well inside the correctness tolerance.
"""

import torch
import triton
import triton.language as tl

from impls import triton_kernel as v1

COMPUTE_DTYPE = torch.float16

BLOCK_N = v1.BLOCK_N                # same tiles as v1, so only split-K differs
TARGET_PROGRAMS_PER_SM = 4          # enough programs that every SM has work, with
                                    # spares to hide memory latency
MAX_SPLIT_K = 32

# Overrides for tests and experiments (analysis/triton_inspect.py). None = the
# defaults described above (automatic SPLIT_K, v1's BLOCK_M rule, BLOCK_N, and
# Triton's own default num_warps / num_stages). The benchmark never sets them.
FORCE_SPLIT_K: int | None = None
FORCE_BLOCK_M: int | None = None
FORCE_BLOCK_N: int | None = None
NUM_WARPS: int | None = None      # warps (groups of 32 threads) per program
NUM_STAGES: int | None = None     # how many loop iterations of loads Triton keeps in flight

_state: dict = {}


# --------------------------------------------------------------------- kernel

@triton.jit
def int4_matmul_splitk_kernel(
    x_ptr,          # activations  [M, K]   fp16
    wq_ptr,         # packed q     [K/2, N] uint8 (v1's format)
    s_ptr,          # scales       [K/G, N] fp16
    out_ptr,        # SPLIT_K == 1: output [M, N] fp16;  SPLIT_K > 1: FP32 buffer [M, N]
    M, N, K,
    K_PER_SPLIT,    # how many K values each split handles (a multiple of BLOCK_K)
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_sg, stride_sn,
    stride_om, stride_on,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    # NEW: a 3-D grid. Axis 1 says which piece of K this program sums.
    pid_n = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    pid_m = tl.program_id(axis=2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_kp = tl.arange(0, BLOCK_K // 2)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # NEW: this program's K range. v1 looped over 0..K; we loop over one
    # 1/SPLIT_K slice. K_PER_SPLIT is a multiple of BLOCK_K (and so k0 stays a
    # multiple of BLOCK_K), which keeps v1's "one scale row per slice" property.
    k_start = pid_k * K_PER_SPLIT
    for k0 in range(k_start, k_start + K_PER_SPLIT, BLOCK_K):
        # ---- identical to v1 from here to the end of the loop ----
        k_even = k0 + 2 * offs_kp
        k_odd = k_even + 1
        x_even = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k_even[None, :] * stride_xk,
            mask=mask_m[:, None], other=0.0)
        x_odd = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k_odd[None, :] * stride_xk,
            mask=mask_m[:, None], other=0.0)
        packed = tl.load(
            wq_ptr + (k0 // 2 + offs_kp)[:, None] * stride_wk + offs_n[None, :] * stride_wn,
            mask=mask_n[None, :], other=0)
        q_even = (packed & 0xF).to(tl.int32) - 8
        q_odd = (packed >> 4).to(tl.int32) - 8
        group = k0 // GROUP_SIZE
        scale = tl.load(s_ptr + group * stride_sg + offs_n * stride_sn,
                        mask=mask_n, other=0.0)
        w_even = q_even.to(tl.float16) * scale[None, :]
        w_odd = q_odd.to(tl.float16) * scale[None, :]
        acc = tl.dot(x_even, w_even, acc)
        acc = tl.dot(x_odd, w_odd, acc)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    mask = mask_m[:, None] & mask_n[None, :]
    # SPLIT_K is a constexpr, so Triton compiles only the branch that applies.
    if SPLIT_K == 1:
        # Whole K summed here: store the FP16 result directly, like v1.
        tl.store(out_ptrs, acc.to(tl.float16), mask=mask)
    else:
        # NEW: only part of K summed here. Add the FP32 partial sum into the
        # shared buffer. atomic_add makes the read-add-write of each element
        # indivisible, so SPLIT_K programs adding to the same element never
        # lose each other's contribution. "relaxed" ordering is enough: we only
        # need every add to land, and the kernel's end makes them all visible.
        tl.atomic_add(out_ptrs, acc, mask=mask, sem="relaxed")


# ------------------------------------------------------------------ prepare

def prepare(weights: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor,
            group_size: int) -> None:
    """Same packing (and the same scope checks) as v1. Not timed."""
    _state.clear()
    _state.update(v1.pack(weights, scales, zeros, group_size))
    # Needed to decide how many programs fill the GPU (see split_k_for).
    _state["num_sms"] = torch.cuda.get_device_properties(weights.device).multi_processor_count


# ---------------------------------------------------------------------- run

def block_m_for(M: int) -> int:
    return FORCE_BLOCK_M or v1.block_m_for(M)


def block_n() -> int:
    return FORCE_BLOCK_N or BLOCK_N


def split_k_for(M: int, N: int, K: int, block_k: int, num_sms: int) -> int:
    """Smallest power of two SPLIT_K giving >= TARGET_PROGRAMS_PER_SM * num_sms
    programs, such that every split gets a whole number of BLOCK_K slices."""
    if FORCE_SPLIT_K is not None:
        return FORCE_SPLIT_K
    programs = triton.cdiv(N, block_n()) * triton.cdiv(M, block_m_for(M))
    k_slices = K // block_k
    split = 1
    while (programs * split < TARGET_PROGRAMS_PER_SM * num_sms
           and split * 2 <= MAX_SPLIT_K
           and k_slices % (split * 2) == 0):
        split *= 2
    return split


def run(x: torch.Tensor) -> torch.Tensor:
    """y [M, N] FP16 = x [M, K] FP16 @ dequant(W). The only timed call."""
    st = _state
    M, K = x.shape
    N = st["N"]
    block_m, bn = block_m_for(M), block_n()
    split_k = split_k_for(M, N, K, st["block_k"], st["num_sms"])
    if (K // st["block_k"]) % split_k:
        raise ValueError(f"SPLIT_K={split_k} must divide K/BLOCK_K = {K // st['block_k']}")

    if split_k == 1:
        out = torch.empty((M, N), dtype=torch.float16, device=x.device)
    else:
        # Atomic adds need a zeroed FP32 buffer to add into.
        out = torch.zeros((M, N), dtype=torch.float32, device=x.device)

    # Only pass num_warps / num_stages when overridden, so Triton's defaults apply.
    launch_opts = {k: v for k, v in (("num_warps", NUM_WARPS), ("num_stages", NUM_STAGES))
                   if v is not None}
    grid = (triton.cdiv(N, bn), split_k, triton.cdiv(M, block_m))
    # The launch returns the compiled kernel; keeping it lets
    # analysis/triton_inspect.py read its register / spill counts.
    st["last_kernel"] = int4_matmul_splitk_kernel[grid](
        x, st["packed"], st["scales"], out,
        M, N, K,
        K // split_k,
        x.stride(0), x.stride(1),
        st["packed"].stride(0), st["packed"].stride(1),
        st["scales"].stride(0), st["scales"].stride(1),
        out.stride(0), out.stride(1),
        GROUP_SIZE=st["group_size"],
        BLOCK_M=block_m, BLOCK_N=bn, BLOCK_K=st["block_k"],
        SPLIT_K=split_k,
        **launch_opts,
    )
    return out if split_k == 1 else out.to(torch.float16)


# ------------------------------------------------- reference (from our packed data)

def reference_weights() -> torch.Tensor:
    """[K, N] FP32 weights decoded from the packed buffer and scales the kernel reads."""
    return v1.reference_weights_from(_state)
