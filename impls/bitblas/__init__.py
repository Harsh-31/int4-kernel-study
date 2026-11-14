"""BitBLAS INT4 x FP16 GEMM (checked against the bitblas 0.1.0.post1 wheel source).

EXCLUDED FROM THE STUDY -- kept for reference only, never run by default.
bitblas 0.1.0's bundled TVM (3rdparty/tvm/build/libtvm*.so) is linked against
the CUDA 12 runtime (libcudart.so.12, libnvrtc.so.12; libnvrtc.so.12 confirmed
missing on CARC), and our environment is CUDA 13 only; we do not mix CUDA 12
and 13 in one environment. common.harness.EXCLUDED_IMPLS keeps it out of
discovery, run_benchmark.py and tests/test_bitblas.py. See CLAUDE.md.

0.1.0.post1 has no GitHub tag (only v0.1.0); it exists only as the PyPI wheel
(uploaded 2025-02-16). The API below was read from that wheel's Python files:
bitblas/ops/general_matmul/__init__.py (MatmulConfig, Matmul) and the decode
code in bitblas/ops/general_matmul/{tirscript,tilelang}/.

What BitBLAS can express for 4-bit weights with per-group FP16 scales
---------------------------------------------------------------------
  W_dtype="uint4", with_scaling=True, with_zeros=True, zeros_mode="original":
      w = (q - zeros[n, k // G]) * scale[n, k // G]
  with zeros an FP16 tensor [N, K/G]. This is exactly our ASYMMETRIC scheme.

  Our SYMMETRIC scheme (zero fixed at 8) has no dedicated type: W_dtype="int4"
  does have the implicit offset of 8 (it decodes as nibble - 8), but
  Matmul.transform_weight asserts `not with_scaling` for int formats, so it
  cannot have per-group scales. Symmetric therefore runs on the same uint4
  path with a zeros tensor filled with 8. The math is identical, but the
  kernel still reads the zeros tensor (unlike Marlin's uint4b8).

  zeros_mode="original" is BitBLAS's default and what its own
  bitblas.Linear.repack_from_gptq produces. "quantized" (4-bit packed zeros,
  w = (q - qz) * s) also exists; switching is one constant below.

Layout: layout="nt", i.e. W is [N, K] (output features first), the transpose
of our [K, N]. Scales/zeros are [N, K/G]. Weights are given to
transform_weight() as one unsigned value per int8; BitBLAS then packs and
permutes them on the CPU (ladder permutation for tensor-core loads, packing 2
values per byte, and the "lop3" nibble interleave for fast decoding).

Numerics: BitBLAS defaults to accum_dtype="float16" (FP16 accumulation). We
use "float32" so BitBLAS accumulates like cuBLAS and Marlin do; with K up to
14336, FP16 accumulation would likely fail our correctness rule. Change
ACCUM_DTYPE to measure the BitBLAS default.

Tuning: prepare() builds the operator and runs BitBLAS's hardware-aware tuning
(compiling many candidate kernels with nvcc), like bitblas.Linear does. Tuned
operators are saved to BitBLAS's on-disk database, so each (shape, group size)
is tuned once and later runs only load it. Set INT4_BITBLAS_CACHE to move the
database (default: BitBLAS's own, ~/.cache/bitblas).
"""

import math
import os

import torch
# Absolute import: this resolves to the installed bitblas package, not to this
# folder (which is only importable as impls.bitblas).
from bitblas import Matmul, MatmulConfig, auto_detect_nvidia_target
from bitblas.cache import get_database_path, global_operator_cache, set_database_path

from common.config import batch_sizes
from common.quantize import dequantize

ZEROS_MODE = "original"   # see module docstring
ACCUM_DTYPE = "float32"   # BitBLAS default is "float16"; see module docstring
TUNE_TOPK = 20            # candidates benchmarked during tuning (bitblas.Linear's value)

_state: dict = {}
_db = {"loaded": False, "target": None}


# ------------------------------------------------------------- operator cache

def _get_or_create_operator(config: MatmulConfig) -> Matmul:
    """Same pattern as bitblas.Linear._get_or_create_bitblas_operator."""
    if not _db["loaded"]:
        if os.environ.get("INT4_BITBLAS_CACHE"):
            set_database_path(os.environ["INT4_BITBLAS_CACHE"])
        _db["target"] = auto_detect_nvidia_target()
        global_operator_cache.load_from_database(get_database_path(), _db["target"])
        _db["loaded"] = True

    op = global_operator_cache.get(config)
    if op is None:
        # Build with the default schedule first, then tune: compiles TUNE_TOPK
        # candidate kernels and keeps the fastest per batch-size bucket.
        op = Matmul(config, target=_db["target"], enable_tuning=False)
        op.hardware_aware_finetune(topk=TUNE_TOPK)
        global_operator_cache.add(config, op)
        global_operator_cache.save_into_database(get_database_path(), _db["target"])
    return op


# ------------------------------------------------------------------ prepare

def prepare(weights: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor,
            group_size: int) -> None:
    """Build/tune the BitBLAS operator and pack the weights (not timed).

    weights: [K, N] uint8 in 0..15, scales: [K/G, N] FP16, zeros: [K/G, N] uint8.
    """
    _state.clear()
    K, N = weights.shape
    config = MatmulConfig(
        # Batch sizes the kernel is tuned for. BitBLAS compiles one dynamic-M
        # kernel and tunes a schedule per listed M. The full list keeps one
        # tuned operator per (layer, group size), shared by early/full sweeps.
        M=list(batch_sizes("full")),
        N=N, K=K,
        A_dtype="float16", W_dtype="uint4", out_dtype="float16",
        accum_dtype=ACCUM_DTYPE,
        layout="nt",
        with_bias=False,
        group_size=group_size,
        with_scaling=True, with_zeros=True, zeros_mode=ZEROS_MODE,
        # fast_decoding and propagate_b left at BitBLAS's defaults (chosen by
        # rule for this dtype pair and GPU).
    )
    op = _get_or_create_operator(config)

    # BitBLAS wants W as [N, K], one unsigned 4-bit value per int8.
    # transform_weight packs/permutes on the CPU and returns on W's device.
    w_nk = weights.t().contiguous().to(torch.int8)
    packed = op.transform_weight(w_nk)

    if ZEROS_MODE != "original":
        raise NotImplementedError("only zeros_mode='original' is wired up")
    _state.update(
        op=op, K=K, N=N, group_size=group_size,
        W=packed,
        scale=scales.t().contiguous(),                       # [N, K/G] FP16
        zeros=zeros.t().contiguous().to(torch.float16),      # [N, K/G] FP16 (8s if symmetric)
        slot_positions=None,                                 # filled lazily
    )


# ---------------------------------------------------------------------- run

def run(x: torch.Tensor) -> torch.Tensor:
    """y [M, N] FP16 = x [M, K] FP16 @ dequant(W)^T. The only timed call."""
    st = _state
    return st["op"].forward(x, st["W"], scale=st["scale"], zeros=st["zeros"])


# ------------------------------------------------ reference (from BitBLAS data)

def reference_weights() -> torch.Tensor:
    """[K, N] FP32 weights decoded from BitBLAS's OWN packed buffers.

    Decodes the packed weight buffer the kernel reads (see _slot_positions)
    and applies BitBLAS's "original" formula w = (q - zeros) * scale using the
    [N, K/G] scale and zeros tensors the kernel reads.
    """
    st = _state
    N, K, G = st["N"], st["K"], st["group_size"]
    if st["slot_positions"] is None:
        st["slot_positions"] = _slot_positions(st["op"], N, K)
    pos = st["slot_positions"]

    nib = _nibbles(st["W"].cpu())
    q = torch.empty(N * K, dtype=torch.uint8)
    q[pos] = nib.to(torch.uint8)          # slot s holds the value of element pos[s]
    q_kn = q.reshape(N, K).t().contiguous()

    zeros = st["zeros"].float()
    if not torch.equal(zeros, zeros.round()):
        raise ValueError("BitBLAS zeros are not integers; cannot express as (q - z) * s")
    z_kn = zeros.round().to(torch.uint8).t().contiguous().cpu()
    s_kn = st["scale"].t().contiguous().cpu()
    return dequantize(q_kn, s_kn, z_kn, G).to(st["W"].device)


def _nibbles(packed: torch.Tensor) -> torch.Tensor:
    """All 4-bit values of a packed buffer, in memory order: byte i gives
    slots 2i (low nibble) and 2i+1 (high nibble). int64, flat."""
    b = packed.contiguous().view(torch.uint8).flatten().to(torch.int64)
    return torch.stack([b & 0xF, b >> 4], dim=-1).flatten()


def _slot_positions(op: Matmul, N: int, K: int) -> torch.Tensor:
    """For every 4-bit slot of the packed buffer, the flat index (n*K + k) of
    the original [N, K] element stored there.

    BitBLAS's weight transform (ladder permutation -> pack 2 per byte -> lop3
    nibble interleave; see Matmul._create_weight_executors) only MOVES 4-bit
    values; it never depends on them. So we can learn the mapping by running
    the same transform on "position-coded" probes: probe d holds base-16
    digit d of each element's index, and the digit that lands in each slot
    tells us where that slot's value came from. ceil(log16(N*K)) probes
    suffice (7 for the largest layer).

    The result is checked to be a true permutation (most value-dependent
    transforms would break that). The definitive check is the GPU test
    test_packed_weights_decode_exactly: decoding the REAL packed weights must
    reproduce our canonical weights exactly.

    What this decoder does and does not verify: it reads the actual buffer,
    scales and zeros the kernel uses, so a wrong orientation, dtype or
    scale/zeros layout handed to BitBLAS shows up. It learns the packing
    layout from BitBLAS's own transform, so a mismatch between that transform
    and the kernel's decode shows up in the kernel correctness check instead.
    """
    n = N * K
    idx = torch.arange(n, dtype=torch.int64)
    pos = torch.zeros(n, dtype=torch.int64)
    for d in range(max(1, math.ceil(math.log(n, 16)))):
        probe = ((idx >> (4 * d)) & 0xF).to(torch.int8).reshape(N, K)
        nib = _nibbles(op.transform_weight(probe).cpu())
        if nib.numel() != n:
            raise ValueError(f"packed buffer holds {nib.numel()} values, expected {n}")
        pos |= nib << (4 * d)
    seen = torch.zeros(n, dtype=torch.bool)
    if int(pos.max()) >= n:
        raise ValueError("BitBLAS weight transform is not a permutation of positions")
    seen[pos] = True
    if not bool(seen.all()):
        raise ValueError("BitBLAS weight transform is not a permutation of positions")
    return pos
