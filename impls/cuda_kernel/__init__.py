"""Our INT4 x FP16 matmul in CUDA, version 1 (kernel: int4_kernel.cu).

Scope: symmetric scheme (w = (q - 8) * s), group sizes that are multiples of
32 (all of ours), K a multiple of 32, any M and N. FP16 in and out, FP32
accumulation, so COMPUTE_DTYPE = float16.

Building
--------
The .cu file is compiled the first time prepare() runs (one-time setup, never
timed), with torch.utils.cpp_extension.load(). That uses:
  * nvcc from CUDA_HOME (carc/env_setup.sh points it at the pip-installed CUDA 13);
  * ninja, the small build tool PyTorch's extension builder requires.
PyTorch caches the compiled library (in ~/.cache/torch_extensions unless
TORCH_EXTENSIONS_DIR says otherwise) and only rebuilds when the source changes.
We compile for both A100 (sm_80) and A40 (sm_86), so the same cached build
works on either node type.

Weight format (built in prepare)
--------------------------------
Column-major with K contiguous, 8 four-bit values per 32-bit word:

    packed[n, k // 8] bits 4j..4j+3  =  q[k, n]  for k = 8 * (k // 8) + j

i.e. one output column's weights are K/2 consecutive bytes. The kernel's warp
reads a column in coalesced 512-byte blocks (see int4_kernel.cu). Scales are
stored transposed, [N, K/G], so each column's scales are consecutive too.
"""

import os
from pathlib import Path

import torch

from common.quantize import SYMMETRIC_ZERO, dequantize

COMPUTE_DTYPE = torch.float16
CHUNK = 32                  # weights per 16-byte load in the kernel
SOURCE = Path(__file__).with_name("int4_kernel.cu")
ARCH_LIST = "8.0;8.6"       # A100 and A40

_state: dict = {}
_ext = None


# ------------------------------------------------------------------- build

def _cudart_link_dir(build_dir: Path) -> list[str]:
    """Extra linker flags if CUDA_HOME lacks an unversioned libcudart.so.

    PyTorch links extensions with -lcudart, which needs a file named exactly
    libcudart.so. pip-installed CUDA ships only libcudart.so.13. In that case
    we put a libcudart.so symlink in OUR build folder (the environment itself
    is never modified) and add that folder to the linker search path.
    """
    from torch.utils.cpp_extension import CUDA_HOME
    if not CUDA_HOME:
        return []
    libs = [Path(CUDA_HOME) / d for d in ("lib64", "lib")]
    if any((d / "libcudart.so").exists() for d in libs):
        return []
    versioned = sorted(p for d in libs if d.is_dir() for p in d.glob("libcudart.so.*"))
    if not versioned:
        return []
    link_dir = build_dir / "cudart_link"
    link_dir.mkdir(parents=True, exist_ok=True)
    link = link_dir / "libcudart.so"
    if not link.exists():
        link.symlink_to(versioned[0])
    return [f"-L{link_dir}"]


def _load():
    """Compile (first time only) and import the CUDA extension."""
    global _ext
    if _ext is not None:
        return _ext
    from torch.utils import cpp_extension
    if not cpp_extension.is_ninja_available():
        raise RuntimeError("impls/cuda_kernel needs ninja to compile: "
                           "python -m pip install ninja   (inside the int4 environment)")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", ARCH_LIST)
    name = "int4_cuda_kernel"
    # The folder PyTorch builds this extension in (a private helper, so fall
    # back to our own folder if a future PyTorch renames it).
    try:
        build_dir = Path(cpp_extension._get_build_directory(name, verbose=False))
    except AttributeError:
        build_dir = Path.home() / ".cache" / "int4_cuda_kernel"
    _ext = cpp_extension.load(name=name, sources=[str(SOURCE)],
                              extra_cuda_cflags=["-O3", "-lineinfo"],
                              extra_ldflags=_cudart_link_dir(build_dir),
                              verbose=False)
    return _ext


# ------------------------------------------------------------------ prepare

def pack(q: torch.Tensor) -> torch.Tensor:
    """[K, N] values 0..15 -> [N, K/8] int32, K contiguous, low nibble = lower k."""
    qt = q.to(torch.uint8).t().contiguous()                 # [N, K]
    bytes_ = (qt[:, 0::2] | (qt[:, 1::2] << 4)).contiguous()   # [N, K/2]: k even low, k odd high
    # Four bytes in a row form one int32 (little-endian: first byte = lowest bits),
    # so nibble j of the word is k = 8 * word + j.
    return bytes_.view(torch.int32)                          # [N, K/8]


def unpack(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of pack(): [N, K/8] int32 -> [K, N] uint8."""
    b = packed.view(torch.uint8)                             # [N, K/2]
    N, K2 = b.shape
    qt = torch.empty((N, 2 * K2), dtype=torch.uint8, device=b.device)
    qt[:, 0::2] = b & 0xF
    qt[:, 1::2] = b >> 4
    return qt.t().contiguous()


def prepare(weights: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor,
            group_size: int) -> None:
    """Pack the weights, transpose the scales, compile the kernel. Not timed."""
    _state.clear()
    K, N = weights.shape
    if not bool(torch.all(zeros == SYMMETRIC_ZERO)):
        raise NotImplementedError("cuda_kernel v1 supports the symmetric scheme only")
    if K % CHUNK or group_size % CHUNK or K % group_size:
        raise NotImplementedError(f"cuda_kernel v1 needs K and the group size to be "
                                  f"multiples of {CHUNK}")
    _load()
    _state.update(K=K, N=N, group_size=group_size, packed=pack(weights),
                  scales_t=scales.t().contiguous())


# ---------------------------------------------------------------------- run

def run(x: torch.Tensor) -> torch.Tensor:
    """y [M, N] FP16 = x [M, K] FP16 @ dequant(W). The only timed call."""
    st = _state
    return _ext.int4_matmul(x, st["packed"], st["scales_t"], st["group_size"])


# ------------------------------------------------- reference (from our packed data)

def reference_weights() -> torch.Tensor:
    """[K, N] FP32 weights decoded from the packed buffer and scales the kernel reads."""
    st = _state
    q = unpack(st["packed"])
    scales = st["scales_t"].t().contiguous()
    z = torch.full_like(scales, SYMMETRIC_ZERO, dtype=torch.uint8)
    return dequantize(q, scales, z, st["group_size"])
