"""TorchAO INT4 weight-only GEMM on CUDA (checked against torchao v0.18.0 and
PyTorch v2.11.0 source on GitHub).

Which TorchAO path
------------------
Int4WeightOnlyConfig (torchao/quantization/quant_api.py, version=2) offers these
packing formats (Int4PackingFormat):
  * PLAIN (the default)  -> Int4Tensor, kernels from the separate `mslk` package
                            (bf16i4bf16_rowwise); BF16 activations only.
  * PRESHUFFLED          -> mslk kernels; BF16 / FP8 activations only.
  * PLAIN_INT32          -> Intel XPU.
  * TILE_PACKED_TO_4D    -> Int4TilePackedTo4dTensor, PyTorch's built-in
                            "tinygemm" kernel torch.ops.aten._weight_int4pack_mm
                            (aten/src/ATen/native/cuda/int4mm.cu, sm80+).
We use TILE_PACKED_TO_4D: it is TorchAO's CUDA INT4 weight-only path that needs
no extra package (the kernel ships with PyTorch, built for CUDA 13).

BF16 only: tinygemm checks `A.dtype() == kBFloat16`, writes BF16 output, and
reads BF16 scales and zeros. So COMPUTE_DTYPE = bfloat16, and correctness is
judged against cuBLAS BF16 (precision-matched rule, CLAUDE.md).

What is timed (reporting caveat, see CLAUDE.md)
-----------------------------------------------
TorchAO's own linear() for this tensor (Int4TilePackedTo4dTensor's
aten.linear override) does: x.to(bfloat16) -> pad K -> _weight_int4pack_mm ->
slice -> .to(x.dtype), through tensor-subclass Python dispatch. For
consistency with Marlin (whose op we also call directly), run() calls
aten._weight_int4pack_mm directly on a BF16 input that prepare_input() made
outside the timed region. Our K and N never need padding, so the computation is
the same; the timing EXCLUDES the two dtype casts and the Python dispatch that
a real TorchAO FP16 model would pay.

How tinygemm represents scales and zero points
----------------------------------------------
  * q: unsigned 4-bit 0..15 (same as ours), packed by
    torch.ops.aten._convert_weight_to_int4pack into int32
    [N/8][K/(8*16)][32][8/2] (inner_k_tiles = 8, the value TorchAO fixes).
  * scale_and_zero: BF16 [K/G, N, 2], built by
    torchao.quantization.utils.pack_tinygemm_scales_and_zeros(scale, zero).
  * Dequantization (kernel: convert_i4x8_to_bf16x2x4 gives q - 8, then
    __hfma2; TorchAO: _dequantize_affine_tinygemm with mid_point 8):
        w = (q - 8) * scale + zero
    i.e. the zero point is a FLOAT offset in the weight domain, not an integer
    level, and the FMA result is rounded once to BF16.

Mapping our schemes, w = (q - z) * s with integer z:
        (q - z) * s = (q - 8) * s + (8 - z) * s
  so scale = s and zero = (8 - z) * s, with q passed through unchanged (no
  re-quantization). Our scales are exact in BF16 (common/quantize.py), so:
  * symmetric (z = 8, the project default): scale exact, zero = 0 exact. The
    stored parameters represent our weights EXACTLY.
  * asymmetric: zero = (8 - z) * s needs up to 12 significant bits and is
    rounded to BF16, so not exact in general.
  In both cases the kernel then rounds each dequantized weight (q - 8)*s + zero
  to BF16, like any BF16 kernel (cuBLAS BF16 does the same to its weights).
"""

import torch
# Absolute import: resolves to the installed torchao package, not to this
# folder (which is only importable as impls.torchao).
from torchao.quantization.utils import (
    pack_tinygemm_scales_and_zeros,
    unpack_tinygemm_scales_and_zeros,
)

COMPUTE_DTYPE = torch.bfloat16
INNER_K_TILES = 8       # fixed by TorchAO for Int4TilePackedTo4dTensor
N_TILE = 8              # tinygemm n-tile on NVIDIA (kNTileSize in int4mm.cu)
K_TILE = 16             # tinygemm k-tile (kKTileSize)
K_ALIGN = 1024          # TorchAO pads K to a multiple of this
TINYGEMM_GROUP_SIZES = (32, 64, 128, 256)
MID_POINT = 8           # tinygemm decodes q as (q - 8)

_state: dict = {}


# ------------------------------------------------------------------ prepare

def prepare(weights: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor,
            group_size: int) -> None:
    """Pack our INT4 weights into TorchAO's tinygemm layout (not timed).

    Same steps as Int4TilePackedTo4dTensor.from_hp after its quantization step,
    but starting from OUR q, scales and zeros instead of re-quantizing.
    """
    _state.clear()
    K, N = weights.shape
    if torch.cuda.get_device_capability(weights.device)[0] < 8:
        raise NotImplementedError("TorchAO tinygemm needs compute capability >= 8.0")
    if group_size not in TINYGEMM_GROUP_SIZES:
        raise NotImplementedError(f"tinygemm group sizes are {TINYGEMM_GROUP_SIZES}")
    if K % K_ALIGN or N % N_TILE:
        # TorchAO would zero-pad; our layers never need it, so keep it simple.
        raise NotImplementedError(f"needs K % {K_ALIGN} == 0 and N % {N_TILE} == 0")

    # 1. [N, K] (F.linear layout), then 2 values per byte exactly as TorchAO does:
    #    even k in the HIGH nibble, odd k in the low nibble.
    q_nk = weights.t().contiguous()
    q_bytes = (q_nk[:, ::2] << 4 | q_nk[:, 1::2]).to(torch.uint8).contiguous()

    # 2. Tensor-core tile layout for the m16n8k16 MMA (GPU op from PyTorch core).
    qdata = torch.ops.aten._convert_weight_to_int4pack(q_bytes, INNER_K_TILES)

    # 3. Scale and float-domain zero, (q - z) * s = (q - 8) * s + (8 - z) * s.
    #    Computed in FP32, then stored in BF16 (the only dtype tinygemm reads).
    #    s itself is BF16-exact; the zero is exact only when z = 8.
    s = scales.t().float()                                    # [N, K/G]
    zero = (MID_POINT - zeros.t().float()) * s
    scale_and_zero = pack_tinygemm_scales_and_zeros(
        s.to(torch.bfloat16).contiguous(), zero.to(torch.bfloat16).contiguous(),
        torch.bfloat16)                                       # [K/G, N, 2]

    _state.update(K=K, N=N, group_size=group_size, qdata=qdata,
                  scale_and_zero=scale_and_zero)


# ---------------------------------------------------------------------- run

def prepare_input(x: torch.Tensor) -> torch.Tensor:
    """FP16 activations -> contiguous BF16, outside the timed region."""
    return x.to(torch.bfloat16).contiguous()


def run(x: torch.Tensor) -> torch.Tensor:
    """y [M, N] BF16 = x [M, K] BF16 @ dequant(W)^T: the tinygemm kernel only."""
    st = _state
    return torch.ops.aten._weight_int4pack_mm(x, st["qdata"], st["group_size"],
                                              st["scale_and_zero"])


# ------------------------------------------------ reference (from TorchAO data)

def reference_weights() -> torch.Tensor:
    """[K, N] FP32 weights decoded from TorchAO's OWN packed tensors.

    q is unpacked from qdata by inverting _convert_weight_to_int4pack (written
    from int4mm.cu, independent of the op); scale and zero come from
    unpack_tinygemm_scales_and_zeros; then tinygemm's formula
    w = (q - 8) * scale + zero, evaluated in FP32.
    """
    st = _state
    K, N, G = st["K"], st["N"], st["group_size"]
    q = unpack_tinygemm_weight(st["qdata"], N, K).float()                  # [N, K]
    scale, zero = unpack_tinygemm_scales_and_zeros(st["scale_and_zero"])   # each [N, K/G, 1]
    scale = scale.squeeze(-1).float().repeat_interleave(G, dim=1)
    zero = zero.squeeze(-1).float().repeat_interleave(G, dim=1)
    return ((q - MID_POINT) * scale + zero).t().contiguous()


def unpack_tinygemm_weight(qdata: torch.Tensor, N: int, K: int) -> torch.Tensor:
    """Inverse of aten._convert_weight_to_int4pack (NVIDIA, inner_k_tiles=8).

    From matrix_to_m16n8k16_Bint4_layout in int4mm.cu: for output word
    out[nTile][kOuter][t][j] (t = lane 0..31, j = pair of k-tiles 0..3),
        n      = nTile*8 + t//4
        kBase0 = (kOuter*8 + 2j) * 16 / 2      (byte index into the [N, K/2] input)
        bytes  = kBase0 + t%4, +4, kBase0+8 + t%4, +4     (v[0..3])
        word   = v[i] low nibble at bits 16+4i, v[i] high nibble at bits 4i
    and each input byte holds (q[n, 2b] << 4) | q[n, 2b+1].
    Returns [N, K] uint8.
    """
    dev = qdata.device
    p = qdata.to(torch.int64) & 0xFFFFFFFF                     # [N/8, K/128, 32, 4]
    n_tiles, k_outer, lanes, pairs = p.shape
    nT = torch.arange(n_tiles, device=dev).view(-1, 1, 1, 1)
    kO = torch.arange(k_outer, device=dev).view(1, -1, 1, 1)
    t = torch.arange(lanes, device=dev).view(1, 1, -1, 1)
    j = torch.arange(pairs, device=dev).view(1, 1, 1, -1)

    n = (nT * N_TILE + t // 4).expand_as(p)
    k_base0 = (kO * INNER_K_TILES + 2 * j) * K_TILE // 2
    byte_idx = [k_base0 + t % 4, k_base0 + t % 4 + 4,
                k_base0 + K_TILE // 2 + t % 4, k_base0 + K_TILE // 2 + t % 4 + 4]

    q = torch.full((N, K), 255, dtype=torch.uint8, device=dev)   # 255 = "not filled"
    for i, b in enumerate(byte_idx):
        b = b.expand_as(p)
        hi = (p >> (4 * i)) & 0xF          # high nibble of v[i] -> even k = 2b
        lo = (p >> (16 + 4 * i)) & 0xF     # low nibble of v[i]  -> odd k  = 2b+1
        q[n, 2 * b] = hi.to(torch.uint8)
        q[n, 2 * b + 1] = lo.to(torch.uint8)
    if bool((q == 255).any()):
        raise ValueError("tinygemm unpack did not cover every weight")
    return q
