"""Our INT4 x FP16 matmul in Triton -- v2 with tuned tile settings.

The SAME kernel as impls/triton_kernel_v2 (v1 + split-K); only the launch
settings differ. Keeping it as its own implementation lets the report show
each step separately: v1 -> v2 (split-K) -> v2 tuned (tile settings).

Settings, chosen from the A100 tile sweep (analysis/triton_inspect.py,
job 14392224, results/triton_inspect_*2026-11-20*.csv):

                  v2 default                 v2 tuned
  BLOCK_M         16 / 32 / 64 (grows with M) 16 always
  BLOCK_N         64                          128
  num_warps       Triton default (4)          8
  num_stages      Triton default (3)          2
  SPLIT_K         automatic                   automatic (recomputed for these tiles)

What the sweep showed:
  * M = 64: every one of the 5 fastest settings used BLOCK_M = 16 with
    BLOCK_N = 128; this was 1.7x faster than v2's default on attn_out
    (177 -> ~101-109 us) and 2.2x on mlp_up (1233 -> 550 us).
  * M = 1: all settings were within about 1 us of each other, so the
    choice hardly matters there; this one was among the fastest.
  * No setting spilled registers (spills = 0 everywhere).
  BLOCK_M = 16 at M = 64 means 4 row-blocks, each reading the weights; the
  sweep shows that is still much faster than one 64-row tile.

M = 16 and the full batch-size list were not in the sweep; the benchmark
measures them. Correctness does not depend on the settings (the tests check).
"""

import torch

from impls import triton_kernel as v1
from impls import triton_kernel_v2 as v2

COMPUTE_DTYPE = torch.float16

BLOCK_M = 16
BLOCK_N = 128
NUM_WARPS = 8
NUM_STAGES = 2

_state: dict = {}


def prepare(weights: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor,
            group_size: int) -> None:
    """Same packing as v1 / v2. Not timed. Own state, so v2 is never affected."""
    _state.clear()
    _state.update(v1.pack(weights, scales, zeros, group_size))
    _state["num_sms"] = torch.cuda.get_device_properties(weights.device).multi_processor_count


def split_k_for(M: int, K: int) -> int:
    st = _state
    return v2.choose_split_k(M, st["N"], K, st["block_k"], st["num_sms"], BLOCK_M, BLOCK_N)


def run(x: torch.Tensor) -> torch.Tensor:
    """y [M, N] FP16 = x [M, K] FP16 @ dequant(W). The only timed call."""
    M, K = x.shape
    return v2.launch(x, _state, BLOCK_M, BLOCK_N, split_k_for(M, K), NUM_WARPS, NUM_STAGES)


def reference_weights() -> torch.Tensor:
    """[K, N] FP32 weights decoded from the packed buffer and scales the kernel reads."""
    return v1.reference_weights_from(_state)
