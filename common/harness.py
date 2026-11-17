"""Benchmark harness: times every implementation the same way (CLAUDE.md rules).

This module holds the timing, metrics and CSV logic. The loop over
implementations, layers and batch sizes lives in run_benchmark.py.

Implementation interface
------------------------
An implementation is any object (usually the package in impls/<name>/) with:

  prepare(weights, scales, zeros, group_size)
      One-time setup such as repacking. NOT timed. Receives our canonical
      layout from common/quantize.py: weights = unpacked INT4 q [K, N] uint8
      (0..15), scales [K/G, N] FP16, zeros [K/G, N] uint8 (all 8 if symmetric).
  run(x)
      x [M, K] on the GPU -> y [M, N]. The only thing that is timed. x is the
      FP16 activations, or whatever prepare_input() returned for them.

Optional attributes:

  COMPUTE_DTYPE = torch.float16 (default) or torch.bfloat16
      The precision the kernel computes in. Selects the precision-matched
      cuBLAS baseline for correctness (common/correctness.py).
  prepare_input(x) -> tensor
      Converts the FP16 activations to what run() expects (e.g. a contiguous
      BF16 copy for a BF16 kernel). Called OUTSIDE the timed region, once per
      batch size, so dtype casts are not timed. Default: x unchanged.
  reference_weights() -> [K, N] tensor
      The weights dequantized from the implementation's OWN packed data
      (after prepare). CLAUDE.md requires each library's correctness reference
      to be built this way so a packing bug cannot hide. If missing, the
      canonical (q, s, z) is dequantized instead and a warning is printed.
  WEIGHT_BITS = 4 or 16
      Bits per stored weight, used for the byte count. Defaults to 4; the
      cuBLAS FP16 and BF16 baselines set 16.
"""

import csv
import importlib
import pkgutil
import statistics
import warnings
from pathlib import Path

import torch

from common.config import REPO_ROOT, load_config
from common.quantize import dequantize

CSV_COLUMNS = [
    "impl", "layer", "K", "N", "M", "group_size", "latency_us", "bandwidth_gbs",
    "pct_peak", "speedup_vs_fp16", "speedup_vs_bf16", "correct", "compute_dtype",
    "gpu", "timestamp",
]
BASELINE_IMPL = "cublas"            # cuBLAS FP16: speedup_vs_fp16 is relative to this
BF16_BASELINE_IMPL = "cublas_bf16"  # cuBLAS BF16: speedup_vs_bf16 is relative to this
BASELINE_IMPLS = (BASELINE_IMPL, BF16_BASELINE_IMPL)   # always run first, in this order

# Implementations kept in impls/ for reference but never run: discover_impls()
# skips them WITHOUT importing them, run_benchmark.py refuses them, and their
# tests skip with "excluded: <reason>". Keep in sync with CLAUDE.md.
EXCLUDED_IMPLS = {
    # bitblas 0.1.0's bundled TVM links libcudart.so.12 / libnvrtc.so.12; our
    # environment is CUDA 13 only and we do not mix CUDA 12 and 13.
    "bitblas": "requires CUDA 12",
}
RESULTS_DIR = REPO_ROOT / "results"
MB = 2**20


# --------------------------------------------------------------------- timing

def make_l2_flush_buffer() -> torch.Tensor:
    """A GPU buffer larger than the L2 cache, READ before every timed run (flush_l2).

    Why: an A100 has a 40 MB L2. Several of our INT4 weight matrices are small
    enough to fit in it (attn_out: 4096*4096/2 = 8 MB). Without a flush, every
    timed run after the first would read the weights from L2 instead of HBM and
    look unrealistically fast. In real decode, the other layers' weights have
    evicted this layer's weights by the time it runs again, so HBM is the
    honest case. Reading a buffer >= 2x the L2 size evicts everything else.

    The buffer is filled once here, and the write-back of that fill finishes
    before any timing starts (we synchronize).
    """
    size = load_config()["harness"]["l2_flush_mb"] * MB
    l2 = getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 0)
    if l2 and size < 2 * l2:
        raise ValueError(f"harness.l2_flush_mb ({size // MB} MB) must be >= 2x the "
                         f"L2 cache ({l2 // MB} MB); increase it in config/workloads.yaml")
    buf = torch.zeros(size, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    return buf


def flush_l2(buf: torch.Tensor) -> None:
    """Evict L2 by READING buf (never writing it). Called before each timed run.

    Why read, not write: writing (e.g. buf.zero_()) also evicts the weights,
    but it leaves L2 full of DIRTY lines, i.e. data not yet saved to HBM. When
    the timed kernel then reads its weights, the GPU must first write those
    dirty lines back to HBM, and that extra traffic lands inside the timed
    region. Reading leaves only CLEAN lines, which can be dropped for free.

    Measured on the A100 (analysis/flush_experiment.py, Nov 2026): the write
    flush inflated latencies by up to 1.65x (cuBLAS FP16, attn_out), while
    "read" and "write then read" gave identical times. That shows the extra
    time was the dirty-line write-back, not the eviction itself.

    sum() over int32 words reads every byte and writes a single scalar.
    """
    buf.view(torch.int32).sum()


def time_runs(fn, n_warmup: int, n_timed: int, flush_buf: torch.Tensor | None = None) -> list[float]:
    """Run fn() n_warmup times untimed, then n_timed times timed. Returns us per run.

    Warm-up: the first calls pay one-time costs we must not measure:
    Triton/JIT compilation and autotuning, cuBLAS heuristic selection, lazy
    CUDA module loading, the caching allocator obtaining memory, and the GPU
    ramping its clocks up from idle.

    CUDA events instead of time.perf_counter(): kernel launches are
    asynchronous, so a CPU timer measures how long it took to *queue* the work,
    not how long the GPU took to do it. Events are timestamps the GPU writes
    itself when it reaches them in the stream, so end - start is the GPU time
    spent between them.

    Event placement: the L2 flush goes BEFORE the start event, so its cost is
    not included. A side benefit is that the flush takes a while on the GPU, so
    the CPU gets ahead and has already queued the start event and run()'s
    kernels by the time the GPU reaches them; this keeps Python launch overhead
    out of the measurement.

    Synchronization: elapsed_time() is only valid once both events have
    actually happened on the GPU. We synchronize once after queuing all runs
    (rather than after each) so the CPU never stalls the GPU between runs.
    """
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()   # warm-up must be fully finished before timing

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_timed)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_timed)]
    for start, end in zip(starts, ends):
        if flush_buf is not None:
            flush_l2(flush_buf)   # read-only flush; see flush_l2 for why not zero_()
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()   # wait until every event has been reached
    return [s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends)]   # ms -> us


def median_us(times_us: list[float]) -> float:
    """Median, not mean: a few runs get delayed by things outside our control
    (other processes, clock changes, OS interrupts). The median ignores those
    outliers, while the mean would be pulled up by them."""
    return statistics.median(times_us)


def measure_latency_us(fn, flush_buf: torch.Tensor) -> float:
    """Median latency of fn() in us, with warm-up and run counts from the config."""
    runs = load_config()["runs"]
    return median_us(time_runs(fn, runs["warmup"], runs["timed"], flush_buf))


def measure_peak_bandwidth() -> float:
    """Measured peak HBM bandwidth in GB/s, from a large device-to-device copy.

    Why measured instead of the spec-sheet number (A100-80GB: 1935 GB/s PCIe,
    2039 GB/s SXM; check which variant the node has with nvidia-smi): no
    real kernel reaches the theoretical peak. A plain copy is the simplest
    memory-bound kernel there is, so it is a fair "best achievable" ceiling for
    % of peak. The buffer is far larger than L2, so no flush is needed.
    """
    runs = load_config()["runs"]
    n = load_config()["harness"]["peak_bw_copy_mb"] * MB
    src = torch.empty(n, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    t = median_us(time_runs(lambda: dst.copy_(src), runs["warmup"], runs["timed"]))
    del src, dst
    return bandwidth_gbs(2 * n, t)   # every byte is read once and written once


# ------------------------------------------------------------------- metrics

def bytes_moved(K: int, N: int, M: int, group_size: int, weight_bits: int, scheme: str) -> int:
    """Minimum bytes a kernel must move between HBM and the SM, per CLAUDE.md.

    INT4:  N*K/2 (weights) + scales + zeros + x in + y out
    FP16:  N*K*2 (weights)                  + x in + y out

    Scales are FP16 (2 bytes each, one per group per column). Zeros are counted
    as 4-bit values (half a byte each) for the asymmetric scheme and not at all
    for the symmetric scheme (zero is the constant 8). Libraries store zeros in
    different formats; using one formula for all of them keeps the bandwidth
    numbers comparable.
    """
    io = M * K * 2 + M * N * 2   # FP16 activations in and outputs out
    if weight_bits == 16:
        return K * N * 2 + io
    if weight_bits != 4:
        raise ValueError(f"unsupported WEIGHT_BITS={weight_bits}")
    n_groups = (K // group_size) * N
    scales = n_groups * 2
    zeros = n_groups // 2 if scheme == "asymmetric" else 0
    return K * N // 2 + scales + zeros + io


def bandwidth_gbs(n_bytes: int, latency_us: float) -> float:
    return n_bytes / (latency_us * 1e-6) / 1e9


# ---------------------------------------------------------------------- CSV

def append_row(path: Path, row: dict) -> None:
    """Append one row, writing the header if the file is new.

    Rows are written as soon as they are measured so a Slurm job that hits its
    time limit still leaves every finished measurement on disk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def default_csv_path(gpu: str, timestamp: str) -> Path:
    slug = "".join(c if c.isalnum() else "-" for c in gpu).strip("-").lower()
    return RESULTS_DIR / f"{slug}_{timestamp.replace(':', '')}.csv"


# ------------------------------------------------------------ implementations

def discover_impls() -> dict:
    """Every package in impls/ that exposes prepare() and run().

    The baselines (BASELINE_IMPLS) are placed first, in that order, so their
    latencies are known before the others need them for speedups. Names in EXCLUDED_IMPLS are skipped before
    import. An implementation whose library is not installed (ImportError,
    e.g. no vllm on a laptop) or whose native libraries fail to load (OSError)
    is skipped with a note.
    """
    import impls
    found = {}
    for info in pkgutil.iter_modules(impls.__path__):
        if info.name in EXCLUDED_IMPLS:
            print(f"[harness] skipping impls/{info.name}: excluded: {EXCLUDED_IMPLS[info.name]}")
            continue
        try:
            mod = importlib.import_module(f"impls.{info.name}")
        except (ImportError, OSError) as e:
            print(f"[harness] skipping impls/{info.name}: {e}")
            continue
        if callable(getattr(mod, "prepare", None)) and callable(getattr(mod, "run", None)):
            found[info.name] = mod
    return dict(sorted(found.items(), key=_baseline_first))


def _baseline_first(item) -> tuple:
    name = item[0]
    return (BASELINE_IMPLS.index(name) if name in BASELINE_IMPLS else len(BASELINE_IMPLS), name)


def compute_dtype(impl) -> torch.dtype:
    """The implementation's COMPUTE_DTYPE (FP16 if it does not declare one)."""
    return getattr(impl, "COMPUTE_DTYPE", torch.float16)


def prepare_input(impl, x: torch.Tensor) -> torch.Tensor:
    """The implementation's run() input for FP16 activations x (not timed)."""
    fn = getattr(impl, "prepare_input", None)
    return fn(x) if callable(fn) else x


def reference_weights(name: str, impl, qw) -> torch.Tensor:
    """Weights for the FP32 reference, from the implementation's own packed data."""
    if callable(getattr(impl, "reference_weights", None)):
        return impl.reference_weights()
    warnings.warn(f"{name} has no reference_weights(); checking against the canonical "
                  "(q, s, z) instead of its own packed weights")
    return dequantize(qw.q, qw.scales, qw.zeros, qw.group_size)
