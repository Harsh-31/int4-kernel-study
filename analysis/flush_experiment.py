"""Diagnostic: does the way we flush L2 change the measured latency?

Background
----------
common/harness.py flushes L2 before every timed run by WRITING zeros to a
256 MB buffer (flush_buf.zero_()). That evicts the weights, as intended, but
it may leave up to a full L2 (40 MB on A100) of DIRTY lines: data that must
be written back to HBM when evicted. If the timed kernel's own reads evict
those lines, the write-back traffic lands inside the timed region and makes
the kernel look slower than it is, most of all for small layers.

This script times the same kernels with four flush variants:

  write            flush_buf.zero_()                    (what the harness does today)
  read             read flush_buf, write nothing        (leaves only CLEAN lines)
  write_then_read  zero_() then read                    (dirty lines written back
                                                         BEFORE the timer starts)
  none             no flush                             (small layers may hit L2;
                                                         a lower bound, not honest)

How to read the result:
  write > read ~ write_then_read   -> dirty-line write-back inflates today's numbers
  write ~ read ~ write_then_read   -> the flush method does not matter

The timing loop is the harness's time_runs() with the flush step made
pluggable; it is copied here so common/harness.py stays unchanged until this
experiment says it should change. Run counts, sizes and seeds all come from
config/workloads.yaml.

Run (GPU node, via Slurm):  sbatch carc/jobs/flush_ab.sh
"""

import argparse
import csv
from datetime import datetime, timezone

import torch

from common.config import batch_sizes, group_sizes, load_config, quant_scheme
from common.correctness import check
from common.harness import (
    RESULTS_DIR,
    bandwidth_gbs,
    bytes_moved,
    compute_dtype,
    default_csv_path,
    discover_impls,
    make_l2_flush_buffer,
    measure_peak_bandwidth,
    median_us,
    prepare_input,
    reference_weights,
)
from common.quantize import random_activations, random_quantized_weights

FLUSH_MODES = ("write", "read", "write_then_read", "none")
COLUMNS = ["impl", "layer", "K", "N", "M", "group_size", "flush_mode", "latency_us",
           "min_us", "max_us", "bandwidth_gbs", "pct_peak", "gpu", "timestamp"]


def make_flush(mode: str, buf: torch.Tensor):
    """Return the function run before each timed iteration for this mode."""
    # Reading as int32 needs 4x fewer elements than uint8 for the same bytes.
    # sum() reads every byte of the buffer and writes only one scalar.
    words = buf.view(torch.int32)
    if mode == "write":
        return lambda: buf.zero_()
    if mode == "read":
        return lambda: words.sum()
    if mode == "write_then_read":
        def flush():
            buf.zero_()
            words.sum()
        return flush
    if mode == "none":
        return None
    raise ValueError(mode)


def time_runs(fn, n_warmup: int, n_timed: int, flush=None) -> list[float]:
    """common.harness.time_runs with a pluggable flush step (see module docstring)."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_timed)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_timed)]
    for start, end in zip(starts, ends):
        if flush is not None:
            flush()                     # before the start event: never timed
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends)]


def main() -> None:
    parser = argparse.ArgumentParser(description="L2 flush A/B experiment.")
    parser.add_argument("--impls", nargs="+", default=["cublas", "marlin"])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("needs a GPU; run it through carc/jobs/flush_ab.sh")

    cfg = load_config()
    runs = cfg["runs"]
    scheme = quant_scheme()
    G = group_sizes("early")[0]
    available = discover_impls()
    impls = {n: available[n] for n in args.impls if n in available}
    missing = sorted(set(args.impls) - set(impls))
    if missing:
        print(f"[flush] not available, skipped: {missing}")

    gpu = torch.cuda.get_device_name(0)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out_csv = RESULTS_DIR / ("flush_ab_" + default_csv_path(gpu, timestamp).name)
    buf = make_l2_flush_buffer()
    flushes = {mode: make_flush(mode, buf) for mode in FLUSH_MODES}
    peak = measure_peak_bandwidth()
    print(f"[flush] GPU {gpu}, measured peak {peak:.0f} GB/s, impls {list(impls)}, writing {out_csv}")

    rows = []
    with torch.inference_mode():
        for layer, shape in cfg["layers"].items():
            K, N = shape["K"], shape["N"]
            qw = random_quantized_weights(K, N, G, scheme, device="cuda")
            for name, impl in impls.items():
                impl.prepare(qw.q, qw.scales, qw.zeros, G)
                w_ref = reference_weights(name, impl, qw)
                bits = getattr(impl, "WEIGHT_BITS", 4)
                for M in batch_sizes("early"):
                    x = random_activations(M, K, device="cuda")
                    x_in = prepare_input(impl, x)
                    if not check(impl.run(x_in), x, w_ref, compute_dtype(impl)).passed:
                        print(f"[flush] {name} {layer} M={M} failed correctness; not timed")
                        continue
                    n_bytes = bytes_moved(K, N, M, G, bits, scheme)
                    # Alternate the modes in one pass per case so slow drifts
                    # (clocks, temperature) affect all four equally.
                    for mode in FLUSH_MODES:
                        t = time_runs(lambda: impl.run(x_in), runs["warmup"], runs["timed"],
                                      flushes[mode])
                        lat = median_us(t)
                        bw = bandwidth_gbs(n_bytes, lat)
                        rows.append(dict(impl=name, layer=layer, K=K, N=N, M=M, group_size=G,
                                         flush_mode=mode, latency_us=round(lat, 3),
                                         min_us=round(min(t), 3), max_us=round(max(t), 3),
                                         bandwidth_gbs=round(bw, 1),
                                         pct_peak=round(100 * bw / peak, 1),
                                         gpu=gpu, timestamp=timestamp))
            del qw
            torch.cuda.empty_cache()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    # Summary: one line per case, median latency for each mode, and how much
    # slower today's "write" flush is than the "read" flush.
    print()
    print(f"{'impl':8s} {'layer':9s} {'M':>3s} | " + " ".join(f"{m:>15s}" for m in FLUSH_MODES)
          + " | write/read")
    by_case = {}
    for r in rows:
        by_case.setdefault((r["impl"], r["layer"], r["M"]), {})[r["flush_mode"]] = r["latency_us"]
    for (name, layer, M), lat in by_case.items():
        ratio = lat["write"] / lat["read"] if lat.get("read") else float("nan")
        print(f"{name:8s} {layer:9s} {M:>3d} | "
              + " ".join(f"{lat.get(m, float('nan')):12.2f} us" for m in FLUSH_MODES)
              + f" | {ratio:6.2f}x")
    print(f"\n[flush] wrote {out_csv}")


if __name__ == "__main__":
    main()
