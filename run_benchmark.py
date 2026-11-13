"""Run the INT4 kernel benchmark (on a GPU node, through a Slurm job in carc/jobs/).

    python run_benchmark.py --early --scheme symmetric
    python run_benchmark.py --impls cublas marlin --scheme symmetric

For every group size, layer and implementation: quantize random weights, call
prepare() once (untimed), then for every batch size check correctness and,
only if it passes, time run() with common/harness.py. Each result is appended
to a CSV in results/ as soon as it is measured.

All shapes, batch sizes, group sizes and run counts come from
config/workloads.yaml.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import torch

from common.config import batch_sizes, group_sizes, load_config, quant_scheme
from common.correctness import check
from common.harness import (
    BASELINE_IMPL,
    append_row,
    bandwidth_gbs,
    bytes_moved,
    default_csv_path,
    discover_impls,
    make_l2_flush_buffer,
    measure_latency_us,
    measure_peak_bandwidth,
    reference_weights,
)
from common.quantize import random_activations, random_quantized_weights


def benchmark(impls: dict, sweep: str = "full", scheme: str | None = None,
              out_csv: Path | None = None) -> Path:
    """Benchmark every implementation on every (group size, layer, batch size).

    `impls` maps a name to an object with prepare()/run(); put BASELINE_IMPL
    first so speedups can be computed. `sweep` is "full" or "early".
    Returns the CSV path.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("needs a GPU; run it through a Slurm job in carc/jobs/")
    cfg = load_config()
    scheme = scheme or quant_scheme()
    gpu = torch.cuda.get_device_name(0)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out_csv = out_csv or default_csv_path(gpu, timestamp)

    flush_buf = make_l2_flush_buffer()
    peak = measure_peak_bandwidth()
    copy_mb = cfg["harness"]["peak_bw_copy_mb"]
    print(f"[bench] GPU: {gpu}")
    print(f"[bench] measured peak bandwidth: {peak:.0f} GB/s ({peak / 1000:.2f} TB/s), "
          f"from a {copy_mb} MB device-to-device copy counted as "
          f"{2 * copy_mb} MB moved (read + write)")
    print(f"[bench] sweep={sweep} scheme={scheme} impls={list(impls)}")
    print(f"[bench] writing {out_csv}")

    # inference_mode: no autograd bookkeeping, which would add CPU overhead
    # and memory to every run() call.
    with torch.inference_mode():
        for G in group_sizes(sweep):
            for layer, shape in cfg["layers"].items():
                K, N = shape["K"], shape["N"]
                qw = random_quantized_weights(K, N, G, scheme, device="cuda")
                fp16_latency = {}   # M -> cuBLAS latency for this layer and G

                for name, impl in impls.items():
                    # One-time setup (repacking etc.), outside all timing.
                    try:
                        impl.prepare(qw.q, qw.scales, qw.zeros, G)
                    except NotImplementedError as e:
                        print(f"[bench] skip {name} {layer} G={G}: {e}")
                        continue
                    w_ref = reference_weights(name, impl, qw)
                    bits = getattr(impl, "WEIGHT_BITS", 4)

                    for M in batch_sizes(sweep):
                        x = random_activations(M, K, device="cuda")
                        row = dict(impl=name, layer=layer, K=K, N=N, M=M, group_size=G,
                                   latency_us="", bandwidth_gbs="", pct_peak="",
                                   speedup_vs_fp16="", gpu=gpu, timestamp=timestamp)

                        # 1. Correctness first: never report a speed for a wrong result.
                        result = check(impl.run(x), x, w_ref)
                        row["correct"] = result.passed
                        if not result.passed:
                            print(f"[bench] FAIL {name:12s} {layer:9s} M={M:<4d} G={G:<4d} "
                                  f"impl {result.impl} vs fp16 {result.baseline}")
                            append_row(out_csv, row)
                            continue

                        # 2. Timing (warm-up, L2 flush, CUDA events, median).
                        lat = measure_latency_us(lambda: impl.run(x), flush_buf)
                        bw = bandwidth_gbs(bytes_moved(K, N, M, G, bits, scheme), lat)
                        if name == BASELINE_IMPL:
                            fp16_latency[M] = lat
                        row.update(
                            latency_us=round(lat, 3),
                            bandwidth_gbs=round(bw, 1),
                            pct_peak=round(100.0 * bw / peak, 1),
                            speedup_vs_fp16=(round(fp16_latency[M] / lat, 3)
                                             if M in fp16_latency else ""),
                        )
                        append_row(out_csv, row)
                        print(f"[bench] {name:12s} {layer:9s} M={M:<4d} G={G:<4d} "
                              f"{lat:9.2f} us {bw:8.1f} GB/s {row['pct_peak']:5}% peak "
                              f"speedup {row['speedup_vs_fp16']}")

                    w_ref = None   # free before the next implementation's prepare()
                del qw
                torch.cuda.empty_cache()
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the INT4 kernel benchmark.")
    parser.add_argument("--early", action="store_true",
                        help="only the early batch sizes and group size from the config")
    parser.add_argument("--scheme", choices=["asymmetric", "symmetric"], default=None,
                        help="defaults to quant_scheme in config/workloads.yaml")
    parser.add_argument("--impls", nargs="+", default=None,
                        help="subset of impls/ to run (default: every implemented one)")
    parser.add_argument("--out", type=Path, default=None,
                        help="CSV path (default: results/<gpu>_<time>.csv)")
    args = parser.parse_args()

    impls = discover_impls()
    if args.impls:
        missing = set(args.impls) - set(impls)
        if missing:
            parser.error(f"not found or not implemented in impls/: {sorted(missing)}; "
                         f"available: {sorted(impls)}")
        impls = {k: v for k, v in impls.items() if k in args.impls}   # keeps baseline first
    if BASELINE_IMPL not in impls:
        print(f"[bench] note: {BASELINE_IMPL} not selected, speedup_vs_fp16 will be empty")

    out = benchmark(impls, sweep="early" if args.early else "full",
                    scheme=args.scheme, out_csv=args.out)
    print(f"[bench] done: {out}")


if __name__ == "__main__":
    main()
