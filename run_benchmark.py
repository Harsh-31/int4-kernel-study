"""Run the INT4 kernel benchmark (on a GPU node, through a Slurm job in carc/jobs/).

    python run_benchmark.py --early
    python run_benchmark.py --impls cublas cublas_bf16 marlin torchao

For every group size, layer and implementation: quantize random weights, call
prepare() once (untimed), then for every batch size prepare the input
(prepare_input(), e.g. a BF16 copy; untimed), check correctness against cuBLAS
in the implementation's own compute dtype and, only if it passes, time run()
with common/harness.py. Each result is appended to a CSV in results/ as soon as
it is measured.

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
    BF16_BASELINE_IMPL,
    EXCLUDED_IMPLS,
    append_row,
    bandwidth_gbs,
    bytes_moved,
    compute_dtype,
    default_csv_path,
    discover_impls,
    make_l2_flush_buffer,
    measure_latency_us,
    measure_peak_bandwidth,
    prepare_input,
    reference_weights,
)

DTYPE_NAMES = {torch.float16: "float16", torch.bfloat16: "bfloat16"}
from common.quantize import random_activations, random_quantized_weights


def benchmark(impls: dict, sweep: str = "full", scheme: str | None = None,
              out_csv: Path | None = None) -> Path:
    """Benchmark every implementation on every (group size, layer, batch size).

    `impls` maps a name to an object with prepare()/run(); put the baselines
    (cublas, cublas_bf16) first so speedups can be computed. `sweep` is "full"
    or "early".
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
                # M -> baseline latency for this layer and G
                base_latency = {BASELINE_IMPL: {}, BF16_BASELINE_IMPL: {}}

                for name, impl in impls.items():
                    # One-time setup (repacking etc.), outside all timing.
                    try:
                        impl.prepare(qw.q, qw.scales, qw.zeros, G)
                    except NotImplementedError as e:
                        print(f"[bench] skip {name} {layer} G={G}: {e}")
                        continue
                    w_ref = reference_weights(name, impl, qw)
                    bits = getattr(impl, "WEIGHT_BITS", 4)
                    dtype = compute_dtype(impl)

                    for M in batch_sizes(sweep):
                        x = random_activations(M, K, device="cuda")
                        # Input in the form run() expects (e.g. BF16), made
                        # OUTSIDE the timed region so casts are not timed.
                        x_in = prepare_input(impl, x)
                        row = dict(impl=name, layer=layer, K=K, N=N, M=M, group_size=G,
                                   latency_us="", bandwidth_gbs="", pct_peak="",
                                   speedup_vs_fp16="", speedup_vs_bf16="",
                                   compute_dtype=DTYPE_NAMES[dtype], gpu=gpu, timestamp=timestamp)

                        # 1. Correctness first, against cuBLAS in the same dtype:
                        #    never report a speed for a wrong result.
                        result = check(impl.run(x_in), x, w_ref, dtype)
                        row["correct"] = result.passed
                        if not result.passed:
                            print(f"[bench] FAIL {name:12s} {layer:9s} M={M:<4d} G={G:<4d} "
                                  f"impl {result.impl} vs cuBLAS {DTYPE_NAMES[dtype]} {result.baseline}")
                            append_row(out_csv, row)
                            continue

                        # 2. Timing (warm-up, L2 flush, CUDA events, median).
                        lat = measure_latency_us(lambda: impl.run(x_in), flush_buf)
                        bw = bandwidth_gbs(bytes_moved(K, N, M, G, bits, scheme), lat)
                        if name in base_latency:
                            base_latency[name][M] = lat
                        fp16_lat = base_latency[BASELINE_IMPL].get(M)
                        bf16_lat = base_latency[BF16_BASELINE_IMPL].get(M)
                        row.update(
                            latency_us=round(lat, 3),
                            bandwidth_gbs=round(bw, 1),
                            pct_peak=round(100.0 * bw / peak, 1),
                            speedup_vs_fp16=round(fp16_lat / lat, 3) if fp16_lat else "",
                            speedup_vs_bf16=round(bf16_lat / lat, 3) if bf16_lat else "",
                        )
                        append_row(out_csv, row)
                        print(f"[bench] {name:12s} {layer:9s} M={M:<4d} G={G:<4d} "
                              f"{lat:9.2f} us {bw:8.1f} GB/s {row['pct_peak']:5}% peak "
                              f"speedup fp16 {row['speedup_vs_fp16']} bf16 {row['speedup_vs_bf16']}")

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
                        help="subset of impls/ to run (default: every implemented one, "
                             f"except the excluded {sorted(EXCLUDED_IMPLS)})")
    parser.add_argument("--out", type=Path, default=None,
                        help="CSV path (default: results/<gpu>_<time>.csv)")
    args = parser.parse_args()

    # Excluded implementations (common.harness.EXCLUDED_IMPLS) are never in the
    # default list, and naming one explicitly is an error, not a silent skip.
    excluded = [name for name in (args.impls or []) if name in EXCLUDED_IMPLS]
    if excluded:
        parser.error("; ".join(f"{name} is excluded: {EXCLUDED_IMPLS[name]}" for name in excluded))

    impls = discover_impls()
    if args.impls:
        missing = set(args.impls) - set(impls)
        if missing:
            parser.error(f"not found or not implemented in impls/: {sorted(missing)}; "
                         f"available: {sorted(impls)}")
        impls = {k: v for k, v in impls.items() if k in args.impls}   # keeps baseline first
    for base, col in ((BASELINE_IMPL, "speedup_vs_fp16"), (BF16_BASELINE_IMPL, "speedup_vs_bf16")):
        if base not in impls:
            print(f"[bench] note: {base} not selected, {col} will be empty")

    out = benchmark(impls, sweep="early" if args.early else "full",
                    scheme=args.scheme, out_csv=args.out)
    print(f"[bench] done: {out}")


if __name__ == "__main__":
    main()
