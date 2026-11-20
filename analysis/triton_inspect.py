"""Why is Triton v2 slow? Evidence that needs no GPU performance counters.

CARC does not let normal users read the GPU's performance counters, so
Nsight Compute cannot profile there (ERR_NVGPUCTRPERM). This script collects
the evidence that does not need counters:

Part A -- what the compiler produced, for v2's default settings.
  Triton reports, for every compiled kernel:
    regs     registers per thread. Each SM has 65536 registers, so e.g.
             128 regs/thread x 128 threads = 16384 per program -> at most 4
             programs fit on one SM at once (fewer programs = less latency hiding).
    spills   values that did not fit in registers and went to slow "local"
             memory (really HBM via the cache). Should be 0.
    shared   shared memory per program, in bytes.
  plus the measured latency and bandwidth, with the harness's timing rules.

Part B -- which setting causes the slowdown? (a small tile sweep)
  Times v2 with every combination of
    BLOCK_M     rows of x per program
    BLOCK_N     output columns per program
    num_warps   threads per program / 32
    num_stages  loads kept in flight by Triton's software pipelining
  checks each one for correctness first, and prints the fastest settings next
  to the default. If e.g. every fast setting has BLOCK_M = 16 at M = 64, the
  large BLOCK_M tile is the problem; if fast settings all use more stages,
  memory latency hiding is.

Layers and batch sizes come from `profiling:` in config/workloads.yaml, the
group size from the early sweep. Writes results/triton_inspect_<gpu>_<ts>.csv.

Run (GPU node, via Slurm):  sbatch carc/jobs/triton_inspect.sh
"""

import csv
import itertools
from datetime import datetime, timezone

import torch
import triton

from common.config import group_sizes, load_config, quant_scheme
from common.correctness import check
from common.harness import (
    RESULTS_DIR,
    bandwidth_gbs,
    bytes_moved,
    default_csv_path,
    make_l2_flush_buffer,
    median_us,
    time_runs,
)
from common.quantize import random_activations, random_quantized_weights
from impls import triton_kernel_v2 as v2

# Part B search space. These are kernel tile settings (not workload numbers).
BLOCK_MS = (16, 32, 64)
BLOCK_NS = (32, 64, 128)
NUM_WARPS = (4, 8)
NUM_STAGES = (2, 3, 4)
TOP = 5

COLUMNS = ["part", "layer", "K", "N", "M", "group_size", "block_m", "block_n",
           "num_warps", "num_stages", "split_k", "regs", "spills", "shared_bytes",
           "latency_us", "bandwidth_gbs", "correct", "error", "gpu", "timestamp"]


def set_knobs(block_m=None, block_n=None, num_warps=None, num_stages=None) -> None:
    v2.FORCE_BLOCK_M, v2.FORCE_BLOCK_N = block_m, block_n
    v2.NUM_WARPS, v2.NUM_STAGES = num_warps, num_stages


def kernel_info() -> dict:
    """Registers, spills, shared memory and launch settings of the last launch."""
    k = v2._state.get("last_kernel")
    if k is None:
        return {}
    if hasattr(k, "_init_handles"):
        k._init_handles()               # fills n_regs / n_spills (lazy in Triton 3)
    md = getattr(k, "metadata", None)

    def meta(name):
        if md is None:
            return None
        return md.get(name) if isinstance(md, dict) else getattr(md, name, None)

    return dict(regs=getattr(k, "n_regs", None), spills=getattr(k, "n_spills", None),
                shared_bytes=meta("shared"), num_warps=meta("num_warps"),
                num_stages=meta("num_stages"))


def measure(x, w_ref, n_bytes, flush_buf, runs) -> dict:
    """Correctness first, then latency with the harness's rules."""
    result = check(v2.run(x), x, w_ref, v2.COMPUTE_DTYPE)
    info = kernel_info()
    row = dict(info, correct=result.passed)
    if result.passed:
        lat = median_us(time_runs(lambda: v2.run(x), runs["warmup"], runs["timed"], flush_buf))
        row.update(latency_us=round(lat, 2), bandwidth_gbs=round(bandwidth_gbs(n_bytes, lat), 1))
    return row


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("needs a GPU; run it through carc/jobs/triton_inspect.sh")
    cfg = load_config()
    runs, prof = cfg["runs"], cfg["profiling"]
    G = group_sizes("early")[0]
    scheme = quant_scheme()
    gpu = torch.cuda.get_device_name(0)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out_csv = RESULTS_DIR / ("triton_inspect_" + default_csv_path(gpu, timestamp).name)
    flush_buf = make_l2_flush_buffer()
    print(f"[inspect] GPU {gpu}, triton {triton.__version__}, writing {out_csv}")

    rows = []
    with torch.inference_mode():
        for layer in prof["layers"]:
            K, N = cfg["layers"][layer]["K"], cfg["layers"][layer]["N"]
            qw = random_quantized_weights(K, N, G, scheme, device="cuda")
            v2.prepare(qw.q, qw.scales, qw.zeros, G)
            w_ref = v2.reference_weights()
            for M in prof["batch_sizes"]:
                x = random_activations(M, K, device="cuda")
                n_bytes = bytes_moved(K, N, M, G, 4, scheme)
                base = dict(layer=layer, K=K, N=N, M=M, group_size=G, gpu=gpu,
                            timestamp=timestamp)

                # ---- Part A: defaults
                set_knobs()
                row = dict(base, part="A_default", block_m=v2.block_m_for(M),
                           block_n=v2.block_n(),
                           split_k=v2.split_k_for(M, N, K, v2._state["block_k"],
                                                  v2._state["num_sms"]))
                row.update(measure(x, w_ref, n_bytes, flush_buf, runs))
                rows.append(row)

                # ---- Part B: sweep (BLOCK_M larger than needed for M only wastes work)
                max_bm = max(16, triton.next_power_of_2(M))
                for bm, bn, nw, ns in itertools.product(BLOCK_MS, BLOCK_NS, NUM_WARPS, NUM_STAGES):
                    if bm > max_bm:
                        continue
                    set_knobs(bm, bn, nw, ns)
                    row = dict(base, part="B_sweep", block_m=bm, block_n=bn,
                               split_k=v2.split_k_for(M, N, K, v2._state["block_k"],
                                                      v2._state["num_sms"]))
                    try:
                        row.update(measure(x, w_ref, n_bytes, flush_buf, runs))
                    except Exception as e:     # e.g. a tile too big for shared memory
                        row.update(correct=False, error=f"{type(e).__name__}: {str(e)[:80]}")
                    row.setdefault("num_warps", nw)
                    row.setdefault("num_stages", ns)
                    rows.append(row)
                set_knobs()
            del qw
            torch.cuda.empty_cache()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    def line(r):
        lat = r.get("latency_us")
        return (f"  BM={r['block_m']:<3} BN={r['block_n']:<4} warps={r.get('num_warps')!s:<2} "
                f"stages={r.get('num_stages')!s:<2} split={r['split_k']:<3} "
                f"regs={r.get('regs')!s:<4} spills={r.get('spills')!s:<4} "
                f"shared={r.get('shared_bytes')!s:<6} "
                + (f"{lat:9.2f} us {r['bandwidth_gbs']:7.1f} GB/s" if lat is not None
                   else f"   {'FAILED: ' + r['error'] if r.get('error') else 'INCORRECT'}"))

    print("\n=== Part A: v2 defaults (what the benchmark runs) ===")
    for r in rows:
        if r["part"] == "A_default":
            print(f"{r['layer']:9s} M={r['M']:<3}" + line(r))

    print(f"\n=== Part B: fastest {TOP} settings per case (default shown first) ===")
    for layer in prof["layers"]:
        for M in prof["batch_sizes"]:
            case = [r for r in rows if r["layer"] == layer and r["M"] == M]
            default = next(r for r in case if r["part"] == "A_default")
            sweep = sorted((r for r in case if r["part"] == "B_sweep" and r.get("latency_us")),
                           key=lambda r: r["latency_us"])
            failed = sum(1 for r in case if r["part"] == "B_sweep" and not r.get("latency_us"))
            print(f"\n{layer} M={M}  ({len(sweep)} settings timed, {failed} failed)")
            print("  default:" + line(default))
            for r in sweep[:TOP]:
                print("  " + line(r).lstrip())
    print(f"\n[inspect] wrote {out_csv}")


if __name__ == "__main__":
    main()
