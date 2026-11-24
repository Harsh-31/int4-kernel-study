"""Figures and tables for the report, from one results CSV.

    python -m analysis.plots                         # newest benchmark CSV in results/
    python -m analysis.plots results/<file>.csv      # a specific one

Writes to analysis/figures/<csv name>/, for every group size in the CSV:

  speedup_G<G>.png    speedup vs cuBLAS FP16 against batch size, one panel per
                      layer, with the IDEAL INT4 speedup (FP16 bytes / INT4 bytes)
                      as a dashed line: how close each kernel gets to the limit.
  pct_peak_G<G>.png   achieved bandwidth as % of the measured peak.
  roofline_G<G>.png   one small panel per implementation: achieved TFLOP/s vs
                      arithmetic intensity, under the memory and compute roofs.
  table_G<G>.md       the numbers behind the charts (latency, speedup, % peak),
                      one table per layer, ready to paste into the report.

CPU only (no GPU needed). Refuses CSVs measured with the old write-based L2
flush, whose latencies are inflated (see CLAUDE.md, "Benchmarking rules").
TorchAO is a BF16 kernel timed without its dtype casts: see the CLAUDE.md
reporting caveat, which the table files repeat.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")           # write files; no display needed
import matplotlib.pyplot as plt  # noqa: E402

from common.config import REPO_ROOT  # noqa: E402
from common.harness import RESULTS_DIR, bytes_moved  # noqa: E402

# Result CSVs from before this time used the write-based flush (CLAUDE.md).
FIRST_VALID_TIMESTAMP = "2026-11-17T00:00:00"
DIAGNOSTIC_PREFIXES = ("flush_ab_", "triton_inspect_")
BASELINES = ("cublas", "cublas_bf16")

# Fixed colour per implementation (dataviz reference palette, categorical
# slots in order; colour follows the implementation, never its rank).
# Each also has its own marker, so identity never depends on colour alone.
STYLE = {
    "marlin":                 ("#2a78d6", "o", "-"),
    "torchao":                ("#eb6834", "s", "-"),
    "triton_kernel":          ("#1baf7a", "v", "--"),
    "triton_kernel_v2":       ("#eda100", "^", "--"),
    "triton_kernel_v2_tuned": ("#e87ba4", "D", "--"),
    "cuda_kernel":            ("#008300", "P", "--"),
}
LABEL = {
    "cublas": "cuBLAS FP16", "cublas_bf16": "cuBLAS BF16", "marlin": "Marlin",
    "torchao": "TorchAO (BF16)*", "triton_kernel": "Triton v1", "triton_kernel_v2": "Triton v2",
    "triton_kernel_v2_tuned": "Triton v2 tuned", "cuda_kernel": "CUDA v1",
}
INK, MUTED, GRID = "#2b2b29", "#6b6a64", "#e4e3de"

# Peak math throughput from NVIDIA's A100 datasheet (dense, no sparsity).
# Memory roof uses the MEASURED peak bandwidth from the CSV instead.
COMPUTE_ROOFS_TFLOPS = {
    "A100": [("FP16 tensor cores", 312.0), ("FP32 CUDA cores", 19.5)],
}


def newest_csv() -> Path:
    files = [p for p in RESULTS_DIR.glob("*.csv") if not p.name.startswith(DIAGNOSTIC_PREFIXES)]
    if not files:
        raise SystemExit(f"no benchmark CSVs in {RESULTS_DIR}")
    return max(files, key=lambda p: p.stat().st_mtime)


def load(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} is empty")
    old = [r["timestamp"] for r in rows if r["timestamp"] < FIRST_VALID_TIMESTAMP]
    if old:
        raise SystemExit(f"{path} was measured with the old write-based L2 flush "
                         f"(timestamp {old[0]}); its numbers must not be reported")
    out = []
    for r in rows:
        if r["correct"] != "True" or not r["latency_us"]:
            continue            # never plot a speed for an incorrect result
        for k in ("K", "N", "M", "group_size"):
            r[k] = int(r[k])
        for k in ("latency_us", "bandwidth_gbs", "pct_peak"):
            r[k] = float(r[k])
        r["speedup_vs_fp16"] = float(r["speedup_vs_fp16"]) if r["speedup_vs_fp16"] else None
        out.append(r)
    return out


def style_axes(ax, title: str) -> None:
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def batch_axis(ax, Ms) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ms)
    ax.set_xticklabels([str(m) for m in Ms])
    ax.set_xlabel("batch size M (tokens)", color=MUTED, fontsize=8)


def by_case(rows):
    """{(layer, impl): sorted rows}, layers in config order."""
    d = defaultdict(list)
    for r in rows:
        d[(r["layer"], r["impl"])].append(r)
    for v in d.values():
        v.sort(key=lambda r: r["M"])
    return d


def layer_order(rows):
    seen = []
    for r in rows:
        if r["layer"] not in seen:
            seen.append(r["layer"])
    return seen


def plot_lines(rows, G, metric, ylabel, title, path, ideal=False, ref=None):
    layers = layer_order(rows)
    cases = by_case(rows)
    Ms = sorted({r["M"] for r in rows})
    fig, axes = plt.subplots(1, len(layers), figsize=(3.4 * len(layers), 3.4), sharey=True)
    axes = axes if len(layers) > 1 else [axes]
    for ax, layer in zip(axes, layers):
        any_row = next(r for r in rows if r["layer"] == layer)
        style_axes(ax, f"{layer}  (K={any_row['K']}, N={any_row['N']})")
        batch_axis(ax, Ms)
        if ideal:   # FP16 bytes / INT4 bytes: the speedup a perfect INT4 kernel would get
            K, N = any_row["K"], any_row["N"]
            ys = [bytes_moved(K, N, m, G, 16, "symmetric") / bytes_moved(K, N, m, G, 4, "symmetric")
                  for m in Ms]
            ax.plot(Ms, ys, color=MUTED, linestyle=":", linewidth=1.5, label="ideal INT4 (bytes ratio)")
        if ref is not None:
            ax.axhline(ref, color=MUTED, linewidth=1.0, label="cuBLAS FP16")
        for impl, (color, marker, ls) in STYLE.items():
            pts = [r for r in cases.get((layer, impl), []) if r[metric] is not None]
            if pts:
                ax.plot([r["M"] for r in pts], [r[metric] for r in pts], color=color,
                        marker=marker, markersize=5, linestyle=ls, linewidth=1.5,
                        label=LABEL[impl])
        if metric == "pct_peak":
            for impl in BASELINES[:1]:
                pts = cases.get((layer, impl), [])
                if pts:
                    ax.plot([r["M"] for r in pts], [r[metric] for r in pts], color=MUTED,
                            marker="x", markersize=5, linewidth=1.5, label=LABEL[impl])
    axes[0].set_ylabel(ylabel, color=MUTED, fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5), frameon=False,
               fontsize=8, labelcolor=INK, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    fig.text(0.99, 0.005, "* TorchAO: BF16 kernel, timed without its FP16<->BF16 casts",
             color=MUTED, fontsize=7, ha="right")
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_roofline(rows, G, path):
    gpu = rows[0]["gpu"]
    peak_gbs = next(r["bandwidth_gbs"] / r["pct_peak"] * 100 for r in rows if r["pct_peak"] > 0)
    roofs = next((v for k, v in COMPUTE_ROOFS_TFLOPS.items() if k in gpu), [])
    impls = ["cublas"] + [i for i in STYLE if any(r["impl"] == i for r in rows)]
    cols = 4
    nrows = -(-len(impls) // cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(3.2 * cols, 3.0 * nrows), sharex=True, sharey=True,
                             squeeze=False)
    for ax in axes.flat[len(impls):]:
        ax.set_visible(False)
    for ax, impl in zip(axes.flat, impls):
        color = STYLE.get(impl, (MUTED,))[0]
        style_axes(ax, LABEL[impl])
        ax.set_xscale("log")
        ax.set_yscale("log")
        pts = []
        for r in rows:
            if r["impl"] != impl:
                continue
            bits = 16 if impl in BASELINES else 4
            flops = 2 * r["M"] * r["N"] * r["K"]
            ai = flops / bytes_moved(r["K"], r["N"], r["M"], G, bits, "symmetric")
            pts.append((ai, flops / (r["latency_us"] * 1e-6) / 1e12))
        xs = [p[0] for p in pts]
        lo, hi = 0.5, 2 * max(max(xs), 300)
        grid = [lo * (hi / lo) ** (i / 60) for i in range(61)]
        top = max(t for _, t in roofs) if roofs else float("inf")
        ax.plot(grid, [min(peak_gbs * x / 1e3, top) for x in grid], color=MUTED, linewidth=1.2)
        for name, t in roofs:
            ax.axhline(t, color=GRID if name.startswith("FP16") else MUTED, linewidth=0.8,
                       linestyle="--")
        ax.scatter(xs, [p[1] for p in pts], s=28, color=color, edgecolors="white", linewidths=1.0,
                   zorder=3)
    # With shared axes matplotlib shows x numbers only on the bottom row, which
    # is partly empty here; label every visible panel instead.
    for ax in axes.flat[:len(impls)]:
        ax.tick_params(labelbottom=True)
        ax.set_xlabel("arithmetic intensity (FLOP / byte)", color=MUTED, fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("achieved TFLOP/s", color=MUTED, fontsize=8)
    roof_text = ", ".join(f"{n} {t:g}" for n, t in roofs)
    fig.suptitle(f"Roofline, G={G}: memory roof = measured {peak_gbs:.0f} GB/s"
                 + (f"; dashed = {roof_text} TFLOP/s" if roofs else ""),
                 color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_table(rows, G, path, source):
    cases = by_case(rows)
    Ms = sorted({r["M"] for r in rows})
    impls = list(BASELINES) + list(STYLE)
    lines = [f"# Results, group size {G}", "",
             f"Source: `{source}` ({rows[0]['gpu']}). Latency in us; speedup vs cuBLAS FP16 "
             "in brackets; % of measured peak bandwidth after the slash.", "",
             "TorchAO is a BF16 kernel timed as its kernel only (no FP16<->BF16 casts, no "
             "Python dispatch); a real FP16 model using TorchAO would be slower.", ""]
    for layer in layer_order(rows):
        any_row = next(r for r in rows if r["layer"] == layer)
        lines += [f"## {layer} (K={any_row['K']}, N={any_row['N']})", "",
                  "| implementation | " + " | ".join(f"M={m}" for m in Ms) + " |",
                  "|---|" + "---|" * len(Ms)]
        for impl in impls:
            pts = {r["M"]: r for r in cases.get((layer, impl), [])}
            if not pts:
                continue
            cells = []
            for m in Ms:
                r = pts.get(m)
                if r is None:
                    cells.append("-")
                else:
                    sp = f" ({r['speedup_vs_fp16']:.2f}x)" if r["speedup_vs_fp16"] else ""
                    cells.append(f"{r['latency_us']:.1f}{sp} / {r['pct_peak']:.0f}%")
            lines.append(f"| {LABEL[impl]} | " + " | ".join(cells) + " |")
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else newest_csv()
    rows = load(src)
    out = REPO_ROOT / "analysis" / "figures" / src.stem
    out.mkdir(parents=True, exist_ok=True)
    for G in sorted({r["group_size"] for r in rows}):
        g_rows = [r for r in rows if r["group_size"] == G]
        plot_lines(g_rows, G, "speedup_vs_fp16", "speedup vs cuBLAS FP16 (x)",
                   f"Speedup over cuBLAS FP16, INT4 group size {G}",
                   out / f"speedup_G{G}.png", ideal=True, ref=1.0)
        plot_lines(g_rows, G, "pct_peak", "% of measured peak bandwidth",
                   f"Achieved memory bandwidth, group size {G}", out / f"pct_peak_G{G}.png")
        plot_roofline(g_rows, G, out / f"roofline_G{G}.png")
        write_table(g_rows, G, out / f"table_G{G}.md", src.name)
    print(f"[plots] {src.name} -> {out}")
    for p in sorted(out.iterdir()):
        print(f"  {p.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
