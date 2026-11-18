"""Summarize Nsight Compute results into one table.

    python -m analysis.ncu_summary results/ncu_<jobid>

Reads every <impl>__<layer>__M<M>.csv in the folder (written by
carc/jobs/profile_ncu.sh with `ncu --import ... --page raw --csv`) and prints,
for every kernel of every case:

  time_us    kernel duration measured by ncu (one call, caches flushed by ncu)
  DRAM%      HBM throughput as % of the GPU's peak. Near 100% = memory-bound and
             doing well; low = the kernel is NOT keeping memory busy.
  DRAM GB/s  bytes actually read from + written to HBM / duration
  MB read    bytes actually read from HBM (compare with the weight size)
  SM%        how busy the SMs' busiest unit is, as % of its peak
  occ%       achieved occupancy: active warps / maximum warps per SM
  regs       registers per thread (high -> fewer warps fit on an SM)
  grid       number of thread blocks (programs) launched
  spill      bytes of local-memory traffic (register spills); 0 is good
  top stalls the three most common reasons warps were waiting, with the
             average number of cycles per issued instruction spent on each

Metrics missing from a report are shown as "-", so this works with any ncu
version / section set. The full human-readable reports (with ncu's own advice)
are the <case>.txt files next to the CSVs.
"""

import csv
import re
import sys
from pathlib import Path

TIME = "gpu__time_duration.sum"
DRAM_PCT = "dram__throughput.avg.pct_of_peak_sustained_elapsed"
DRAM_READ = "dram__bytes_read.sum"
DRAM_WRITE = "dram__bytes_write.sum"
SM_PCT = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
OCC = "sm__warps_active.avg.pct_of_peak_sustained_active"
REGS = "launch__registers_per_thread"
GRID = "launch__grid_size"
SPILL = re.compile(r"local_op_(ld|st)\.sum$|op_local_(ld|st)\.sum$")
STALL = re.compile(r"^smsp__average_warps?_issue_stalled_([a-z_]+?)_per_issue_active\.ratio$")

UNIT_SCALE = {  # to base units: seconds, bytes
    "nsecond": 1e-9, "usecond": 1e-6, "msecond": 1e-3, "second": 1.0,
    "byte": 1.0, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9,
    "KB": 1e3, "MB": 1e6, "GB": 1e9,
}


def _num(text: str) -> float | None:
    try:
        return float(text.replace(",", ""))
    except (AttributeError, ValueError):
        return None


def _base(row: dict, units: dict, name: str) -> float | None:
    """Metric value converted to seconds / bytes using ncu's unit row."""
    value = _num(row.get(name, ""))
    if value is None:
        return None
    return value * UNIT_SCALE.get(units.get(name, ""), 1.0)


def read_case(path: Path) -> list[dict]:
    """One dict per profiled kernel, with the summary fields filled in."""
    with open(path, newline="") as f:
        lines = [line for line in f if line.startswith('"')]   # skip ncu's ==PROF== lines
    rows = list(csv.DictReader(lines))
    if not rows:
        return []
    units, kernels = rows[0], rows[1:]      # ncu's raw CSV: second row holds the units
    out = []
    for r in kernels:
        t = _base(r, units, TIME)
        rd = _base(r, units, DRAM_READ)
        wr = _base(r, units, DRAM_WRITE)
        spill = [_base(r, units, c) for c in r if SPILL.search(c)]
        stalls = []
        for col, val in r.items():
            m = STALL.match(col)
            if m and _num(val) is not None:
                stalls.append((_num(val), m.group(1)))
        stalls.sort(reverse=True)
        name = re.sub(r"<.*", "", r.get("Kernel Name", "?")).replace("void ", "")
        out.append(dict(
            kernel=name[:34],
            time_us=t * 1e6 if t else None,
            dram_pct=_num(r.get(DRAM_PCT, "")),
            dram_gbs=((rd or 0) + (wr or 0)) / t / 1e9 if t and rd is not None else None,
            mb_read=rd / 1e6 if rd is not None else None,
            sm_pct=_num(r.get(SM_PCT, "")),
            occ=_num(r.get(OCC, "")),
            regs=_num(r.get(REGS, "")),
            grid=_num(r.get(GRID, "")),
            spill=sum(s for s in spill if s is not None) if spill else None,
            stalls=", ".join(f"{n} {v:.1f}" for v, n in stalls[:3]) or "-",
        ))
    return out


def fmt(value, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def main() -> None:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if folder is None or not folder.is_dir():
        raise SystemExit("usage: python -m analysis.ncu_summary results/ncu_<jobid>")
    files = sorted(folder.glob("*__*__M*.csv"))
    if not files:
        raise SystemExit(f"no <impl>__<layer>__M<M>.csv files in {folder}")

    print(f"{'case':30s} {'kernel':34s} {'time_us':>8s} {'DRAM%':>6s} {'GB/s':>6s} "
          f"{'MB read':>8s} {'SM%':>5s} {'occ%':>5s} {'regs':>5s} {'grid':>6s} "
          f"{'spill':>6s}  top stalls (cycles/instr)")
    for path in files:
        impl, layer, m = path.stem.split("__")
        case = f"{impl} {layer} {m}"
        kernels = read_case(path)
        if not kernels:
            print(f"{case:30s} (no kernels recorded; see {path.with_suffix('.txt').name})")
            continue
        for k in kernels:
            print(f"{case:30s} {k['kernel']:34s} {fmt(k['time_us'], '8.1f')} "
                  f"{fmt(k['dram_pct'], '6.1f')} {fmt(k['dram_gbs'], '6.0f')} "
                  f"{fmt(k['mb_read'], '8.2f')} {fmt(k['sm_pct'], '5.1f')} "
                  f"{fmt(k['occ'], '5.1f')} {fmt(k['regs'], '5.0f')} {fmt(k['grid'], '6.0f')} "
                  f"{fmt(k['spill'], '6.0f')}  {k['stalls']}")
            case = ""          # print the case name only on its first kernel


if __name__ == "__main__":
    main()
