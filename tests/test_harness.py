"""Tests for common/harness.py and run_benchmark.py.

The byte-count, bandwidth and CSV logic run on CPU. The end-to-end sweep uses
small dummy implementations and needs a GPU (run via Slurm).
"""

import csv
import types

import pytest
import torch

from common.config import group_sizes, load_config
from common.harness import (
    BASELINE_IMPL,
    BASELINE_IMPLS,
    BF16_BASELINE_IMPL,
    CSV_COLUMNS,
    append_row,
    bandwidth_gbs,
    bytes_moved,
    compute_dtype,
    default_csv_path,
    discover_impls,
    flush_l2,
    make_l2_flush_buffer,
    median_us,
    time_runs,
    prepare_input,
)
from common.quantize import dequantize
from run_benchmark import benchmark


def _attn_out():
    s = load_config()["layers"]["attn_out"]
    return s["K"], s["N"]


def test_bytes_int4_asymmetric():
    K, N = _attn_out()
    G, M = group_sizes("early")[0], 1
    groups = (K // G) * N
    expected = K * N // 2 + groups * 2 + groups // 2 + M * K * 2 + M * N * 2
    assert bytes_moved(K, N, M, G, 4, "asymmetric") == expected


def test_bytes_int4_symmetric_has_no_zeros():
    K, N = _attn_out()
    G, M = group_sizes("early")[0], 1
    groups = (K // G) * N
    asym = bytes_moved(K, N, M, G, 4, "asymmetric")
    assert asym - bytes_moved(K, N, M, G, 4, "symmetric") == groups // 2


def test_bytes_fp16_ignores_group_size_and_is_about_4x():
    K, N = _attn_out()
    a = bytes_moved(K, N, 1, group_sizes("full")[0], 16, "symmetric")
    b = bytes_moved(K, N, 1, group_sizes("full")[-1], 16, "symmetric")
    assert a == b
    assert 3.5 < a / bytes_moved(K, N, 1, group_sizes("early")[0], 4, "symmetric") < 4.0


def test_bad_weight_bits_raises():
    with pytest.raises(ValueError):
        bytes_moved(128, 128, 1, 32, 8, "symmetric")


def test_bandwidth_units():
    # 1e9 bytes in 1e6 us (= 1 s) is 1 GB/s
    assert bandwidth_gbs(10**9, 10**6) == pytest.approx(1.0)


def test_median_ignores_outliers():
    assert median_us([10.0, 11.0, 12.0, 1000.0, 9.0]) == 11.0


def test_append_row_writes_header_once(tmp_path):
    path = tmp_path / "sub" / "r.csv"
    row = {c: "" for c in CSV_COLUMNS} | {"impl": "x", "correct": True}
    append_row(path, row)
    append_row(path, row)
    with open(path) as f:
        rows = list(csv.reader(f))
    assert rows[0] == CSV_COLUMNS
    assert len(rows) == 3


def test_default_csv_path_is_filesystem_safe():
    p = default_csv_path("NVIDIA A100-SXM4-80GB", "2026-11-12T10:00:00+00:00")
    assert p.parent.name == "results"
    assert ":" not in p.name and " " not in p.name and p.suffix == ".csv"


def test_discover_impls_puts_baselines_first_in_order():
    names = list(discover_impls())
    present = [b for b in BASELINE_IMPLS if b in names]
    assert names[:len(present)] == present
    assert BASELINE_IMPLS == (BASELINE_IMPL, BF16_BASELINE_IMPL)


def test_compute_dtype_and_prepare_input_defaults():
    plain = types.SimpleNamespace()                          # declares nothing
    x = torch.zeros(2, 4, dtype=torch.float16)
    assert compute_dtype(plain) == torch.float16
    assert prepare_input(plain, x) is x
    bf16 = types.SimpleNamespace(COMPUTE_DTYPE=torch.bfloat16,
                                 prepare_input=lambda t: t.to(torch.bfloat16))
    assert compute_dtype(bf16) == torch.bfloat16
    assert prepare_input(bf16, x).dtype == torch.bfloat16


# ------------------------------------------------------------- GPU end-to-end

class _DequantMatmul:
    """Dummy implementation: dequantize in prepare(), matmul in `dtype` in run()."""

    def __init__(self, weight_bits=4, corrupt=False, dtype=torch.float16):
        self.WEIGHT_BITS = weight_bits
        self.COMPUTE_DTYPE = dtype
        self.corrupt = corrupt

    def prepare(self, weights, scales, zeros, group_size):
        self.w = dequantize(weights, scales, zeros, group_size, self.COMPUTE_DTYPE)

    def prepare_input(self, x):
        return x.to(self.COMPUTE_DTYPE).contiguous()

    def run(self, x):
        y = x @ self.w
        return y + 1.0 if self.corrupt else y

    def reference_weights(self):
        return self.w


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (run via Slurm)")
def test_sweep_end_to_end(tmp_path):
    out = tmp_path / "r.csv"
    impls = {
        BASELINE_IMPL: _DequantMatmul(weight_bits=16),
        BF16_BASELINE_IMPL: _DequantMatmul(weight_bits=16, dtype=torch.bfloat16),
        "dummy": _DequantMatmul(),
        "dummy_bf16": _DequantMatmul(dtype=torch.bfloat16),
        "broken": _DequantMatmul(corrupt=True),
    }
    benchmark(impls, sweep="early", scheme="symmetric", out_csv=out)
    with open(out) as f:
        rows = list(csv.DictReader(f))

    assert list(rows[0]) == CSV_COLUMNS
    n = len(group_sizes("early")) * len(load_config()["layers"]) * len(load_config()["batch_sizes"]["early"])
    assert len(rows) == len(impls) * n
    for r in rows:
        if r["impl"] == "broken":
            assert r["correct"] == "False" and r["latency_us"] == ""
            continue
        assert r["correct"] == "True" and float(r["latency_us"]) > 0
        assert float(r["speedup_vs_fp16"]) > 0
        # cublas runs before cublas_bf16 exists, so its bf16 speedup is empty.
        if r["impl"] != BASELINE_IMPL:
            assert float(r["speedup_vs_bf16"]) > 0
        assert r["compute_dtype"] == ("bfloat16" if "bf16" in r["impl"] else "float16")
        if r["impl"] == BASELINE_IMPL:
            assert float(r["speedup_vs_fp16"]) == 1.0
        if r["impl"] == BF16_BASELINE_IMPL:
            assert float(r["speedup_vs_bf16"]) == 1.0


# ------------------------------------------------------------- excluded impls

def test_excluded_impls_are_never_discovered_or_imported(monkeypatch):
    """Excluded implementations stay out even if their library IS importable
    (bitblas is still installed in the CARC environment), and are skipped
    before import so their native libraries are never loaded."""
    import sys
    import types

    from common.harness import EXCLUDED_IMPLS

    for name in EXCLUDED_IMPLS:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))   # pretend installed
        monkeypatch.delitem(sys.modules, f"impls.{name}", raising=False)
    found = discover_impls()
    for name in EXCLUDED_IMPLS:
        assert name not in found
        assert f"impls.{name}" not in sys.modules


def test_run_benchmark_refuses_excluded_impls(monkeypatch, capsys):
    import run_benchmark
    from common.harness import EXCLUDED_IMPLS

    for name, reason in EXCLUDED_IMPLS.items():
        monkeypatch.setattr("sys.argv", ["run_benchmark.py", "--impls", "cublas", name])
        with pytest.raises(SystemExit) as exc:
            run_benchmark.main()
        assert exc.value.code == 2   # argparse error
        assert f"{name} is excluded: {reason}" in capsys.readouterr().err


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (run via Slurm)")
def test_flush_reads_but_never_writes_the_buffer():
    """The flush must be read-only (see flush_l2): the buffer stays all zeros."""
    buf = make_l2_flush_buffer()
    buf[::4096] = 7                     # a recognisable pattern
    before = buf.clone()
    flush_l2(buf)
    torch.cuda.synchronize()
    assert torch.equal(buf, before)
    times = time_runs(lambda: None, n_warmup=1, n_timed=3, flush_buf=buf)
    assert len(times) == 3 and all(t >= 0 for t in times)
    assert torch.equal(buf, before)
