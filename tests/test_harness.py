"""Tests for common/harness.py and run_benchmark.py.

The byte-count, bandwidth and CSV logic run on CPU. The end-to-end sweep uses
small dummy implementations and needs a GPU (run via Slurm).
"""

import csv

import pytest
import torch

from common.config import group_sizes, load_config
from common.harness import (
    BASELINE_IMPL,
    CSV_COLUMNS,
    append_row,
    bandwidth_gbs,
    bytes_moved,
    default_csv_path,
    discover_impls,
    median_us,
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


def test_discover_impls_puts_baseline_first():
    names = list(discover_impls())
    if BASELINE_IMPL in names:
        assert names[0] == BASELINE_IMPL


# ------------------------------------------------------------- GPU end-to-end

class _DequantMatmul:
    """Dummy implementation: dequantize in prepare(), FP16 matmul in run()."""

    def __init__(self, weight_bits=4, corrupt=False):
        self.WEIGHT_BITS = weight_bits
        self.corrupt = corrupt

    def prepare(self, weights, scales, zeros, group_size):
        self.w = dequantize(weights, scales, zeros, group_size, torch.float16)

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
        "dummy": _DequantMatmul(),
        "broken": _DequantMatmul(corrupt=True),
    }
    benchmark(impls, sweep="early", scheme="symmetric", out_csv=out)
    with open(out) as f:
        rows = list(csv.DictReader(f))

    n = len(group_sizes("early")) * len(load_config()["layers"]) * len(load_config()["batch_sizes"]["early"])
    assert len(rows) == 3 * n
    for r in rows:
        if r["impl"] == "broken":
            assert r["correct"] == "False" and r["latency_us"] == ""
        else:
            assert r["correct"] == "True" and float(r["latency_us"]) > 0
            assert float(r["speedup_vs_fp16"]) > 0
        if r["impl"] == BASELINE_IMPL:
            assert float(r["speedup_vs_fp16"]) == 1.0
