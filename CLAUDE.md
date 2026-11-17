# INT4 Kernel Study — Project Context

## What this project is
A single-GPU study of INT4 weight-only (W4A16) matrix-multiply kernels for LLM decode.
We compare existing kernels (Marlin, TorchAO) against cuBLAS FP16 and our own
Triton and CUDA kernels, and explain the results with roofline and Nsight analysis.
Goal is a correct, well-explained study — NOT beating Marlin.

### Excluded implementations
- **BitBLAS — excluded: requires CUDA 12.** BitBLAS 0.1.0's bundled TVM is linked against
  the CUDA 12 runtime (`libcudart.so.12`, `libnvrtc.so.12`; `libnvrtc.so.12` confirmed missing
  on CARC). Our environment is CUDA 13 only, and we do not mix CUDA 12 and 13 in one environment.
  `impls/bitblas/` stays in the repo for reference but is never run: it is listed in
  `EXCLUDED_IMPLS` in `common/harness.py`, so it is not in the default implementation list,
  `run_benchmark.py --impls bitblas` is refused, it is not in any job script, and
  `tests/test_bitblas.py` skips with "excluded: requires CUDA 12". Do not install CUDA 12
  libraries to re-enable it.

## Hardware
- USC CARC Discovery cluster, Slurm scheduler.
- Final numbers: NVIDIA A100-80GB (request with `--constraint=a100-80gb`).
- Development: A40 (debug partition is OK).
- NEVER run GPU work on the login node. All GPU runs go through Slurm job scripts in `carc/jobs/`.

## Workloads (Llama-3-8B layer shapes, K -> N)
- attn_qkv:  4096 -> 6144
- attn_out:  4096 -> 4096
- mlp_up:    4096 -> 28672  (gate + up fused)
- mlp_down:  14336 -> 4096
- Batch sizes (M): 1, 2, 4, 8, 16, 32, 64, 128, 256 (early benchmark: 1, 16, 64)
- Group sizes: 32, 64, 128 (early benchmark: 128)
- Use random FP16 matrices with a fixed seed for speed tests. Real model weights only for quality checks.
- Quantization scheme: **symmetric** by default (`quant_scheme` in `config/workloads.yaml`):
  w = (q - 8) * s, q in 0..15, one FP16 scale per group. Asymmetric (integer zero points)
  remains available.
- Every scale is rounded UP to a value exactly representable in both FP16 and BF16
  (`common/quantize.py`), so FP16 and BF16 kernels receive identical weights and scales.
  Rounding up keeps every weight within half a step; it changes the overall quantization
  error by less than 0.3%.

## Benchmarking rules (every implementation, no exceptions)
- Exclude one-time setup from timing (weight repacking, compilation, autotuning), and
  input preparation (e.g. casting FP16 activations to BF16 for a BF16 kernel, via
  `prepare_input()`).
- Warm up before timing.
- Time with CUDA events; synchronize before reading the timer.
- Flush L2 cache between runs so weights come from HBM. Flush by READING a buffer larger
  than L2 (`flush_l2` in `common/harness.py`), never by writing one: a write flush leaves dirty
  lines whose write-back to HBM lands inside the timed region. On the A100 that inflated
  latencies by up to 1.65x (`analysis/flush_experiment.py`). Result CSVs from before
  2026-11-16 used the write flush and must not be reported.
- Report the median of at least 100 runs.
- Record: latency (us), effective bandwidth (GB/s), % of measured peak bandwidth, speedup vs
  cuBLAS FP16 (`impls/cublas`) and vs cuBLAS BF16 (`impls/cublas_bf16`).
- Bytes for INT4 = N*K/2 (weights) + scales/zeros + activations in + outputs out.
- Save raw results to `results/*.csv` with columns:
  impl, layer, K, N, M, group_size, latency_us, bandwidth_gbs, pct_peak, speedup_vs_fp16,
  speedup_vs_bf16, correct, compute_dtype, gpu, timestamp

### Reporting caveats
- **TorchAO is timed as its kernel only.** TorchAO's own `linear()` for its INT4 tensor casts
  FP16 activations to BF16, calls `aten._weight_int4pack_mm` through tensor-subclass Python
  dispatch, and casts the output back. For consistency with Marlin (whose op is also called
  directly), `impls/torchao` calls `aten._weight_int4pack_mm` directly on a BF16 input prepared
  outside the timed region. Its numbers therefore EXCLUDE the two dtype casts and the Python
  dispatch overhead that a real FP16 model using TorchAO would pay. State this whenever
  TorchAO numbers are reported. (`tests/test_torchao.py` checks the direct call computes
  exactly what TorchAO's `linear()` computes.)

## Correctness rules
- Reference: dequantize the SAME packed INT4 weights in PyTorch (using the library's own
  formula, e.g. w = (q - z) * s), matmul in FP32.
- Build each library's reference from that library's own packed weights and scales.
- **Precision-matched rule:** every implementation declares its compute dtype
  (`COMPUTE_DTYPE`, FP16 by default, or BF16). It passes if its max-abs and relative-L2 error
  are both <= 2x the error of cuBLAS **in that same dtype** on the same weights and the same
  reference (cuBLAS FP16 for FP16 kernels, cuBLAS BF16 for BF16 kernels).
- Why: BF16 has 8 significant bits to FP16's 11, so a BF16-only kernel (TorchAO's tinygemm) has
  roughly 7x the error of cuBLAS FP16 by construction, and failed the old FP16-only rule for its
  precision, not for a bug. Comparing each kernel with cuBLAS in its own dtype still catches real
  errors (wrong layout, wrong scales, bad accumulation) while judging both precisions fairly.
  Report BF16 kernels' accuracy as BF16-level, not FP16-level.
- Never report a speed number for an implementation that fails correctness.

## Repo layout
```
int4-kernel-study/
├── CLAUDE.md
├── README.md
├── config/
│   └── workloads.yaml        # layer sizes, batch sizes, group sizes, runs, seed
├── common/
│   ├── harness.py            # timing: warm-up, L2 flush, CUDA events, median
│   ├── correctness.py        # FP32 reference + error checks
│   └── quantize.py           # random INT4 weights + scales/zeros
├── impls/
│   ├── cublas/               # FP16 baseline (no quantization)
│   ├── cublas_bf16/          # BF16 baseline (no quantization), for BF16 kernels
│   ├── marlin/               # via vLLM's Marlin ops
│   ├── bitblas/              # EXCLUDED (requires CUDA 12); kept for reference, never run
│   ├── torchao/
│   ├── triton_kernel/        # our Triton kernel
│   └── cuda_kernel/          # our CUDA kernel
├── carc/
│   ├── env_setup.sh          # conda environment + module loads
│   └── jobs/                 # Slurm job scripts
├── tests/                    # correctness tests (pytest) for every implementation
├── results/                  # raw CSVs (never edit by hand)
└── analysis/                 # plots, roofline, tables for the report
```

### Rules for this layout
- All numbers (shapes, batch sizes, group sizes, run counts, seed) live ONLY in `config/workloads.yaml`. Never hard-code them elsewhere.
- Shared logic lives in `common/`. Implementation folders never re-implement timing or correctness.
- Every folder in `impls/` exposes the same interface:
  - `prepare(weights, scales, zeros, group_size)` — one-time setup such as repacking. NOT timed.
  - `run(x)` — the matrix multiply only. Timed by `common/harness.py`.
  - Optional: `COMPUTE_DTYPE` (FP16 default, or BF16; selects the correctness baseline),
    `prepare_input(x)` (e.g. FP16 -> BF16 cast; NOT timed), `reference_weights()` (weights
    decoded from the library's own packed data), `WEIGHT_BITS` (4 default, 16 for cuBLAS).
- Adding an implementation = adding one folder with that interface; the harness loops over all of them
  (except those in `EXCLUDED_IMPLS`, see "Excluded implementations" above).
- Cluster-specific files (module loads, Slurm scripts) stay in `carc/`, so the code itself is not tied to CARC.

## Working style
- Small, reviewable changes. Explain non-obvious GPU code in comments.
- Ask before adding new dependencies.
- When unsure about a library API, check its docs or source instead of guessing.
- For our CUDA kernel: the team writes the core logic; help by explaining, reviewing and debugging.
