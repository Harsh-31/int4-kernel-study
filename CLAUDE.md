# INT4 Kernel Study — Project Context

## What this project is
A single-GPU study of INT4 weight-only (W4A16) matrix-multiply kernels for LLM decode.
We compare existing kernels (Marlin, BitBLAS, TorchAO) against cuBLAS FP16 and our own
Triton and CUDA kernels, and explain the results with roofline and Nsight analysis.
Goal is a correct, well-explained study — NOT beating Marlin.

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

## Benchmarking rules (every implementation, no exceptions)
- Exclude one-time setup from timing (weight repacking, compilation, autotuning).
- Warm up before timing.
- Time with CUDA events; synchronize before reading the timer.
- Flush L2 cache between runs so weights come from HBM.
- Report the median of at least 100 runs.
- Record: latency (us), effective bandwidth (GB/s), % of measured peak bandwidth, speedup vs cuBLAS FP16.
- Bytes for INT4 = N*K/2 (weights) + scales/zeros + activations in + outputs out.
- Save raw results to `results/*.csv` with columns:
  impl, layer, K, N, M, group_size, latency_us, bandwidth_gbs, pct_peak, speedup_vs_fp16, correct, gpu, timestamp

## Correctness rules
- Reference: dequantize the SAME packed INT4 weights in PyTorch (w = (q - z) * s), matmul in FP32.
- Build each library's reference from that library's own packed weights and scales.
- Pass if max-abs and relative-L2 error are <= 2x cuBLAS FP16's error against the same reference.
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
│   ├── marlin/               # via vLLM's Marlin ops
│   ├── bitblas/
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
- Adding an implementation = adding one folder with that interface; the harness loops over all of them.
- Cluster-specific files (module loads, Slurm scripts) stay in `carc/`, so the code itself is not tied to CARC.

## Working style
- Small, reviewable changes. Explain non-obvious GPU code in comments.
- Ask before adding new dependencies.
- When unsure about a library API, check its docs or source instead of guessing.
- For our CUDA kernel: the team writes the core logic; help by explaining, reviewing and debugging.
