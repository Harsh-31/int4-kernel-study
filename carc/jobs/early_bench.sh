#!/bin/bash
# Early benchmark: GPU tests, then the selected implementations on the early sweep.
#
# Submit from the repo root (Slurm resolves the relative --output path and
# $SLURM_SUBMIT_DIR from where sbatch is called):
#
#   sbatch carc/jobs/early_bench.sh                        # cublas marlin (default)
#   INT4_IMPLS="cublas" sbatch carc/jobs/early_bench.sh    # choose implementations
#
# INT4_IMPLS is a space-separated list of folder names in impls/. Keep cublas
# in it: speedup_vs_fp16 is computed against cuBLAS and is empty without it.
#
# The log goes to results/early_bench_<jobid>.log.
#
# Environment location and caches (INT4_ENV_PREFIX, PIP_CACHE_DIR,
# CONDA_PKGS_DIRS, HF_HOME) are set in carc/env_setup.sh, which this job
# sources, so the paths live in one place. Override any of them at submit time:
#
#   INT4_ENV_PREFIX=/project2/.../other-env sbatch carc/jobs/early_bench.sh
#
# (sbatch passes your shell's environment variables to the job by default.)

#SBATCH --job-name=int4-early
#SBATCH --account=prasanna_933
#SBATCH --partition=gpu
#SBATCH --constraint=a100-80gb
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=a100:1    # one A100; the constraint picks the 80 GB variant
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=results/early_bench_%j.log

# -e: stop at the first failure, so the benchmark never runs after a failing
# test. pipefail: a failure inside a pipe counts. (No -u: see env_setup.sh.)
set -eo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1   # print progress to the log as it happens

# env_setup.sh defaults the CPU math libraries to 1 thread (for login nodes).
# Here we own SLURM_CPUS_PER_TASK cores, so use them: the random weights are
# generated on the CPU before being moved to the GPU.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

echo "=== job $SLURM_JOB_ID on $(hostname) at $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
source carc/env_setup.sh

echo
echo "=== nvidia-smi ==="
nvidia-smi

echo
echo "=== tests (GPU tests run here; they are skipped on machines without a GPU) ==="
# -rs lists anything still skipped, so a GPU test that skipped unexpectedly is visible.
python -m pytest tests -v -rs

echo
# Split the space-separated list into an array, so each name is passed to
# --impls as its own argument.
read -r -a IMPLS <<< "${INT4_IMPLS:-cublas marlin}"
echo "=== early benchmark: ${IMPLS[*]}, symmetric scheme ==="
python run_benchmark.py --early --scheme symmetric --impls "${IMPLS[@]}"

echo
echo "=== done at $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
