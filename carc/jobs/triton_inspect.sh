#!/bin/bash
# Triton v2 diagnosis that needs no GPU performance counters
# (analysis/triton_inspect.py): register / spill counts from the compiler, and
# a small tile-size sweep. CARC blocks the counters Nsight Compute needs
# (ERR_NVGPUCTRPERM), so this is our main evidence for what to fix in v3.
#
# Submit from the repo root:
#
#   sbatch carc/jobs/triton_inspect.sh
#
# The log goes to results/triton_inspect_<jobid>.log, the data to
# results/triton_inspect_*.csv.

#SBATCH --job-name=int4-inspect
#SBATCH --account=prasanna_933
#SBATCH --partition=gpu
#SBATCH --constraint=a100-80gb
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=a100:1    # one A100; the constraint picks the 80 GB variant
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=results/triton_inspect_%j.log

set -eo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

echo "=== job $SLURM_JOB_ID on $(hostname) at $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
source carc/env_setup.sh
nvidia-smi --query-gpu=name,driver_version --format=csv

echo "=== Triton v2 tests ==="
python -m pytest tests/test_triton_kernel_v2.py -q -rs

echo "=== Triton v2 inspection ==="
python -m analysis.triton_inspect

echo "=== done at $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
