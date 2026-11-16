#!/bin/bash
# L2 flush A/B experiment (analysis/flush_experiment.py): times cuBLAS and
# Marlin on the early sweep with four flush variants, to check whether the
# harness's write-based flush inflates latencies.
#
# Submit from the repo root:
#
#   sbatch carc/jobs/flush_ab.sh
#   FLUSH_IMPLS="cublas" sbatch carc/jobs/flush_ab.sh     # choose implementations
#
# The log goes to results/flush_ab_<jobid>.log, the data to results/flush_ab_*.csv.

#SBATCH --job-name=int4-flush-ab
#SBATCH --account=prasanna_933
#SBATCH --partition=gpu
#SBATCH --constraint=a100-80gb
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=a100:1    # one A100; the constraint picks the 80 GB variant
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=results/flush_ab_%j.log

# -e: stop at the first failure. pipefail: a failure inside a pipe counts.
set -eo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

echo "=== job $SLURM_JOB_ID on $(hostname) at $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
source carc/env_setup.sh
nvidia-smi --query-gpu=name,memory.total,clocks.sm,clocks.mem --format=csv

read -r -a IMPLS <<< "${FLUSH_IMPLS:-cublas marlin}"
echo "=== flush experiment: ${IMPLS[*]} ==="
python -m analysis.flush_experiment --impls "${IMPLS[@]}"

echo "=== done at $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
