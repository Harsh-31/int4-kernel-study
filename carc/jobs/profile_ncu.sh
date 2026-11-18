#!/bin/bash
# Nsight Compute (ncu) profiles of the cases listed under `profiling:` in
# config/workloads.yaml, then one summary table (analysis/ncu_summary.py).
#
# Submit from the repo root:
#
#   sbatch carc/jobs/profile_ncu.sh
#   NCU=/path/to/ncu sbatch carc/jobs/profile_ncu.sh           # use a specific ncu
#   NCU_MODULE=<module name> sbatch carc/jobs/profile_ncu.sh   # module that provides ncu
#
# Finding ncu (first match wins): $NCU, `ncu` on PATH, the module in
# $NCU_MODULE, then common install folders. A module is loaded only inside a
# subshell, just to find the ncu program's path, so it never changes this
# job's CUDA libraries or Python environment (ncu is a separate program that
# attaches to our Python process; nothing gets linked into the env).
#
# Output, in results/ncu_<jobid>/, one set per case <impl>__<layer>__M<M>:
#   .ncu-rep  full report (open in the Nsight Compute GUI; git-ignored)
#   .txt      ncu's human-readable details page, including its own advice
#   .csv      every raw metric (read by analysis/ncu_summary.py)
# The log (results/profile_ncu_<jobid>.log) ends with the summary table.

#SBATCH --job-name=int4-ncu
#SBATCH --account=prasanna_933
#SBATCH --partition=gpu
#SBATCH --constraint=a100-80gb
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=a100:1    # one A100; the constraint picks the 80 GB variant
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=results/profile_ncu_%j.log

set -eo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

echo "=== job $SLURM_JOB_ID on $(hostname) at $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
source carc/env_setup.sh
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem --format=csv

# ---------------------------------------------------------------- find ncu
if [ -z "$NCU" ]; then
    NCU="$(command -v ncu || true)"
fi
if [ -z "$NCU" ] && [ -n "$NCU_MODULE" ]; then
    NCU="$(module load "$NCU_MODULE" >/dev/null 2>&1 && command -v ncu || true)"
fi
if [ -z "$NCU" ]; then
    for candidate in /usr/local/cuda*/bin/ncu /opt/nvidia/nsight-compute/*/ncu \
                     /usr/local/NVIDIA-Nsight-Compute*/ncu; do
        if [ -x "$candidate" ]; then NCU="$candidate"; fi   # last match = newest version
    done
fi
if [ -z "$NCU" ]; then
    echo "ERROR: Nsight Compute (ncu) not found. On the login node, run"
    echo "    module spider nsight      and      module spider cuda"
    echo "and resubmit with NCU_MODULE=<a module that provides ncu> or NCU=/path/to/ncu."
    exit 1
fi
echo "=== ncu: $NCU ==="
"$NCU" --version

# ------------------------------------------------------------------ cases
OUT="results/ncu_${SLURM_JOB_ID}"
mkdir -p "$OUT"
# One "impl layer M" line per case, read from the config.
mapfile -t CASES < <(python - <<'EOF'
from common.config import load_config
p = load_config()["profiling"]
for impl in p["impls"]:
    for layer in p["layers"]:
        for m in p["batch_sizes"]:
            print(impl, layer, m)
EOF
)
echo "=== ${#CASES[@]} cases -> $OUT ==="

# ncu options:
#   --profile-from-start off  record only between torch.cuda.profiler.start()/stop()
#                             in analysis/profile_case.py, i.e. one run() call
#   --set full                every section (memory, compute, occupancy, warp stalls, ...)
#   --cache-control all       flush caches before each kernel (ncu's default; same
#                             idea as our harness: weights come from HBM)
#   --clock-control none      leave clocks as they are, as in the benchmark (ncu's
#                             default would lock them to the lower base clock)
failed=0
for case in "${CASES[@]}"; do
    read -r impl layer m <<< "$case"
    name="${impl}__${layer}__M${m}"
    echo
    echo "--- $name ---"
    if "$NCU" --profile-from-start off --set full --cache-control all --clock-control none \
              -f -o "$OUT/$name" \
              python -m analysis.profile_case --impl "$impl" --layer "$layer" --M "$m" \
              > "$OUT/$name.ncu.log" 2>&1; then
        "$NCU" --import "$OUT/$name.ncu-rep" --page raw --csv > "$OUT/$name.csv"
        "$NCU" --import "$OUT/$name.ncu-rep" --page details > "$OUT/$name.txt"
        grep -E "^\[profile\]|==PROF== Disconnected" "$OUT/$name.ncu.log" || true
    else
        failed=1
        echo "FAILED; last lines of $OUT/$name.ncu.log:"
        tail -n 15 "$OUT/$name.ncu.log"
        if grep -q "ERR_NVGPUCTRPERM" "$OUT/$name.ncu.log"; then
            echo "ERROR: this node does not allow access to GPU performance counters."
            echo "Ask CARC support to enable profiling (NVreg_RestrictProfilingToAdminUsers=0)."
            exit 1
        fi
    fi
done

echo
echo "=== summary ==="
python -m analysis.ncu_summary "$OUT"
echo
echo "=== done at $(date -u '+%Y-%m-%d %H:%M:%S UTC') (failed cases: $failed) ==="
