#!/bin/bash
# Conda environment for the INT4 kernel study on USC CARC Discovery.
#
# Two ways to use this file (both from the repo root):
#
#   bash carc/env_setup.sh      # CREATE the environment and install packages (once)
#   source carc/env_setup.sh    # LOAD modules and ACTIVATE the environment
#                               # (what the Slurm jobs in carc/jobs/ do)
#
# Creating the environment only installs packages; it does no GPU work, so it
# can run on the login node. Torch will report no GPU there, which is expected.
#
# Options (environment variables; set any of them before running/sourcing to override):
#   INT4_ENV_PREFIX  full path of the environment
#                    (default: /project2/prasanna_933/hsharma9/int4-env).
#                    /project2 instead of home because vLLM + PyTorch take
#                    roughly 10 GB. Set it to an EMPTY string to use a named
#                    environment (INT4_ENV) in conda's default location instead.
#   INT4_ENV         environment name, only used when INT4_ENV_PREFIX is empty
#                    (default: int4-study)
#   PYTHON_VERSION   Python for a new environment (default: 3.11)
#   PIP_CACHE_DIR    pip download cache   (default: /scratch1/hsharma9/cache/pip)
#   CONDA_PKGS_DIRS  conda package cache  (default: /scratch1/hsharma9/cache/conda_pkgs)
#   HF_HOME          Hugging Face cache for model weights (default: /scratch1/hsharma9/hf_cache)
#
# The caches go on /scratch1 so multi-GB downloads (PyTorch/vLLM wheels, model
# weights) neither fill the home quota nor get re-downloaded on every install.

# "${VAR-default}" (no colon) keeps an explicitly empty INT4_ENV_PREFIX empty,
# which selects the named-environment mode below.
export INT4_ENV_PREFIX="${INT4_ENV_PREFIX-/project2/prasanna_933/hsharma9/int4-env}"
INT4_ENV="${INT4_ENV:-int4-study}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch1/hsharma9/cache/pip}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/scratch1/hsharma9/cache/conda_pkgs}"
export HF_HOME="${HF_HOME:-/scratch1/hsharma9/hf_cache}"
mkdir -p "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$HF_HOME"

if [[ -n "${INT4_ENV_PREFIX:-}" ]]; then
    CONDA_TARGET=(-p "$INT4_ENV_PREFIX")
    ENV_LABEL="$INT4_ENV_PREFIX"
else
    CONDA_TARGET=(-n "$INT4_ENV")
    ENV_LABEL="$INT4_ENV"
fi

load_modules() {
    module load conda/25.11.0 || return 1
    # cuda/12.6.3 depends on gcc/13.3.0 in some module trees; if loading it
    # directly fails, load the compiler first and retry.
    if ! module load cuda/12.6.3 2>/dev/null; then
        echo "[env] cuda/12.6.3 not found directly; loading gcc/13.3.0 first"
        module load gcc/13.3.0 && module load cuda/12.6.3 || return 1
    fi
    # Make `conda activate` work in non-interactive shells (Slurm jobs).
    eval "$(conda shell.bash hook)"
}

env_exists() {
    if [[ -n "${INT4_ENV_PREFIX:-}" ]]; then
        [[ -d "$INT4_ENV_PREFIX/conda-meta" ]]
    else
        conda env list | awk '{print $1}' | grep -qx "$INT4_ENV"
    fi
}

print_versions() {
    # importlib.metadata reads the installed version without importing the
    # package, so this works on the login node even for packages (vLLM,
    # BitBLAS) that probe for a GPU on import.
    python - <<'EOF'
import sys
from importlib.metadata import PackageNotFoundError, version

print(f"python        {sys.version.split()[0]}")
for pkg in ("torch", "triton", "vllm", "bitblas", "torchao", "pyyaml", "pytest"):
    try:
        print(f"{pkg:13s} {version(pkg)}")
    except PackageNotFoundError:
        print(f"{pkg:13s} NOT INSTALLED")

import torch
print(f"torch CUDA    {torch.version.cuda}   (GPU visible here: {torch.cuda.is_available()})")
EOF
}

# ---------------------------------------------------------------- source mode
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    load_modules || { echo "[env] module load failed" >&2; return 1; }
    conda activate "${CONDA_TARGET[1]}" || { echo "[env] cannot activate $ENV_LABEL; create it with: bash carc/env_setup.sh" >&2; return 1; }
    echo "[env] activated $ENV_LABEL ($(python --version))"
    echo "[env] PIP_CACHE_DIR=$PIP_CACHE_DIR CONDA_PKGS_DIRS=$CONDA_PKGS_DIRS HF_HOME=$HF_HOME"
    return 0
fi

# ---------------------------------------------------------------- create mode
# -e: stop at the first failing command; pipefail: a failure inside a pipe counts.
# (No -u: Lmod and conda's activate scripts reference unset variables.)
set -eo pipefail

load_modules

if env_exists; then
    echo "[env] $ENV_LABEL already exists; installing into it"
else
    echo "[env] creating $ENV_LABEL with Python $PYTHON_VERSION"
    [[ -n "${INT4_ENV_PREFIX:-}" ]] && mkdir -p "$(dirname "$INT4_ENV_PREFIX")"
    conda create -y "${CONDA_TARGET[@]}" "python=$PYTHON_VERSION"
fi
conda activate "${CONDA_TARGET[1]}"
python -m pip install --upgrade pip

# 1. vLLM first. vLLM pins an exact PyTorch (and Triton) version; installing it
#    first lets pip pick that version instead of whatever torch is newest.
#    Downloads are cached in $PIP_CACHE_DIR on /scratch1, so re-running is fast.
python -m pip install vllm

# 2. Freeze the torch/triton versions vLLM chose, and pass them as a pip
#    constraint below. Without this, installing BitBLAS or TorchAO could
#    silently replace torch with a version vLLM's compiled kernels don't match.
#    With it, pip fails loudly instead if a package needs a different torch.
CONSTRAINTS="$(mktemp)"
trap 'rm -f "$CONSTRAINTS"' EXIT
python -m pip freeze | grep -iE '^(torch|triton)==' > "$CONSTRAINTS"
echo "[env] pinning versions chosen by vLLM:"
cat "$CONSTRAINTS"

# 3. The rest. pyyaml loads config/workloads.yaml; pytest runs tests/.
python -m pip install -c "$CONSTRAINTS" triton bitblas torchao pyyaml pytest

echo
echo "[env] installed versions in $ENV_LABEL:"
print_versions
