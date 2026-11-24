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
# Both modes check that `python` is the environment's own interpreter and fail
# loudly if not. When sourced from a batch script (non-interactive shell), a
# failure EXITS the job, so nothing runs with the wrong Python. In an
# interactive shell it only returns an error, so your terminal stays open.
#
# Options (environment variables; set any of them before running/sourcing to override):
#   INT4_ENV_PREFIX  full path of the environment
#                    (default: /project2/prasanna_933/hsharma9/int4-env).
#                    /project2 instead of home because vLLM + PyTorch take
#                    roughly 10 GB. Set it to an EMPTY string to use a named
#                    environment (INT4_ENV) in conda's default location instead.
#   INT4_ENV         environment name, only used when INT4_ENV_PREFIX is empty
#                    (default: int4-study)
#   PYTHON_VERSION   Python for a NEW environment (default: 3.11, the version of
#                    the existing int4-env). Ignored if the environment exists.
#   PIP_CACHE_DIR    pip download cache   (default: /scratch1/hsharma9/cache/pip)
#   CONDA_PKGS_DIRS  conda package cache  (default: /scratch1/hsharma9/cache/conda_pkgs)
#   HF_HOME          Hugging Face cache for model weights (default: /scratch1/hsharma9/hf_cache)
#   OPENBLAS_NUM_THREADS, OMP_NUM_THREADS, MKL_NUM_THREADS
#                    CPU threads for numpy/torch math libraries (default: 1 each).
#                    Job scripts can raise them to $SLURM_CPUS_PER_TASK.
#
# The caches go on /scratch1 so multi-GB downloads (PyTorch/vLLM wheels, model
# weights) neither fill the home quota nor get re-downloaded on every install.
#
# CUDA toolkit
# ------------
# We do NOT load a CARC cuda module. The CUDA compiler (nvcc) comes from pip,
# at exactly the CUDA version PyTorch was built with (PyTorch declares it as a
# `cuda-toolkit==X` dependency). Those wheels install into
# site-packages/nvidia/cu<major>/{bin,include,lib}, which is used as CUDA_HOME.
# A CARC module would give a different CUDA version than PyTorch's, and kernels
# we compile (our CUDA kernel) must match the runtime torch loads.
#
# BitBLAS is NOT installed: it is excluded from the study because bitblas 0.1.0
# needs the CUDA 12 runtime (libcudart.so.12, libnvrtc.so.12), and we do not
# mix CUDA 12 and 13 in one environment. See CLAUDE.md.

# "${VAR-default}" (no colon) keeps an explicitly empty INT4_ENV_PREFIX empty,
# which selects the named-environment mode below.
export INT4_ENV_PREFIX="${INT4_ENV_PREFIX-/project2/prasanna_933/hsharma9/int4-env}"
INT4_ENV="${INT4_ENV:-int4-study}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch1/hsharma9/cache/pip}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/scratch1/hsharma9/cache/conda_pkgs}"
export HF_HOME="${HF_HOME:-/scratch1/hsharma9/hf_cache}"
mkdir -p "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$HF_HOME"

# CPU thread pools. OpenBLAS (loaded by numpy), OpenMP and MKL each start one
# thread per core by default: 64 on a CARC node. Login nodes cap the number of
# threads/processes per user, so importing numpy there fails to create its
# threads and Python crashes. One thread each is plenty for installing and
# checking versions; GPU benchmarks barely use the CPU math libraries.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# Ignore ~/.local/lib/pythonX.Y/site-packages. Python adds that folder to the
# import path automatically, so a `pip install --user torch` from some other
# session would silently shadow the environment's own torch.
export PYTHONNOUSERSITE=1

if [[ -n "${INT4_ENV_PREFIX:-}" ]]; then
    CONDA_TARGET=(-p "$INT4_ENV_PREFIX")
    ENV_LABEL="$INT4_ENV_PREFIX"
else
    CONDA_TARGET=(-n "$INT4_ENV")
    ENV_LABEL="$INT4_ENV"
fi

# CUDA extras installed alongside PyTorch's own CUDA libraries:
#   nvcc   the compiler driver, ptxas, nvlink, fatbinary
#   nvvm   cicc + libnvvm: the device-code compiler nvcc calls internally
#          (nvcc alone cannot compile a kernel without it)
#   crt    nvcc's built-in headers (crt/host_config.h, ...)
#   cudart CUDA runtime headers and library (cuda_runtime.h, libcudart)
#   cccl   CUB/Thrust/libcu++ headers, used by many CUDA kernels
CUDA_EXTRAS="nvcc,nvvm,crt,cudart,cccl"

env_fail() {
    # Print an error. In a non-interactive shell (Slurm job, `bash` script)
    # exit immediately; in an interactive shell just return 1 so the user's
    # terminal is not closed. Callers write `env_fail "..." || return 1`.
    echo "[env] ERROR: $*" >&2
    [[ $- == *i* ]] || exit 1
    return 1
}

load_modules() {
    module load conda/25.11.0 || return 1
    # nvcc needs a host C++ compiler. gcc 13 is supported by CUDA 13.0 and new
    # enough for PyTorch's C++17 headers; the OS gcc may be older. Not fatal:
    # without it, nvcc falls back to the system gcc.
    module load gcc/13.3.0 || echo "[env] warning: gcc/13.3.0 not loaded; using system gcc" >&2
    # Make `conda activate` work in non-interactive shells (Slurm jobs).
    eval "$(conda shell.bash hook)"
}

activate_env() {
    # sbatch copies the submitting shell's environment into the job. If an
    # environment was already active there, `conda activate` of the same
    # environment is treated as a re-activation and does not move its bin/
    # back to the front of PATH, even though `module load` just put other
    # bin/ folders (with their own python) ahead of it. Deactivating
    # everything first makes this a clean activation every time.
    local i
    for i in 1 2 3 4 5; do
        [[ "${CONDA_SHLVL:-0}" -gt 0 ]] || break
        conda deactivate
    done
    conda activate "${CONDA_TARGET[1]}" || return 1
    hash -r   # forget any python path bash cached before activation
}

check_python() {
    # Verify `python` is the environment's own interpreter, and show where it is.
    local expected_prefix py sys_prefix
    if [[ -n "${INT4_ENV_PREFIX:-}" ]]; then
        expected_prefix="$INT4_ENV_PREFIX"
    else
        expected_prefix="${CONDA_PREFIX:-}"
        if [[ "$(basename "$expected_prefix")" != "$INT4_ENV" ]]; then
            echo "[env] CONDA_PREFIX='$expected_prefix' is not the '$INT4_ENV' environment" >&2
            return 1
        fi
    fi
    # `|| true`: under a caller's `set -e`, a failure here must still reach the
    # diagnostic message below instead of silently exiting.
    py="$(command -v python)" || true
    sys_prefix="$(python -c 'import sys; print(sys.prefix)' 2>/dev/null)" || true
    echo "[env] which python:   $py"
    echo "[env] python version: $(python --version 2>&1)"
    echo "[env] sys.prefix:     $sys_prefix"
    if [[ -z "$py" || "$(realpath "$py")" != "$(realpath "$expected_prefix/bin/python")" \
          || "$(realpath "$sys_prefix")" != "$(realpath "$expected_prefix")" ]]; then
        echo "[env] expected the interpreter of $expected_prefix; every python on PATH:" >&2
        type -a python >&2 || echo "  (none)" >&2
        return 1
    fi
}

cuda_root() {
    # site-packages/nvidia/cu<major> for the CUDA major version torch was built with.
    python - <<'EOF'
import sysconfig
import torch
major = (torch.version.cuda or "").split(".")[0]
print(f"{sysconfig.get_paths()['purelib']}/nvidia/cu{major}" if major else "")
EOF
}

setup_cuda_home() {
    # Point CUDA_HOME/PATH at the pip-installed toolkit. CUDA_HOME is what
    # torch.utils.cpp_extension and most build scripts use to find nvcc and
    # the CUDA headers.
    local root
    root="$(cuda_root)" || return 1
    if [[ -n "$root" && -x "$root/bin/nvcc" ]]; then
        export CUDA_HOME="$root"
        export PATH="$root/bin:$PATH"
    else
        echo "[env] warning: no pip-installed nvcc at '$root/bin/nvcc'; rerun: bash carc/env_setup.sh" >&2
    fi
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
    # package, so this works on the login node even for packages (vLLM) that
    # probe for a GPU on import.
    python - <<'EOF'
import sys
from importlib.metadata import PackageNotFoundError, version

print(f"python            {sys.version.split()[0]}  ({sys.executable})")
for pkg in ("torch", "triton", "vllm", "torchao", "cuda-toolkit",
            "nvidia-cuda-nvcc", "pyyaml", "pytest", "matplotlib"):
    try:
        print(f"{pkg:17s} {version(pkg)}")
    except PackageNotFoundError:
        print(f"{pkg:17s} NOT INSTALLED")

import torch
print(f"torch CUDA        {torch.version.cuda}   (GPU visible here: {torch.cuda.is_available()})")
EOF
}

# ---------------------------------------------------------------- source mode
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    load_modules || { env_fail "module load failed" || return 1; }
    activate_env || { env_fail "cannot activate $ENV_LABEL; create it with: bash carc/env_setup.sh" || return 1; }
    check_python || { env_fail "python is not the interpreter of $ENV_LABEL (see above)" || return 1; }
    setup_cuda_home
    echo "[env] activated $ENV_LABEL"
    echo "[env] CUDA_HOME=${CUDA_HOME:-unset} ($(nvcc --version 2>/dev/null | grep -o 'release [0-9.]*' || echo 'nvcc not found'))"
    echo "[env] threads: OPENBLAS=$OPENBLAS_NUM_THREADS OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS"
    echo "[env] PIP_CACHE_DIR=$PIP_CACHE_DIR CONDA_PKGS_DIRS=$CONDA_PKGS_DIRS HF_HOME=$HF_HOME"
    return 0
fi

# ---------------------------------------------------------------- create mode
# -e: stop at the first failing command; pipefail: a failure inside a pipe counts.
# (No -u: Lmod and conda's activate scripts reference unset variables.)
set -eo pipefail

load_modules

if env_exists; then
    echo "[env] $ENV_LABEL already exists; installing into it (its Python version is kept)"
else
    echo "[env] creating $ENV_LABEL with Python $PYTHON_VERSION"
    [[ -n "${INT4_ENV_PREFIX:-}" ]] && mkdir -p "$(dirname "$INT4_ENV_PREFIX")"
    conda create -y "${CONDA_TARGET[@]}" "python=$PYTHON_VERSION"
fi
activate_env
# Must pass BEFORE installing anything, or pip would install into the wrong Python.
check_python || env_fail "python is not the interpreter of $ENV_LABEL (see above)"
python -m pip install --upgrade pip

# 1. vLLM first. vLLM pins an exact PyTorch (and Triton) version; installing it
#    first lets pip pick that version instead of whatever torch is newest.
#    Downloads are cached in $PIP_CACHE_DIR on /scratch1, so re-running is fast.
python -m pip install vllm

# 2. Freeze the torch/triton versions vLLM chose, and pass them as a pip
#    constraint below. Without this, installing TorchAO could
#    silently replace torch with a version vLLM's compiled kernels don't match.
#    With it, pip fails loudly instead if a package needs a different torch.
CONSTRAINTS="$(mktemp)"
trap 'rm -f "$CONSTRAINTS"' EXIT
python -m pip freeze | grep -iE '^(torch|triton)==' > "$CONSTRAINTS"
echo "[env] pinning versions chosen by vLLM:"
cat "$CONSTRAINTS"

# 3. The rest. pyyaml loads config/workloads.yaml; pytest runs tests/;
#    matplotlib draws the report figures (analysis/plots.py).
#    (No bitblas: excluded, see the CUDA toolkit note at the top.)
python -m pip install -c "$CONSTRAINTS" triton torchao pyyaml pytest matplotlib

# 4. CUDA compiler matching PyTorch. Read the exact cuda-toolkit version torch
#    depends on (e.g. torch 2.11.0+cu130 requires cuda-toolkit==13.0.2) and
#    install the compiler extras of that same version, so nvcc, its headers and
#    the runtime libraries torch already uses all come from one CUDA release.
CUDA_TOOLKIT_VERSION="$(python - <<'EOF'
import re
from importlib.metadata import requires
for req in requires("torch") or []:
    m = re.match(r"cuda-toolkit(\[[^\]]*\])?\s*==\s*([0-9.]+)", req)
    if m:
        print(m.group(2))
        break
EOF
)"
if [[ -z "$CUDA_TOOLKIT_VERSION" ]]; then
    env_fail "installed torch does not declare a cuda-toolkit==X dependency; cannot pick a matching nvcc automatically"
fi
echo "[env] installing cuda-toolkit[$CUDA_EXTRAS]==$CUDA_TOOLKIT_VERSION (matches torch)"
python -m pip install -c "$CONSTRAINTS" "cuda-toolkit[$CUDA_EXTRAS]==$CUDA_TOOLKIT_VERSION"

echo
echo "[env] installed versions in $ENV_LABEL:"
print_versions

echo
setup_cuda_home
echo "[env] CUDA_HOME=$CUDA_HOME"
echo "[env] nvcc --version:"
nvcc --version
TORCH_CUDA="$(python -c 'import torch; print(torch.version.cuda)')"
NVCC_CUDA="$(nvcc --version | grep -o 'release [0-9.]*' | awk '{print $2}')"
if [[ "$TORCH_CUDA" == "$NVCC_CUDA" ]]; then
    echo "[env] OK: nvcc CUDA $NVCC_CUDA matches torch CUDA $TORCH_CUDA"
else
    echo "[env] WARNING: nvcc CUDA $NVCC_CUDA does not match torch CUDA $TORCH_CUDA" >&2
fi
