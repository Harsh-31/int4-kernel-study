"""Load config/workloads.yaml.

Every number used by the study (shapes, batch sizes, group sizes, run counts, seed)
comes from that file. Import from here instead of hard-coding values.

    from common.config import load_config, batch_sizes, group_sizes
    cfg = load_config()
    for name, shape in cfg["layers"].items():
        K, N = shape["K"], shape["N"]
"""

from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "workloads.yaml"

_SWEEPS = ("full", "early")
QUANT_SCHEMES = ("asymmetric", "symmetric")


@lru_cache(maxsize=None)
def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """Read and lightly validate the workload config. Cached per path."""
    with open(path) as f:
        cfg = yaml.safe_load(f)

    for key in ("layers", "batch_sizes", "group_sizes", "quant_scheme", "runs", "harness", "seed"):
        if key not in cfg:
            raise KeyError(f"{path}: missing top-level key '{key}'")
    for name, shape in cfg["layers"].items():
        if set(shape) != {"K", "N"}:
            raise ValueError(f"{path}: layer '{name}' must define exactly K and N")
    for key in ("batch_sizes", "group_sizes"):
        for sweep in _SWEEPS:
            if sweep not in cfg[key]:
                raise KeyError(f"{path}: '{key}' is missing the '{sweep}' list")
    for key in ("warmup", "timed"):
        if key not in cfg["runs"]:
            raise KeyError(f"{path}: 'runs' is missing '{key}'")
    for key in ("l2_flush_mb", "peak_bw_copy_mb"):
        if key not in cfg["harness"]:
            raise KeyError(f"{path}: 'harness' is missing '{key}'")
    # null is allowed while the scheme is still undecided.
    if cfg["quant_scheme"] is not None and cfg["quant_scheme"] not in QUANT_SCHEMES:
        raise ValueError(f"{path}: quant_scheme must be one of {QUANT_SCHEMES} or null, "
                         f"got '{cfg['quant_scheme']}'")

    return cfg


def _check_sweep(sweep: str) -> None:
    if sweep not in _SWEEPS:
        raise ValueError(f"sweep must be one of {_SWEEPS}, got '{sweep}'")


def batch_sizes(sweep: str = "full") -> list[int]:
    """Batch sizes (M) for the 'full' or 'early' sweep."""
    _check_sweep(sweep)
    return list(load_config()["batch_sizes"][sweep])


def group_sizes(sweep: str = "full") -> list[int]:
    """Quantization group sizes for the 'full' or 'early' sweep."""
    _check_sweep(sweep)
    return list(load_config()["group_sizes"][sweep])


def quant_scheme() -> str:
    """The INT4 scheme from the config. Raises if it has not been chosen yet."""
    scheme = load_config()["quant_scheme"]
    if scheme is None:
        raise ValueError(
            "quant_scheme is not set in config/workloads.yaml; set it to one of "
            f"{QUANT_SCHEMES} or pass scheme= explicitly"
        )
    return scheme
