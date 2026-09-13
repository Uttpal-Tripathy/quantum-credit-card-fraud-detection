"""Centralized configuration loading for QGFDA.

All YAML configs live under ``configs/`` at the repository root and are never
hard-coded elsewhere in the codebase. This module resolves the repo root,
loads/caches the YAML files, and exposes typed accessors plus environment
variable handling (via python-dotenv) for secrets such as the IBM Quantum
token, which must never be hard-coded.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upward from ``start`` until a directory containing ``configs/`` is found."""
    current = (start or Path(__file__).resolve()).parent
    for candidate in [current, *current.parents]:
        if (candidate / "configs").is_dir() and (candidate / "src").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate repository root (expected a directory containing "
        "both 'configs/' and 'src/')."
    )


REPO_ROOT = find_repo_root()
CONFIGS_DIR = REPO_ROOT / "configs"

# Load .env once at import time. Missing .env is fine (all values optional).
load_dotenv(REPO_ROOT / ".env")


@functools.lru_cache(maxsize=None)
def _load_yaml(name: str) -> dict[str, Any]:
    path = CONFIGS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_datasets_config() -> dict[str, Any]:
    return _load_yaml("datasets.yaml")["datasets"]


def load_models_config() -> dict[str, Any]:
    return _load_yaml("models.yaml")


def load_quantum_config() -> dict[str, Any]:
    return _load_yaml("quantum.yaml")


def load_experiments_config() -> dict[str, Any]:
    return _load_yaml("experiments.yaml")


def get_dataset_spec(dataset_key: str) -> dict[str, Any]:
    """Return the config block for one dataset, resolving relative paths to absolute."""
    datasets = load_datasets_config()
    if dataset_key not in datasets:
        raise KeyError(
            f"Unknown dataset '{dataset_key}'. Available: {sorted(datasets)}"
        )
    spec = dict(datasets[dataset_key])
    for path_key in ("path", "transaction_path", "identity_path"):
        if spec.get(path_key):
            spec[path_key] = str(REPO_ROOT / spec[path_key])
    return spec


def get_env(name: str, default: str | None = None) -> str | None:
    """Read an environment variable, never logging or hard-coding its value."""
    return os.environ.get(name, default)


def clear_config_cache() -> None:
    """Clear cached YAML reads (useful in tests that write temp configs)."""
    _load_yaml.cache_clear()
