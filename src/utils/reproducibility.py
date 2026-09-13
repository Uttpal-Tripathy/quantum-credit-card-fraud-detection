"""Reproducibility helpers: global seeding and software version capture."""

from __future__ import annotations

import platform
import random
import sys
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version


def set_global_seed(seed: int = 42) -> None:
    """Seed every RNG this project touches (python, numpy; sklearn/xgboost/
    lightgbm/qiskit take a seed explicitly per call rather than globally)."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def _pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


@dataclass
class SoftwareVersions:
    python: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)
    numpy: str = field(default_factory=lambda: _pkg_version("numpy"))
    pandas: str = field(default_factory=lambda: _pkg_version("pandas"))
    scikit_learn: str = field(default_factory=lambda: _pkg_version("scikit-learn"))
    xgboost: str = field(default_factory=lambda: _pkg_version("xgboost"))
    lightgbm: str = field(default_factory=lambda: _pkg_version("lightgbm"))
    qiskit: str = field(default_factory=lambda: _pkg_version("qiskit"))
    qiskit_machine_learning: str = field(
        default_factory=lambda: _pkg_version("qiskit-machine-learning")
    )
    qiskit_aer: str = field(default_factory=lambda: _pkg_version("qiskit-aer"))
    qiskit_ibm_runtime: str = field(
        default_factory=lambda: _pkg_version("qiskit-ibm-runtime")
    )

    def to_dict(self) -> dict[str, str]:
        return {
            "python": self.python,
            "platform": self.platform,
            "numpy": self.numpy,
            "pandas": self.pandas,
            "scikit_learn": self.scikit_learn,
            "xgboost": self.xgboost,
            "lightgbm": self.lightgbm,
            "qiskit": self.qiskit,
            "qiskit_machine_learning": self.qiskit_machine_learning,
            "qiskit_aer": self.qiskit_aer,
            "qiskit_ibm_runtime": self.qiskit_ibm_runtime,
        }


def capture_software_versions() -> dict[str, str]:
    return SoftwareVersions().to_dict()
