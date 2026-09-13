"""Experiment tracking: every run is recorded as a structured JSON file under
experiments/results/ AND appended as one row to
experiments/results/experiment_registry.csv, per the project's reproducibility
requirements (section 24/28 of the spec).
"""

from __future__ import annotations

import csv
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config.settings import REPO_ROOT
from src.utils.reproducibility import capture_software_versions

RESULTS_DIR = REPO_ROOT / "experiments" / "results"
REGISTRY_PATH = RESULTS_DIR / "experiment_registry.csv"

REGISTRY_FIELDS = [
    "experiment_id",
    "timestamp_utc",
    "experiment_name",
    "dataset",
    "model",
    "random_seed",
    "preprocessing_version",
    "n_features_selected",
    "qubits",
    "circuit_depth",
    "shots",
    "optimizer",
    "backend",
    "noise_enabled",
    "training_time_s",
    "inference_time_s",
    "pr_auc",
    "roc_auc",
    "recall",
    "precision",
    "f1",
    "fpr",
    "mcc",
    "expected_loss",
    "status",
    "error",
]


@dataclass
class ExperimentRecord:
    experiment_name: str
    dataset: str
    model: str
    random_seed: int = 42
    preprocessing_version: str = "v1"
    selected_features: list[str] = field(default_factory=list)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    qubits: int | None = None
    circuit_depth: int | None = None
    transpiled_depth: int | None = None
    shots: int | None = None
    optimizer: str | None = None
    backend: str | None = None
    noise_enabled: bool = False
    training_time_s: float | None = None
    inference_time_s: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    status: str = "completed"
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    software_versions: dict[str, str] = field(default_factory=capture_software_versions)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_registry_row(self) -> dict[str, Any]:
        m = self.metrics
        return {
            "experiment_id": self.experiment_id,
            "timestamp_utc": self.timestamp_utc,
            "experiment_name": self.experiment_name,
            "dataset": self.dataset,
            "model": self.model,
            "random_seed": self.random_seed,
            "preprocessing_version": self.preprocessing_version,
            "n_features_selected": len(self.selected_features) if self.selected_features else None,
            "qubits": self.qubits,
            "circuit_depth": self.circuit_depth,
            "shots": self.shots,
            "optimizer": self.optimizer,
            "backend": self.backend,
            "noise_enabled": self.noise_enabled,
            "training_time_s": self.training_time_s,
            "inference_time_s": self.inference_time_s,
            "pr_auc": m.get("pr_auc"),
            "roc_auc": m.get("roc_auc"),
            "recall": m.get("recall"),
            "precision": m.get("precision"),
            "f1": m.get("f1"),
            "fpr": m.get("fpr"),
            "mcc": m.get("mcc"),
            "expected_loss": m.get("expected_loss"),
            "status": self.status,
            "error": self.error,
        }


def save_experiment(record: ExperimentRecord) -> Path:
    """Write the full JSON record and append a summary row to the CSV registry."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = RESULTS_DIR / f"{record.experiment_name}_{record.experiment_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record.to_json_dict(), f, indent=2, default=str)

    _append_registry_row(record.to_registry_row())
    return json_path


def _append_registry_row(row: dict[str, Any]) -> None:
    file_exists = REGISTRY_PATH.exists()
    with open(REGISTRY_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_registry() -> list[dict[str, Any]]:
    if not REGISTRY_PATH.exists():
        return []
    with open(REGISTRY_PATH, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
