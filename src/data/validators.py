"""Data-quality and leakage validation.

Every dataset loaded through src/data/loaders.py should be run through
`validate_dataset` before preprocessing. Findings are returned rather than
raised (except for schema-breaking issues) so callers can log/display them —
per the spec's requirement to *document* leakage risk rather than silently
"handle" it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.loaders import LoadedDataset

# Columns known (per docs/methodology.md) to leak the label if left in the
# feature set for a given dataset key.
KNOWN_LEAKAGE_COLUMNS: dict[str, list[str]] = {
    "ulb": [],
    "ieee_cis": [],
    "paysim": ["isFlaggedFraud"],
    "banksim": [],
    "fraud_detection_handbook": ["TX_FRAUD_SCENARIO"],
}


@dataclass
class ValidationReport:
    dataset_key: str
    n_rows: int
    n_columns: int
    n_duplicate_rows: int
    missing_value_fraction: dict[str, float]
    constant_columns: list[str]
    high_cardinality_columns: list[str]
    potential_leakage_columns: list[str]
    class_balance: dict[str, float]
    warnings: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not self.potential_leakage_columns and not self.constant_columns

    def summary(self) -> str:
        lines = [
            f"Dataset: {self.dataset_key}",
            f"  rows={self.n_rows}  columns={self.n_columns}  duplicates={self.n_duplicate_rows}",
            f"  class balance: {self.class_balance}",
        ]
        if self.constant_columns:
            lines.append(f"  constant columns (no signal): {self.constant_columns}")
        if self.potential_leakage_columns:
            lines.append(f"  POTENTIAL LEAKAGE columns: {self.potential_leakage_columns}")
        if self.high_cardinality_columns:
            lines.append(f"  high-cardinality categoricals: {self.high_cardinality_columns}")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


def validate_dataset(
    dataset: LoadedDataset,
    high_cardinality_threshold: int = 100,
) -> ValidationReport:
    frame = dataset.frame
    warnings: list[str] = []

    n_duplicates = int(frame.duplicated().sum())
    if n_duplicates > 0:
        warnings.append(f"{n_duplicates} exact duplicate rows found (not auto-dropped).")

    missing_fraction = (frame.isna().mean()).to_dict()
    high_missing = {c: f for c, f in missing_fraction.items() if f > 0.5}
    if high_missing:
        warnings.append(f"Columns with >50% missing values: {list(high_missing)}")

    constant_columns = [
        c for c in frame.columns
        if c != dataset.target and frame[c].nunique(dropna=False) <= 1
    ]

    high_cardinality = [
        c for c in dataset.categorical_features
        if c in frame.columns and frame[c].nunique(dropna=True) > high_cardinality_threshold
    ]

    known_leakage = [c for c in KNOWN_LEAKAGE_COLUMNS.get(dataset.key, []) if c in frame.columns]
    correlation_leakage = _detect_correlation_leakage(frame, dataset.target)
    potential_leakage = sorted(set(known_leakage) | set(correlation_leakage))

    if dataset.target not in frame.columns:
        raise ValueError(f"Target column '{dataset.target}' not present in dataset frame.")

    class_counts = frame[dataset.target].value_counts(normalize=True).to_dict()
    class_balance = {str(k): float(v) for k, v in class_counts.items()}
    minority_share = min(class_balance.values()) if class_balance else 0.0
    if minority_share < 0.01:
        warnings.append(
            f"Severe class imbalance (minority class = {minority_share:.4%}); "
            f"accuracy must not be used as the primary metric (see docs/methodology.md)."
        )

    return ValidationReport(
        dataset_key=dataset.key,
        n_rows=len(frame),
        n_columns=frame.shape[1],
        n_duplicate_rows=n_duplicates,
        missing_value_fraction=missing_fraction,
        constant_columns=constant_columns,
        high_cardinality_columns=high_cardinality,
        potential_leakage_columns=potential_leakage,
        class_balance=class_balance,
        warnings=warnings,
    )


def _detect_correlation_leakage(
    frame: pd.DataFrame, target: str, threshold: float = 0.98
) -> list[str]:
    """Flag numeric features near-perfectly correlated with the target as
    candidate leakage (a heuristic, not a proof — human review still required)."""
    if target not in frame.columns:
        return []
    numeric = frame.select_dtypes(include=[np.number])
    if target not in numeric.columns or numeric.shape[1] < 2:
        return []
    with np.errstate(invalid="ignore"):
        correlations = numeric.corr()[target].abs()
    flagged = [
        c for c, v in correlations.items()
        if c != target and pd.notna(v) and v >= threshold
    ]
    return flagged
