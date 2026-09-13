"""Dataset loaders behind a common interface.

Every loader returns a `LoadedDataset`: a feature/target DataFrame pair plus
metadata (time column, amount column, categorical features) pulled from
configs/datasets.yaml. No dataset path is ever hard-coded here — see
src/config/settings.get_dataset_spec.

None of the real Kaggle datasets ship with this repository (they require a
Kaggle account and, for IEEE-CIS, competition acceptance). If the raw file is
missing, loaders raise a clear FileNotFoundError pointing at
docs/reproducibility.md rather than silently fabricating data. For tests,
demos, and CI — where no internet/Kaggle credentials are available — use
`load_synthetic(dataset_key)`, which generates a small schema-compatible
synthetic sample and is clearly labeled as synthetic everywhere it is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.settings import get_dataset_spec


@dataclass
class LoadedDataset:
    """Common container returned by every loader in this module."""

    key: str
    frame: pd.DataFrame
    target: str
    time_column: str | None
    amount_column: str | None
    categorical_features: list[str] = field(default_factory=list)
    is_synthetic: bool = False
    source_note: str = ""

    @property
    def feature_columns(self) -> list[str]:
        return [c for c in self.frame.columns if c != self.target]

    def X_y(self) -> tuple[pd.DataFrame, pd.Series]:
        return self.frame[self.feature_columns], self.frame[self.target]


def _missing_file_error(dataset_key: str, path: str, source_url: str) -> FileNotFoundError:
    return FileNotFoundError(
        f"Raw data file for dataset '{dataset_key}' not found at: {path}\n"
        f"Download it from {source_url}, or generate a synthetic sample for "
        f"development/testing with `load_synthetic('{dataset_key}')`.\n"
        f"See docs/reproducibility.md for full setup instructions."
    )


def load_ulb(path: str | Path | None = None) -> LoadedDataset:
    """Load the ULB Credit Card Fraud dataset (primary reproducibility benchmark)."""
    spec = get_dataset_spec("ulb")
    resolved = Path(path) if path is not None else Path(spec["path"])
    if not resolved.exists():
        raise _missing_file_error("ulb", str(resolved), spec["source_url"])

    frame = pd.read_csv(resolved)
    return LoadedDataset(
        key="ulb",
        frame=frame,
        target=spec["target"],
        time_column=spec.get("time_column"),
        amount_column=spec.get("amount_column"),
        categorical_features=spec.get("categorical_features", []),
        source_note=spec["source_url"],
    )


def load_ieee_cis(
    transaction_path: str | Path | None = None,
    identity_path: str | Path | None = None,
) -> LoadedDataset:
    """Load and merge IEEE-CIS transaction + identity tables on TransactionID."""
    spec = get_dataset_spec("ieee_cis")
    tx_path = Path(transaction_path) if transaction_path is not None else Path(spec["transaction_path"])
    id_path = Path(identity_path) if identity_path is not None else Path(spec["identity_path"])

    if not tx_path.exists():
        raise _missing_file_error("ieee_cis", str(tx_path), spec["source_url"])

    transactions = pd.read_csv(tx_path)
    if id_path.exists():
        identity = pd.read_csv(id_path)
        merged = transactions.merge(identity, how="left", on=spec["join_key"])
    else:
        merged = transactions  # identity table is optional per IEEE-CIS rules

    return LoadedDataset(
        key="ieee_cis",
        frame=merged,
        target=spec["target"],
        time_column=spec.get("time_column"),
        amount_column=spec.get("amount_column"),
        categorical_features=spec.get("categorical_features", []),
        source_note=spec["source_url"],
    )


def load_paysim(path: str | Path | None = None) -> LoadedDataset:
    spec = get_dataset_spec("paysim")
    resolved = Path(path) if path is not None else Path(spec["path"])
    if not resolved.exists():
        raise _missing_file_error("paysim", str(resolved), spec["source_url"])

    frame = pd.read_csv(resolved)
    drop_cols = [c for c in spec.get("drop_columns", []) if c in frame.columns]
    frame = frame.drop(columns=drop_cols)

    return LoadedDataset(
        key="paysim",
        frame=frame,
        target=spec["target"],
        time_column=spec.get("time_column"),
        amount_column=spec.get("amount_column"),
        categorical_features=spec.get("categorical_features", []),
        source_note=spec["source_url"],
    )


def load_banksim(path: str | Path | None = None) -> LoadedDataset:
    spec = get_dataset_spec("banksim")
    resolved = Path(path) if path is not None else Path(spec["path"])
    if not resolved.exists():
        raise _missing_file_error("banksim", str(resolved), spec["source_url"])

    frame = pd.read_csv(resolved)
    # BankSim ships quoted categorical values like "'es_transportation'"
    for col in frame.select_dtypes(include="object").columns:
        frame[col] = frame[col].str.strip("'\"")

    return LoadedDataset(
        key="banksim",
        frame=frame,
        target=spec["target"],
        time_column=spec.get("time_column"),
        amount_column=spec.get("amount_column"),
        categorical_features=spec.get("categorical_features", []),
        source_note=spec["source_url"],
    )


def load_fraud_detection_handbook(path: str | Path | None = None) -> LoadedDataset:
    spec = get_dataset_spec("fraud_detection_handbook")
    resolved = Path(path) if path is not None else Path(spec["path"])
    if not resolved.exists():
        raise _missing_file_error("fraud_detection_handbook", str(resolved), spec["source_url"])

    frame = pd.read_csv(resolved)
    drop_cols = [c for c in spec.get("drop_columns", []) if c in frame.columns]
    frame = frame.drop(columns=drop_cols)

    return LoadedDataset(
        key="fraud_detection_handbook",
        frame=frame,
        target=spec["target"],
        time_column=spec.get("time_column"),
        amount_column=spec.get("amount_column"),
        categorical_features=spec.get("categorical_features", []),
        source_note=spec["source_url"],
    )


_LOADERS = {
    "ulb": load_ulb,
    "ieee_cis": load_ieee_cis,
    "paysim": load_paysim,
    "banksim": load_banksim,
    "fraud_detection_handbook": load_fraud_detection_handbook,
}


def load_dataset(dataset_key: str) -> LoadedDataset:
    """Dispatch to the correct loader by key (see configs/datasets.yaml)."""
    if dataset_key not in _LOADERS:
        raise KeyError(f"Unknown dataset '{dataset_key}'. Available: {sorted(_LOADERS)}")
    return _LOADERS[dataset_key]()


def load_synthetic(dataset_key: str = "ulb", n_samples: int = 5000, fraud_rate: float = 0.0173,
                    random_state: int = 42) -> LoadedDataset:
    """Generate a small, schema-compatible SYNTHETIC sample for tests/demos.

    This is NOT real transaction data and must never be used to report
    research results. It exists solely so the pipeline (loaders ->
    preprocessing -> models -> evaluation) can be exercised and unit-tested
    without requiring a Kaggle download. `is_synthetic=True` is propagated so
    downstream code (dashboard, reports) can label outputs accordingly.
    """
    rng = np.random.default_rng(random_state)
    n_fraud = max(1, int(n_samples * fraud_rate))
    n_legit = n_samples - n_fraud

    if dataset_key == "ulb":
        legit = rng.normal(loc=0.0, scale=1.0, size=(n_legit, 28))
        fraud = rng.normal(loc=0.6, scale=1.8, size=(n_fraud, 28))
        V = np.vstack([legit, fraud])
        amount = np.concatenate([
            rng.gamma(shape=2.0, scale=40.0, size=n_legit),
            rng.gamma(shape=2.0, scale=150.0, size=n_fraud),
        ])
        label = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])

        # Shuffle (V, amount, label) together BEFORE assigning a monotonic
        # Time column, so fraud is scattered uniformly across the time range
        # instead of clustering at the end (which would happen if sorted
        # Time were assigned positionally to this legit-then-fraud-ordered
        # array) — that spurious label<->time correlation previously made
        # every early rolling-window fold contain zero fraud rows.
        shuffle_idx = rng.permutation(n_samples)
        V, amount, label = V[shuffle_idx], amount[shuffle_idx], label[shuffle_idx]
        time = np.sort(rng.uniform(0, 172792, size=n_samples))

        frame = pd.DataFrame(V, columns=[f"V{i}" for i in range(1, 29)])
        frame.insert(0, "Time", time)
        frame["Amount"] = amount
        frame["Class"] = label.astype(int)
        frame = frame.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

        return LoadedDataset(
            key="ulb",
            frame=frame,
            target="Class",
            time_column="Time",
            amount_column="Amount",
            categorical_features=[],
            is_synthetic=True,
            source_note="SYNTHETIC sample generated for testing; not real ULB data.",
        )

    raise NotImplementedError(
        f"Synthetic generator for '{dataset_key}' is not implemented yet; "
        f"only 'ulb' schema is currently supported."
    )
