"""Dataset-agnostic feature engineering built only from information available
at inference time (no post-hoc / label-derived features).

Kept intentionally generic (time-of-day, log-amount, rolling velocity) so the
same functions apply across ULB, IEEE-CIS, PaySim, BankSim, and the Fraud
Detection Handbook simulator, given only a time column and an amount column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_amount_features(frame: pd.DataFrame, amount_column: str) -> pd.DataFrame:
    """log1p-transformed amount (fraud amounts are heavy-tailed) plus a
    z-scored amount within the frame."""
    out = frame.copy()
    if amount_column not in out.columns:
        return out
    out[f"{amount_column}_log1p"] = np.log1p(out[amount_column].clip(lower=0))
    mean, std = out[amount_column].mean(), out[amount_column].std() or 1.0
    out[f"{amount_column}_zscore"] = (out[amount_column] - mean) / std
    return out


def add_time_features(frame: pd.DataFrame, time_column: str, unit: str = "seconds") -> pd.DataFrame:
    """Cyclical hour-of-day / day encoding from a raw elapsed-time column
    (ULB's `Time` and PaySim/BankSim's `step` are both elapsed units, not
    timestamps, so we derive hour-of-day via modulo rather than parsing a
    calendar date)."""
    out = frame.copy()
    if time_column not in out.columns:
        return out

    seconds_per_hour = 3600
    if unit == "hours":
        hour_of_day = out[time_column] % 24
    else:
        hour_of_day = (out[time_column] // seconds_per_hour) % 24

    out[f"{time_column}_hour_sin"] = np.sin(2 * np.pi * hour_of_day / 24)
    out[f"{time_column}_hour_cos"] = np.cos(2 * np.pi * hour_of_day / 24)
    return out


def add_velocity_features(
    frame: pd.DataFrame,
    entity_column: str,
    time_column: str,
    amount_column: str,
    windows: tuple[int, ...] = (5,),
) -> pd.DataFrame:
    """Rolling transaction count and mean amount per entity (e.g. card/customer
    id) over the preceding `windows[i]` transactions, computed causally (only
    using past rows) to avoid look-ahead leakage. Requires the frame to already
    be sorted by time_column."""
    out = frame.copy()
    if entity_column not in out.columns or time_column not in out.columns:
        return out
    out = out.sort_values(time_column)

    for w in windows:
        grouped = out.groupby(entity_column)[amount_column]
        out[f"{entity_column}_rolling_mean_{w}"] = (
            grouped.transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        )
        out[f"{entity_column}_rolling_count_{w}"] = (
            grouped.transform(lambda s: s.shift(1).rolling(w, min_periods=1).count())
        )
    out[[c for c in out.columns if c.startswith(f"{entity_column}_rolling")]] = (
        out[[c for c in out.columns if c.startswith(f"{entity_column}_rolling")]].fillna(0)
    )
    return out.sort_index()


def engineer_features(
    frame: pd.DataFrame,
    time_column: str | None,
    amount_column: str | None,
    entity_column: str | None = None,
) -> pd.DataFrame:
    """Apply the standard, leakage-safe feature-engineering stack for whichever
    columns are actually present in this dataset."""
    out = frame
    if amount_column:
        out = add_amount_features(out, amount_column)
    if time_column:
        out = add_time_features(out, time_column)
    if entity_column and time_column and amount_column:
        out = add_velocity_features(out, entity_column, time_column, amount_column)
    return out
