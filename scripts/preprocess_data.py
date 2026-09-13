#!/usr/bin/env python
"""Validate + clean a raw dataset and cache the cleaned frame under
data/interim/ and data/processed/, so experiment scripts don't repeat
cleaning/validation on every run.

    python scripts/preprocess_data.py --dataset ulb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import REPO_ROOT  # noqa: E402
from src.data.loaders import load_dataset, load_synthetic  # noqa: E402
from src.data.preprocessing import clean_dataframe  # noqa: E402
from src.data.validators import validate_dataset  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger("scripts.preprocess_data")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and clean a raw dataset.")
    parser.add_argument("--dataset", default="ulb",
                         choices=["ulb", "ieee_cis", "paysim", "banksim", "fraud_detection_handbook"])
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data instead of a real download.")
    args = parser.parse_args()

    if args.synthetic:
        dataset = load_synthetic(args.dataset)
    else:
        try:
            dataset = load_dataset(args.dataset)
        except FileNotFoundError as exc:
            logger.error(str(exc))
            sys.exit(1)

    report = validate_dataset(dataset)
    logger.info(report.summary())
    if report.potential_leakage_columns:
        logger.warning(f"Potential leakage columns detected: {report.potential_leakage_columns}. "
                        f"Review docs/methodology.md before proceeding.")

    cleaned = clean_dataframe(dataset.frame, dataset.target)

    interim_path = REPO_ROOT / "data" / "interim" / f"{args.dataset}_cleaned.csv"
    interim_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(interim_path, index=False)
    logger.info(f"Wrote cleaned data ({len(cleaned)} rows) -> {interim_path}")

    processed_path = REPO_ROOT / "data" / "processed" / f"{args.dataset}_validation_report.txt"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.write_text(report.summary(), encoding="utf-8")
    logger.info(f"Wrote validation report -> {processed_path}")


if __name__ == "__main__":
    main()
