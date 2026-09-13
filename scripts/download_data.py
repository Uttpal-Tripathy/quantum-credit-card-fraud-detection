#!/usr/bin/env python
"""Download raw datasets via the Kaggle API into data/raw/, matching the
paths declared in configs/datasets.yaml. Requires KAGGLE_USERNAME/KAGGLE_KEY
in .env (see .env.example) or ~/.kaggle/kaggle.json.

This project never bundles the real datasets (Kaggle terms of use +
IEEE-CIS competition-acceptance requirements) and every other script falls
back to a clearly-labeled synthetic sample if the raw file is missing, so
this script is optional, not a prerequisite for exploring the codebase.

    python scripts/download_data.py --dataset ulb
    python scripts/download_data.py --dataset all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import REPO_ROOT, get_env, load_datasets_config  # noqa: E402

KAGGLE_REFS = {
    "ulb": ("dataset", "mlg-ulb/creditcardfraud"),
    "ieee_cis": ("competition", "ieee-fraud-detection"),
    "paysim": ("dataset", "ealaxi/paysim1"),
    "banksim": ("dataset", "ealaxi/banksim1"),
}


def _check_kaggle_credentials() -> bool:
    has_env = bool(get_env("KAGGLE_USERNAME")) and bool(get_env("KAGGLE_KEY"))
    has_file = (Path.home() / ".kaggle" / "kaggle.json").exists()
    return has_env or has_file


def download_one(dataset_key: str) -> None:
    if dataset_key not in KAGGLE_REFS:
        print(f"[skip] '{dataset_key}' has no Kaggle reference (e.g. fraud_detection_handbook uses a "
              f"simulator instead — see docs/reproducibility.md).")
        return

    kind, ref = KAGGLE_REFS[dataset_key]
    dest_dir = REPO_ROOT / "data" / "raw"
    dest_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["kaggle"] + (["competitions", "download", "-c", ref] if kind == "competition"
                         else ["datasets", "download", "-d", ref])
    print(f"[download] {' '.join(cmd)} -> {dest_dir}")
    subprocess.run(cmd, cwd=dest_dir, check=True)

    for zip_path in dest_dir.glob("*.zip"):
        print(f"[extract] {zip_path.name}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest_dir)
        zip_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw datasets via the Kaggle API.")
    parser.add_argument("--dataset", default="ulb",
                         choices=["ulb", "ieee_cis", "paysim", "banksim", "fraud_detection_handbook", "all"])
    args = parser.parse_args()

    if not _check_kaggle_credentials():
        print("ERROR: No Kaggle credentials found. Set KAGGLE_USERNAME/KAGGLE_KEY in .env "
              "(see .env.example) or place kaggle.json in ~/.kaggle/. Aborting.")
        sys.exit(1)

    datasets_cfg = load_datasets_config()
    targets = list(datasets_cfg) if args.dataset == "all" else [args.dataset]
    for key in targets:
        download_one(key)

    print("Done. See configs/datasets.yaml for expected file paths.")


if __name__ == "__main__":
    main()
