# data/

This directory holds no committed data files (see `.gitignore`) — datasets
are either downloaded locally or generated synthetically at runtime.

- `raw/` — original downloaded files, exactly as obtained from Kaggle /
  the Fraud Detection Handbook simulator. Paths are declared in
  `configs/datasets.yaml` and populated via `scripts/download_data.py`.
- `interim/` — intermediate cleaned-but-not-yet-split data, written by
  `scripts/preprocess_data.py`.
- `processed/` — validation reports and any fully processed artifacts.

None of these are committed to git (financial transaction data must never
be checked into version control, and IEEE-CIS's competition rules prohibit
redistribution). If a raw file is missing, every loader in
`src/data/loaders.py` falls back to a clearly-labeled synthetic sample
(`load_synthetic`) so the rest of the pipeline can still run — see
`docs/reproducibility.md` for full dataset setup instructions.
