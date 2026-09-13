# Reproducibility

## Verified environment (as of 2026-09)

Every dependency in `requirements.txt` was resolved and installed fresh
(`pip index versions <pkg>` + a clean venv install) rather than copied from
an old tutorial. Exact versions observed in the reference environment:

| Package | Version |
|---|---|
| Python | 3.11.9 |
| numpy | 2.4.6 |
| pandas | 2.3.3 |
| scikit-learn | 1.9.0 |
| xgboost | 3.2.0 |
| lightgbm | 4.7.0 |
| qiskit | 2.5.2 |
| qiskit-machine-learning | 0.9.1 |
| qiskit-aer | 0.17.2 |
| qiskit-ibm-runtime | 0.49.0 |

Every experiment record also embeds the software versions active at run
time (`src/utils/reproducibility.capture_software_versions`), so results
remain traceable even as dependencies are upgraded later.

**Qiskit API note**: this project targets the Qiskit 2.x function-style
circuit-library API (`qiskit.circuit.library.zz_feature_map`,
`real_amplitudes`, `efficient_su2`, `n_local`, ...) and V2 primitives
(`BaseSamplerV2`/`BaseEstimatorV2`), not the deprecated 0.x/1.x class-style
API (`ZZFeatureMap`, `RealAmplitudes`, ...) or V1 primitives. If you see
tutorials using the old class-style API, they predate this project's target
version — check `python -c "import qiskit; print(qiskit.__version__)"`
before following them.

## Setting up the environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Register the Jupyter kernel (needed to open the notebooks with the project's
own venv):

```bash
python -m ipykernel install --user --name qgfda --display-name "QGFDA (.venv)"
```

## Dataset setup

Copy `.env.example` to `.env` and fill in `KAGGLE_USERNAME`/`KAGGLE_KEY`
(from your Kaggle account -> Settings -> API -> Create New Token), then:

```bash
python scripts/download_data.py --dataset ulb              # primary benchmark
python scripts/download_data.py --dataset ieee_cis          # requires competition acceptance
python scripts/download_data.py --dataset paysim
python scripts/download_data.py --dataset banksim
```

The Fraud Detection Handbook dataset is not on Kaggle — generate it via the
handbook's own simulator
(https://fraud-detection-handbook.github.io/fraud-detection-handbook/) and
place the output CSV at the path declared in `configs/datasets.yaml`.

**No real dataset is required to explore this codebase.** Every loader has a
synthetic fallback (`src.data.loaders.load_synthetic`) that every script and
the dashboard use automatically when the real file is missing — always
clearly labeled `is_synthetic=True` / a `_synthetic` dataset suffix in
recorded results.

## IBM Quantum setup (optional)

1. Create an IBM Quantum account and obtain an API token.
2. In `.env`, set `IBM_QUANTUM_TOKEN=<your token>` (and `IBM_QUANTUM_INSTANCE`
   if you use a non-default instance).
3. Pass `backend=ibm_quantum` to any quantum experiment. If the token is
   missing or the service is unreachable, execution automatically falls back
   to the noisy simulator with a logged warning — nothing crashes.

Never commit `.env`. `.gitignore` already excludes it.

## Random seeds

`configs/experiments.yaml: evaluation.random_seed` (default 42) is the
project-wide default, propagated to `src.utils.reproducibility.set_global_seed`
(Python `random` + NumPy) and passed explicitly to every sklearn/XGBoost/
LightGBM/Qiskit call that accepts a seed. Every CLI script accepts `--seed`
to override it. Every experiment record stores the exact seed used.

## Experiment registry

Every experiment run appends one row to
`experiments/results/experiment_registry.csv` (columns: experiment id,
timestamp, dataset, model, seed, preprocessing version, selected features
count, qubits, circuit depth, shots, optimizer, backend, noise flag,
training/inference time, and the full metric suite) and writes the full
structured record to `experiments/results/<name>_<id>.json`. This registry
is the single source of truth `scripts/generate_report.py` reads from.

## Running the test suite

```bash
pytest tests/ -m "not hardware"
```

74 tests as of this writing, none requiring IBM Quantum credentials. The one
`@pytest.mark.hardware`-marked test is skipped by default and only runs (and
only passes) with a valid `IBM_QUANTUM_TOKEN` set.

## Known non-determinism sources

- IBM Quantum hardware jobs (queueing, device calibration drift) are
  inherently non-reproducible run-to-run; only simulator-backed runs are
  bit-for-bit reproducible given a fixed seed.
- QAOA/VQC/QNN training uses gradient-free classical optimizers (COBYLA by
  default) whose exact trajectory can vary slightly across NumPy/SciPy
  versions even with a fixed seed; treat reported quantum metrics as
  reproducible in distribution, not to the last decimal place, across
  environments with different pinned versions.
