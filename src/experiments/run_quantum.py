"""Experiments B/C/D/E (quantum kernel, QSVC, VQC, QNN) runner.

Quantum models train in O(n) to O(n^2) circuit evaluations, so this runner
always subsamples the training set to a tractable size (`train_sample_size`)
and requires an explicit, small `qubits` count (features are trimmed/PCA'd
down to fit) — this project never silently attempts a qubit count a
simulator can't handle in reasonable time.

    python -m src.experiments.run_quantum --model vqc --qubits 6 --dataset ulb --synthetic
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from src.config.settings import load_experiments_config, load_quantum_config
from src.data.loaders import load_dataset, load_synthetic
from src.data.imbalance import stratified_subsample
from src.data.preprocessing import clean_dataframe, fit_transform_split, infer_feature_types, split_data
from src.data.validators import validate_dataset
from src.evaluation.metrics import compute_metrics
from src.hybrid.cost_sensitive_decision import CostModel
from src.quantum.qnn_model import QNNModel
from src.quantum.qsvc_model import QSVCModel
from src.quantum.quantum_pca import classical_pca
from src.quantum.vqc_model import VQCModel
from src.utils.experiment_tracker import ExperimentRecord, save_experiment
from src.utils.logging import get_logger
from src.utils.reproducibility import set_global_seed

logger = get_logger("experiments.run_quantum")

MODEL_REGISTRY = {"qsvc": QSVCModel, "vqc": VQCModel, "qnn": QNNModel}


def _reduce_to_qubits(X: np.ndarray, n_qubits: int, random_state: int) -> np.ndarray:
    """If the feature space is wider than the qubit budget, reduce it via
    classical PCA (a documented, honest default — see
    src/quantum/quantum_pca.py for why genuine quantum PCA isn't used here
    at this scale)."""
    if X.shape[1] <= n_qubits:
        return X
    return classical_pca(X, n_components=n_qubits, random_state=random_state).transformed


def run_quantum_experiment(
    dataset_key: str = "ulb",
    model_name: str = "vqc",
    qubits: int = 6,
    use_synthetic: bool = False,
    backend: str | None = None,
    shots: int | None = None,
    train_sample_size: int = 200,
    test_sample_size: int = 300,
    random_seed: int | None = None,
) -> dict:
    from src.experiments.run_classical import load_dataset_with_fallback

    exp_cfg = load_experiments_config()
    quantum_cfg = load_quantum_config()["quantum"]
    random_seed = random_seed if random_seed is not None else exp_cfg["evaluation"]["random_seed"]
    set_global_seed(random_seed)

    backend = backend or quantum_cfg["backend"]
    shots = shots or quantum_cfg["shots"]

    dataset = load_dataset_with_fallback(dataset_key, use_synthetic)
    report = validate_dataset(dataset)
    logger.info(report.summary())

    frame = clean_dataframe(dataset.frame, dataset.target)
    numeric, categorical = infer_feature_types(frame, dataset.target, dataset.categorical_features)
    splits = split_data(
        frame, dataset.target, strategy="random",
        test_size=exp_cfg["evaluation"]["test_size"],
        validation_size=exp_cfg["evaluation"]["validation_size"],
        random_state=random_seed,
    )

    X_train_full, X_others, transformer, meta = fit_transform_split(
        splits["train"], {"test": splits["test"]}, dataset.target, numeric, categorical
    )
    y_train_full = splits["train"][dataset.target].values
    y_test_full = splits["test"][dataset.target].values

    train_idx = stratified_subsample(X_train_full, y_train_full, train_sample_size, random_state=random_seed)
    test_idx = stratified_subsample(X_others["test"], y_test_full, test_sample_size, random_state=random_seed)

    X_train = _reduce_to_qubits(X_train_full[train_idx], qubits, random_seed)
    X_test = _reduce_to_qubits(X_others["test"][test_idx], qubits, random_seed)
    y_train, y_test = y_train_full[train_idx], y_test_full[test_idx]

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown quantum model '{model_name}'. Choose from {sorted(MODEL_REGISTRY)}.")

    logger.info(f"Training {model_name} on {dataset_key} ({qubits} qubits, backend={backend}, "
                f"n_train={len(y_train)}, n_test={len(y_test)})...")

    model_kwargs = dict(
        num_qubits=qubits,
        feature_map_name=quantum_cfg["feature_map"],
        feature_map_reps=quantum_cfg["feature_map_reps"],
        entanglement=quantum_cfg["entanglement"],
        backend=backend,
        shots=shots,
        seed=random_seed,
    )
    if model_name in ("vqc", "qnn"):
        model_kwargs.update(
            ansatz_name=quantum_cfg["ansatz"],
            ansatz_reps=quantum_cfg["ansatz_reps"],
            optimizer=quantum_cfg["optimizer"],
            optimizer_maxiter=quantum_cfg["optimizer_maxiter"],
        )
    if model_name == "qnn":
        model_kwargs["qnn_type"] = quantum_cfg["qnn_type"]

    cost_model = CostModel.from_config(exp_cfg["cost_model"])

    try:
        model = MODEL_REGISTRY[model_name](**model_kwargs)
        fit_result = model.fit(X_train, y_train)
        proba, inference_time = model.predict_proba_timed(X_test)

        metrics = compute_metrics(
            y_test, proba, threshold=0.5,
            fraud_loss=cost_model.fraud_loss, false_positive_cost=cost_model.false_positive_cost,
        )

        record = ExperimentRecord(
            experiment_name=f"quantum_{model_name}",
            dataset=dataset_key + ("_synthetic" if dataset.is_synthetic else ""),
            model=model_name,
            random_seed=random_seed,
            qubits=qubits,
            circuit_depth=getattr(fit_result, "circuit_depth", None),
            transpiled_depth=getattr(fit_result, "transpiled_depth", None),
            shots=shots,
            optimizer=getattr(fit_result, "optimizer", None),
            backend=model.primitives.backend_info.backend_name,
            noise_enabled=model.primitives.backend_info.is_noisy,
            training_time_s=fit_result.training_time_s,
            inference_time_s=inference_time / max(len(X_test), 1),
            metrics=metrics.to_dict(),
            status="completed",
            extra={"is_synthetic": dataset.is_synthetic, "n_train": len(y_train), "n_test": len(y_test)},
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"Quantum model '{model_name}' failed.")
        record = ExperimentRecord(
            experiment_name=f"quantum_{model_name}", dataset=dataset_key, model=model_name,
            random_seed=random_seed, qubits=qubits, shots=shots, backend=backend,
            status="failed", error=str(exc),
        )

    save_experiment(record)
    logger.info(f"{model_name}: {record.metrics if record.status == 'completed' else record.error}")
    return record.to_registry_row()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run quantum model experiments (B/C/D/E).")
    parser.add_argument("--dataset", default="ulb")
    parser.add_argument("--model", default="vqc", choices=["qsvc", "vqc", "qnn"])
    parser.add_argument("--qubits", type=int, default=6)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--backend", default=None, choices=[None, "simulator", "noisy_simulator", "ibm_quantum"])
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--train-sample-size", type=int, default=200)
    parser.add_argument("--test-sample-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    run_quantum_experiment(
        dataset_key=args.dataset,
        model_name=args.model,
        qubits=args.qubits,
        use_synthetic=args.synthetic,
        backend=args.backend,
        shots=args.shots,
        train_sample_size=args.train_sample_size,
        test_sample_size=args.test_sample_size,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
