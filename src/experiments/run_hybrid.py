"""Experiments H/I (classical+quantum fusion, full quantum-gated hybrid) and
the ablation matrix (spec section 33) runner.

Ablation arms A and B (`use_quantum=False`) never touch the quantum path at
all — they run the classical model alone (optionally on a selected feature
subset) so the ablation table's "classical-only" baselines aren't taxed with
irrelevant quantum training cost. Arm C (`quantum_only`) runs the quantum
model alone with no classical fast path or fusion. Arms D-I go through the
full QGFDA object, with the uncertainty gate and cost-sensitive engine
enabled/disabled per arm to match spec section 33's ablation definitions.

    python -m src.experiments.run_hybrid --dataset ulb --synthetic --arm I_full_qgfda
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from src.classical.xgboost_model import XGBoostModel
from src.config.settings import load_experiments_config, load_quantum_config
from src.data.imbalance import stratified_subsample
from src.data.preprocessing import clean_dataframe, fit_transform_split, infer_feature_types, split_data
from src.data.validators import validate_dataset
from src.evaluation.ablation import ABLATION_ARMS
from src.evaluation.metrics import compute_metrics
from src.hybrid.cost_sensitive_decision import CostModel
from src.hybrid.qgfda import QGFDA, QGFDAConfig
from src.quantum.qubo_feature_selection import qafs_select_features
from src.quantum.vqc_model import VQCModel
from src.utils.experiment_tracker import ExperimentRecord, save_experiment
from src.utils.logging import get_logger
from src.utils.reproducibility import set_global_seed

logger = get_logger("experiments.run_hybrid")


def _select_quantum_features(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
    feature_names: list[str], n_qubits: int, method: str, exp_cfg: dict, random_seed: int,
) -> list[int]:
    if method == "qafs":
        candidates = qafs_select_features(
            X_train, y_train, X_val, y_val, feature_names,
            candidate_counts=[n_qubits], weights=exp_cfg["feature_selection"]["qafs_weights"],
            primitives=None, random_state=random_seed,
        )
        if candidates:
            return candidates[0].selected_indices
        logger.warning("QAFS returned no candidates; falling back to mutual-information ranking.")
    from sklearn.feature_selection import mutual_info_classif

    mi = mutual_info_classif(X_train, y_train, random_state=random_seed)
    return list(np.argsort(-mi)[:n_qubits])


def run_hybrid_experiment(
    dataset_key: str = "ulb",
    arm_name: str = "I_full_qgfda",
    qubits: int = 4,
    use_synthetic: bool = False,
    random_seed: int | None = None,
    quantum_train_sample_size: int | None = None,
    optimizer_maxiter: int | None = None,
) -> dict:
    from src.experiments.run_classical import load_dataset_with_fallback

    if arm_name not in ABLATION_ARMS:
        raise ValueError(f"Unknown ablation arm '{arm_name}'. Choose from {sorted(ABLATION_ARMS)}.")
    arm = ABLATION_ARMS[arm_name]

    exp_cfg = load_experiments_config()
    quantum_cfg = load_quantum_config()["quantum"]
    random_seed = random_seed if random_seed is not None else exp_cfg["evaluation"]["random_seed"]
    quantum_train_sample_size = quantum_train_sample_size or 200
    optimizer_maxiter = optimizer_maxiter or quantum_cfg["optimizer_maxiter"]
    set_global_seed(random_seed)

    dataset = load_dataset_with_fallback(dataset_key, use_synthetic)
    report = validate_dataset(dataset)
    logger.info(report.summary())

    frame = clean_dataframe(dataset.frame, dataset.target)
    numeric, categorical = infer_feature_types(frame, dataset.target, dataset.categorical_features)
    splits = split_data(
        frame, dataset.target, strategy="random",
        test_size=exp_cfg["evaluation"]["test_size"], validation_size=exp_cfg["evaluation"]["validation_size"],
        random_state=random_seed,
    )
    X_train, X_others, transformer, meta = fit_transform_split(
        splits["train"], {"val": splits["val"], "test": splits["test"]}, dataset.target, numeric, categorical
    )
    y_train = splits["train"][dataset.target].values
    y_val = splits["val"][dataset.target].values
    y_test = splits["test"][dataset.target].values

    feature_names = meta.output_feature_names or [f"f{i}" for i in range(X_train.shape[1])]
    cost_model = CostModel.from_config(exp_cfg["cost_model"])

    selected_indices: list[int] | None = None
    if arm.get("use_feature_selection"):
        fs_method = arm.get("feature_selection_method", "classical")
        fs_train_idx = stratified_subsample(X_train, y_train, min(400, len(y_train)), random_state=random_seed)
        fs_val_idx = stratified_subsample(X_others["val"], y_val, min(200, len(y_val)), random_state=random_seed)
        selected_indices = _select_quantum_features(
            X_train[fs_train_idx], y_train[fs_train_idx], X_others["val"][fs_val_idx], y_val[fs_val_idx],
            feature_names, qubits, fs_method, exp_cfg, random_seed,
        )
        logger.info(f"Ablation arm '{arm_name}': selected features = {[feature_names[i] for i in selected_indices]}")

    try:
        if not arm.get("use_quantum"):
            record = _run_classical_only_arm(
                arm_name, arm, dataset_key, dataset, X_train, y_train, X_others["test"], y_test,
                selected_indices, feature_names, cost_model, random_seed,
            )
        elif arm.get("quantum_only"):
            record = _run_quantum_only_arm(
                arm_name, arm, dataset_key, dataset, X_train, y_train, X_others["test"], y_test,
                selected_indices, feature_names, quantum_cfg, cost_model, random_seed,
                quantum_train_sample_size, optimizer_maxiter,
            )
        else:
            record = _run_full_hybrid_arm(
                arm_name, arm, dataset_key, dataset, X_train, y_train, X_others["val"], y_val,
                X_others["test"], y_test, selected_indices or list(range(min(qubits, X_train.shape[1]))),
                feature_names, quantum_cfg, exp_cfg, cost_model, random_seed,
                quantum_train_sample_size, optimizer_maxiter,
            )
    except Exception as exc:  # noqa: BLE001 - one failed arm must not abort a full ablation sweep
        logger.exception(f"Ablation arm '{arm_name}' failed.")
        record = ExperimentRecord(
            experiment_name=arm_name, dataset=dataset_key, model="qgfda",
            random_seed=random_seed, status="failed", error=str(exc),
        )

    save_experiment(record)
    logger.info(f"{arm_name}: {record.metrics if record.status == 'completed' else record.error}")
    return record.to_registry_row()


def _run_classical_only_arm(
    arm_name, arm, dataset_key, dataset, X_train, y_train, X_test, y_test,
    selected_indices, feature_names, cost_model, random_seed,
) -> ExperimentRecord:
    Xtr = X_train[:, selected_indices] if selected_indices is not None else X_train
    Xte = X_test[:, selected_indices] if selected_indices is not None else X_test

    model = XGBoostModel(calibrate=False, random_state=random_seed)
    fit_result = model.fit(Xtr, y_train)
    proba, inference_time = model.predict_proba_timed(Xte)
    metrics = compute_metrics(y_test, proba, threshold=0.5,
                               fraud_loss=cost_model.fraud_loss, false_positive_cost=cost_model.false_positive_cost)

    return ExperimentRecord(
        experiment_name=arm_name,
        dataset=dataset_key + ("_synthetic" if dataset.is_synthetic else ""),
        model="xgboost",
        random_seed=random_seed,
        selected_features=[feature_names[i] for i in selected_indices] if selected_indices else [],
        training_time_s=fit_result.training_time_s,
        inference_time_s=inference_time / max(len(Xte), 1),
        metrics=metrics.to_dict(),
        status="completed",
        extra={"is_synthetic": dataset.is_synthetic, "arm_description": arm["description"], "quantum_fraction": 0.0},
    )


def _run_quantum_only_arm(
    arm_name, arm, dataset_key, dataset, X_train, y_train, X_test, y_test,
    selected_indices, feature_names, quantum_cfg, cost_model, random_seed,
    train_sample_size, optimizer_maxiter,
) -> ExperimentRecord:
    idx = selected_indices or list(range(min(6, X_train.shape[1])))
    Xtr = X_train[:, idx]
    Xte = X_test[:, idx]

    train_idx = stratified_subsample(Xtr, y_train, min(train_sample_size, len(y_train)), random_state=random_seed)

    model = VQCModel(
        num_qubits=len(idx), feature_map_name=quantum_cfg["feature_map"], feature_map_reps=quantum_cfg["feature_map_reps"],
        ansatz_name=quantum_cfg["ansatz"], ansatz_reps=quantum_cfg["ansatz_reps"],
        optimizer=quantum_cfg["optimizer"], optimizer_maxiter=optimizer_maxiter,
        backend=quantum_cfg["backend"], shots=quantum_cfg["shots"], seed=random_seed,
    )
    fit_result = model.fit(Xtr[train_idx], y_train[train_idx])
    proba, inference_time = model.predict_proba_timed(Xte)
    metrics = compute_metrics(y_test, proba, threshold=0.5,
                               fraud_loss=cost_model.fraud_loss, false_positive_cost=cost_model.false_positive_cost)

    return ExperimentRecord(
        experiment_name=arm_name,
        dataset=dataset_key + ("_synthetic" if dataset.is_synthetic else ""),
        model="vqc",
        random_seed=random_seed,
        selected_features=[feature_names[i] for i in idx],
        qubits=len(idx),
        circuit_depth=fit_result.circuit_depth,
        transpiled_depth=fit_result.transpiled_depth,
        shots=quantum_cfg["shots"],
        optimizer=quantum_cfg["optimizer"],
        backend=model.primitives.backend_info.backend_name,
        noise_enabled=model.primitives.backend_info.is_noisy,
        training_time_s=fit_result.training_time_s,
        inference_time_s=inference_time / max(len(Xte), 1),
        metrics=metrics.to_dict(),
        status="completed",
        extra={"is_synthetic": dataset.is_synthetic, "arm_description": arm["description"], "quantum_fraction": 1.0},
    )


def _run_full_hybrid_arm(
    arm_name, arm, dataset_key, dataset, X_train, y_train, X_val, y_val, X_test, y_test,
    quantum_feature_indices, feature_names, quantum_cfg, exp_cfg, cost_model, random_seed,
    quantum_train_sample_size, optimizer_maxiter,
) -> ExperimentRecord:
    gate_threshold = exp_cfg["hybrid"]["gate_threshold"] if arm.get("use_gate") else 0.0
    risk_band = tuple(exp_cfg["hybrid"]["gate_risk_band"]) if arm.get("use_gate") else (0.0, 1.0)
    effective_cost_model = cost_model if arm.get("cost_sensitive") else CostModel(fraud_loss=0, false_positive_cost=0, review_cost=0)

    classical_model = XGBoostModel(calibrate=False, random_state=random_seed)
    quantum_model = VQCModel(
        num_qubits=len(quantum_feature_indices),
        feature_map_name=quantum_cfg["feature_map"], feature_map_reps=quantum_cfg["feature_map_reps"],
        ansatz_name=quantum_cfg["ansatz"], ansatz_reps=quantum_cfg["ansatz_reps"],
        optimizer=quantum_cfg["optimizer"], optimizer_maxiter=optimizer_maxiter,
        backend=quantum_cfg["backend"], shots=quantum_cfg["shots"], seed=random_seed,
    )

    config = QGFDAConfig(
        quantum_feature_indices=quantum_feature_indices,
        gate_threshold=gate_threshold,
        risk_band=risk_band,
        max_quantum_fraction=exp_cfg["hybrid"]["max_quantum_fraction"] if arm.get("use_gate") else 1.0,
        fusion_method=exp_cfg["hybrid"]["fusion_method"],
        fusion_weights=exp_cfg["hybrid"]["fusion_weights"],
        cost_model=effective_cost_model,
        threshold_search_grid_points=30,
        quantum_train_sample_size=quantum_train_sample_size,
        random_state=random_seed,
    )

    start = time.perf_counter()
    model = QGFDA(classical_model, quantum_model, config)
    model.fit(X_train, y_train, X_val, y_val)
    training_time = time.perf_counter() - start
    result = model.evaluate(X_test, y_test)

    return ExperimentRecord(
        experiment_name=arm_name,
        dataset=dataset_key + ("_synthetic" if dataset.is_synthetic else ""),
        model="qgfda",
        random_seed=random_seed,
        selected_features=[feature_names[i] for i in quantum_feature_indices],
        qubits=len(quantum_feature_indices),
        shots=quantum_cfg["shots"],
        optimizer=quantum_cfg["optimizer"],
        backend=quantum_cfg["backend"],
        training_time_s=training_time,
        inference_time_s=result["avg_classical_latency_s"] + result["avg_quantum_latency_s"],
        metrics=result["binary_metrics"].to_dict(),
        status="completed",
        extra={
            "is_synthetic": dataset.is_synthetic,
            "quantum_fraction": result["quantum_fraction"],
            "decision_summary": vars(result["decision_summary"]),
            "arm_description": arm["description"],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hybrid QGFDA / ablation experiments.")
    parser.add_argument("--dataset", default="ulb")
    parser.add_argument("--arm", default="I_full_qgfda", choices=sorted(ABLATION_ARMS))
    parser.add_argument("--qubits", type=int, default=4)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--all-arms", action="store_true", help="Run every ablation arm in sequence.")
    parser.add_argument("--quantum-train-sample-size", type=int, default=None)
    parser.add_argument("--optimizer-maxiter", type=int, default=None)
    args = parser.parse_args()

    arms = sorted(ABLATION_ARMS) if args.all_arms else [args.arm]
    for arm in arms:
        run_hybrid_experiment(
            dataset_key=args.dataset, arm_name=arm, qubits=args.qubits,
            use_synthetic=args.synthetic, random_seed=args.seed,
            quantum_train_sample_size=args.quantum_train_sample_size,
            optimizer_maxiter=args.optimizer_maxiter,
        )


if __name__ == "__main__":
    main()
