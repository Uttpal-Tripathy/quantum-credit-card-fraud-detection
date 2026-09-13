"""Experiment J (noise robustness) runner: sweeps shot count and
ideal-vs-noisy backend for a quantum model, recording PR-AUC/F1/recall/FPR,
circuit depth, execution time, and run-to-run stability (spec section 10/16).

    python -m src.experiments.run_noise --dataset ulb --synthetic --model vqc --qubits 4
"""

from __future__ import annotations

import argparse

import numpy as np

from src.config.settings import load_experiments_config, load_quantum_config
from src.data.imbalance import stratified_subsample
from src.data.preprocessing import clean_dataframe, fit_transform_split, infer_feature_types, split_data
from src.evaluation.metrics import compute_metrics
from src.hybrid.cost_sensitive_decision import CostModel
from src.quantum.qnn_model import QNNModel
from src.quantum.qsvc_model import QSVCModel
from src.quantum.quantum_pca import classical_pca
from src.quantum.vqc_model import VQCModel
from src.utils.experiment_tracker import ExperimentRecord, save_experiment
from src.utils.logging import get_logger
from src.utils.reproducibility import set_global_seed

logger = get_logger("experiments.run_noise")

MODEL_REGISTRY = {"qsvc": QSVCModel, "vqc": VQCModel, "qnn": QNNModel}


def run_noise_sweep(
    dataset_key: str = "ulb",
    model_name: str = "vqc",
    qubits: int = 4,
    use_synthetic: bool = True,
    shots_sweep: list[int] | None = None,
    backends: list[str] | None = None,
    n_repeats: int = 2,
    train_sample_size: int = 80,
    test_sample_size: int = 150,
    random_seed: int | None = None,
) -> list[dict]:
    from src.experiments.run_classical import load_dataset_with_fallback

    exp_cfg = load_experiments_config()
    quantum_cfg = load_quantum_config()["quantum"]
    random_seed = random_seed if random_seed is not None else exp_cfg["evaluation"]["random_seed"]
    set_global_seed(random_seed)

    shots_sweep = shots_sweep or load_quantum_config()["shots_sweep"]
    backends = backends or ["simulator", "noisy_simulator"]

    dataset = load_dataset_with_fallback(dataset_key, use_synthetic)
    frame = clean_dataframe(dataset.frame, dataset.target)
    numeric, categorical = infer_feature_types(frame, dataset.target, dataset.categorical_features)
    splits = split_data(frame, dataset.target, strategy="random",
                         test_size=exp_cfg["evaluation"]["test_size"],
                         validation_size=exp_cfg["evaluation"]["validation_size"], random_state=random_seed)
    X_train_full, X_others, transformer, meta = fit_transform_split(
        splits["train"], {"test": splits["test"]}, dataset.target, numeric, categorical
    )
    y_train_full = splits["train"][dataset.target].values
    y_test_full = splits["test"][dataset.target].values

    train_idx = stratified_subsample(X_train_full, y_train_full, train_sample_size, random_state=random_seed)
    test_idx = stratified_subsample(X_others["test"], y_test_full, test_sample_size, random_state=random_seed)

    X_train = X_train_full[train_idx]
    X_test = X_others["test"][test_idx]
    if X_train.shape[1] > qubits:
        pca = classical_pca(X_train, n_components=qubits, random_state=random_seed)
        X_train = pca.transformed
        X_test = classical_pca(X_test, n_components=qubits, random_state=random_seed).transformed
    y_train, y_test = y_train_full[train_idx], y_test_full[test_idx]

    cost_model = CostModel.from_config(exp_cfg["cost_model"])
    results = []

    for backend in backends:
        for shots in shots_sweep:
            run_metrics = []
            circuit_depth = transpiled_depth = None
            for repeat in range(n_repeats):
                seed = random_seed + repeat
                model_kwargs = dict(
                    num_qubits=qubits, feature_map_name=quantum_cfg["feature_map"],
                    feature_map_reps=quantum_cfg["feature_map_reps"], backend=backend, shots=shots, seed=seed,
                )
                if model_name in ("vqc", "qnn"):
                    model_kwargs.update(
                        ansatz_name=quantum_cfg["ansatz"], ansatz_reps=quantum_cfg["ansatz_reps"],
                        optimizer=quantum_cfg["optimizer"], optimizer_maxiter=min(quantum_cfg["optimizer_maxiter"], 60),
                    )
                if model_name == "qnn":
                    model_kwargs["qnn_type"] = quantum_cfg["qnn_type"]

                model = MODEL_REGISTRY[model_name](**model_kwargs)
                fit_result = model.fit(X_train, y_train)
                proba, inference_time = model.predict_proba_timed(X_test)
                metrics = compute_metrics(y_test, proba, threshold=0.5,
                                           fraud_loss=cost_model.fraud_loss, false_positive_cost=cost_model.false_positive_cost)
                run_metrics.append(metrics.to_dict())
                circuit_depth = getattr(fit_result, "circuit_depth", None)
                transpiled_depth = getattr(fit_result, "transpiled_depth", None)

            pr_aucs = [m["pr_auc"] for m in run_metrics if not np.isnan(m["pr_auc"])]
            stability_std = float(np.std(pr_aucs)) if len(pr_aucs) > 1 else 0.0
            mean_metrics = {k: float(np.nanmean([m[k] for m in run_metrics])) for k in run_metrics[0] if isinstance(run_metrics[0][k], (int, float))}

            record = ExperimentRecord(
                experiment_name=f"J_noise_robustness_{model_name}",
                dataset=dataset_key + ("_synthetic" if dataset.is_synthetic else ""),
                model=model_name,
                random_seed=random_seed,
                qubits=qubits,
                circuit_depth=circuit_depth,
                transpiled_depth=transpiled_depth,
                shots=shots,
                backend=backend,
                noise_enabled=(backend == "noisy_simulator"),
                metrics=mean_metrics,
                status="completed",
                extra={"is_synthetic": dataset.is_synthetic, "n_repeats": n_repeats, "pr_auc_stability_std": stability_std},
            )
            save_experiment(record)
            logger.info(f"backend={backend} shots={shots}: pr_auc={mean_metrics.get('pr_auc'):.3f} "
                        f"(std={stability_std:.3f}) fpr={mean_metrics.get('fpr'):.3f}")
            results.append(record.to_registry_row())

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run noise-robustness sweep (Experiment J).")
    parser.add_argument("--dataset", default="ulb")
    parser.add_argument("--model", default="vqc", choices=["qsvc", "vqc", "qnn"])
    parser.add_argument("--qubits", type=int, default=4)
    parser.add_argument("--synthetic", action="store_true", default=True)
    parser.add_argument("--shots", type=int, nargs="+", default=None)
    parser.add_argument("--backends", nargs="+", default=None, choices=["simulator", "noisy_simulator", "ibm_quantum"])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    run_noise_sweep(
        dataset_key=args.dataset, model_name=args.model, qubits=args.qubits, use_synthetic=args.synthetic,
        shots_sweep=args.shots, backends=args.backends, n_repeats=args.repeats, random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
