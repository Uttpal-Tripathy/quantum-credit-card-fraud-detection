"""Experiment A (classical baseline) runner: trains every requested classical
model on a dataset, evaluates with the full metric suite, tunes a cost-aware
decision threshold, and records results via the experiment tracker.

Invoked by scripts/run_benchmark.py; also runnable standalone:
    python -m src.experiments.run_classical --dataset ulb --model all
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from src.classical.calibration import tune_threshold_expected_loss
from src.classical.lightgbm_model import LightGBMModel
from src.classical.logistic_regression import LogisticRegressionModel
from src.classical.random_forest import RandomForestModel
from src.classical.xgboost_model import XGBoostModel
from src.config.settings import load_experiments_config
from src.data.loaders import load_dataset, load_synthetic
from src.data.preprocessing import clean_dataframe, fit_transform_split, infer_feature_types, split_data
from src.data.validators import validate_dataset
from src.evaluation.metrics import compute_metrics
from src.hybrid.cost_sensitive_decision import CostModel
from src.utils.experiment_tracker import ExperimentRecord, save_experiment
from src.utils.logging import get_logger
from src.utils.reproducibility import set_global_seed

logger = get_logger("experiments.run_classical")

MODEL_REGISTRY = {
    "logistic_regression": LogisticRegressionModel,
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel,
}


def load_dataset_with_fallback(dataset_key: str, use_synthetic: bool = False):
    if use_synthetic:
        logger.warning(f"Using SYNTHETIC data for '{dataset_key}' (development/testing only).")
        return load_synthetic(dataset_key)
    try:
        return load_dataset(dataset_key)
    except FileNotFoundError as exc:
        logger.warning(f"{exc}\nFalling back to a SYNTHETIC sample so the pipeline can still run.")
        return load_synthetic(dataset_key)


def run_classical_experiment(
    dataset_key: str = "ulb",
    model_names: list[str] | None = None,
    use_synthetic: bool = False,
    split_strategy: str = "random",
    calibrate: bool = True,
    random_seed: int | None = None,
) -> list[dict]:
    exp_cfg = load_experiments_config()
    random_seed = random_seed if random_seed is not None else exp_cfg["evaluation"]["random_seed"]
    set_global_seed(random_seed)

    dataset = load_dataset_with_fallback(dataset_key, use_synthetic)
    report = validate_dataset(dataset)
    logger.info(report.summary())

    frame = clean_dataframe(dataset.frame, dataset.target)
    numeric, categorical = infer_feature_types(frame, dataset.target, dataset.categorical_features)

    splits = split_data(
        frame, dataset.target,
        strategy=split_strategy if dataset.time_column else "random",
        time_column=dataset.time_column,
        test_size=exp_cfg["evaluation"]["test_size"],
        validation_size=exp_cfg["evaluation"]["validation_size"],
        random_state=random_seed,
    )

    X_train, X_others, transformer, meta = fit_transform_split(
        splits["train"], {"test": splits["test"]}, dataset.target, numeric, categorical
    )
    y_train = splits["train"][dataset.target].values
    y_test = splits["test"][dataset.target].values
    X_test = X_others["test"]

    cost_model = CostModel.from_config(exp_cfg["cost_model"])
    model_names = model_names or list(MODEL_REGISTRY)

    results = []
    for name in model_names:
        if name not in MODEL_REGISTRY:
            logger.warning(f"Unknown classical model '{name}', skipping.")
            continue

        logger.info(f"Training {name} on {dataset_key} (synthetic={dataset.is_synthetic})...")
        model = MODEL_REGISTRY[name](calibrate=calibrate)

        try:
            fit_result = model.fit(X_train, y_train)
            start = time.perf_counter()
            proba = model.predict_proba(X_test)
            inference_time = (time.perf_counter() - start) / max(len(X_test), 1)

            threshold_result = tune_threshold_expected_loss(
                y_test, proba, cost_model.fraud_loss, cost_model.false_positive_cost
            )
            metrics = compute_metrics(
                y_test, proba, threshold=threshold_result.threshold,
                fraud_loss=cost_model.fraud_loss, false_positive_cost=cost_model.false_positive_cost,
            )

            record = ExperimentRecord(
                experiment_name=f"A_classical_baseline_{name}",
                dataset=dataset_key + ("_synthetic" if dataset.is_synthetic else ""),
                model=name,
                random_seed=random_seed,
                hyperparameters=model.config,
                training_time_s=fit_result.training_time_s,
                inference_time_s=inference_time,
                metrics=metrics.to_dict(),
                status="completed",
                extra={"is_synthetic": dataset.is_synthetic, "decision_threshold": threshold_result.threshold},
            )
        except Exception as exc:  # noqa: BLE001 - a failed model must not crash the whole sweep
            logger.exception(f"Model '{name}' failed.")
            record = ExperimentRecord(
                experiment_name=f"A_classical_baseline_{name}",
                dataset=dataset_key,
                model=name,
                random_seed=random_seed,
                status="failed",
                error=str(exc),
            )

        save_experiment(record)
        results.append(record.to_registry_row())
        logger.info(f"{name}: {record.metrics if record.status == 'completed' else record.error}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run classical baseline experiments (Experiment A).")
    parser.add_argument("--dataset", default="ulb", choices=["ulb", "ieee_cis", "paysim", "banksim", "fraud_detection_handbook"])
    parser.add_argument("--model", default="all", help="Model name or 'all'.")
    parser.add_argument("--synthetic", action="store_true", help="Force use of synthetic data (no download needed).")
    parser.add_argument("--split", default="random", choices=["random", "temporal"])
    parser.add_argument("--no-calibrate", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    model_names = None if args.model == "all" else [args.model]
    run_classical_experiment(
        dataset_key=args.dataset,
        model_names=model_names,
        use_synthetic=args.synthetic,
        split_strategy=args.split,
        calibrate=not args.no_calibrate,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
