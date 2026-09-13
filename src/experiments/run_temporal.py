"""Experiment K (temporal drift) runner: chronological split + rolling-window
forward-chaining evaluation, reporting PR-AUC degradation over time and PSI/
KS drift statistics between the first and last windows (spec sections 15-16).

    python -m src.experiments.run_temporal --dataset ulb --synthetic --model xgboost
"""

from __future__ import annotations

import argparse

import numpy as np

from src.classical.lightgbm_model import LightGBMModel
from src.classical.logistic_regression import LogisticRegressionModel
from src.classical.random_forest import RandomForestModel
from src.classical.xgboost_model import XGBoostModel
from src.config.settings import load_experiments_config
from src.data.preprocessing import clean_dataframe, fit_transform_split, infer_feature_types
from src.evaluation.statistical_tests import drift_status, population_stability_index
from src.evaluation.temporal_validation import degradation_summary, evaluate_rolling_windows
from src.utils.experiment_tracker import ExperimentRecord, save_experiment
from src.utils.logging import get_logger
from src.utils.reproducibility import set_global_seed

logger = get_logger("experiments.run_temporal")

MODEL_REGISTRY = {
    "logistic_regression": LogisticRegressionModel,
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel,
}


def run_temporal_experiment(
    dataset_key: str = "ulb",
    model_name: str = "xgboost",
    use_synthetic: bool = True,
    n_folds: int = 5,
    random_seed: int | None = None,
) -> dict:
    from src.experiments.run_classical import load_dataset_with_fallback

    exp_cfg = load_experiments_config()
    random_seed = random_seed if random_seed is not None else exp_cfg["evaluation"]["random_seed"]
    set_global_seed(random_seed)

    dataset = load_dataset_with_fallback(dataset_key, use_synthetic)
    if not dataset.time_column:
        raise ValueError(f"Dataset '{dataset_key}' has no time_column configured; temporal validation requires one.")

    frame = clean_dataframe(dataset.frame, dataset.target)
    numeric, categorical = infer_feature_types(frame, dataset.target, dataset.categorical_features)
    time_col = dataset.time_column
    temporal_cfg = exp_cfg["temporal"]

    def fit_predict(train_frame, test_frame):
        X_train, X_others, _, _ = fit_transform_split(
            train_frame, {"test": test_frame}, dataset.target, numeric, categorical
        )
        y_train = train_frame[dataset.target].values
        y_test = test_frame[dataset.target].values
        model = MODEL_REGISTRY[model_name](calibrate=False)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_others["test"])
        return y_test, proba

    fold_results = evaluate_rolling_windows(
        frame, time_col, dataset.target, fit_predict,
        n_folds=n_folds,
        train_window_fraction=temporal_cfg["train_window_fraction"],
        step_fraction=temporal_cfg["step_fraction"],
    )
    if not fold_results:
        raise RuntimeError("No rolling-window folds produced; dataset may be too small for the configured window fractions.")

    degradation = degradation_summary(fold_results)

    first_window_frame = frame.sort_values(time_col).iloc[: fold_results[0].n_train]
    last_window_frame = frame.sort_values(time_col).iloc[-fold_results[-1].n_test:]
    amount_col = dataset.amount_column
    psi = None
    ks_status = None
    if amount_col and amount_col in frame.columns:
        psi = population_stability_index(
            first_window_frame[amount_col].values, last_window_frame[amount_col].values
        )
        ks_status = drift_status(psi, temporal_cfg.get("psi_warning", 0.1), temporal_cfg.get("psi_critical", 0.25))

    for i, fold in enumerate(fold_results):
        record = ExperimentRecord(
            experiment_name=f"K_temporal_drift_{model_name}",
            dataset=dataset_key + ("_synthetic" if dataset.is_synthetic else ""),
            model=model_name,
            random_seed=random_seed,
            metrics=fold.metrics.to_dict(),
            status="completed",
            extra={
                "is_synthetic": dataset.is_synthetic,
                "window_index": fold.window_index,
                "train_range": fold.train_range,
                "test_range": fold.test_range,
                "n_train": fold.n_train,
                "n_test": fold.n_test,
            },
        )
        save_experiment(record)
        logger.info(f"window {i}: pr_auc={fold.metrics.pr_auc:.3f} recall={fold.metrics.recall:.3f}")

    logger.info(f"Degradation summary: {degradation}")
    if psi is not None:
        logger.info(f"Amount-distribution PSI (first vs last window) = {psi:.4f} -> status={ks_status}")

    return {
        "fold_results": fold_results,
        "degradation_summary": degradation,
        "amount_psi": psi,
        "drift_status": ks_status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run temporal drift validation (Experiment K).")
    parser.add_argument("--dataset", default="ulb")
    parser.add_argument("--model", default="xgboost", choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--synthetic", action="store_true", default=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    run_temporal_experiment(
        dataset_key=args.dataset, model_name=args.model, use_synthetic=args.synthetic,
        n_folds=args.folds, random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
