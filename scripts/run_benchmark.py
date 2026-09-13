#!/usr/bin/env python
"""Run a batch of experiments and print/export the section-20 comparison
table (Model x Dataset x PR-AUC/Recall/Precision/F1/FPR/Latency/Qubits/
Depth/Shots), sourced from experiments/results/experiment_registry.csv.

    python scripts/run_benchmark.py --dataset ulb --synthetic --suite quick
    python scripts/run_benchmark.py --dataset ulb --synthetic --suite full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config.settings import REPO_ROOT  # noqa: E402
from src.experiments.run_classical import run_classical_experiment  # noqa: E402
from src.experiments.run_hybrid import run_hybrid_experiment  # noqa: E402
from src.experiments.run_quantum import run_quantum_experiment  # noqa: E402
from src.utils.experiment_tracker import load_registry  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger("scripts.run_benchmark")

COMPARISON_COLUMNS = [
    "model", "dataset", "pr_auc", "recall", "precision", "f1", "fpr",
    "inference_time_s", "qubits", "circuit_depth", "shots",
]


def run_quick_suite(dataset: str, synthetic: bool, seed: int) -> None:
    """A fast sanity suite: all 4 classical baselines + one small VQC run +
    the full QGFDA hybrid arm, on tiny sample sizes. Meant for CI / local
    verification, not for publishable numbers (use --suite full for that)."""
    run_classical_experiment(dataset_key=dataset, use_synthetic=synthetic, random_seed=seed)
    run_quantum_experiment(dataset_key=dataset, model_name="vqc", qubits=3, use_synthetic=synthetic,
                            train_sample_size=60, test_sample_size=100, random_seed=seed)
    run_hybrid_experiment(dataset_key=dataset, arm_name="I_full_qgfda", qubits=3, use_synthetic=synthetic,
                           random_seed=seed, quantum_train_sample_size=60, optimizer_maxiter=25)


def run_full_suite(dataset: str, synthetic: bool, seed: int) -> None:
    """The full experiment matrix (spec section 19): classical baselines,
    every quantum model, and the complete ablation study A-I. This is
    significantly slower — quantum training/kernel evaluation dominates."""
    run_classical_experiment(dataset_key=dataset, use_synthetic=synthetic, random_seed=seed)
    for quantum_model in ("qsvc", "vqc", "qnn"):
        run_quantum_experiment(dataset_key=dataset, model_name=quantum_model, qubits=6, use_synthetic=synthetic,
                                train_sample_size=200, test_sample_size=300, random_seed=seed)
    from src.evaluation.ablation import ABLATION_ARMS

    for arm in sorted(ABLATION_ARMS):
        run_hybrid_experiment(dataset_key=dataset, arm_name=arm, qubits=6, use_synthetic=synthetic, random_seed=seed)


def print_comparison_table() -> None:
    records = load_registry()
    if not records:
        print("No experiments recorded yet — run some experiments first.")
        return

    df = pd.DataFrame(records)
    available = [c for c in COMPARISON_COLUMNS if c in df.columns]
    df = df[available]
    for col in ["pr_auc", "recall", "precision", "f1", "fpr", "inference_time_s"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(4)

    print("\n=== QGFDA Experiment Comparison Table ===")
    print(df.to_string(index=False))

    out_path = REPO_ROOT / "experiments" / "results" / "comparison_table.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the QGFDA experiment benchmark suite.")
    parser.add_argument("--dataset", default="ulb")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--suite", default="quick", choices=["quick", "full"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger.info(f"Running '{args.suite}' benchmark suite on dataset='{args.dataset}' "
                f"(synthetic={args.synthetic})")
    if args.suite == "quick":
        run_quick_suite(args.dataset, args.synthetic, args.seed)
    else:
        run_full_suite(args.dataset, args.synthetic, args.seed)

    print_comparison_table()


if __name__ == "__main__":
    main()
