"""Resource-vs-performance analysis: the core evidence for/against the
project's research hypothesis (does spending more quantum resource — qubits,
depth, shots — actually buy better fraud-detection performance, and at what
latency cost?).

Produces both aggregated DataFrames (for the dashboard) and saved PNG figures
under experiments/figures/ (for the paper / README).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config.settings import REPO_ROOT

FIGURES_DIR = REPO_ROOT / "experiments" / "figures"


def _registry_to_frame(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    numeric_cols = [
        "pr_auc", "roc_auc", "recall", "precision", "f1", "fpr", "mcc",
        "expected_loss", "qubits", "circuit_depth", "shots",
        "training_time_s", "inference_time_s",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _line_plot(df: pd.DataFrame, x: str, y: str, title: str, filename: str, hue: str | None = None) -> Path | None:
    if df.empty or x not in df.columns or y not in df.columns:
        return None
    plotted = df.dropna(subset=[x, y])
    if plotted.empty:
        return None

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if hue and hue in plotted.columns:
        for key, group in plotted.groupby(hue):
            group = group.sort_values(x)
            ax.plot(group[x], group[y], marker="o", label=str(key))
        ax.legend(title=hue, fontsize=8)
    else:
        agg = plotted.groupby(x, as_index=False)[y].mean().sort_values(x)
        ax.plot(agg[x], agg[y], marker="o", color="#7C3AED")

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = FIGURES_DIR / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_performance_vs_qubits(records: list[dict], metric: str = "pr_auc") -> Path | None:
    df = _registry_to_frame(records)
    return _line_plot(df, "qubits", metric, f"{metric.upper()} vs Qubit Count", f"perf_vs_qubits_{metric}.png", hue="model")


def plot_performance_vs_depth(records: list[dict], metric: str = "pr_auc") -> Path | None:
    df = _registry_to_frame(records)
    return _line_plot(df, "circuit_depth", metric, f"{metric.upper()} vs Circuit Depth", f"perf_vs_depth_{metric}.png", hue="model")


def plot_performance_vs_shots(records: list[dict], metric: str = "pr_auc") -> Path | None:
    df = _registry_to_frame(records)
    return _line_plot(df, "shots", metric, f"{metric.upper()} vs Shot Count", f"perf_vs_shots_{metric}.png", hue="model")


def plot_performance_vs_latency(records: list[dict], metric: str = "pr_auc") -> Path | None:
    df = _registry_to_frame(records)
    return _line_plot(df, "inference_time_s", metric, f"{metric.upper()} vs Inference Latency", f"perf_vs_latency_{metric}.png", hue="model")


def plot_performance_vs_noise(records: list[dict], metric: str = "pr_auc") -> Path | None:
    df = _registry_to_frame(records)
    if "noise_enabled" not in df.columns:
        return None
    df["noise_enabled"] = df["noise_enabled"].astype(str)
    return _line_plot(df, "noise_enabled", metric, f"{metric.upper()} vs Noise", f"perf_vs_noise_{metric}.png", hue="model")


def plot_performance_vs_features(records: list[dict], metric: str = "pr_auc") -> Path | None:
    df = _registry_to_frame(records)
    return _line_plot(df, "n_features_selected", metric, f"{metric.upper()} vs Number of Selected Features",
                       f"perf_vs_features_{metric}.png", hue="model")


def generate_all_resource_plots(records: list[dict], metric: str = "pr_auc") -> list[Path]:
    plot_fns = [
        plot_performance_vs_qubits,
        plot_performance_vs_depth,
        plot_performance_vs_shots,
        plot_performance_vs_latency,
        plot_performance_vs_noise,
        plot_performance_vs_features,
    ]
    paths = []
    for fn in plot_fns:
        p = fn(records, metric)
        if p is not None:
            paths.append(p)
    return paths
