"""Page 3 — Classical vs Quantum: head-to-head charts across recorded
experiments (experiments/results/experiment_registry.csv)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.data import get_registry_df
from dashboard.components.theme import inject_theme, render_header

st.set_page_config(page_title="Classical vs Quantum", page_icon="⚖️", layout="wide")
inject_theme()
render_header("Classical vs Quantum", "Head-to-head comparison across every recorded experiment run.")

df = get_registry_df()
if df.empty:
    st.warning(
        "No experiments recorded yet. Run `python scripts/run_benchmark.py --dataset ulb --synthetic "
        "--suite quick` (or the full suite) to populate experiments/results/experiment_registry.csv, "
        "then reload this page.",
        icon="⚠️",
    )
    st.stop()

df["family"] = df["model"].apply(
    lambda m: "Quantum" if m in ("qsvc", "vqc", "qnn") else ("Hybrid" if m == "qgfda" else "Classical")
)

metric = st.selectbox("Metric", ["pr_auc", "f1", "recall", "precision", "fpr", "inference_time_s"], index=0)
plot_df = df.dropna(subset=[metric])

if plot_df.empty:
    st.info("No completed runs have a value for this metric yet.")
else:
    fig = px.bar(
        plot_df, x="model", y=metric, color="family",
        color_discrete_map={"Classical": "#4C72B0", "Quantum": "#C44E52", "Hybrid": "#A855F7"},
        hover_data=["dataset", "experiment_name", "timestamp_utc"],
        title=f"{metric.upper()} by model",
    )
    fig.update_layout(template="plotly_dark", plot_bgcolor="#131a29", paper_bgcolor="#131a29")
    st.plotly_chart(fig, width='stretch')

st.markdown("#### All metrics, side by side")
metrics_cols = ["model", "family", "dataset", "pr_auc", "f1", "recall", "precision", "fpr", "inference_time_s"]
st.dataframe(df[[c for c in metrics_cols if c in df.columns]].sort_values("pr_auc", ascending=False),
             width='stretch', hide_index=True)

st.markdown("#### Latency vs PR-AUC trade-off")
if {"inference_time_s", "pr_auc"}.issubset(df.columns):
    lat_df = df.dropna(subset=["inference_time_s", "pr_auc"])
    if not lat_df.empty:
        fig2 = px.scatter(
            lat_df, x="inference_time_s", y="pr_auc", color="family", size_max=18,
            hover_data=["model", "dataset"], log_x=True,
            title="PR-AUC vs per-transaction inference latency",
        )
        fig2.update_layout(template="plotly_dark", plot_bgcolor="#131a29", paper_bgcolor="#131a29")
        st.plotly_chart(fig2, width='stretch')
