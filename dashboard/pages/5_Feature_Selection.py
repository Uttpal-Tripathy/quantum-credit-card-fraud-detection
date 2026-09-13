"""Page 5 — Feature Selection: QAFS candidate feature subsets and the
performance/qubit-cost trade-off across feature counts."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.theme import inject_theme, render_header
from src.data.imbalance import stratified_subsample
from src.data.loaders import load_synthetic
from src.data.preprocessing import clean_dataframe, fit_transform_split, infer_feature_types, random_split
from src.quantum.qubo_feature_selection import qafs_select_features

st.set_page_config(page_title="Feature Selection", page_icon="🧬", layout="wide")
inject_theme()
render_header("Feature Selection", "Quantum-Aware Feature Selection (QAFS): performance vs qubit cost.")


@st.cache_data(show_spinner="Running QAFS across candidate feature counts...")
def run_qafs_demo(candidate_counts: tuple[int, ...], seed: int = 42):
    ds = load_synthetic("ulb", n_samples=4000, random_state=seed)
    frame = clean_dataframe(ds.frame, ds.target)
    numeric, categorical = infer_feature_types(frame, ds.target, ds.categorical_features)
    splits = random_split(frame, ds.target, test_size=0.2, validation_size=0.15, random_state=seed)
    X_train, X_others, transformer, meta = fit_transform_split(
        splits["train"], {"val": splits["val"]}, ds.target, numeric, categorical
    )
    y_train = splits["train"][ds.target].values
    y_val = splits["val"][ds.target].values
    feature_names = meta.output_feature_names

    tr_idx = stratified_subsample(X_train, y_train, 300, random_state=seed)
    val_idx = stratified_subsample(X_others["val"], y_val, 150, random_state=seed)

    weights = {"alpha": 1.0, "beta": 0.5, "gamma": 0.2, "delta": 0.3, "epsilon": 0.2, "zeta": 0.2}
    candidates = qafs_select_features(
        X_train[tr_idx], y_train[tr_idx], X_others["val"][val_idx], y_val[val_idx],
        feature_names, list(candidate_counts), weights, random_state=seed,
    )
    return candidates


candidate_counts = st.multiselect("Candidate feature counts", [2, 4, 6, 8, 10], default=[2, 4, 6, 8])
if not candidate_counts:
    st.info("Select at least one candidate feature count.")
    st.stop()

candidates = run_qafs_demo(tuple(sorted(candidate_counts)))

rows = [{
    "n_features": c.n_features, "qubit_cost": c.qubit_cost, "pr_auc": c.pr_auc, "fpr": c.fpr,
    "circuit_depth": c.circuit_depth, "latency_ms": c.latency_s * 1000, "J": c.J,
    "solver": c.solver_method, "selected_features": ", ".join(f.replace("numeric__", "") for f in c.selected_features),
} for c in candidates]
df = pd.DataFrame(rows)

st.dataframe(df, width='stretch', hide_index=True)

col1, col2 = st.columns(2)
with col1:
    fig = px.line(df, x="n_features", y="pr_auc", markers=True, title="PR-AUC vs number of selected features")
    fig.update_layout(template="plotly_dark", plot_bgcolor="#131a29", paper_bgcolor="#131a29")
    st.plotly_chart(fig, width='stretch')
with col2:
    fig2 = px.line(df, x="n_features", y="J", markers=True, title="QAFS objective J vs number of features (lower is better)")
    fig2.update_layout(template="plotly_dark", plot_bgcolor="#131a29", paper_bgcolor="#131a29")
    st.plotly_chart(fig2, width='stretch')

st.caption(
    "PR-AUC/FPR/ExpectedLoss terms in J come from a fast classical logistic-regression surrogate "
    "trained on each candidate subset — an explicit, documented approximation for ranking subsets "
    "cheaply (see src/quantum/qubo_feature_selection.py). QubitCost/CircuitDepth/Latency are real "
    "measurements from the actual feature-map circuit."
)
