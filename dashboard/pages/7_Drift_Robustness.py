"""Page 7 — Drift & Robustness: PSI/KS drift monitoring and controlled
adversarial-perturbation robustness experiments (defensive research only —
no real-world attack instructions, per spec section 17)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.data import get_demo_qgfda
from dashboard.components.theme import inject_theme, render_header
from src.evaluation.statistical_tests import drift_status, ks_drift_test, population_stability_index

st.set_page_config(page_title="Drift & Robustness", page_icon="📉", layout="wide")
inject_theme()
render_header("Drift & Robustness", "Concept-drift monitoring and adversarial-perturbation sensitivity.")

demo = get_demo_qgfda()
frame = demo["frame"]

st.markdown("#### Concept drift monitor (Amount distribution, first vs second half of demo set)")
half = len(frame) // 2
reference = frame["amount"].values[:half]
current = frame["amount"].values[half:]

psi = population_stability_index(reference, current)
ks_result = ks_drift_test(reference, current, alpha=0.01)
status = drift_status(psi, psi_warning=0.1, psi_critical=0.25)

status_color = {"NORMAL": "🟢", "WARNING": "🟡", "CRITICAL": "🔴"}[status]
col1, col2, col3 = st.columns(3)
col1.metric("PSI (Amount)", f"{psi:.4f}")
col2.metric("KS statistic", f"{ks_result.statistic:.4f}", f"p={ks_result.p_value:.2e}")
col3.metric("Drift status", f"{status_color} {status}")

if status != "NORMAL":
    st.warning("Drift detected above the WARNING threshold — recommend scheduling model retraining.", icon="⚠️")
else:
    st.success("No significant drift detected in this window.", icon="✅")

fig = px.histogram(
    pd.DataFrame({"amount": np.concatenate([reference, current]),
                  "window": ["reference"] * len(reference) + ["current"] * len(current)}),
    x="amount", color="window", barmode="overlay", nbins=40, log_x=True,
    title="Amount distribution: reference vs current window",
)
fig.update_layout(template="plotly_dark", plot_bgcolor="#131a29", paper_bgcolor="#131a29")
st.plotly_chart(fig, width='stretch')

st.divider()
st.markdown("#### Adversarial robustness: controlled feature perturbation")
st.caption(
    "Perturbs Amount by a chosen +/-% and measures how much the classical model's fraud score "
    "changes and how often the APPROVE/BLOCK decision flips — a defensive robustness check, "
    "not an attack-generation tool."
)

perturbation_pct = st.slider("Perturbation magnitude (%)", 1, 50, 10) / 100.0
model = demo["model"]


@st.cache_data(show_spinner="Running perturbation sweep...")
def perturbation_sweep(pct: float, seed: int = 42):
    X_sample = model.classical_model.model  # noqa: F841 - ensures model is fitted; unused directly
    rng = np.random.default_rng(seed)
    n = 300
    # Re-derive a small synthetic batch through the same preprocessing the demo model was fit on.
    from src.data.loaders import load_synthetic
    from src.data.preprocessing import clean_dataframe, fit_transform_split, infer_feature_types, random_split

    ds = load_synthetic("ulb", n_samples=2000, random_state=seed + 1)
    frame_ = clean_dataframe(ds.frame, ds.target)
    numeric, categorical = infer_feature_types(frame_, ds.target, ds.categorical_features)
    splits = random_split(frame_, ds.target, test_size=0.2, validation_size=0.1, random_state=seed + 1)
    X_train, X_others, transformer, meta = fit_transform_split(
        splits["train"], {"test": splits["test"]}, ds.target, numeric, categorical
    )
    X_test = X_others["test"][:n]
    amount_col_idx = numeric.index("Amount") if "Amount" in numeric else 0

    base_scores = model.classical_model.predict_proba(X_test)
    perturbed = X_test.copy()
    perturbed[:, amount_col_idx] = perturbed[:, amount_col_idx] * (1 + pct)
    perturbed_scores = model.classical_model.predict_proba(perturbed)

    score_delta = perturbed_scores - base_scores
    flips = ((base_scores >= 0.5) != (perturbed_scores >= 0.5)).mean()
    return score_delta, flips


score_delta, flip_rate = perturbation_sweep(perturbation_pct)

col1, col2 = st.columns(2)
col1.metric("Mean |score change|", f"{np.abs(score_delta).mean():.4f}")
col2.metric("Decision flip rate", f"{flip_rate:.2%}")

fig2 = px.histogram(score_delta, nbins=40, title=f"Fraud-score change under +{perturbation_pct:.0%} Amount perturbation")
fig2.update_layout(template="plotly_dark", plot_bgcolor="#131a29", paper_bgcolor="#131a29", showlegend=False,
                    xaxis_title="Score change (perturbed - original)")
st.plotly_chart(fig2, width='stretch')
