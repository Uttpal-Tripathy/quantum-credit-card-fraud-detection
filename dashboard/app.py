"""QUANTUM FINANCIAL FRAUD INTELLIGENCE CENTER — Page 1: Executive Overview.

Run with:  streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from dashboard.components.data import get_demo_qgfda
from dashboard.components.theme import inject_theme, render_header

st.set_page_config(
    page_title="Quantum Financial Fraud Intelligence Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_theme()

st.sidebar.markdown("### 🛡️ QUANTUM FINANCIAL\nFRAUD INTELLIGENCE CENTER")
st.sidebar.caption("QGFDA Research Dashboard")
st.sidebar.divider()
st.sidebar.caption(
    "Navigate via the pages listed above. All data on every page is synthetic "
    "or read from local experiment result files — no real financial data is used."
)

render_header(
    "Executive Overview",
    "Quantum-Gated Fraud Detection Architecture (QGFDA) — live summary on a synthetic demo dataset.",
)

with st.spinner("Loading demo pipeline (trains once per session)..."):
    demo = get_demo_qgfda()

result = demo["result"]
frame = demo["frame"]
metrics = result["binary_metrics"]
decision_summary = result["decision_summary"]

n_total = len(frame)
n_fraud = int(frame["actual_fraud"].sum())
fraud_rate = n_fraud / n_total if n_total else 0.0

row1 = st.columns(4)
row1[0].metric("Total Transactions (demo)", f"{n_total:,}")
row1[1].metric("Fraud Cases", f"{n_fraud:,}")
row1[2].metric("Fraud Rate", f"{fraud_rate:.3%}")
row1[3].metric("Quantum Inference %", f"{result['quantum_fraction']:.2%}")

row2 = st.columns(4)
row2[0].metric("PR-AUC (primary metric)", f"{metrics.pr_auc:.3f}")
row2[1].metric("Recall", f"{metrics.recall:.3f}")
row2[2].metric("Precision", f"{metrics.precision:.3f}")
row2[3].metric("F1", f"{metrics.f1:.3f}")

row3 = st.columns(4)
row3[0].metric("False Positive Rate", f"{metrics.fpr:.4f}")
row3[1].metric("Avg. Latency (classical)", f"{result['avg_classical_latency_s']*1000:.3f} ms")
row3[2].metric("Avg. Latency (quantum path)", f"{result['avg_quantum_latency_s']*1000:.2f} ms")
row3[3].metric("Qubits Used", f"{demo['model'].quantum_model.num_qubits}")

st.divider()
col_a, col_b = st.columns([2, 1])
with col_a:
    st.markdown("#### Decision distribution")
    dc = st.columns(3)
    dc[0].metric("APPROVE", f"{decision_summary.n_approve:,}", f"{decision_summary.approve_rate:.1%}")
    dc[1].metric("REVIEW", f"{decision_summary.n_review:,}", f"{decision_summary.review_rate:.1%}")
    dc[2].metric("BLOCK", f"{decision_summary.n_block:,}", f"{decision_summary.block_rate:.1%}")
    st.caption(f"Total expected loss on demo test set: ${decision_summary.expected_loss:,.2f}")

with col_b:
    st.markdown("#### Research hypothesis")
    st.markdown(
        "> *Selective quantum-classical inference may improve the fraud-detection "
        "accuracy-false-positive-resource trade-off for difficult transactions "
        "under constrained quantum resources.*\n\n"
        "This project does **not** assume or claim quantum advantage — every "
        "number on this dashboard is produced by running the actual pipeline, "
        "and Page 9 (Research Results) reports pending experiments honestly "
        "rather than fabricating them."
    )

st.info(
    "This overview runs on a small **synthetic** demo sample for responsiveness. "
    "For real ULB/IEEE-CIS/PaySim/BankSim benchmark results, run "
    "`python scripts/run_benchmark.py` and revisit Page 9 - Research Results.",
    icon="ℹ️",
)
