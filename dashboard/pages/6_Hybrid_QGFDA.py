"""Page 6 — Hybrid QGFDA: pipeline visualization and routing/latency
breakdown for the full quantum-gated architecture."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import plotly.graph_objects as go
import streamlit as st

from dashboard.components.data import get_demo_qgfda
from dashboard.components.theme import inject_theme, render_header

st.set_page_config(page_title="Hybrid QGFDA", page_icon="🧩", layout="wide")
inject_theme()
render_header("Hybrid QGFDA Architecture", "Classical Fast Path -> Uncertainty Gate -> Quantum Path -> Risk Fusion -> Decision.")

FLOW_HTML = """
<style>
.qgfda-flow { display:flex; flex-direction:column; align-items:center; gap:6px; margin: 10px 0 24px 0; }
.qgfda-node {
  background: linear-gradient(135deg, rgba(34,211,238,0.12), rgba(168,85,247,0.12));
  border: 1px solid #1f2937; border-radius: 10px; padding: 10px 18px;
  text-align:center; min-width: 340px; font-size: 0.92rem;
}
.qgfda-node b { color:#22d3ee; }
.qgfda-branch { display:flex; gap:24px; justify-content:center; width:100%; }
.qgfda-arrow { color:#6b7280; font-size:1.1rem; }
</style>
<div class="qgfda-flow">
  <div class="qgfda-node"><b>Transaction</b></div>
  <div class="qgfda-arrow">&#8595;</div>
  <div class="qgfda-node">Preprocessing + Quantum-Aware Feature Selection (QAFS)</div>
  <div class="qgfda-arrow">&#8595;</div>
  <div class="qgfda-node"><b>Classical Fast Path</b> (XGBoost, scores every transaction)</div>
  <div class="qgfda-arrow">&#8595;</div>
  <div class="qgfda-node"><b>Uncertainty Gate</b> (confidence-band routing, label never used)</div>
  <div class="qgfda-arrow">&#8595;</div>
  <div class="qgfda-branch">
    <div class="qgfda-node">High confidence &#8594; keep classical score</div>
    <div class="qgfda-node">Low confidence &#8594; <b>Quantum ML</b> (VQC / QSVC / QNN)</div>
  </div>
  <div class="qgfda-arrow">&#8595;</div>
  <div class="qgfda-node"><b>Risk Fusion</b> (weighted / confidence-weighted / stacked)</div>
  <div class="qgfda-arrow">&#8595;</div>
  <div class="qgfda-node"><b>Cost-Sensitive Decision Engine</b></div>
  <div class="qgfda-arrow">&#8595;</div>
  <div class="qgfda-node" style="border-color:#a855f7;">APPROVE &nbsp;/&nbsp; REVIEW &nbsp;/&nbsp; BLOCK</div>
</div>
"""
st.markdown(FLOW_HTML, unsafe_allow_html=True)

demo = get_demo_qgfda()
result = demo["result"]
frame = demo["frame"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("% processed classically", f"{(1 - result['quantum_fraction']):.2%}")
col2.metric("% routed to QML", f"{result['quantum_fraction']:.2%}")
col3.metric("Avg. classical latency", f"{result['avg_classical_latency_s']*1000:.3f} ms")
col4.metric("Avg. quantum latency", f"{result['avg_quantum_latency_s']*1000:.2f} ms")

st.markdown("#### Routing funnel")
n_total = len(frame)
n_routed = int(frame["routed_to_quantum"].sum())
n_classical_only = n_total - n_routed

fig = go.Figure(go.Funnel(
    y=["All transactions", "Classical fast path only", "Routed to quantum path"],
    x=[n_total, n_classical_only, n_routed],
    marker={"color": ["#4C72B0", "#55A868", "#C44E52"]},
))
fig.update_layout(template="plotly_dark", plot_bgcolor="#131a29", paper_bgcolor="#131a29",
                   title="Transaction routing funnel (demo dataset)")
st.plotly_chart(fig, width='stretch')

st.markdown("#### Quantum contribution among routed transactions")
routed = frame[frame["routed_to_quantum"]]
if len(routed):
    avg_shift = (routed["final_risk"] - routed["classical_risk"]).abs().mean()
    st.metric("Avg. |final risk - classical-only risk| on routed rows", f"{avg_shift:.4f}")
    st.caption(
        "How much the quantum path's score changed the final decision, on average, for the "
        "subset of transactions the uncertainty gate actually routed to it."
    )
else:
    st.info("No transactions were routed to the quantum path in this demo run.")
