"""Page 2 — Live Transaction Monitor: per-transaction risk scores and
decisions on a synthetic demo stream. Anonymized transaction IDs only —
never real card numbers, CVVs, or account numbers (spec section 22)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import streamlit as st

from dashboard.components.data import get_demo_qgfda
from dashboard.components.theme import decision_pill, inject_theme, render_header

st.set_page_config(page_title="Live Transaction Monitor", page_icon="📡", layout="wide")
inject_theme()
render_header("Live Transaction Monitor", "Synthetic demo transaction stream scored by QGFDA.")

demo = get_demo_qgfda()
frame = demo["frame"].copy()

col1, col2, col3 = st.columns(3)
decision_filter = col1.multiselect("Filter by decision", ["APPROVE", "REVIEW", "BLOCK"], default=["REVIEW", "BLOCK"])
routed_only = col2.checkbox("Quantum-routed only", value=False)
n_rows = col3.slider("Rows to display", 10, 200, 50, step=10)

view = frame
if decision_filter:
    view = view[view["decision"].isin(decision_filter)]
if routed_only:
    view = view[view["routed_to_quantum"]]
view = view.sort_values("final_risk", ascending=False).head(n_rows).reset_index(drop=True)

st.caption(f"Showing {len(view)} of {len(frame)} synthetic demo transactions.")

display = view[["transaction_id", "amount", "classical_risk", "quantum_risk", "final_risk", "routed_to_quantum", "decision"]].copy()
display["amount"] = display["amount"].map(lambda v: f"${v:,.2f}" if pd.notna(v) else "—")
for col in ["classical_risk", "quantum_risk", "final_risk"]:
    display[col] = display[col].map(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
display["routed_to_quantum"] = display["routed_to_quantum"].map({True: "🔮 quantum", False: "⚡ classical"})
display.columns = ["Transaction ID", "Amount", "Classical Risk", "Quantum Risk", "Final Risk", "Path", "Decision"]

st.dataframe(
    display,
    width='stretch',
    hide_index=True,
    column_config={
        "Decision": st.column_config.TextColumn("Decision"),
    },
)

st.markdown("#### Decision legend")
st.markdown(
    f"{decision_pill('APPROVE')} auto-approved &nbsp;&nbsp; "
    f"{decision_pill('REVIEW')} sent to analyst review &nbsp;&nbsp; "
    f"{decision_pill('BLOCK')} blocked",
    unsafe_allow_html=True,
)
