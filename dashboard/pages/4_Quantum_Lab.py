"""Page 4 — Quantum Lab: circuit structure and configuration for the
quantum path (feature map, ansatz, qubits, shots, optimizer, backend, noise)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from dashboard.components.data import get_demo_qgfda
from dashboard.components.theme import inject_theme, render_header
from src.quantum.circuit_metrics import compute_circuit_metrics
from src.quantum.feature_maps import build_feature_map, FEATURE_MAP_NAMES
from src.quantum.ansatzes import build_ansatz, ANSATZ_NAMES

st.set_page_config(page_title="Quantum Lab", page_icon="⚛️", layout="wide")
inject_theme()
render_header("Quantum Lab", "Inspect the quantum circuit powering the QGFDA quantum path.")

demo = get_demo_qgfda()
quantum_model = demo["model"].quantum_model

col1, col2, col3, col4 = st.columns(4)
col1.metric("Qubits", quantum_model.num_qubits)
col2.metric("Shots", quantum_model.shots)
col3.metric("Optimizer", quantum_model.optimizer_name.upper())
col4.metric("Backend", quantum_model.primitives.backend_info.backend_name)

st.markdown("#### Demo pipeline circuit (feature map + ansatz, as trained)")
full_circuit = demo["feature_map"].compose(demo["ansatz"]) if demo["ansatz"] is not None else demo["feature_map"]
metrics = compute_circuit_metrics(full_circuit)

mcol1, mcol2, mcol3, mcol4 = st.columns(4)
mcol1.metric("Logical depth", metrics.logical_depth)
mcol2.metric("Transpiled depth", metrics.transpiled_depth)
mcol3.metric("Two-qubit gates", metrics.two_qubit_gate_count)
mcol4.metric("Noise enabled", "Yes" if quantum_model.primitives.backend_info.is_noisy else "No")

fig = full_circuit.decompose().draw("mpl", style="iqp")
st.pyplot(fig, width='stretch')

st.divider()
st.markdown("#### Build your own circuit")
build_col1, build_col2, build_col3, build_col4 = st.columns(4)
fm_name = build_col1.selectbox("Feature map", FEATURE_MAP_NAMES, index=0)
ansatz_name = build_col2.selectbox("Ansatz", ANSATZ_NAMES, index=0)
n_qubits = build_col3.slider("Qubits", 2, 10, 4)
reps = build_col4.slider("Repetitions", 1, 4, 2)

custom_fm = build_feature_map(fm_name, n_qubits, reps)
custom_ansatz = build_ansatz(ansatz_name, n_qubits, reps)
custom_circuit = custom_fm.compose(custom_ansatz)
custom_metrics = compute_circuit_metrics(custom_circuit)

st.caption(f"Logical depth: {custom_metrics.logical_depth} · Transpiled depth: {custom_metrics.transpiled_depth} "
           f"· Two-qubit gates: {custom_metrics.two_qubit_gate_count}")
fig2 = custom_circuit.decompose().draw("mpl", style="iqp")
st.pyplot(fig2, width='stretch')
