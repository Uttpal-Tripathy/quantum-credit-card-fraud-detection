"""Page 8 — Research Experiments: configure and explicitly launch a single
experiment run. Nothing here executes automatically — every run requires
pressing "Run Experiment" (spec section 21: "Do not execute expensive
hardware jobs automatically. Require explicit user action.")."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from dashboard.components.theme import inject_theme, render_header

st.set_page_config(page_title="Research Experiments", page_icon="🧪", layout="wide")
inject_theme()
render_header("Research Experiments", "Configure and launch a single experiment run on demand.")

st.warning(
    "Runs execute in this Streamlit process and block the UI until finished. Real IBM Quantum "
    "hardware runs (backend='ibm_quantum') require IBM_QUANTUM_TOKEN in .env and consume real "
    "quantum-computing time/queue allocation — only select it when you intend to use it.",
    icon="⚠️",
)

with st.form("experiment_form"):
    col1, col2, col3 = st.columns(3)
    dataset = col1.selectbox("Dataset", ["ulb", "ieee_cis", "paysim", "banksim", "fraud_detection_handbook"])
    experiment_type = col2.selectbox("Experiment type", ["classical", "quantum", "hybrid_ablation", "noise_sweep", "temporal"])
    use_synthetic = col3.checkbox("Force synthetic data", value=True,
                                   help="Uncheck only if the real dataset file exists under data/raw/.")

    col4, col5, col6 = st.columns(3)
    model = col4.selectbox("Model", ["logistic_regression", "random_forest", "xgboost", "lightgbm",
                                      "qsvc", "vqc", "qnn"])
    qubits = col5.slider("Qubit count", 2, 10, 4)
    ablation_arm = col6.selectbox("Ablation arm (hybrid only)",
                                   ["A_classical_only", "B_classical_plus_feature_selection", "C_quantum_only",
                                    "D_classical_plus_quantum_fusion", "E_hybrid_without_gate",
                                    "F_hybrid_with_uncertainty_gate", "G_hybrid_plus_qafs",
                                    "H_hybrid_plus_cost_sensitive", "I_full_qgfda"], index=8)

    col7, col8, col9 = st.columns(3)
    backend = col7.selectbox("Backend", ["simulator", "noisy_simulator", "ibm_quantum"])
    shots = col8.select_slider("Shots", options=[128, 256, 512, 1024, 2048], value=256)
    seed = col9.number_input("Random seed", value=42, step=1)

    submitted = st.form_submit_button("▶ Run Experiment", type="primary")

if submitted:
    with st.spinner(f"Running {experiment_type} experiment... this may take a while for quantum models."):
        try:
            if experiment_type == "classical":
                from src.experiments.run_classical import run_classical_experiment
                rows = run_classical_experiment(
                    dataset_key=dataset, model_names=[model] if model in
                    ("logistic_regression", "random_forest", "xgboost", "lightgbm") else None,
                    use_synthetic=use_synthetic, random_seed=int(seed),
                )
            elif experiment_type == "quantum":
                from src.experiments.run_quantum import run_quantum_experiment
                qmodel = model if model in ("qsvc", "vqc", "qnn") else "vqc"
                rows = [run_quantum_experiment(
                    dataset_key=dataset, model_name=qmodel, qubits=qubits, use_synthetic=use_synthetic,
                    backend=backend, shots=shots, train_sample_size=150, test_sample_size=200,
                    random_seed=int(seed),
                )]
            elif experiment_type == "hybrid_ablation":
                from src.experiments.run_hybrid import run_hybrid_experiment
                rows = [run_hybrid_experiment(
                    dataset_key=dataset, arm_name=ablation_arm, qubits=qubits, use_synthetic=use_synthetic,
                    random_seed=int(seed), quantum_train_sample_size=150, optimizer_maxiter=60,
                )]
            elif experiment_type == "noise_sweep":
                from src.experiments.run_noise import run_noise_sweep
                qmodel = model if model in ("qsvc", "vqc", "qnn") else "vqc"
                rows = run_noise_sweep(
                    dataset_key=dataset, model_name=qmodel, qubits=qubits, use_synthetic=use_synthetic,
                    shots_sweep=[shots], backends=[backend], n_repeats=1, random_seed=int(seed),
                )
            else:
                from src.experiments.run_temporal import run_temporal_experiment
                cmodel = model if model in ("logistic_regression", "random_forest", "xgboost", "lightgbm") else "xgboost"
                result = run_temporal_experiment(
                    dataset_key=dataset, model_name=cmodel, use_synthetic=use_synthetic, random_seed=int(seed),
                )
                rows = [{"degradation_summary": result["degradation_summary"], "drift_status": result["drift_status"]}]

            st.success("Experiment completed and recorded in experiments/results/experiment_registry.csv.", icon="✅")
            st.json(rows)
            st.cache_data.clear()
        except Exception as exc:  # noqa: BLE001 - surface any failure to the user directly
            st.error(f"Experiment failed: {exc}", icon="❌")
            st.exception(exc)
else:
    st.info("Configure an experiment above and click **Run Experiment** to launch it.", icon="👆")
