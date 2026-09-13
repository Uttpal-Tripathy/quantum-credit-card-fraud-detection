# Experiment Matrix

Every experiment below writes one JSON record to `experiments/results/` and
one row to `experiments/results/experiment_registry.csv`
(`src/utils/experiment_tracker.py`). Run any of them via the CLI scripts
(`scripts/run_benchmark.py`) or directly via `src/experiments/*.py`, or
launch a single configured run from the dashboard's **Research Experiments**
page.

| ID | Name | Entry point | What it tests |
|---|---|---|---|
| A | Classical baseline | `src.experiments.run_classical` | LR/RF/XGBoost/LightGBM on the full feature set |
| B | Quantum kernel | `src.experiments.run_quantum --model qsvc` (kernel matrix) | Fidelity quantum kernel quality |
| C | QSVC | `src.experiments.run_quantum --model qsvc` | Quantum kernel -> SVC |
| D | VQC | `src.experiments.run_quantum --model vqc` | Feature map -> ansatz -> variational classifier |
| E | QNN | `src.experiments.run_quantum --model qnn` | EstimatorQNN / SamplerQNN + classical output layer |
| F | Classical feature selection | ablation arm `B_classical_plus_feature_selection` | Mutual-information selection vs all-features |
| G | Quantum-aware feature selection | ablation arm `G_hybrid_plus_qafs` | QAFS (QUBO/QAOA) vs classical selection |
| H | Classical + quantum fusion | ablation arm `D_classical_plus_quantum_fusion` | Fusing scores for every transaction (no gate) |
| I | Quantum-gated hybrid | ablation arm `I_full_qgfda` | Full QGFDA (gate + fusion + cost-sensitive engine) |
| J | Noise robustness | `src.experiments.run_noise` | Shots x ideal/noisy backend sweep |
| K | Temporal drift | `src.experiments.run_temporal` | Rolling-window forward-chaining PR-AUC degradation |
| L | Cross-dataset validation | run A/I across multiple `--dataset` values | Generalization beyond ULB |
| M | Real IBM Quantum hardware | any quantum experiment with `--backend ibm_quantum` | Real-device behavior (requires `IBM_QUANTUM_TOKEN`) |

## Ablation study (arms A-I)

Defined in `src/evaluation/ablation.py::ABLATION_ARMS` and executed via
`src/experiments/run_hybrid.py --arm <name> [--all-arms]`:

| Arm | Feature selection | Gate | Quantum | Cost-sensitive |
|---|---|---|---|---|
| A_classical_only | No | No | No | No |
| B_classical_plus_feature_selection | Yes (MI) | No | No | No |
| C_quantum_only | Yes (MI) | No | Yes (quantum score = final score) | No |
| D_classical_plus_quantum_fusion | Yes (MI) | No (100% routed) | Yes | No |
| E_hybrid_without_gate | Yes (MI) | No (100% routed) | Yes | No |
| F_hybrid_with_uncertainty_gate | Yes (MI) | Yes | Yes | No |
| G_hybrid_plus_qafs | Yes (QAFS) | Yes | Yes | No |
| H_hybrid_plus_cost_sensitive | Yes (QAFS) | Yes | Yes | Yes |
| I_full_qgfda | Yes (QAFS) | Yes | Yes | Yes |

Arms A and B never touch the quantum path (no wasted quantum training cost
for a classical-only baseline). Arm C uses the quantum model alone with no
classical fusion. Arms D-I go through the full `QGFDA` orchestrator with the
gate/cost-sensitivity toggled to match the arm definition.

## Quick vs full benchmark suites

`scripts/run_benchmark.py --suite quick` runs a fast sanity suite (all 4
classical baselines + 1 small VQC run + the full QGFDA arm) suitable for
local verification in a couple of minutes. `--suite full` runs the complete
matrix (all classical baselines, QSVC/VQC/QNN, and all 9 ablation arms) —
budget significantly more time, dominated by quantum kernel evaluation and
VQC/QNN optimizer iterations.

## Resource sweeps

`configs/quantum.yaml` defines the default sweeps used by resource-analysis
plots (`src/evaluation/resource_analysis.py`):

- `shots_sweep: [128, 256, 512, 1024, 2048]`
- `qubit_sweep: [2, 4, 6, 8, 10, 12]`
- `feature_selection.candidate_counts: [2, 4, 6, 8, 10, 12, 16]`

Each produces a "performance vs resource" plot saved to
`experiments/figures/`.
