"""Ablation study definitions (spec section 33) and result aggregation.

Each ablation "arm" is a named configuration of the hybrid pipeline. The
functions in this module do not run the pipeline themselves (that's
src/experiments/run_hybrid.py's job) — they define the arms and turn a set of
already-recorded ExperimentRecords into a comparison table.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

ABLATION_ARMS: dict[str, dict[str, object]] = {
    "A_classical_only": {
        "description": "Classical model only, all features, no gate, no quantum path.",
        "use_feature_selection": False,
        "use_gate": False,
        "use_quantum": False,
        "cost_sensitive": False,
    },
    "B_classical_plus_feature_selection": {
        "description": "Classical model + quantum-aware/classical feature selection.",
        "use_feature_selection": True,
        "use_gate": False,
        "use_quantum": False,
        "cost_sensitive": False,
    },
    "C_quantum_only": {
        "description": "Quantum model (QSVC/VQC/QNN) only, on selected features.",
        "use_feature_selection": True,
        "use_gate": False,
        "use_quantum": True,
        "quantum_only": True,
        "cost_sensitive": False,
    },
    "D_classical_plus_quantum_fusion": {
        "description": "Classical + quantum scores fused for every transaction (no gating).",
        "use_feature_selection": True,
        "use_gate": False,
        "use_quantum": True,
        "cost_sensitive": False,
    },
    "E_hybrid_without_gate": {
        "description": "Same as D — fusion applied uniformly, gate disabled (routes 100% to quantum).",
        "use_feature_selection": True,
        "use_gate": False,
        "use_quantum": True,
        "cost_sensitive": False,
    },
    "F_hybrid_with_uncertainty_gate": {
        "description": "Uncertainty gate routes only low-confidence transactions to quantum.",
        "use_feature_selection": True,
        "use_gate": True,
        "use_quantum": True,
        "cost_sensitive": False,
    },
    "G_hybrid_plus_qafs": {
        "description": "Gate + Quantum-Aware Feature Selection (QAFS) specifically.",
        "use_feature_selection": True,
        "feature_selection_method": "qafs",
        "use_gate": True,
        "use_quantum": True,
        "cost_sensitive": False,
    },
    "H_hybrid_plus_cost_sensitive": {
        "description": "Gate + QAFS + cost-sensitive APPROVE/REVIEW/BLOCK decision engine.",
        "use_feature_selection": True,
        "feature_selection_method": "qafs",
        "use_gate": True,
        "use_quantum": True,
        "cost_sensitive": True,
    },
    "I_full_qgfda": {
        "description": "Full QGFDA: QAFS + fast classical path + uncertainty gate + quantum path "
                        "+ risk fusion + cost-sensitive 3-way decision engine.",
        "use_feature_selection": True,
        "feature_selection_method": "qafs",
        "use_gate": True,
        "use_quantum": True,
        "cost_sensitive": True,
        "full_architecture": True,
    },
}


@dataclass
class AblationRow:
    arm: str
    description: str
    pr_auc: float | None
    recall: float | None
    fpr: float | None
    expected_loss: float | None
    avg_latency_ms: float | None
    quantum_fraction: float | None


def build_ablation_table(records: list[dict]) -> pd.DataFrame:
    """`records` is a list of registry-row-like dicts (see
    experiment_tracker.ExperimentRecord.to_registry_row), each tagged with an
    `experiment_name` matching one of ABLATION_ARMS's keys. Missing arms are
    filled with None and rendered as 'RESULT PENDING' by the report generator
    — results are never fabricated."""
    by_arm = {r.get("experiment_name"): r for r in records}
    rows = []
    for arm, spec in ABLATION_ARMS.items():
        r = by_arm.get(arm)
        rows.append(AblationRow(
            arm=arm,
            description=str(spec["description"]),
            pr_auc=float(r["pr_auc"]) if r and r.get("pr_auc") not in (None, "") else None,
            recall=float(r["recall"]) if r and r.get("recall") not in (None, "") else None,
            fpr=float(r["fpr"]) if r and r.get("fpr") not in (None, "") else None,
            expected_loss=float(r["expected_loss"]) if r and r.get("expected_loss") not in (None, "") else None,
            avg_latency_ms=None,
            quantum_fraction=None,
        ))
    return pd.DataFrame([vars(r) for r in rows])
