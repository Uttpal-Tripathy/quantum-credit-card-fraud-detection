"""Research Experiments + Results endpoints.

`run_experiment` executes synchronously and can take anywhere from seconds
(classical) to minutes (quantum/hybrid) — mirroring the dashboard's
"Research Experiments" page, nothing here runs automatically. Route
handlers are plain `def` (not `async def`) so FastAPI/Starlette runs them
in a worker thread, keeping the rest of the API responsive while a run is
in progress.
"""

from __future__ import annotations

import csv
import io

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api import services
from api.schemas import RunExperimentRequest

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


@router.get("")
def list_experiments() -> list[dict]:
    return services.experiment_registry_rows()


@router.get("/export.csv")
def export_csv() -> StreamingResponse:
    rows = services.experiment_registry_rows()
    buf = io.StringIO()
    if rows:
        writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]), media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=qgfda_experiment_registry.csv"},
    )


@router.post("/run")
def run_experiment(payload: RunExperimentRequest) -> dict:
    try:
        if payload.experiment_type == "classical":
            from src.experiments.run_classical import run_classical_experiment

            model_names = [payload.model] if payload.model in (
                "logistic_regression", "random_forest", "xgboost", "lightgbm") else None
            rows = run_classical_experiment(
                dataset_key=payload.dataset, model_names=model_names,
                use_synthetic=payload.use_synthetic, random_seed=payload.seed,
            )
            return {"rows": rows}

        if payload.experiment_type == "quantum":
            from src.experiments.run_quantum import run_quantum_experiment

            qmodel = payload.model if payload.model in ("qsvc", "vqc", "qnn") else "vqc"
            row = run_quantum_experiment(
                dataset_key=payload.dataset, model_name=qmodel, qubits=payload.qubits,
                use_synthetic=payload.use_synthetic, backend=payload.backend, shots=payload.shots,
                train_sample_size=150, test_sample_size=200, random_seed=payload.seed,
            )
            return {"rows": [row]}

        if payload.experiment_type == "hybrid_ablation":
            from src.experiments.run_hybrid import run_hybrid_experiment

            row = run_hybrid_experiment(
                dataset_key=payload.dataset, arm_name=payload.ablation_arm, qubits=payload.qubits,
                use_synthetic=payload.use_synthetic, random_seed=payload.seed,
                quantum_train_sample_size=150, optimizer_maxiter=60,
            )
            return {"rows": [row]}

        if payload.experiment_type == "noise_sweep":
            from src.experiments.run_noise import run_noise_sweep

            qmodel = payload.model if payload.model in ("qsvc", "vqc", "qnn") else "vqc"
            rows = run_noise_sweep(
                dataset_key=payload.dataset, model_name=qmodel, qubits=payload.qubits,
                use_synthetic=payload.use_synthetic, shots_sweep=[payload.shots], backends=[payload.backend],
                n_repeats=1, random_seed=payload.seed,
            )
            return {"rows": rows}

        from src.experiments.run_temporal import run_temporal_experiment

        cmodel = payload.model if payload.model in (
            "logistic_regression", "random_forest", "xgboost", "lightgbm") else "xgboost"
        result = run_temporal_experiment(
            dataset_key=payload.dataset, model_name=cmodel, use_synthetic=payload.use_synthetic,
            random_seed=payload.seed,
        )
        return {"degradation_summary": result["degradation_summary"], "drift_status": result["drift_status"]}

    except Exception as exc:  # noqa: BLE001 - surface any failure to the client directly
        raise HTTPException(status_code=500, detail=str(exc)) from exc
