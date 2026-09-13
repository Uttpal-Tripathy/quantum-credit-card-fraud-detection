"""QGFDA live web app: FastAPI backend serving both the JSON API and the
static HTML5/CSS/JS frontend.

Run with:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Or:
    python -m api.main
"""

from __future__ import annotations

import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

load_dotenv(REPO_ROOT / ".env")

from api import services  # noqa: E402
from api.routers import drift, experiments, feature_selection, overview, quantum, transactions  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger("api.main")

FRONTEND_DIR = REPO_ROOT / "frontend"
CORS_ORIGINS = [o.strip() for o in os.environ.get("API_CORS_ORIGINS", "*").split(",") if o.strip()]


def _prewarm_demo_state() -> None:
    try:
        services.get_demo_state()
    except Exception:  # noqa: BLE001 - a failed prewarm must not crash the server
        logger.exception("Failed to prewarm the demo QGFDA pipeline; it will build lazily on first request.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting QGFDA API — prewarming demo pipeline in the background...")
    threading.Thread(target=_prewarm_demo_state, daemon=True).start()
    yield
    logger.info("Shutting down QGFDA API.")


app = FastAPI(
    title="QGFDA API",
    description="Quantum-Gated Fraud Detection Architecture — live backend. Research prototype, not for "
                 "production financial authorization.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok", "service": "qgfda-api", "notice": "Research Prototype — Not for Production Financial Authorization."}


app.include_router(overview.router)
app.include_router(transactions.router)
app.include_router(quantum.router)
app.include_router(feature_selection.router)
app.include_router(drift.router)
app.include_router(experiments.router)

if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    logger.warning(f"Frontend directory not found at {FRONTEND_DIR}; only the JSON API will be served.")


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))
    reload_enabled = os.environ.get("API_RELOAD", "false").lower() == "true"
    uvicorn.run("api.main:app", host=host, port=port, reload=reload_enabled)
