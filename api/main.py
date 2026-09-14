"""QGFDA live web app: FastAPI backend serving both the JSON API and the
static HTML5/CSS/JS frontend, plus a real-time WebSocket transaction feed.

Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000          # development
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1   # production (see note below)

Or:
    python -m api.main

Production note: the demo pipeline singleton (api/services.py) and the
WebSocket connection manager (api/live_feed.py) both live in this process's
memory. That is correct and efficient for a SINGLE worker process, but is
NOT automatically shared across multiple worker processes. Scale this
service horizontally by running multiple single-worker containers behind a
load balancer with sticky WebSocket routing (or migrate the connection
manager to a pub/sub broker such as Redis) rather than raising --workers on
one process. See docs/reproducibility.md and the Dockerfile for the
recommended deployment shape.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

load_dotenv(REPO_ROOT / ".env")

from api import db, services  # noqa: E402
from api.live_feed import broadcaster  # noqa: E402
from api.routers import drift, experiments, feature_selection, live, overview, quantum, transactions  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger("api.main")

FRONTEND_DIR = REPO_ROOT / "frontend"
ENVIRONMENT = os.environ.get("ENVIRONMENT", "development").lower()
IS_PRODUCTION = ENVIRONMENT == "production"
CORS_ORIGINS = [o.strip() for o in os.environ.get("API_CORS_ORIGINS", "*").split(",") if o.strip()]
RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "20"))
RATE_LIMITED_PATHS = ("/api/experiments/run", "/api/quantum/circuit", "/api/feature-selection")


def _prewarm_demo_state() -> None:
    try:
        services.get_demo_state()
    except Exception:  # noqa: BLE001 - a failed prewarm must not crash the server
        logger.exception("Failed to prewarm the demo QGFDA pipeline; it will build lazily on first request.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting QGFDA API (environment={ENVIRONMENT})...")
    db.init_db()
    threading.Thread(target=_prewarm_demo_state, daemon=True).start()
    broadcaster.start()
    yield
    await broadcaster.stop()
    logger.info("Shutting down QGFDA API.")


app = FastAPI(
    title="QGFDA API",
    description="Quantum-Gated Fraud Detection Architecture — live backend. Research prototype, not for "
                 "production financial authorization.",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if not IS_PRODUCTION else None,
    redoc_url="/redoc" if not IS_PRODUCTION else None,
    openapi_url="/openapi.json" if not IS_PRODUCTION else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Structured access logging: method, path, status, latency — the
    minimum observability needed to operate this in production."""

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(f'{request.method} {request.url.path} -> {response.status_code} ({duration_ms:.1f}ms)')
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory fixed-window rate limiter for the genuinely
    expensive endpoints (experiment runs, circuit rendering, QAFS) —
    protects a single-process deployment from being knocked over by a
    request flood without pulling in Redis/slowapi for a research demo."""

    def __init__(self, app):
        super().__init__(app)
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    async def dispatch(self, request: Request, call_next):
        if request.url.path in RATE_LIMITED_PATHS:
            client_ip = request.client.host if request.client else "unknown"
            key = f"{client_ip}:{request.url.path}"
            now = time.time()
            with self._lock:
                window = [t for t in self._hits.get(key, []) if now - t < 60.0]
                if len(window) >= RATE_LIMIT_PER_MINUTE:
                    return JSONResponse(
                        status_code=429,
                        content={"detail": f"Rate limit exceeded ({RATE_LIMIT_PER_MINUTE}/min) for this endpoint. Try again shortly."},
                    )
                window.append(now)
                self._hits[key] = window
        return await call_next(request)


app.add_middleware(RateLimitMiddleware)
app.add_middleware(RequestLoggingMiddleware)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception on {request.method} {request.url.path}")
    detail = "Internal server error." if IS_PRODUCTION else f"{type(exc).__name__}: {exc}"
    return JSONResponse(status_code=500, content={"detail": detail})


@app.get("/api/health", tags=["meta"])
def health() -> dict:
    """Liveness probe — always returns 200 once the process is up, even
    while the demo pipeline is still warming up. Use /api/ready to check
    whether the pipeline itself is ready to serve real inference."""
    return {"status": "ok", "service": "qgfda-api", "environment": ENVIRONMENT,
            "notice": "Research Prototype — Not for Production Financial Authorization."}


@app.get("/api/ready", tags=["meta"])
def ready() -> JSONResponse:
    """Readiness probe — 200 once the demo QGFDA pipeline has finished
    warming up (loaded from disk or freshly trained), 503 otherwise. Never
    triggers a build itself, so it can't block on the first call."""
    if services.is_ready():
        return JSONResponse(status_code=200, content={"ready": True})
    return JSONResponse(status_code=503, content={"ready": False, "detail": "Demo pipeline still warming up."})


app.include_router(overview.router)
app.include_router(transactions.router)
app.include_router(quantum.router)
app.include_router(feature_selection.router)
app.include_router(drift.router)
app.include_router(experiments.router)
app.include_router(live.router)

if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    logger.warning(f"Frontend directory not found at {FRONTEND_DIR}; only the JSON API will be served.")


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))
    reload_enabled = os.environ.get("API_RELOAD", "false").lower() == "true" and not IS_PRODUCTION
    uvicorn.run("api.main:app", host=host, port=port, reload=reload_enabled)
