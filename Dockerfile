# Production image for the QGFDA live web app (FastAPI + HTML5 frontend).
#
# Build:  docker build -t qgfda-api .
# Run:    docker run -p 8000:8000 --env-file .env qgfda-api
#
# NOTE ON SCALING: the in-process demo-pipeline singleton and WebSocket
# connection manager (api/services.py, api/live_feed.py) live in this
# container's memory. Run exactly ONE uvicorn worker per container (as
# below) and scale out by running multiple containers behind a load
# balancer with sticky WebSocket routing, rather than raising --workers
# inside a single container. See api/main.py's module docstring.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ENVIRONMENT=production \
    API_HOST=0.0.0.0 \
    API_PORT=8000

WORKDIR /app

# matplotlib (used for quantum circuit rendering) needs a minimal set of
# system libraries even with the non-interactive Agg backend.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfreetype6 \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY frontend/ ./frontend/
COPY src/ ./src/
COPY configs/ ./configs/
COPY data/README.md ./data/README.md

# Non-root runtime user.
RUN useradd --create-home --shell /bin/bash qgfda \
    && mkdir -p /app/models/demo_pipeline /app/api/data /app/experiments/logs \
    && chown -R qgfda:qgfda /app
USER qgfda

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health', timeout=3)" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
