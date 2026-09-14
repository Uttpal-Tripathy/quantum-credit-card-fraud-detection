"""Lightweight SQLite persistence for the real-time live transaction feed.

Every transaction scored by the WebSocket broadcaster (api/live_feed.py) or
the REST scoring endpoint is written here, so the feed survives process
restarts and every connected client (or a client that connects late) sees a
consistent history — not just whatever happens to still be in memory.

Uses the stdlib `sqlite3` module (no new dependency) in WAL mode for decent
concurrent read/write behavior under a single-process deployment. A fresh
connection is opened per call rather than shared across threads — simpler
and safer than manual locking, and cheap enough at this request volume.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from src.config.settings import REPO_ROOT

DB_PATH = REPO_ROOT / "api" / "data" / "live_feed.sqlite3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS live_transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    amount REAL,
    classical_risk REAL NOT NULL,
    quantum_risk REAL,
    final_risk REAL NOT NULL,
    routed_to_quantum INTEGER NOT NULL,
    decision TEXT NOT NULL,
    synthetic_ground_truth_label INTEGER,
    classical_latency_ms REAL,
    quantum_latency_ms REAL
);
CREATE INDEX IF NOT EXISTS idx_live_transactions_created_at ON live_transactions(created_at DESC);
"""


@contextmanager
def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.row_factory = sqlite3.Row
    # Idempotent and cheap enough to run per-connection: makes this module
    # self-healing regardless of whether the FastAPI app's lifespan (which
    # also calls init_db() once at startup) actually ran first — e.g. under
    # a test client used outside a `with` block, or a standalone script.
    conn.executescript(_SCHEMA)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect():
        pass


def insert_transaction(record: dict) -> int:
    """Insert one scored transaction. `record` matches the shape returned by
    api.services.score_new_transaction, plus a `created_at` ISO timestamp."""
    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO live_transactions
                (transaction_id, created_at, amount, classical_risk, quantum_risk, final_risk,
                 routed_to_quantum, decision, synthetic_ground_truth_label, classical_latency_ms, quantum_latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["transaction_id"], record["created_at"], record.get("amount"),
                record["classical_risk"], record.get("quantum_risk"), record["final_risk"],
                int(record["routed_to_quantum"]), record["decision"],
                record.get("synthetic_ground_truth_label"),
                record.get("classical_latency_ms"), record.get("quantum_latency_ms"),
            ),
        )
        return cursor.lastrowid


def fetch_recent(limit: int = 50) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM live_transactions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]


def count_total() -> int:
    with _connect() as conn:
        return conn.execute("SELECT COUNT(*) AS n FROM live_transactions").fetchone()["n"]


def clear_all() -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM live_transactions")
