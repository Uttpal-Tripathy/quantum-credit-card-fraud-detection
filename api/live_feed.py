"""Real-time live transaction feed: a background asyncio task that
periodically scores a fresh synthetic transaction through the actual QGFDA
pipeline, persists it (api/db.py), and pushes it to every connected
WebSocket client. This is the "real-time" piece of the live web app — the
frontend's Live Transactions tab no longer has to poll; it receives pushes.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from datetime import datetime, timezone

from fastapi import WebSocket

from api import db, services
from src.utils.logging import get_logger

logger = get_logger("api.live_feed")


class ConnectionManager:
    """Tracks connected WebSocket clients and broadcasts to all of them,
    dropping any connection that errors out (client disconnected, etc.)."""

    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)
        logger.info(f"WebSocket client connected ({len(self._connections)} active).")

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)
        logger.info(f"WebSocket client disconnected ({len(self._connections)} active).")

    async def broadcast(self, message: dict) -> None:
        payload = json.dumps(message)
        async with self._lock:
            targets = list(self._connections)
        for ws in targets:
            try:
                await ws.send_text(payload)
            except Exception:  # noqa: BLE001 - a dead connection must not break the broadcast loop
                await self.disconnect(ws)

    @property
    def active_count(self) -> int:
        return len(self._connections)


manager = ConnectionManager()


def _score_and_persist_one() -> dict:
    """Runs synchronously (called via asyncio.to_thread) — draws a fresh
    synthetic transaction, scores it through the live pipeline, and persists
    it to SQLite. A randomized amount multiplier gives the feed visible
    variety instead of every transaction looking identical."""
    multiplier = round(random.uniform(0.3, 6.0), 2)
    record = services.score_new_transaction(amount_multiplier=multiplier)
    record["created_at"] = datetime.now(timezone.utc).isoformat()
    db.insert_transaction(record)
    return record


class LiveFeedBroadcaster:
    """Owns the background task that drives the feed. Started/stopped from
    api/main.py's lifespan so it starts with the app and shuts down cleanly."""

    def __init__(self, interval_seconds: float | None = None) -> None:
        self.interval_seconds = interval_seconds or float(os.environ.get("LIVE_FEED_INTERVAL_SECONDS", "3"))
        self._task: asyncio.Task | None = None

    async def _run(self) -> None:
        logger.info(f"Live feed broadcaster started (interval={self.interval_seconds}s).")
        while True:
            try:
                record = await asyncio.to_thread(_score_and_persist_one)
                await manager.broadcast({"type": "transaction", "data": record})
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - one bad tick must not kill the feed forever
                logger.exception("Live feed tick failed; will retry after the usual interval.")
            await asyncio.sleep(self.interval_seconds)

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            logger.info("Live feed broadcaster stopped.")


broadcaster = LiveFeedBroadcaster()
