"""Real-time endpoints: the WebSocket push feed and its REST counterparts
(recent history + connection/feed status)."""

from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api import db
from api.live_feed import broadcaster, manager

router = APIRouter(tags=["live"])


@router.websocket("/ws/live")
async def ws_live(websocket: WebSocket) -> None:
    """Server-push feed: every ~few seconds (LIVE_FEED_INTERVAL_SECONDS) a
    freshly scored synthetic transaction is broadcast to all connected
    clients as `{"type": "transaction", "data": {...}}`. Clients aren't
    expected to send anything; we still await receive to detect disconnects."""
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(websocket)


@router.get("/api/transactions/live/recent", tags=["transactions"])
def recent_live_transactions(limit: int = 50) -> list[dict]:
    return db.fetch_recent(limit=min(limit, 500))


@router.get("/api/transactions/live/status", tags=["transactions"])
def live_feed_status() -> dict:
    return {
        "active_websocket_connections": manager.active_count,
        "total_recorded": db.count_total(),
        "feed_interval_seconds": broadcaster.interval_seconds,
    }
