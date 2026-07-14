"""Small, durable SQLite projections for dashboard live updates.

This module deliberately has no route or queue ownership.  A server route can
provide its request object and a factory for the canonical ``ResultsDB``.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

from pydantic import BaseModel

from ..resultsdb import ResultsDB
from .security import is_loopback_host


DEFAULT_SNAPSHOT_LIMIT = 100
MAX_SNAPSHOT_LIMIT = 250
_HEARTBEAT = "keep-alive"


class StatusResponse(BaseModel):
    worker: str
    gpu_available: bool
    queued: int
    running: int
    runs: int

    class Config:
        extra = "forbid"


def stream_client_allowed(client_host: str | None) -> bool:
    """Keep the live stream loopback-only even when other routes allow remote."""
    return is_loopback_host(client_host)


def build_status_payload(
    db_factory: Callable[[], ResultsDB], *, worker: str, gpu_available: bool
) -> dict[str, str | bool | int]:
    """Return the durable, intentionally small dashboard status projection."""
    store = db_factory()
    with store._conn() as con:
        queued = con.execute(
            "SELECT COUNT(*) FROM lab_jobs WHERE status='queued'"
        ).fetchone()[0]
        running = con.execute(
            "SELECT COUNT(*) FROM lab_jobs WHERE status='running'"
        ).fetchone()[0]
        runs = con.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    return {
        "worker": str(worker),
        "gpu_available": bool(gpu_available),
        "queued": int(queued),
        "running": int(running),
        "runs": int(runs),
    }


def live_snapshot(
    db_factory: Callable[[], ResultsDB], *, limit: int = DEFAULT_SNAPSHOT_LIMIT
) -> dict[str, object]:
    """Project bounded durable queue/live state and a content-addressed token.

    The token hashes the projected values, rather than timestamps, so an update
    remains observable even when SQLite's timestamp precision is one second.
    Connections are scoped to this function and closed before the snapshot is
    returned.
    """
    if not 1 <= int(limit) <= MAX_SNAPSHOT_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_SNAPSHOT_LIMIT}")
    store = db_factory()
    with store._conn() as con:
        jobs = [
            dict(row)
            for row in con.execute(
                "SELECT id, status, updated_ts, run_key, worker_id, "
                "completed_step, run_uuid FROM lab_jobs "
                "ORDER BY updated_ts DESC, id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        ]
        runs = [
            dict(row)
            for row in con.execute(
                "SELECT run_key, run_uuid, status, current_step, total_steps, "
                "updated_ts, last_train_loss, last_val_ppl, "
                "primary_metric_name, last_primary_metric_value FROM live_runs "
                "ORDER BY updated_ts DESC, run_key ASC LIMIT ?",
                (int(limit),),
            ).fetchall()
        ]
    content = {"jobs": jobs, "live_runs": runs}
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        **content,
        "change_token": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    }


def encode_sse_event(
    event: str, payload: object, *, event_id: str | None = None
) -> str:
    """Encode one SSE event using the event-stream field grammar."""
    if "\r" in event or "\n" in event:
        raise ValueError("SSE event names cannot contain newlines")
    if event_id is not None and ("\r" in event_id or "\n" in event_id):
        raise ValueError("SSE event IDs cannot contain newlines")
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    prefix = f"id: {event_id}\n" if event_id is not None else ""
    return f"{prefix}event: {event}\ndata: {body}\n\n"


def encode_sse_comment(comment: str = _HEARTBEAT) -> str:
    """Encode a bounded SSE comment, one protocol line per input line."""
    lines = str(comment).replace("\r", "").split("\n")
    return "".join(f": {line}\n" for line in lines) + "\n"


async def _is_disconnected(request: Any) -> bool:
    result = request.is_disconnected()
    return bool(await result) if inspect.isawaitable(result) else bool(result)


async def jobs_event_stream(
    request: Any,
    *,
    db_factory: Callable[[], ResultsDB],
    poll_seconds: float = 1.0,
    heartbeat_seconds: float = 15.0,
    snapshot_limit: int = DEFAULT_SNAPSHOT_LIMIT,
    sleep: Callable[[float], Any] = asyncio.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> AsyncIterator[str]:
    """Yield an initial jobs event, deltas, and bounded keep-alive comments."""
    if poll_seconds <= 0 or heartbeat_seconds <= 0:
        raise ValueError("poll_seconds and heartbeat_seconds must be positive")

    previous_token: str | None = None
    last_heartbeat = monotonic()
    while not await _is_disconnected(request):
        snapshot = live_snapshot(db_factory, limit=snapshot_limit)
        token = str(snapshot["change_token"])
        if token != previous_token:
            yield encode_sse_event("jobs", snapshot, event_id=token)
            previous_token = token
            last_heartbeat = monotonic()
        elif monotonic() - last_heartbeat >= heartbeat_seconds:
            yield encode_sse_comment()
            last_heartbeat = monotonic()

        result = sleep(poll_seconds)
        if inspect.isawaitable(result):
            await result
