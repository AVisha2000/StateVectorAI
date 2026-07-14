import asyncio
import json

from qllm.dashboard.live_stream import (
    build_status_payload,
    encode_sse_comment,
    encode_sse_event,
    jobs_event_stream,
    live_snapshot,
    stream_client_allowed,
)
from qllm.resultsdb import ResultsDB


def _factory(path):
    return lambda: ResultsDB(path)


def _job(status: str, name: str) -> dict:
    return {
        "status": status,
        "preset_id": "classical-small",
        "dataset_name": "default-text",
        "run_name": name,
        "seed": 0,
        "steps": 2,
        "eval_every": 1,
    }


def test_status_payload_has_exact_keys_types_and_durable_counts(tmp_path):
    path = tmp_path / "stream.db"
    db = ResultsDB(path)
    db.create_lab_jobs([_job("queued", "queued"), _job("running", "running")])
    with db._conn() as con:
        con.execute("INSERT INTO runs (ts, suite, variant, dataset, seed, steps, n_params) VALUES (?,?,?,?,?,?,?)", ("2026-01-01T00:00:00", "s", "v", "d", 0, 1, 1))
        con.execute("INSERT INTO runs (ts, suite, variant, dataset, seed, steps, n_params) VALUES (?,?,?,?,?,?,?)", ("2026-01-01T00:00:00", "s", "v", "d", 1, 1, 1))

    payload = build_status_payload(_factory(path), worker="worker-1", gpu_available=1)

    assert set(payload) == {"worker", "gpu_available", "queued", "running", "runs"}
    assert payload == {"worker": "worker-1", "gpu_available": True, "queued": 1, "running": 1, "runs": 2}
    assert isinstance(payload["worker"], str)
    assert isinstance(payload["gpu_available"], bool)
    assert all(isinstance(payload[key], int) for key in ("queued", "running", "runs"))


def test_stream_client_gate_is_always_loopback_only(monkeypatch):
    monkeypatch.setenv("QLLM_ALLOW_REMOTE", "1")
    assert stream_client_allowed("127.0.0.1")
    assert stream_client_allowed("::1")
    assert not stream_client_allowed("192.0.2.10")
    assert not stream_client_allowed(None)


def test_live_snapshot_token_is_stable_and_detects_same_second_change(tmp_path):
    path = tmp_path / "stream.db"
    db = ResultsDB(path)
    job_id = db.create_lab_job(_job("queued", "same-second"))
    db.start_run("run-1", "run", "suite", "variant", "data", 0, 5)

    first = live_snapshot(_factory(path))
    assert live_snapshot(_factory(path)) == first
    assert set(first["jobs"][0]) == {"id", "status", "updated_ts", "run_key", "worker_id", "completed_step", "run_uuid"}
    assert set(first["live_runs"][0]) == {
        "run_key",
        "run_uuid",
        "status",
        "current_step",
        "total_steps",
        "updated_ts",
        "last_train_loss",
        "last_val_ppl",
        "primary_metric_name",
        "last_primary_metric_value",
    }
    assert first["live_runs"][0]["primary_metric_name"] == "val_ppl"
    assert first["live_runs"][0]["last_primary_metric_value"] is None

    with db._conn() as con:
        con.execute("UPDATE lab_jobs SET completed_step=1, updated_ts=? WHERE id=?", (first["jobs"][0]["updated_ts"], job_id))
    changed = live_snapshot(_factory(path))
    assert changed["jobs"][0]["updated_ts"] == first["jobs"][0]["updated_ts"]
    assert changed["change_token"] != first["change_token"]


def test_sse_encoding_is_protocol_compliant():
    assert encode_sse_event("jobs", {"id": 7}) == 'event: jobs\ndata: {"id":7}\n\n'
    assert encode_sse_event("jobs", {"id": 7}, event_id="abc") == (
        'id: abc\nevent: jobs\ndata: {"id":7}\n\n'
    )
    assert encode_sse_comment("still\nalive") == ": still\n: alive\n\n"


class _Request:
    def __init__(self, disconnected):
        self._disconnected = iter(disconnected)

    async def is_disconnected(self):
        return next(self._disconnected)


def test_jobs_event_stream_emits_deltas_only_and_stops_on_disconnect(tmp_path):
    path = tmp_path / "stream.db"
    db = ResultsDB(path)
    job_id = db.create_lab_job(_job("queued", "stream"))
    request = _Request([False, False, True])
    calls = []

    async def sleep(_seconds):
        calls.append(_seconds)
        if len(calls) == 1:
            with ResultsDB(path)._conn() as con:
                con.execute("UPDATE lab_jobs SET completed_step=1 WHERE id=?", (job_id,))

    async def collect():
        return [chunk async for chunk in jobs_event_stream(request, db_factory=_factory(path), poll_seconds=0.01, heartbeat_seconds=99, sleep=sleep)]

    events = asyncio.run(collect())
    assert len(events) == 2
    assert all(event.startswith("id: ") for event in events)
    first = json.loads(events[0].split("data: ", 1)[1])
    second = json.loads(events[1].split("data: ", 1)[1])
    assert first["change_token"] != second["change_token"]
    assert calls == [0.01, 0.01]


def test_jobs_event_stream_emits_heartbeat_when_unchanged(tmp_path):
    path = tmp_path / "stream.db"
    ResultsDB(path).create_lab_job(_job("queued", "heartbeat"))
    request = _Request([False, False, True])
    times = iter([0.0, 0.0, 16.0, 16.0])

    async def sleep(_seconds):
        return None

    async def collect():
        return [
            chunk
            async for chunk in jobs_event_stream(
                request,
                db_factory=_factory(path),
                poll_seconds=0.01,
                heartbeat_seconds=15.0,
                sleep=sleep,
                monotonic=lambda: next(times),
            )
        ]

    events = asyncio.run(collect())
    assert len(events) == 2
    assert events[0].startswith("id: ")
    assert events[1] == ": keep-alive\n\n"
