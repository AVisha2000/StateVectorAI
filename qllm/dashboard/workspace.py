"""Run-workspace payloads for the QLLM Lab UI."""
from __future__ import annotations

import json
from collections import defaultdict

from ..resultsdb import ResultsDB
from .datasets import get_dataset
from .presets import preset_meta


def _decode_config(row: dict | None) -> dict:
    if not row:
        return {}
    try:
        return json.loads(row.get("config_json") or "{}")
    except json.JSONDecodeError:
        return {}


def _curve(db: ResultsDB, run_key: str | None) -> dict:
    if not run_key:
        return {}
    series: dict[str, list] = defaultdict(list)
    for step in db.fetch_steps(run_key):
        series[step["name"]].append({"step": step["step"], "value": step["value"]})
    return dict(series)


def _live(db: ResultsDB, run_key: str | None) -> dict | None:
    if not run_key:
        return None
    with db._conn() as con:
        row = con.execute(
            "SELECT * FROM live_runs WHERE run_key=?", (run_key,)
        ).fetchone()
    return dict(row) if row is not None else None


def _final_run(db: ResultsDB, job: dict | None) -> dict | None:
    if not job:
        return None
    if job.get("status") != "done":
        return None
    variant = job["preset_id"]
    if job.get("run_key"):
        parts = str(job["run_key"]).split("/")
        if len(parts) >= 2:
            variant = parts[1]
    run = db.get_run(
        "lab",
        variant,
        job["dataset_name"],
        int(job["seed"]),
        int(job["steps"]),
    )
    if run:
        run["config"] = _decode_config(run)
    return run


def _job(db: ResultsDB, job_id: int | None) -> dict | None:
    if job_id is None:
        return None
    row = db.get_lab_job(int(job_id))
    if not row:
        return None
    row["config"] = _decode_config(row)
    return row


def _job_payload(db: ResultsDB, job: dict | None) -> dict | None:
    if not job:
        return None
    preset = preset_meta(job["preset_id"])
    return {
        "job": job,
        "preset": preset,
        "dataset": get_dataset(db, job["dataset_name"]),
        "live": _live(db, job.get("run_key")),
        "curve": _curve(db, job.get("run_key")),
        "final_run": _final_run(db, job),
    }


def comparison_payload(db: ResultsDB, job_id: int) -> dict:
    job = _job(db, job_id)
    if not job:
        return {"available": False, "reason": "job not found"}
    other = _job(db, job.get("compare_to_job_id"))
    if not other:
        return {
            "available": False,
            "reason": "no linked classical comparison",
            "candidate": _job_payload(db, job),
            "baseline": None,
            "deltas": None,
        }

    if job.get("comparison_role") == "baseline":
        baseline, candidate = job, other
    else:
        candidate, baseline = job, other
    candidate_payload = _job_payload(db, candidate)
    baseline_payload = _job_payload(db, baseline)
    candidate_run = candidate_payload["final_run"] if candidate_payload else None
    baseline_run = baseline_payload["final_run"] if baseline_payload else None
    deltas = None
    if candidate_run and baseline_run:
        deltas = {
            "val_loss": _delta(candidate_run, baseline_run, "val_loss"),
            "val_ppl": _delta(candidate_run, baseline_run, "val_ppl"),
            "val_bpc": _delta(candidate_run, baseline_run, "val_bpc"),
            "wall_seconds": _delta(candidate_run, baseline_run, "wall_seconds"),
            "n_params": _delta(candidate_run, baseline_run, "n_params"),
        }
    return {
        "available": True,
        "candidate": candidate_payload,
        "baseline": baseline_payload,
        "deltas": deltas,
    }


def _delta(candidate: dict, baseline: dict, key: str):
    if candidate.get(key) is None or baseline.get(key) is None:
        return None
    return candidate[key] - baseline[key]


def workspace_payload(db: ResultsDB, job_id: int) -> dict | None:
    job = _job(db, job_id)
    if not job:
        return None
    payload = _job_payload(db, job)
    payload["comparison"] = comparison_payload(db, job_id)
    return payload
