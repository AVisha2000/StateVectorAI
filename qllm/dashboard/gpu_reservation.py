"""GPU reservation metadata for local dashboard jobs."""
from __future__ import annotations

import json

from ..resultsdb import ResultsDB

ACTIVE_STATES = {"queued", "running"}
HIGH_MEMORY_BANDS = {"high", "extreme"}


def reservation_metadata(device_target: str, estimate: dict) -> dict:
    """Build queue-time reservation metadata from a resource estimate."""
    target = (device_target or "auto").lower()
    band = estimate.get("band") or "unknown"
    high_memory = band in HIGH_MEMORY_BANDS
    requires_gpu = target == "gpu"
    return {
        "required": requires_gpu,
        "lane": "exclusive-gpu" if requires_gpu else "standard",
        "state": "queued" if requires_gpu else "not_required",
        "reason": "gpu target requested" if requires_gpu else "no gpu target requested",
        "high_memory": high_memory,
        "memory_warning": (
            f"High-memory quantum simulation estimate: {band}."
            if high_memory else ""
        ),
    }


def apply_reservation_config(config: dict, metadata: dict) -> dict:
    out = dict(config)
    out["lab.gpu_reservation.required"] = bool(metadata["required"])
    out["lab.gpu_reservation.lane"] = metadata["lane"]
    out["lab.gpu_reservation.state"] = metadata["state"]
    out["lab.gpu_reservation.reason"] = metadata["reason"]
    out["lab.resource.high_memory_warning"] = metadata["memory_warning"]
    out["lab.resource.high_memory"] = bool(metadata["high_memory"])
    return out


def update_reservation_state(config: dict, state: str, job_id: int | None = None) -> dict:
    out = dict(config)
    if out.get("lab.gpu_reservation.required"):
        out["lab.gpu_reservation.state"] = state
        if job_id is not None and state == "active":
            out["lab.gpu_reservation.owner_job_id"] = int(job_id)
    return out


def _decode_config(job: dict) -> dict:
    config = job.get("config")
    if config is not None:
        return config
    try:
        return json.loads(job.get("config_json") or "{}")
    except json.JSONDecodeError:
        return {}


def job_reservation(job: dict) -> dict:
    config = _decode_config(job)
    required = bool(config.get("lab.gpu_reservation.required"))
    band = config.get("lab.resource.band") or "unknown"
    high_memory = bool(config.get("lab.resource.high_memory")) or band in HIGH_MEMORY_BANDS
    status = job.get("status")
    state = config.get("lab.gpu_reservation.state")
    if required and status == "running":
        state = "active"
    elif required and status == "queued":
        state = "queued"
    elif required and status in {"done", "error", "cancelled"}:
        state = "released"
    return {
        "required": required,
        "lane": config.get("lab.gpu_reservation.lane") or ("exclusive-gpu" if required else "standard"),
        "state": state or ("not_required" if not required else "queued"),
        "reason": config.get("lab.gpu_reservation.reason") or "",
        "high_memory": high_memory,
        "memory_warning": config.get("lab.resource.high_memory_warning") or (
            f"High-memory quantum simulation estimate: {band}." if high_memory else ""
        ),
        "resource_band": band,
        "resource_score": config.get("lab.resource.score"),
    }


def gpu_reservation_status(db: ResultsDB, limit: int = 1000) -> dict:
    jobs = db.fetch_lab_jobs(limit=limit)
    enriched = []
    for job in jobs:
        reservation = job_reservation(job)
        if job.get("status") in ACTIVE_STATES and (
            reservation["required"] or reservation["high_memory"]
        ):
            enriched.append({
                "id": job["id"],
                "run_name": job["run_name"],
                "preset_id": job["preset_id"],
                "dataset_name": job["dataset_name"],
                "status": job["status"],
                "device_target": job.get("device_target") or "auto",
                "reservation": reservation,
            })
    running = [j for j in enriched if j["status"] == "running" and j["reservation"]["required"]]
    waiting = [j for j in enriched if j["status"] == "queued" and j["reservation"]["required"]]
    high_memory = [j for j in enriched if j["reservation"]["high_memory"]]
    owner = running[0] if running else None
    if owner:
        state = "active"
    elif waiting:
        state = "waiting"
    else:
        state = "idle"
    return {
        "mode": "exclusive",
        "state": state,
        "owner": owner,
        "waiting": waiting[:8],
        "waiting_count": len(waiting),
        "high_memory_jobs": high_memory[:8],
        "high_memory_count": len(high_memory),
        "summary": (
            f"GPU lane reserved by job #{owner['id']}"
            if owner else (
                f"{len(waiting)} GPU job(s) waiting for the exclusive lane"
                if waiting else "GPU reservation lane is idle"
            )
        ),
    }
