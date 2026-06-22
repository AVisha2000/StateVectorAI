"""Workflow-oriented lab summaries for the dashboard."""
from __future__ import annotations

import json
from collections import Counter

from ..research_protocol import classify_claim, resource_normalized_delta
from ..resultsdb import ResultsDB
from .analogues import analogue_status_for_job
from .evidence import comparison_evidence_ladder
from .gpu_reservation import gpu_reservation_status, job_reservation
from .model_graph import model_family, uses_quantum_config
from .presets import preset_meta
from .workspace import comparison_payload


def _quantum_scale(job: dict) -> dict | None:
    config = job.get("config") or {}
    try:
        qubits = int(config.get("lab.quantum_override.n_qubits"))
        depth = int(config.get("lab.quantum_override.n_circuit_layers"))
    except (TypeError, ValueError):
        return None
    return {"n_qubits": qubits, "n_circuit_layers": depth}


def _job_variant(job: dict) -> str:
    if job.get("run_key"):
        parts = str(job["run_key"]).split("/")
        if len(parts) >= 2:
            return parts[1]
    scale = _quantum_scale(job)
    if scale:
        return f"{job['preset_id']}-q{scale['n_qubits']}-d{scale['n_circuit_layers']}"
    return job["preset_id"]


def _final_run_for_job(db: ResultsDB, job: dict) -> dict | None:
    return db.get_run(
        "lab",
        _job_variant(job),
        job["dataset_name"],
        int(job["seed"]),
        int(job["steps"]),
    )


def enrich_job(job: dict, db: ResultsDB | None = None) -> dict:
    out = dict(job)
    config = out.get("config")
    if config is None:
        try:
            config = json.loads(out.get("config_json") or "{}")
        except json.JSONDecodeError:
            config = {}
    out["config"] = config
    try:
        preset = preset_meta(out["preset_id"])
    except Exception:
        preset = {}
    out["kind"] = preset.get("kind", "unknown")
    out["uses_quantum"] = uses_quantum_config(config)
    out["model_family"] = model_family(config)
    out["resource_band"] = config.get("lab.resource.band")
    out["resource_score"] = config.get("lab.resource.score")
    out["gpu_reservation"] = job_reservation(out)
    out.update(analogue_status_for_job(db, out))
    if out.get("compare_to_job_id"):
        out["comparison_state"] = "linked"
    elif out.get("analogue_state") == "missing":
        out["comparison_state"] = "missing"
    elif preset.get("classical_twin_id"):
        out["comparison_state"] = "available"
    else:
        out["comparison_state"] = "none"
    out["elapsed_or_wall_seconds"] = None
    return out


def fairness_flags(candidate: dict | None, baseline: dict | None) -> dict:
    if not candidate or not baseline:
        return {
            "complete": False,
            "same_dataset": False,
            "same_seed": False,
            "same_steps": False,
            "same_eval_interval": False,
            "same_device_target": False,
            "role_validation": False,
            "parameter_delta_ratio": None,
        }
    cjob, bjob = candidate["job"], baseline["job"]
    cparams = (candidate.get("final_run") or {}).get("n_params")
    bparams = (baseline.get("final_run") or {}).get("n_params")
    cconfig = cjob.get("config") or {}
    bconfig = bjob.get("config") or {}
    training_fields = (
        "train.batch_size",
        "train.seq_len",
        "train.lr",
        "train.weight_decay",
        "train.grad_clip",
    )
    preprocessing_fields = (
        "data.kind",
        "data.corpus_path",
        "data.val_fraction",
    )
    matched_config_fields = {
        key: cconfig.get(key) == bconfig.get(key)
        for key in (*training_fields, *preprocessing_fields)
    }
    ratio = None
    if cparams is not None and bparams:
        ratio = (cparams - bparams) / max(abs(bparams), 1)
    return {
        "complete": bool(candidate.get("final_run") and baseline.get("final_run")),
        "same_dataset": cjob.get("dataset_name") == bjob.get("dataset_name"),
        "same_seed": int(cjob.get("seed", -1)) == int(bjob.get("seed", -2)),
        "same_steps": int(cjob.get("steps", -1)) == int(bjob.get("steps", -2)),
        "same_eval_interval": int(cjob.get("eval_every", -1)) == int(bjob.get("eval_every", -2)),
        "same_device_target": (cjob.get("device_target") or "auto") == (bjob.get("device_target") or "auto"),
        "same_training_budget": all(
            matched_config_fields[key] for key in training_fields
        ),
        "same_preprocessing": all(
            matched_config_fields[key] for key in preprocessing_fields
        ),
        "matched_config_fields": matched_config_fields,
        "role_validation": (
            cjob.get("comparison_role") == "candidate"
            and bjob.get("comparison_role") == "baseline"
        ),
        "parameter_delta_ratio": ratio,
    }


def verdict_for_comparison(payload: dict) -> dict:
    if not payload.get("available"):
        return {"label": "incomplete", "reason": payload.get("reason", "comparison missing")}
    flags = fairness_flags(payload.get("candidate"), payload.get("baseline"))
    required = [
        "same_dataset", "same_seed", "same_steps", "same_eval_interval",
        "same_device_target", "same_training_budget", "same_preprocessing",
        "role_validation",
    ]
    if not flags["complete"]:
        return {"label": "incomplete", "reason": "one or both runs have not finished", "fairness": flags}
    if not all(flags[k] for k in required):
        return {"label": "insufficient fairness", "reason": "protocol fields do not match", "fairness": flags}
    delta = (payload.get("deltas") or {}).get("val_ppl")
    if delta is None:
        return {"label": "needs review", "reason": "validation perplexity is unavailable", "fairness": flags}
    verdict = classify_claim(
        fairness=flags,
        single_delta=-float(delta),  # val_ppl is lower-is-better.
        metric_name="validation perplexity",
    )
    verdict["fairness"] = flags
    return verdict


def _resource_normalized_for_payload(payload: dict) -> dict | None:
    candidate = payload.get("candidate") or {}
    baseline = payload.get("baseline") or {}
    crun = candidate.get("final_run")
    brun = baseline.get("final_run")
    if not crun or not brun:
        return None
    if crun.get("val_ppl") is None or brun.get("val_ppl") is None:
        return None
    return resource_normalized_delta(
        candidate_metric=float(crun["val_ppl"]),
        baseline_metric=float(brun["val_ppl"]),
        candidate_wall_seconds=crun.get("wall_seconds"),
        baseline_wall_seconds=brun.get("wall_seconds"),
        lower_is_better=True,
    )


def comparison_research_payload(db: ResultsDB, job_id: int) -> dict:
    payload = comparison_payload(db, job_id)
    verdict = verdict_for_comparison(payload)
    payload["fairness"] = verdict.get("fairness") or fairness_flags(
        payload.get("candidate"), payload.get("baseline")
    )
    payload["verdict"] = {k: v for k, v in verdict.items() if k != "fairness"}
    payload["resource_normalized"] = _resource_normalized_for_payload(payload)
    payload["evidence_ladder"] = comparison_evidence_ladder(payload)
    return payload


def lab_overview(db: ResultsDB, status_payload: dict) -> dict:
    jobs = [enrich_job(j, db) for j in db.fetch_lab_jobs(limit=200)]
    counts = Counter(j["status"] for j in jobs)
    active = [j for j in jobs if j["status"] in {"queued", "running"}][:8]
    failed = [j for j in jobs if j["status"] == "error"][:6]
    linked = [j for j in jobs if j.get("compare_to_job_id") and j.get("comparison_role") != "baseline"]
    recent_comparisons = []
    for job in linked[:5]:
        payload = comparison_research_payload(db, int(job["id"]))
        recent_comparisons.append({
            "job_id": job["id"],
            "candidate": payload.get("candidate", {}).get("job") if payload.get("candidate") else None,
            "baseline": payload.get("baseline", {}).get("job") if payload.get("baseline") else None,
            "deltas": payload.get("deltas"),
            "verdict": payload.get("verdict"),
            "fairness": payload.get("fairness"),
        })
    with db._conn() as con:
        rows = con.execute(
            "SELECT variant, dataset, MIN(val_ppl) best_ppl, MAX(ts) last_ts "
            "FROM runs GROUP BY variant, dataset ORDER BY best_ppl ASC LIMIT 5"
        ).fetchall()
    return {
        "counts": dict(counts),
        "active_jobs": active,
        "recent_failed_jobs": failed,
        "recent_comparisons": recent_comparisons,
        "gpu_status": status_payload.get("gpu", {}),
        "gpu_reservation": gpu_reservation_status(db),
        "leaderboard_highlights": [dict(r) for r in rows],
    }


def scaling_tests_overview(db: ResultsDB) -> list[dict]:
    jobs = [enrich_job(j, db) for j in db.fetch_lab_jobs(limit=500)]
    groups: dict[str, list[dict]] = {}
    for job in jobs:
        if not job.get("group_id") or not _quantum_scale(job):
            continue
        groups.setdefault(job["group_id"], []).append(job)
    out = []
    for group_id, items in groups.items():
        if len(items) < 2:
            continue
        scales = [_quantum_scale(j) for j in items]
        counts = Counter(j["status"] for j in items)
        out.append({
            "group_id": group_id,
            "count": len(items),
            "preset_id": items[0]["preset_id"],
            "dataset_name": items[0]["dataset_name"],
            "seed": items[0]["seed"],
            "steps": items[0]["steps"],
            "device_target": items[0].get("device_target") or "auto",
            "statuses": dict(counts),
            "qubits": sorted({s["n_qubits"] for s in scales if s}),
            "depths": sorted({s["n_circuit_layers"] for s in scales if s}),
            "updated_ts": max(j.get("updated_ts") or "" for j in items),
        })
    return sorted(out, key=lambda item: item["updated_ts"], reverse=True)


def scaling_test_payload(db: ResultsDB, group_id: str) -> dict:
    jobs = [
        enrich_job(j, db) for j in db.fetch_lab_jobs(limit=500)
        if j.get("group_id") == group_id and _quantum_scale(enrich_job(j, db))
    ]
    if not jobs:
        return {"available": False, "reason": "scaling test not found", "group_id": group_id}
    points = []
    for job in jobs:
        scale = _quantum_scale(job) or {}
        final_run = _final_run_for_job(db, job)
        points.append({
            "job": job,
            "n_qubits": scale.get("n_qubits"),
            "n_circuit_layers": scale.get("n_circuit_layers"),
            "scale": (scale.get("n_qubits") or 0) * (scale.get("n_circuit_layers") or 0),
            "status": job["status"],
            "variant": _job_variant(job),
            "val_loss": final_run.get("val_loss") if final_run else None,
            "val_ppl": final_run.get("val_ppl") if final_run else None,
            "val_bpc": final_run.get("val_bpc") if final_run else None,
            "wall_seconds": final_run.get("wall_seconds") if final_run else None,
            "n_params": final_run.get("n_params") if final_run else None,
        })
    points.sort(key=lambda p: (p["n_qubits"], p["n_circuit_layers"]))
    complete = [p for p in points if p["val_ppl"] is not None]
    best = min(complete, key=lambda p: p["val_ppl"]) if complete else None
    return {
        "available": True,
        "group_id": group_id,
        "preset_id": jobs[0]["preset_id"],
        "dataset_name": jobs[0]["dataset_name"],
        "seed": jobs[0]["seed"],
        "steps": jobs[0]["steps"],
        "device_target": jobs[0].get("device_target") or "auto",
        "points": points,
        "best": best,
        "complete_count": len(complete),
        "total_count": len(points),
    }
