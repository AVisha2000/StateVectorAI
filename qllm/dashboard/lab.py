"""Workflow-oriented lab summaries for the dashboard."""
from __future__ import annotations

import json
from collections import Counter

from ..claims import get_claim, infer_claim_id
from ..research_protocol import (
    classify_claim,
    evaluate_analogue_ladder,
    evaluate_fairness,
    normalize_seed_axes,
    resource_normalized_delta,
    two_stream_metric_contract,
)
from ..resultsdb import ResultsDB
from .analogues import analogue_status_for_job
from .evidence import comparison_evidence_ladder
from .gpu_reservation import gpu_reservation_status, job_reservation
from .model_graph import model_family, uses_quantum_config
from .presets import preset_meta
from .workspace import comparison_payload


_UNASSIGNED_COMPONENT_SCHEMA = {
    "schema_id": "unassigned_component_swap_v1",
    "required_equal": [
        "data.*",
        "job.dataset_name",
        "job.device_target",
        "job.eval_every",
        "job.seed",
        "job.steps",
        "seed_axes.generator",
        "seed_axes.initialization",
        "seed_axes.minibatch",
        "seed_axes.split",
        "train.batch_size",
        "train.eval_batches",
        "train.grad_clip",
        "train.lr",
        "train.seq_len",
        "train.weight_decay",
    ],
    "intentional_differences": [
        {
            "path": path,
            "reason": "Narrow automatic quantum-to-classical component swap.",
        }
        for path in (
            "model.arch",
            "model.rnn_hidden",
            "model.attn_type",
            "model.embed_type",
            "model.encoder_kind",
            "model.ffn_type",
            "model.head_type",
            "model.blocks.*.attn_type",
            "model.blocks.*.ffn_type",
            "model.blocks.*.quantum.*",
            "seed_axes.*circuit",
        )
    ],
}
PAIRABLE_VAL_PPL_METRIC_TYPES = frozenset({
    "strict_autoregressive_next_token",
    "validation_perplexity",
})


def _quantum_scale(job: dict) -> dict | None:
    config = job.get("config") or {}
    try:
        qubits = int(
            config.get("lab.quantum_override.n_qubits")
            if config.get("lab.quantum_override.n_qubits") is not None
            else config.get("lab.study_cell.n_qubits")
        )
        depth = int(
            config.get("lab.quantum_override.n_circuit_layers")
            if config.get("lab.quantum_override.n_circuit_layers") is not None
            else config.get("lab.study_cell.n_circuit_layers")
        )
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
    claim_id = infer_claim_id(
        explicit=config.get("research.claim_id"),
        preset_id=out.get("preset_id"),
    )
    claim = get_claim(claim_id) if claim_id else None
    seed_axes = config.get("research.seed_axes")
    if not isinstance(seed_axes, dict):
        seed_axes = normalize_seed_axes(
            int(out.get("seed", 0)),
            generator_seed=config.get("data.gen_seed"),
            data_kind=config.get("data.kind"),
            circuit_applicable=uses_quantum_config(config),
        )
    out["claim_id"] = claim_id
    out["claim"] = claim
    out["metric_type"] = config.get("research.metric_type") or (claim or {}).get("metric_type")
    out["seed_axes"] = seed_axes
    out["assessment_status"] = "unassigned" if claim_id is None else "descriptive"
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
    job = (candidate or {}).get("job") or {}
    config = job.get("config") or {}
    claim_id = infer_claim_id(
        explicit=config.get("research.claim_id"),
        preset_id=job.get("preset_id"),
    )
    claim = get_claim(claim_id) if claim_id else None
    schema = (claim or {}).get("fairness_schema")
    if claim is None and config.get("lab.analogue.type") == "component_swap":
        schema = _UNASSIGNED_COMPONENT_SCHEMA
    return evaluate_fairness(
        candidate,
        baseline,
        schema=schema,
    )


def verdict_for_comparison(payload: dict, metric_type: str | None = None) -> dict:
    if not payload.get("available"):
        return {"label": "incomplete", "reason": payload.get("reason", "comparison missing")}
    flags = fairness_flags(payload.get("candidate"), payload.get("baseline"))
    if not flags["complete"]:
        return {"label": "incomplete", "reason": "one or both runs have not finished", "fairness": flags}
    if metric_type not in PAIRABLE_VAL_PPL_METRIC_TYPES:
        return {
            "label": "unsupported metric",
            "claim_level": "descriptive",
            "assessment_status": "unsupported",
            "reason": (
                f"dashboard comparison inference does not extract {metric_type!r}; "
                "validation perplexity is not relabeled as that metric"
            ),
            "fairness": flags,
        }
    if not flags.get("valid"):
        return {
            "label": "insufficient fairness",
            "reason": "one or more undisclosed protocol mismatches remain",
            "fairness": flags,
        }
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
    candidate = payload.get("candidate") or {}
    candidate_job = candidate.get("job") or {}
    candidate_config = candidate_job.get("config") or {}
    claim_id = infer_claim_id(
        explicit=candidate_config.get("research.claim_id"),
        preset_id=candidate_job.get("preset_id"),
    )
    claim = get_claim(claim_id) if claim_id else None
    contracts = []
    for role in ("candidate", "baseline"):
        job = (payload.get(role) or {}).get("job") or {}
        contract = two_stream_metric_contract(
            suite="lab",
            config=job.get("config") or {},
        )
        if contract:
            contracts.append(contract)
    metric_contract = next(
        (contract for contract in contracts if contract["rerun_required"]),
        contracts[0] if contracts else None,
    )
    payload["metric_contract"] = metric_contract
    effective_metric_type = (
        (metric_contract or {}).get("metric_type")
        or candidate_config.get("research.metric_type")
        or (claim or {}).get("metric_type")
    )
    verdict = verdict_for_comparison(payload, effective_metric_type)
    if metric_contract and metric_contract["rerun_required"]:
        verdict = {
            "label": "rerun required",
            "claim_level": "invalid",
            "assessment_status": "rerun_required",
            "reason": metric_contract["limitation"],
        }
    payload["fairness"] = verdict.get("fairness") or fairness_flags(
        payload.get("candidate"), payload.get("baseline")
    )
    payload["verdict"] = {k: v for k, v in verdict.items() if k != "fairness"}
    payload["resource_normalized"] = _resource_normalized_for_payload(payload)
    payload["claim_id"] = claim_id
    payload["claim"] = claim
    payload["metric_type"] = effective_metric_type
    payload["seed_axes"] = (payload["fairness"].get("seed_axes") or {}).get(
        "candidate"
    )
    payload["fairness_mismatches"] = payload["fairness"].get("mismatches") or []
    payload["analogue_ladder"] = evaluate_analogue_ladder(
        candidate=payload.get("candidate"),
        baseline=payload.get("baseline"),
        fairness=payload["fairness"],
        claim=claim,
    )
    payload["assessment_status"] = payload["verdict"].get(
        "assessment_status"
    ) or ("smoke" if claim_id is None else "descriptive")
    payload["evidence_ladder"] = comparison_evidence_ladder(payload)
    return payload


def _leaderboard_highlights(db: ResultsDB) -> list[dict]:
    """Return current-contract highlights without promoting invalid history."""
    with db._conn() as con:
        rows = con.execute(
            "SELECT suite, variant, dataset, val_ppl, ts, config_json "
            "FROM runs WHERE val_ppl IS NOT NULL"
        ).fetchall()
    grouped: dict[tuple[str, str], dict] = {}
    for raw in rows:
        row = dict(raw)
        try:
            config = json.loads(row.get("config_json") or "{}")
        except json.JSONDecodeError:
            config = {}
        contract = two_stream_metric_contract(
            suite=row.get("suite", ""),
            config=config,
        )
        if contract and contract["rerun_required"]:
            continue
        key = (row["variant"], row["dataset"])
        current = grouped.get(key)
        if current is None:
            grouped[key] = {
                "variant": row["variant"],
                "dataset": row["dataset"],
                "best_ppl": row["val_ppl"],
                "last_ts": row["ts"],
            }
            continue
        current["best_ppl"] = min(current["best_ppl"], row["val_ppl"])
        current["last_ts"] = max(current["last_ts"], row["ts"])
    return sorted(grouped.values(), key=lambda item: item["best_ppl"])[:5]


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
    return {
        "counts": dict(counts),
        "active_jobs": active,
        "recent_failed_jobs": failed,
        "recent_comparisons": recent_comparisons,
        "gpu_status": status_payload.get("gpu", {}),
        "gpu_reservation": gpu_reservation_status(db),
        "leaderboard_highlights": _leaderboard_highlights(db),
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
        metric_contract = two_stream_metric_contract(
            suite="lab",
            config=job.get("config") or {},
        )
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
            "metric_contract": metric_contract,
            "rerun_required": bool(
                metric_contract and metric_contract["rerun_required"]
            ),
        })
    points.sort(key=lambda p: (p["n_qubits"], p["n_circuit_layers"]))
    complete = [
        point for point in points
        if point["val_ppl"] is not None and not point["rerun_required"]
    ]
    best = min(complete, key=lambda p: p["val_ppl"]) if complete else None
    protocol_warnings = sorted({
        point["metric_contract"]["limitation"]
        for point in points
        if point["rerun_required"] and point["metric_contract"]
    })
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
        "protocol_warnings": protocol_warnings,
        "complete_count": len(complete),
        "total_count": len(points),
    }
