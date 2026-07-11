"""Run-workspace payloads for the QLLM Lab UI."""
from __future__ import annotations

import json
from collections import defaultdict

from ..claims import get_claim, infer_claim_id
from ..research_protocol import normalize_seed_axes
from ..research_protocol import two_stream_metric_contract
from ..resultsdb import ResultsDB
from .analogues import analogue_status_for_job
from .datasets import get_dataset
from .model_graph import model_family, uses_quantum_config
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
    row.update(analogue_status_for_job(db, row))
    return row


def _model_spec_meta(db: ResultsDB, preset_id: str, config: dict) -> dict:
    spec = None
    if str(preset_id).startswith("model-spec:"):
        try:
            spec_id = int(str(preset_id).split(":", 1)[1])
            spec = db.get_model_spec(spec_id)
        except ValueError:
            spec = None
    label = spec["name"] if spec else preset_id
    uses_quantum = uses_quantum_config(config)
    family = model_family(config)
    return {
        "id": preset_id,
        "label": label,
        "kind": "quantum" if uses_quantum else "classical",
        "cost": "Generated model spec",
        "summary": f"Editable {family} model spec.",
        "description": (spec.get("notes") if spec else None) or "Model spec generated from the editable model builder.",
        "architecture": family,
        "quantum_role": "Config-driven quantum components." if uses_quantum else "None. Classical analogue or baseline.",
        "recommended_use": "Use with matched dataset, seed, steps, preprocessing, and resource-cost context.",
        "risks": "Generated specs should be interpreted through the fairness and comparison panels.",
        "classical_twin_id": None,
        "classical_analogue": None,
        "comparison_policy": "none",
        "quantum_controls": {"enabled": False},
        "defaults": {
            "steps": config.get("train.steps"),
            "seed": config.get("train.seed"),
            "eval_every": config.get("train.eval_every"),
            "run_name": config.get("tracking.run_name"),
        },
        "config": config,
    }


def _job_payload(db: ResultsDB, job: dict | None) -> dict | None:
    if not job:
        return None
    try:
        preset = preset_meta(job["preset_id"])
    except Exception:
        preset = _model_spec_meta(db, job["preset_id"], job.get("config") or {})
    metric_contract = two_stream_metric_contract(
        suite="lab",
        config=job.get("config") or {},
    )
    config = job.get("config") or {}
    claim_id = infer_claim_id(
        explicit=config.get("research.claim_id"),
        preset_id=job.get("preset_id"),
    )
    claim = get_claim(claim_id) if claim_id else None
    seed_axes = config.get("research.seed_axes")
    if not isinstance(seed_axes, dict):
        seed_axes = normalize_seed_axes(
            int(job.get("seed", 0)),
            generator_seed=config.get("data.gen_seed"),
            data_kind=config.get("data.kind"),
            circuit_applicable=uses_quantum_config(config),
        )
    return {
        "job": job,
        "preset": preset,
        "dataset": get_dataset(db, job["dataset_name"]),
        "live": _live(db, job.get("run_key")),
        "curve": _curve(db, job.get("run_key")),
        "final_run": _final_run(db, job),
        "metric_contract": metric_contract,
        "claim_id": claim_id,
        "claim": claim,
        "metric_type": (
            (metric_contract or {}).get("metric_type")
            or config.get("research.metric_type")
            or (claim or {}).get("metric_type")
        ),
        "seed_axes": seed_axes,
        "assessment_status": "unassigned" if claim_id is None else "descriptive",
    }


def _comparison_metric_contract(*rows: dict | None) -> dict | None:
    contracts = [
        row.get("metric_contract")
        for row in rows
        if row and row.get("metric_contract")
    ]
    return next(
        (contract for contract in contracts if contract["rerun_required"]),
        contracts[0] if contracts else None,
    )


def comparison_payload(db: ResultsDB, job_id: int) -> dict:
    job = _job(db, job_id)
    if not job:
        return {"available": False, "reason": "job not found"}
    other = _job(db, job.get("compare_to_job_id"))
    if not other:
        candidate_payload = _job_payload(db, job)
        return {
            "available": False,
            "reason": "no linked classical comparison",
            "candidate": candidate_payload,
            "baseline": None,
            "deltas": None,
            "metric_contract": _comparison_metric_contract(candidate_payload),
            "claim_id": (candidate_payload or {}).get("claim_id"),
            "claim": (candidate_payload or {}).get("claim"),
            "metric_type": (candidate_payload or {}).get("metric_type"),
            "seed_axes": (candidate_payload or {}).get("seed_axes"),
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
    metric_contract = _comparison_metric_contract(
        candidate_payload,
        baseline_payload,
    )
    return {
        "available": True,
        "candidate": candidate_payload,
        "baseline": baseline_payload,
        "deltas": deltas,
        "metric_contract": metric_contract,
        "claim_id": (candidate_payload or {}).get("claim_id"),
        "claim": (candidate_payload or {}).get("claim"),
        "metric_type": (
            (metric_contract or {}).get("metric_type")
            or (candidate_payload or {}).get("metric_type")
        ),
        "seed_axes": (candidate_payload or {}).get("seed_axes"),
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
