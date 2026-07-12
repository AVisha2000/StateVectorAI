"""Run-workspace payloads for the QLLM Lab UI."""
from __future__ import annotations

from ..claims import get_claim, infer_claim_id
from ..research_protocol import normalize_seed_axes
from ..research_protocol import two_stream_metric_contract
from ..resultsdb import ResultsDB
from .analogues import analogue_status_for_job
from .datasets import get_dataset
from .evidence import (
    interpretation_warnings,
    job_durability_payload,
    run_resource_payload,
)
from .model_graph import model_family, uses_quantum_config
from .presets import preset_meta
from ._shared import curve as _curve
from ._shared import decode_config as _decode_config


def _live(
    db: ResultsDB, run_key: str | None, run_uuid: str | None = None
) -> dict | None:
    if not run_key:
        return None
    with db._conn() as con:
        if run_uuid is None:
            row = con.execute(
                "SELECT * FROM live_runs WHERE run_key=? AND run_uuid IS NULL",
                (run_key,),
            ).fetchone()
        else:
            row = con.execute(
                "SELECT * FROM live_runs WHERE run_key=? AND run_uuid=?",
                (run_key, run_uuid),
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
        run_uuid=job.get("run_uuid"),
    )
    if run:
        run["config"] = _decode_config(run)
        if run.get("run_uuid"):
            stored_manifest = db.get_run_manifest(str(run["run_uuid"]))
            if stored_manifest:
                run["manifest"] = stored_manifest.get("manifest")
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


def _job_payload(
    db: ResultsDB, job: dict | None, *, include_curves: bool = True
) -> dict | None:
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
    final_run = _final_run(db, job)
    durability = job_durability_payload(job)
    resources = run_resource_payload(final_run)
    payload = {
        "job": job,
        "preset": preset,
        "dataset": get_dataset(db, job["dataset_name"]),
        "live": _live(db, job.get("run_key"), job.get("run_uuid")),
        "curve": (
            _curve(db, job.get("run_key"), job.get("run_uuid"))
            if include_curves
            else {}
        ),
        "final_run": final_run,
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
        "manifest": durability["manifest"],
        "durability": durability,
        **resources,
    }
    payload["interpretation_warnings"] = list(
        durability["interpretation_warnings"]
    )
    return payload


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


def comparison_payload(
    db: ResultsDB, job_id: int, *, include_curves: bool = True
) -> dict:
    job = _job(db, job_id)
    if not job:
        payload = {"available": False, "reason": "job not found"}
        payload["interpretation_warnings"] = interpretation_warnings(
            available=False, baseline_linked=False
        )
        return payload
    other = _job(db, job.get("compare_to_job_id"))
    if not other:
        candidate_payload = _job_payload(
            db, job, include_curves=include_curves
        )
        payload = {
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
        payload["interpretation_warnings"] = interpretation_warnings(
            available=False,
            baseline_linked=False,
            candidate_uses_quantum=uses_quantum_config(job.get("config") or {}),
        )
        return payload

    if job.get("comparison_role") == "baseline":
        baseline, candidate = job, other
    else:
        candidate, baseline = job, other
    candidate_payload = _job_payload(
        db, candidate, include_curves=include_curves
    )
    baseline_payload = _job_payload(
        db, baseline, include_curves=include_curves
    )
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
    payload = {
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
    payload["paired_stats"] = None
    payload["equivalence"] = None
    payload["power"] = None
    payload["interpretation_warnings"] = interpretation_warnings(
        available=True,
        independent_pairs=1 if candidate_run and baseline_run else None,
        baseline_linked=True,
        candidate_uses_quantum=uses_quantum_config(candidate.get("config") or {}),
        metric_contract=metric_contract,
        metric_type=payload.get("metric_type"),
    )
    return payload


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
    payload["interpretation_warnings"] = [
        *(payload.get("interpretation_warnings") or []),
        *((payload["comparison"] or {}).get("interpretation_warnings") or []),
    ]
    return payload
