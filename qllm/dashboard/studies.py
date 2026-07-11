"""First-class study protocols and multi-run evidence summaries."""
from __future__ import annotations

import math
import uuid
from collections import Counter, defaultdict
from statistics import mean, pstdev
from typing import Any

from ..claims import get_claim, infer_claim_id
from ..research_protocol import (
    classify_claim,
    evaluate_analogue_ladder,
    evaluate_fairness,
    paired_improvements,
    paired_power_plan,
    paired_stats,
    practical_equivalence,
)
from ..resultsdb import ResultsDB
from .datasets import get_dataset
from .evidence import study_evidence_ladder
from .lab import (
    PAIRABLE_VAL_PPL_METRIC_TYPES,
    comparison_research_payload,
    enrich_job,
)
from .presets import preset_meta
from .runner import ExperimentQueue


def _ints(values: list[Any], *, name: str, minimum: int = 0) -> list[int]:
    out = []
    for value in values:
        try:
            item = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} values must be integers.") from exc
        if item < minimum:
            raise ValueError(f"{name} values must be at least {minimum}.")
        out.append(item)
    return sorted(set(out))


def _study_protocol(payload: dict) -> dict:
    datasets = payload.get("dataset_names") or payload.get("datasets") or ["default-text"]
    if isinstance(datasets, str):
        datasets = [datasets]
    seeds = _ints(payload.get("seeds") or [0, 1, 2], name="Seed", minimum=0)
    if not seeds:
        raise ValueError("At least one seed is required.")
    sweep = payload.get("sweep") or {}
    qubits = _ints(sweep.get("qubits") or [], name="Qubit", minimum=1)
    depths = _ints(sweep.get("depths") or [], name="Depth", minimum=1)
    if len(seeds) * max(len(datasets), 1) * max(len(qubits), 1) * max(len(depths), 1) > 96:
        raise ValueError("Study grids are capped at 96 candidate jobs.")
    control_ids = payload.get("control_preset_ids") or []
    if isinstance(control_ids, str):
        control_ids = [control_ids]
    candidate_preset_id = str(payload.get("candidate_preset_id") or "").strip()
    explicit_claim_id = payload.get("claim_id")
    claim_id = infer_claim_id(
        explicit=(str(explicit_claim_id) if explicit_claim_id else None),
        preset_id=candidate_preset_id or None,
    )
    if explicit_claim_id and claim_id is None:
        raise ValueError(f"Unknown or ambiguous claim_id '{explicit_claim_id}'.")
    claim = get_claim(claim_id) if claim_id else None
    metric_type = str(
        payload.get("metric_type")
        or (claim or {}).get("metric_type")
        or "strict_autoregressive_next_token"
    )
    if claim and metric_type != claim["metric_type"]:
        raise ValueError(
            f"metric_type must match claim '{claim_id}': {claim['metric_type']}"
        )
    if metric_type not in PAIRABLE_VAL_PPL_METRIC_TYPES:
        raise ValueError(
            f"Study paired inference does not support metric_type '{metric_type}'; "
            "use a metric-specific runner instead of relabeling validation perplexity."
        )
    baseline_policy = str(payload.get("baseline_policy") or "analogue").strip()
    if baseline_policy not in {"analogue", "none"}:
        raise ValueError("baseline_policy must be 'analogue' or 'none'.")
    analysis = (claim or {}).get("analysis_settings") or {}
    return {
        "name": (payload.get("name") or "Untitled study").strip(),
        "research_question": (payload.get("research_question") or "").strip(),
        "task": (payload.get("task") or "").strip(),
        "description": (payload.get("description") or "").strip(),
        "dataset_names": [str(item) for item in datasets if str(item).strip()],
        "candidate_preset_id": candidate_preset_id,
        "baseline_policy": baseline_policy,
        "control_preset_ids": [str(item) for item in control_ids if str(item).strip()],
        "seeds": seeds,
        "steps": int(payload.get("steps") or 50),
        "eval_every": int(payload.get("eval_every") or 10),
        "checkpoint_every": int(payload.get("checkpoint_every") or 0),
        "batch_size": (
            int(payload["batch_size"]) if payload.get("batch_size") not in (None, "") else None
        ),
        "seq_len": (
            int(payload["seq_len"]) if payload.get("seq_len") not in (None, "") else None
        ),
        "device_target": str(payload.get("device_target") or "auto").strip().lower(),
        "queue_now": bool(payload.get("queue_now", True)),
        "queue_analogues": bool(payload.get("queue_analogues", True)),
        "sweep": {"qubits": qubits, "depths": depths},
        "metrics": payload.get("metrics") or ["val_ppl", "wall_seconds", "n_params"],
        "claim_id": claim_id,
        "claim": claim,
        "metric_type": metric_type,
        "seed_axes": payload.get("seed_axes"),
        "analysis_settings": analysis,
    }


def create_study(db: ResultsDB, queue: ExperimentQueue, payload: dict) -> dict:
    protocol = _study_protocol(payload)
    if not protocol["name"]:
        raise ValueError("Study name is required.")
    if not protocol["candidate_preset_id"]:
        raise ValueError("Candidate preset is required.")
    preset_meta(protocol["candidate_preset_id"])
    for control_id in protocol["control_preset_ids"]:
        preset_meta(control_id)
    for dataset in protocol["dataset_names"]:
        if get_dataset(db, dataset) is None:
            raise ValueError(f"Unknown dataset '{dataset}'")
    if protocol["steps"] < 1:
        raise ValueError("Steps must be at least 1.")
    if protocol["eval_every"] < 1:
        raise ValueError("Eval interval must be at least 1.")

    group_id = uuid.uuid4().hex
    study_id = db.create_study({
        "name": protocol["name"],
        "research_question": protocol["research_question"],
        "task": protocol["task"],
        "description": protocol["description"],
        "dataset_names": protocol["dataset_names"],
        "candidate_preset_id": protocol["candidate_preset_id"],
        "baseline_policy": protocol["baseline_policy"],
        "control_preset_ids": protocol["control_preset_ids"],
        "seeds": protocol["seeds"],
        "sweep": protocol["sweep"],
        "status": "draft",
        "group_id": group_id,
        "protocol": protocol,
    })
    if protocol["queue_now"]:
        queue_study(db, queue, study_id)
    return study_payload(db, study_id)


def _grid_points(protocol: dict, supports_quantum_grid: bool = True) -> list[dict]:
    if not supports_quantum_grid:
        return [{"n_qubits": None, "n_circuit_layers": None}]
    qubits = protocol.get("sweep", {}).get("qubits") or [None]
    depths = protocol.get("sweep", {}).get("depths") or [None]
    return [
        {"n_qubits": q, "n_circuit_layers": d}
        for q in qubits
        for d in depths
    ]


def queue_study(db: ResultsDB, queue: ExperimentQueue, study_id: int) -> dict:
    study = db.get_study(study_id)
    if study is None:
        raise ValueError(f"Unknown study {study_id}")
    protocol = study.get("protocol") or {}
    if db.fetch_study_jobs(study_id):
        return study_payload(db, study_id)

    group_id = study["group_id"]
    queued = 0
    candidate_meta = preset_meta(study["candidate_preset_id"])
    supports_quantum_grid = bool(candidate_meta.get("quantum_controls", {}).get("enabled"))
    for dataset in study["dataset_names"]:
        for seed in study["seeds"]:
            for point in _grid_points(protocol, supports_quantum_grid):
                suffix = f"{dataset}-s{seed}"
                overrides = None
                if point["n_qubits"] is not None and point["n_circuit_layers"] is not None:
                    suffix += f"-q{point['n_qubits']}-d{point['n_circuit_layers']}"
                    overrides = {
                        "n_qubits": point["n_qubits"],
                        "n_circuit_layers": point["n_circuit_layers"],
                    }
                job = queue.submit(
                    preset_id=study["candidate_preset_id"],
                    dataset_name=dataset,
                    run_name=f"{study['name']}-candidate-{suffix}",
                    seed=int(seed),
                    steps=int(protocol.get("steps") or 50),
                    eval_every=int(protocol.get("eval_every") or 10),
                    device_target=protocol.get("device_target") or "auto",
                    queue_classical_comparison=(
                        study["baseline_policy"] == "analogue"
                        and bool(protocol.get("queue_analogues", True))
                    ),
                    quantum_overrides=overrides,
                    group_id=group_id,
                    batch_size=protocol.get("batch_size"),
                    seq_len=protocol.get("seq_len"),
                    claim_id=protocol.get("claim_id"),
                    seed_axes=protocol.get("seed_axes"),
                    metric_type=protocol.get("metric_type"),
                    checkpoint_every=int(protocol.get("checkpoint_every") or 0),
                )
                db.add_study_job(study_id, job["id"], "candidate", point)
                queued += 1
                if job.get("comparison_job"):
                    db.add_study_job(study_id, job["comparison_job"]["id"], "baseline", point)
                for control_id in study.get("control_preset_ids") or []:
                    control = queue.submit(
                        preset_id=control_id,
                        dataset_name=dataset,
                        run_name=f"{study['name']}-control-{control_id}-{suffix}",
                        seed=int(seed),
                        steps=int(protocol.get("steps") or 50),
                        eval_every=int(protocol.get("eval_every") or 10),
                        device_target=protocol.get("device_target") or "auto",
                        queue_classical_comparison=False,
                        group_id=group_id,
                        batch_size=protocol.get("batch_size"),
                        seq_len=protocol.get("seq_len"),
                        claim_id=protocol.get("claim_id"),
                        seed_axes=protocol.get("seed_axes"),
                        metric_type=protocol.get("metric_type"),
                        checkpoint_every=int(protocol.get("checkpoint_every") or 0),
                        experiment_uuid=job.get("experiment_uuid"),
                    )
                    db.update_lab_job(control["id"], comparison_role="control")
                    db.add_study_job(study_id, control["id"], "control", point)
                    queued += 1
    db.update_study(study_id, status="queued")
    payload = study_payload(db, study_id)
    payload["queued_count"] = queued
    return payload


def list_studies(db: ResultsDB) -> list[dict]:
    return [study_payload(db, int(row["id"]), include_jobs=False) for row in db.fetch_studies()]


def _final_run_for_job(db: ResultsDB, job: dict) -> dict | None:
    if job.get("status") != "done":
        return None
    variant = job["preset_id"]
    if job.get("run_key"):
        parts = str(job["run_key"]).split("/")
        if len(parts) >= 2:
            variant = parts[1]
    return db.get_run(
        "lab",
        variant,
        job["dataset_name"],
        int(job["seed"]),
        int(job["steps"]),
        run_uuid=job.get("run_uuid"),
    )


def _matched_controls_for_observation(
    controls: list[dict],
    observation: dict,
    claim: dict | None,
) -> list[dict]:
    """Return only completed controls matched to this exact study cell/pair."""
    payload = observation["payload"]
    candidate = payload.get("candidate") or {}
    candidate_job = candidate.get("job") or {}
    sweep = observation.get("study_sweep") or {}
    matches: list[dict] = []
    for control in controls:
        if not control.get("final_run"):
            continue
        if control.get("claim_id") != observation.get("claim_id"):
            continue
        if control.get("metric_type") != observation.get("metric_type"):
            continue
        if control.get("dataset_name") != candidate_job.get("dataset_name"):
            continue
        if int(control.get("seed", -1)) != int(candidate_job.get("seed", -2)):
            continue
        if (control.get("study_sweep") or {}) != sweep:
            continue
        control_job = dict(control)
        control_job["comparison_role"] = "baseline"
        candidate_for_control = {
            **candidate,
            "job": {**candidate_job, "comparison_role": "candidate"},
        }
        control_side = {
            "job": control_job,
            "final_run": control.get("final_run"),
        }
        report = evaluate_fairness(
            candidate_for_control,
            control_side,
            schema=(claim or {}).get("fairness_schema"),
        )
        if not (report.get("complete") and report.get("valid")):
            continue
        matched = dict(control)
        matched["control_match"] = {
            "valid": True,
            "claim_id": observation.get("claim_id"),
            "metric_type": observation.get("metric_type"),
            "dataset": candidate_job.get("dataset_name"),
            "seed": candidate_job.get("seed"),
            "sweep": sweep,
            "fairness": report,
        }
        matches.append(matched)
    return matches


def _aggregate_analogue_ladders(ladders: list[dict]) -> dict:
    if not ladders:
        return {"required_complete": False, "missing_required": [], "rungs": []}
    ordered_ids: list[str] = []
    for ladder in ladders:
        for rung in ladder.get("rungs") or []:
            if rung.get("id") not in ordered_ids:
                ordered_ids.append(rung.get("id"))
    rungs = []
    for rung_id in ordered_ids:
        per_pair = [
            next(
                (row for row in ladder.get("rungs") or [] if row.get("id") == rung_id),
                {"id": rung_id, "status": "unknown", "required": False},
            )
            for ladder in ladders
        ]
        statuses = [str(row.get("status") or "unknown") for row in per_pair]
        status = (
            "met"
            if all(item == "met" for item in statuses)
            else "not_met"
            if any(item == "not_met" for item in statuses)
            else "unknown"
        )
        aggregate = dict(per_pair[0])
        aggregate["status"] = status
        aggregate["pair_statuses"] = statuses
        aggregate["pairs_assessed"] = len(per_pair)
        rungs.append(aggregate)
    missing = [
        row["id"] for row in rungs
        if row.get("required") and row.get("status") != "met"
    ]
    return {
        "required_complete": not missing,
        "missing_required": missing,
        "rungs": rungs,
        "pairs_assessed": len(ladders),
    }


def _evidence_for_jobs(
    db: ResultsDB,
    jobs: list[dict],
    protocol: dict | None = None,
) -> dict:
    protocol = protocol or {}
    claim_id = protocol.get("claim_id")
    claim = get_claim(claim_id) if claim_id else None
    candidate_jobs = [job for job in jobs if job.get("study_role") == "candidate"]
    controls = [job for job in jobs if job.get("study_role") == "control"]
    comparisons = []
    cells: dict[tuple, list[dict]] = defaultdict(list)
    fair = 0
    complete = 0
    rerun_required = 0
    observed_metric_types: set[str | None] = set()
    observed_claim_ids: set[str | None] = set()
    for job in candidate_jobs:
        payload = comparison_research_payload(db, int(job["id"]))
        if not payload.get("available"):
            comparisons.append({
                "job_id": job["id"],
                "available": False,
                "reason": payload.get("reason"),
                "fairness_mismatches": payload.get("fairness_mismatches") or [],
            })
            continue
        flags = payload.get("fairness") or {}
        deltas_payload = payload.get("deltas") or {}
        delta = deltas_payload.get("val_ppl")
        cfinal = (payload.get("candidate") or {}).get("final_run")
        bfinal = (payload.get("baseline") or {}).get("final_run")
        if cfinal and bfinal:
            complete += 1
        metric_contract = payload.get("metric_contract") or {}
        needs_rerun = bool(metric_contract.get("rerun_required"))
        if needs_rerun:
            rerun_required += 1
        is_fair = bool(flags.get("valid")) and bool(flags.get("complete")) and not needs_rerun
        if is_fair:
            fair += 1
        metric_type = (
            payload.get("metric_type")
            or metric_contract.get("metric_type")
            or protocol.get("metric_type")
        )
        payload_claim_id = payload.get("claim_id") or claim_id
        observed_metric_types.add(metric_type)
        observed_claim_ids.add(payload_claim_id)
        sweep = job.get("study_sweep") or {}
        cell = (
            payload_claim_id,
            metric_type,
            job.get("dataset_name"),
            sweep.get("n_qubits"),
            sweep.get("n_circuit_layers"),
        )
        comparison_row = {
            "job_id": job["id"],
            "available": True,
            "fair": is_fair,
            "rerun_required": needs_rerun,
            "claim_id": payload_claim_id,
            "metric_type": metric_type,
            "delta_val_ppl": delta,
            "comparison_link": f"/comparisons/{job['id']}",
            "fairness_mismatches": payload.get("fairness_mismatches") or [],
            "analysis_eligible": metric_type in PAIRABLE_VAL_PPL_METRIC_TYPES,
            "cell": {
                "claim_id": payload_claim_id,
                "metric_type": metric_type,
                "dataset": job.get("dataset_name"),
                "sweep": sweep,
            },
        }
        comparisons.append(comparison_row)
        if (
            is_fair
            and cfinal
            and bfinal
            and delta is not None
            and metric_type in PAIRABLE_VAL_PPL_METRIC_TYPES
        ):
            cells[cell].append({
                "seed": int(job["seed"]),
                "candidate_score": float(cfinal["val_ppl"]),
                "baseline_score": float(bfinal["val_ppl"]),
                "payload": payload,
                "study_sweep": sweep,
                "claim_id": payload_claim_id,
                "metric_type": metric_type,
            })

    analyses = []
    for cell, observations in sorted(cells.items(), key=lambda item: str(item[0])):
        cell_claim = get_claim(cell[0]) if cell[0] else None
        settings = (cell_claim or {}).get("analysis_settings") or {}
        by_seed: dict[int, list[dict]] = defaultdict(list)
        for observation in observations:
            by_seed[observation["seed"]].append(observation)
        unique = [rows[0] for _, rows in sorted(by_seed.items()) if len(rows) == 1]
        duplicates = sorted(seed for seed, rows in by_seed.items() if len(rows) != 1)
        candidate_scores = [row["candidate_score"] for row in unique]
        baseline_scores = [row["baseline_score"] for row in unique]
        stats = (
            paired_stats(
                candidate_scores,
                baseline_scores,
                alpha=float(settings.get("alpha", 0.05)),
                bootstrap_seed=int(settings.get("bootstrap_seed", 0)),
                bootstrap_resamples=int(settings.get("bootstrap_resamples", 20_000)),
                sign_flip_seed=int(settings.get("sign_flip_seed", 0)),
                sign_flip_draws=int(settings.get("sign_flip_draws", 20_000)),
            ).as_dict()
            if unique else None
        )
        margin = settings.get("practical_equivalence_margin")
        equivalence = (
            practical_equivalence(stats, margin=float(margin))
            if stats and margin not in (None, 0)
            else {"status": "not_assessed", "equivalent": False, "margin": margin}
        )
        improvements = (
            paired_improvements(candidate_scores, baseline_scores)
            if unique else []
        )
        power = (
            paired_power_plan(
                improvements,
                smallest_useful_effect=float(margin),
                alpha=float(settings.get("alpha", 0.05)),
                power=float(settings.get("target_power", 0.8)),
            )
            if unique and margin not in (None, 0)
            else {
                "status": "not_assessed",
                "observed_pairs": len(unique),
                "recommended_pairs": None,
                "adequately_powered": False,
            }
        )
        ladders = [
            evaluate_analogue_ladder(
                candidate=row["payload"].get("candidate"),
                baseline=row["payload"].get("baseline"),
                fairness=row["payload"].get("fairness"),
                controls=_matched_controls_for_observation(
                    controls, row, cell_claim
                ),
                claim=cell_claim,
            )
            for row in unique
        ]
        ladder = _aggregate_analogue_ladders(ladders)
        verdict = (
            classify_claim(
                fairness={
                    "same_dataset": True,
                    "same_seed": True,
                    "same_steps": True,
                    "same_eval_interval": True,
                    "same_device_target": True,
                    "role_validation": True,
                    "valid": not duplicates,
                },
                paired=stats,
                min_pairs=int(settings.get("minimum_confirmatory_pairs", 6)),
                equivalence=equivalence,
                power=power,
                analogue_ladder=ladder,
                metric_name="validation perplexity",
            )
            if stats else {
                "label": "incomplete",
                "claim_level": "incomplete",
                "assessment_status": "incomplete",
                "reason": "no unique fair paired observations",
            }
        )
        analyses.append({
            "claim_id": cell[0],
            "claim": cell_claim,
            "metric_type": cell[1],
            "dataset": cell[2],
            "sweep": {"n_qubits": cell[3], "n_circuit_layers": cell[4]},
            "eligible_pairs": len(observations),
            "independent_pairs": len(unique),
            "duplicate_seeds": duplicates,
            "paired_stats": stats,
            "equivalence": equivalence,
            "power": power,
            "analogue_ladder": ladder,
            "verdict": verdict,
            "assessment_status": verdict.get("assessment_status"),
            "wins": sum(
                row["candidate_score"] < row["baseline_score"] for row in unique
            ),
            "mean_delta_val_ppl": (
                mean(
                    row["candidate_score"] - row["baseline_score"]
                    for row in unique
                )
                if unique else None
            ),
            "std_delta_val_ppl": (
                pstdev([
                    row["candidate_score"] - row["baseline_score"]
                    for row in unique
                ])
                if unique else None
            ),
        })

    cell_metrics = observed_metric_types
    cell_claim_ids = observed_claim_ids
    mixed_metrics = len(cell_metrics) > 1 or bool(
        cell_metrics and cell_metrics != {protocol.get("metric_type")}
    )
    mixed_claims = len(cell_claim_ids) > 1 or bool(
        cell_claim_ids and cell_claim_ids != {claim_id}
    )
    primary = (
        analyses[0]
        if len(analyses) == 1 and not mixed_metrics and not mixed_claims
        else None
    )
    if rerun_required:
        label = "rerun required"
        reason = f"{rerun_required} comparison(s) require a current-protocol rerun"
    elif mixed_metrics or mixed_claims:
        label = "invalid protocol"
        reason = "mixed claim IDs or metric types cannot be aggregated"
    elif primary:
        label = primary["verdict"]["label"]
        reason = primary["verdict"]["reason"]
    elif analyses:
        label = "multiple analysis cells"
        reason = "sweep and dataset cells are reported separately and are not pooled"
    else:
        label = "incomplete"
        reason = "no complete matched candidate/baseline comparisons yet"
    wins = int(primary.get("wins") or 0) if primary else 0
    mean_delta = primary.get("mean_delta_val_ppl") if primary else None
    std_delta = primary.get("std_delta_val_ppl") if primary else None
    aggregate_available = bool(primary and primary.get("paired_stats"))
    aggregate_mismatches = [
        {"job_id": row.get("job_id"), **mismatch}
        for row in comparisons
        for mismatch in row.get("fairness_mismatches") or []
    ]
    observed_seed_axes = [
        {
            "job_id": job.get("id"),
            "role": job.get("study_role"),
            "seed": job.get("seed"),
            "axes": job.get("seed_axes"),
        }
        for job in sorted(jobs, key=lambda item: int(item.get("id", 0)))
    ]

    evidence = {
        "label": label,
        "reason": reason,
        "candidate_count": len(candidate_jobs),
        "complete_pairs": complete,
        "fair_pairs": fair,
        "eligible_pairs": fair,
        "independent_pairs": primary.get("independent_pairs") if primary else None,
        "analysis_cell_count": len(analyses),
        "rerun_required_pairs": rerun_required,
        "wins": wins,
        "aggregate_available": aggregate_available,
        "mean_delta_val_ppl": mean_delta,
        "std_delta_val_ppl": std_delta,
        "comparisons": comparisons,
        "analyses": analyses,
        "paired_stats": primary.get("paired_stats") if primary else None,
        "equivalence": primary.get("equivalence") if primary else None,
        "power": primary.get("power") if primary else None,
        "analogue_ladder": primary.get("analogue_ladder") if primary else None,
        "claim_id": claim_id,
        "claim": claim,
        "metric_type": primary.get("metric_type") if primary else protocol.get("metric_type"),
        "assessment_status": (
            "rerun_required"
            if rerun_required
            else primary.get("assessment_status")
            if primary
            else "invalid"
            if (mixed_metrics or mixed_claims)
            else "incomplete"
        ),
        "mixed_metric_types": mixed_metrics,
        "mixed_claim_ids": mixed_claims,
        "fairness_mismatches": aggregate_mismatches,
        "fairness_mismatch_count": len(aggregate_mismatches),
        "seed_axes": {
            "requested": protocol.get("seed_axes"),
            "observed": observed_seed_axes,
        },
    }
    evidence["ladder"] = study_evidence_ladder(evidence)
    return evidence


def _resource_band_counts(jobs: list[dict]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for job in jobs:
        band = (job.get("config") or {}).get("lab.resource.band")
        if band:
            counts[str(band)] += 1
    return dict(counts)


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _completed_role_summary(jobs: list[dict], role: str) -> dict:
    rows = [job for job in jobs if job.get("study_role") == role and job.get("final_run")]
    walls = [float(job["final_run"]["wall_seconds"]) for job in rows if job["final_run"].get("wall_seconds") is not None]
    params = [float(job["final_run"]["n_params"]) for job in rows if job["final_run"].get("n_params") is not None]
    qubits = []
    depths = []
    for job in rows:
        config = job.get("config") or {}
        if config.get("model.quantum.n_qubits") is not None:
            qubits.append(float(config["model.quantum.n_qubits"]))
        if config.get("model.quantum.n_circuit_layers") is not None:
            depths.append(float(config["model.quantum.n_circuit_layers"]))
    return {
        "role": role,
        "completed_jobs": len(rows),
        "mean_wall_seconds": _mean(walls),
        "mean_n_params": _mean(params),
        "mean_qubits": _mean(qubits),
        "mean_depth": _mean(depths),
        "resource_bands": _resource_band_counts(rows),
    }


def _study_limitations(study: dict, jobs: list[dict], evidence: dict) -> list[str]:
    limitations: list[str] = []
    if evidence.get("rerun_required_pairs"):
        limitations.append(
            f"{evidence['rerun_required_pairs']} comparison pair(s) use an "
            "obsolete side-information metric and require a causal rerun."
        )
    for step in evidence.get("ladder") or []:
        if not step.get("ok"):
            limitations.append(f"{step['label']}: {step['detail']}")
    active = [job for job in jobs if job.get("status") in {"queued", "running"}]
    if active:
        limitations.append(f"{len(active)} study job(s) are still queued or running.")
    failures = [job for job in jobs if job.get("status") == "error"]
    if failures:
        limitations.append(f"{len(failures)} study job(s) failed and need review.")
    analogue_notes: list[str] = []
    for job in jobs:
        for item in ((job.get("analogue") or {}).get("known_limitations") or []):
            if item not in analogue_notes:
                analogue_notes.append(item)
    limitations.extend(analogue_notes)
    if not study.get("task"):
        limitations.append("Task-specific framing is missing; interpret evidence as model/dataset-specific rather than a general quantum advantage claim.")
    return limitations


def _pair_report_rows(db: ResultsDB, jobs: list[dict]) -> list[dict]:
    rows = []
    for job in jobs:
        if job.get("study_role") != "candidate":
            continue
        payload = comparison_research_payload(db, int(job["id"]))
        metric_contract = payload.get("metric_contract") or {}
        candidate = payload.get("candidate") or {}
        baseline = payload.get("baseline") or {}
        cjob = candidate.get("job") or {}
        bjob = baseline.get("job") or {}
        crun = candidate.get("final_run") or {}
        brun = baseline.get("final_run") or {}
        rows.append({
            "candidate_job_id": cjob.get("id") or job["id"],
            "baseline_job_id": bjob.get("id"),
            "dataset": cjob.get("dataset_name") or job.get("dataset_name"),
            "seed": cjob.get("seed") or job.get("seed"),
            "grid": job.get("study_sweep") or {},
            "available": payload.get("available", False),
            "fair": bool((payload.get("fairness") or {}).get("valid"))
            and bool((payload.get("fairness") or {}).get("complete"))
            and not metric_contract.get("rerun_required", False),
            "rerun_required": bool(metric_contract.get("rerun_required")),
            "metric_type": payload.get("metric_type") or metric_contract.get("metric_type"),
            "claim_id": payload.get("claim_id"),
            "seed_axes": payload.get("seed_axes"),
            "fairness_mismatches": payload.get("fairness_mismatches") or [],
            "analogue_ladder": payload.get("analogue_ladder"),
            "verdict_label": (payload.get("verdict") or {}).get("label"),
            "delta_val_ppl": (payload.get("deltas") or {}).get("val_ppl"),
            "delta_wall_seconds": (payload.get("deltas") or {}).get("wall_seconds"),
            "candidate_val_ppl": crun.get("val_ppl"),
            "baseline_val_ppl": brun.get("val_ppl"),
            "comparison_link": f"/comparisons/{job['id']}" if payload.get("available") else None,
            "reason": payload.get("reason") or (payload.get("verdict") or {}).get("reason"),
        })
    return rows


def _report_markdown(report: dict) -> str:
    protocol = report["protocol"]
    verdict = report["verdict"]
    stats = report["statistics"]
    lines = [
        f"# Study Report: {report['name']}",
        "",
        f"Research question: {report['research_question'] or 'Multi-run quantum/classical study'}",
        "",
        "## Verdict",
        f"- Label: {verdict['label']}",
        f"- Reason: {verdict['reason']}",
        f"- Fair pairs: {stats['fair_pairs']}",
        f"- Aggregate available: {stats.get('aggregate_available', False)}",
        f"- Rerun-required pairs: {stats.get('rerun_required_pairs', 0)}",
        f"- Candidate wins: {stats['wins']}",
        f"- Mean delta val ppl: {stats['mean_delta_val_ppl'] if stats['mean_delta_val_ppl'] is not None else '-'}",
        f"- Std delta val ppl: {stats['std_delta_val_ppl'] if stats['std_delta_val_ppl'] is not None else '-'}",
        "",
        "## Protocol",
        f"- Candidate preset: {protocol['candidate_preset_id']}",
        f"- Task: {protocol['task'] or '-'}",
        f"- Datasets: {', '.join(protocol['dataset_names']) or '-'}",
        f"- Seeds: {', '.join(str(v) for v in protocol['seeds']) or '-'}",
        f"- Device target: {protocol['device_target']}",
        f"- Steps / eval: {protocol['steps']} / {protocol['eval_every']}",
        f"- Batch / seq len: {protocol['batch_size'] or '-'} / {protocol['seq_len'] or '-'}",
        f"- Sweep qubits: {', '.join(str(v) for v in protocol['sweep'].get('qubits') or []) or '-'}",
        f"- Sweep depths: {', '.join(str(v) for v in protocol['sweep'].get('depths') or []) or '-'}",
        "",
        "## Limitations",
    ]
    limitations = report.get("limitations") or ["No additional limitations recorded."]
    lines.extend(f"- {item}" for item in limitations)
    return "\n".join(lines)


def _resolved_study_protocol(study: dict) -> dict:
    """Add inferred M04 fields in memory without rewriting legacy rows."""
    protocol = dict(study.get("protocol") or {})
    claim_id = protocol.get("claim_id") or infer_claim_id(
        preset_id=study.get("candidate_preset_id")
    )
    claim = get_claim(claim_id) if claim_id else None
    protocol["claim_id"] = claim_id
    protocol["claim"] = claim
    protocol["metric_type"] = (
        protocol.get("metric_type")
        or (claim or {}).get("metric_type")
        or "strict_autoregressive_next_token"
    )
    protocol.setdefault("seed_axes", None)
    protocol["analysis_settings"] = (
        protocol.get("analysis_settings")
        or (claim or {}).get("analysis_settings")
        or {}
    )
    return protocol


def study_payload(db: ResultsDB, study_id: int, include_jobs: bool = True) -> dict:
    study = db.get_study(study_id)
    if study is None:
        raise KeyError(f"Unknown study {study_id}")
    raw_jobs = db.fetch_study_jobs(study_id) if include_jobs else []
    jobs = []
    for row in raw_jobs:
        job = enrich_job(row, db)
        job["study_role"] = row.get("role")
        job["study_sweep"] = row.get("study_sweep") or {}
        final = _final_run_for_job(db, row)
        job["final_run"] = final
        jobs.append(job)
    counts = Counter(job["status"] for job in jobs)
    role_counts = Counter(job.get("study_role") for job in jobs)
    stored_protocol = study.get("protocol") or {}
    resolved_protocol = _resolved_study_protocol(study)
    evidence = _evidence_for_jobs(
        db, jobs, resolved_protocol
    ) if include_jobs else {
        "label": "pending",
        "reason": "open the study to inspect evidence",
    }
    return {
        "id": study["id"],
        "name": study["name"],
        "research_question": study.get("research_question"),
        "task": study.get("task"),
        "description": study.get("description"),
        "dataset_names": study.get("dataset_names") or [],
        "candidate_preset_id": study["candidate_preset_id"],
        "baseline_policy": study["baseline_policy"],
        "control_preset_ids": study.get("control_preset_ids") or [],
        "seeds": study.get("seeds") or [],
        "sweep": study.get("sweep") or {},
        "status": study["status"],
        "group_id": study["group_id"],
        "protocol": stored_protocol,
        "resolved_protocol": resolved_protocol,
        "job_counts": dict(counts),
        "role_counts": dict(role_counts),
        "job_count": len(jobs) if include_jobs else len(db.fetch_study_jobs(study_id)),
        "jobs": jobs,
        "evidence": evidence,
        "claim_id": evidence.get("claim_id", resolved_protocol.get("claim_id")),
        "claim": evidence.get("claim", resolved_protocol.get("claim")),
        "metric_type": evidence.get(
            "metric_type", resolved_protocol.get("metric_type")
        ),
        "seed_axes": evidence.get("seed_axes") or {
            "requested": resolved_protocol.get("seed_axes"),
            "observed": [],
        },
        "fairness_mismatches": evidence.get("fairness_mismatches") or [],
        "fairness_mismatch_count": evidence.get("fairness_mismatch_count", 0),
        "paired_stats": evidence.get("paired_stats"),
        "equivalence": evidence.get("equivalence"),
        "power": evidence.get("power"),
        "analyses": evidence.get("analyses") or [],
        "assessment_status": evidence.get("assessment_status"),
    }


def study_report_payload(db: ResultsDB, study_id: int) -> dict:
    payload = study_payload(db, study_id, include_jobs=True)
    candidate_meta = preset_meta(payload["candidate_preset_id"])
    controls_meta = [preset_meta(item) for item in payload.get("control_preset_ids") or []]
    jobs = payload["jobs"]
    evidence = payload["evidence"]
    pair_rows = _pair_report_rows(db, jobs)
    statistics = {
        "candidate_jobs": payload["role_counts"].get("candidate", 0),
        "baseline_jobs": payload["role_counts"].get("baseline", 0),
        "control_jobs": payload["role_counts"].get("control", 0),
        "fair_pairs": evidence.get("fair_pairs", 0),
        "complete_pairs": evidence.get("complete_pairs", 0),
        "rerun_required_pairs": evidence.get("rerun_required_pairs", 0),
        "wins": evidence.get("wins", 0),
        "aggregate_available": bool(evidence.get("aggregate_available")),
        "independent_pairs": evidence.get("independent_pairs"),
        "analysis_cell_count": evidence.get("analysis_cell_count", 0),
        "win_rate": (
            float(evidence.get("wins", 0)) / float(evidence.get("fair_pairs", 1))
            if evidence.get("aggregate_available") and evidence.get("fair_pairs")
            else None
        ),
        "mean_delta_val_ppl": evidence.get("mean_delta_val_ppl"),
        "std_delta_val_ppl": evidence.get("std_delta_val_ppl"),
        "paired_stats": evidence.get("paired_stats"),
        "equivalence": evidence.get("equivalence"),
        "power": evidence.get("power"),
    }
    resource_summary = {
        "candidate": _completed_role_summary(jobs, "candidate"),
        "baseline": _completed_role_summary(jobs, "baseline"),
        "control": _completed_role_summary(jobs, "control"),
    }
    report = {
        "id": payload["id"],
        "name": payload["name"],
        "status": payload["status"],
        "research_question": payload.get("research_question"),
        "protocol": {
            "task": payload.get("task"),
            "dataset_names": payload.get("dataset_names") or [],
            "candidate_preset_id": payload["candidate_preset_id"],
            "baseline_policy": payload["baseline_policy"],
            "control_preset_ids": payload.get("control_preset_ids") or [],
            "seeds": payload.get("seeds") or [],
            "sweep": payload.get("sweep") or {},
            "steps": payload["protocol"].get("steps"),
            "eval_every": payload["protocol"].get("eval_every"),
            "batch_size": payload["protocol"].get("batch_size"),
            "seq_len": payload["protocol"].get("seq_len"),
            "device_target": payload["protocol"].get("device_target") or "auto",
            "group_id": payload["group_id"],
            "claim_id": payload.get("claim_id"),
            "metric_type": payload.get("metric_type"),
            "seed_axes": payload.get("seed_axes"),
        },
        "candidate": {
            "id": candidate_meta["id"],
            "label": candidate_meta["label"],
            "kind": candidate_meta["kind"],
            "architecture": candidate_meta["architecture"],
            "quantum_role": candidate_meta["quantum_role"],
            "recommended_use": candidate_meta["recommended_use"],
            "risks": candidate_meta["risks"],
        },
        "controls": [
            {
                "id": item["id"],
                "label": item["label"],
                "kind": item["kind"],
                "architecture": item["architecture"],
                "risks": item["risks"],
            }
            for item in controls_meta
        ],
        "verdict": {
            "label": evidence.get("label") or "pending",
            "reason": evidence.get("reason") or "report pending",
            "ladder": evidence.get("ladder") or [],
        },
        "statistics": statistics,
        "resource_summary": resource_summary,
        "pair_rows": pair_rows,
        "limitations": _study_limitations(payload, jobs, evidence),
        "claim_id": payload.get("claim_id"),
        "claim": payload.get("claim"),
        "metric_type": payload.get("metric_type"),
        "seed_axes": payload.get("seed_axes"),
        "paired_stats": evidence.get("paired_stats"),
        "equivalence": evidence.get("equivalence"),
        "power": evidence.get("power"),
        "analyses": evidence.get("analyses") or [],
        "analogue_ladder": evidence.get("analogue_ladder"),
        "fairness_mismatches": evidence.get("fairness_mismatches") or [],
        "fairness_mismatch_count": evidence.get("fairness_mismatch_count", 0),
        "assessment_status": evidence.get("assessment_status"),
    }
    report["markdown"] = _report_markdown(report)
    return report
