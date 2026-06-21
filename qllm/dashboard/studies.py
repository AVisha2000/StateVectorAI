"""First-class study protocols and multi-run evidence summaries."""
from __future__ import annotations

import math
import uuid
from collections import Counter, defaultdict
from statistics import mean, pstdev
from typing import Any

from ..resultsdb import ResultsDB
from .datasets import get_dataset
from .lab import comparison_research_payload, enrich_job
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
    return {
        "name": (payload.get("name") or "Untitled study").strip(),
        "research_question": (payload.get("research_question") or "").strip(),
        "task": (payload.get("task") or "").strip(),
        "description": (payload.get("description") or "").strip(),
        "dataset_names": [str(item) for item in datasets if str(item).strip()],
        "candidate_preset_id": str(payload.get("candidate_preset_id") or "").strip(),
        "baseline_policy": str(payload.get("baseline_policy") or "analogue").strip(),
        "control_preset_ids": [str(item) for item in control_ids if str(item).strip()],
        "seeds": seeds,
        "steps": int(payload.get("steps") or 50),
        "eval_every": int(payload.get("eval_every") or 10),
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
                    )
                    db.update_lab_job(control["id"], comparison_role="frozen_control")
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
    return db.get_run("lab", variant, job["dataset_name"], int(job["seed"]), int(job["steps"]))


def _evidence_for_jobs(db: ResultsDB, jobs: list[dict]) -> dict:
    candidate_jobs = [job for job in jobs if job.get("study_role") == "candidate"]
    comparisons = []
    deltas = []
    fair = 0
    complete = 0
    for job in candidate_jobs:
        payload = comparison_research_payload(db, int(job["id"]))
        if not payload.get("available"):
            comparisons.append({
                "job_id": job["id"],
                "available": False,
                "reason": payload.get("reason"),
            })
            continue
        flags = payload.get("fairness") or {}
        deltas_payload = payload.get("deltas") or {}
        delta = deltas_payload.get("val_ppl")
        cfinal = (payload.get("candidate") or {}).get("final_run")
        bfinal = (payload.get("baseline") or {}).get("final_run")
        if cfinal and bfinal:
            complete += 1
        required = [
            "same_dataset", "same_seed", "same_steps", "same_eval_interval",
            "same_device_target", "same_training_budget", "same_preprocessing",
            "role_validation",
        ]
        is_fair = bool(flags) and all(flags.get(key) for key in required)
        if is_fair:
            fair += 1
        if delta is not None and is_fair:
            deltas.append(float(delta))
        comparisons.append({
            "job_id": job["id"],
            "available": True,
            "fair": is_fair,
            "delta_val_ppl": delta,
            "comparison_link": f"/comparisons/{job['id']}",
        })

    label = "incomplete"
    reason = "no complete matched candidate/baseline comparisons yet"
    if deltas:
        wins = sum(1 for delta in deltas if delta < 0)
        mean_delta = mean(deltas)
        std_delta = pstdev(deltas) if len(deltas) > 1 else 0.0
        if wins == len(deltas) and len(deltas) >= 3:
            label = "promising study"
            reason = "candidate wins every fair matched seed in this study"
        elif wins > len(deltas) / 2:
            label = "repeated improvement"
            reason = "candidate wins a majority of fair matched seeds"
        elif mean_delta >= 0:
            label = "negative"
            reason = "baseline matches or beats the candidate on average"
        else:
            label = "inconclusive"
            reason = "candidate has mixed or low-sample improvements"
    else:
        wins = 0
        mean_delta = None
        std_delta = None

    return {
        "label": label,
        "reason": reason,
        "candidate_count": len(candidate_jobs),
        "complete_pairs": complete,
        "fair_pairs": fair,
        "wins": wins,
        "mean_delta_val_ppl": mean_delta,
        "std_delta_val_ppl": std_delta,
        "comparisons": comparisons,
        "ladder": [
            {"label": "matched baseline", "ok": fair > 0, "detail": f"{fair} fair pair(s)"},
            {"label": "multi-seed evidence", "ok": len(deltas) >= 3, "detail": f"{len(deltas)} fair completed seed(s)"},
            {"label": "candidate better", "ok": bool(deltas) and wins > len(deltas) / 2, "detail": f"{wins}/{len(deltas) if deltas else 0} wins"},
            {"label": "variance reviewed", "ok": std_delta is not None and len(deltas) >= 2, "detail": "-" if std_delta is None else f"std {std_delta:.3f}"},
            {"label": "cost still required", "ok": False, "detail": "resource cost is inspected per comparison"},
        ],
    }


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
    evidence = _evidence_for_jobs(db, jobs) if include_jobs else {
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
        "protocol": study.get("protocol") or {},
        "job_counts": dict(counts),
        "role_counts": dict(role_counts),
        "job_count": len(jobs) if include_jobs else len(db.fetch_study_jobs(study_id)),
        "jobs": jobs,
        "evidence": evidence,
    }
