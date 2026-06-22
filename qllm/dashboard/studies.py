"""First-class study protocols and multi-run evidence summaries."""
from __future__ import annotations

import math
import uuid
from collections import Counter, defaultdict
from statistics import mean, pstdev
from typing import Any

from ..resultsdb import ResultsDB
from .datasets import get_dataset
from .evidence import study_evidence_ladder
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

    evidence = {
        "label": label,
        "reason": reason,
        "candidate_count": len(candidate_jobs),
        "complete_pairs": complete,
        "fair_pairs": fair,
        "wins": wins,
        "mean_delta_val_ppl": mean_delta,
        "std_delta_val_ppl": std_delta,
        "comparisons": comparisons,
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
            "fair": bool((payload.get("fairness") or {}).get("same_dataset")) and (payload.get("verdict") or {}).get("label") != "insufficient fairness",
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
        "wins": evidence.get("wins", 0),
        "win_rate": (
            float(evidence.get("wins", 0)) / float(evidence.get("fair_pairs", 1))
            if evidence.get("fair_pairs") else None
        ),
        "mean_delta_val_ppl": evidence.get("mean_delta_val_ppl"),
        "std_delta_val_ppl": evidence.get("std_delta_val_ppl"),
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
    }
    report["markdown"] = _report_markdown(report)
    return report
