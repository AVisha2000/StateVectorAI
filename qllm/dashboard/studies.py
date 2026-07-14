"""First-class study protocols and multi-run evidence summaries."""
from __future__ import annotations

import math
import uuid
from collections import Counter, defaultdict
from statistics import mean, pstdev
from typing import Any

from ..claims import get_claim, infer_claim_id
from ..registry import TASK_TYPES, metric_type_spec
from ..research_protocol import (
    classify_claim,
    evaluate_analogue_ladder,
    evaluate_fairness,
    paired_improvements,
    paired_power_plan,
    paired_stats,
    practical_equivalence,
    with_legacy_assessment_alias,
)
from ..resultsdb import ResultsDB
from .datasets import get_dataset
from .evidence import (
    interpretation_warnings,
    job_durability_payload,
    run_resource_payload,
    study_evidence_ladder,
)
from .lab import (
    comparison_research_payload,
    enrich_job,
)
from ._shared import primary_metric_value
from .presets import preset_meta
from .runner import ExperimentQueue
from ._shared import ground_state_solver_competition_readiness


SEQUENCE_TASK_TYPE = "sequence_modeling"


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
    requested_datasets = payload.get("dataset_names") or payload.get("datasets")
    datasets = requested_datasets or ["default-text"]
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
    requested_task_type = payload.get("task_type")
    claim_task_type = (claim or {}).get("task_type")
    if requested_task_type is None:
        if claim and claim_task_type is None:
            raise ValueError(
                f"Claim '{claim_id}' is cross-cutting; task_type must be "
                "provided explicitly for the study."
            )
        task_type = claim_task_type or SEQUENCE_TASK_TYPE
    else:
        task_type = requested_task_type
    if not isinstance(task_type, str) or task_type not in TASK_TYPES:
        raise ValueError(
            f"task_type must be one of: {', '.join(TASK_TYPES)}; "
            f"got {task_type!r}."
        )
    if claim_task_type is not None and task_type != claim_task_type:
        raise ValueError(
            f"task_type must match claim '{claim_id}': {claim_task_type}"
        )
    if task_type == "ground_state" and requested_datasets is None:
        try:
            candidate_config = preset_meta(candidate_preset_id).get("config") or {}
        except KeyError:
            candidate_config = {}
        instance_id = candidate_config.get("problem.instance_id")
        if instance_id:
            datasets = [str(instance_id)]
    metric_type = str(
        payload.get("metric_type")
        or (claim or {}).get("metric_type")
        or "strict_autoregressive_next_token"
    )
    if claim and metric_type != claim["metric_type"]:
        raise ValueError(
            f"metric_type must match claim '{claim_id}': {claim['metric_type']}"
        )
    metric_spec = metric_type_spec(metric_type)
    if metric_spec is None:
        raise ValueError(
            f"Study execution does not support metric_type '{metric_type}'; "
            "use a registered metric-specific runner instead of relabeling "
            "another value."
        )
    default_baseline_policy = "none" if task_type == "ground_state" else "analogue"
    baseline_policy = str(
        payload.get("baseline_policy") or default_baseline_policy
    ).strip()
    if baseline_policy not in {"analogue", "none"}:
        raise ValueError("baseline_policy must be 'analogue' or 'none'.")
    if (
        task_type != "ground_state"
        and baseline_policy != "none"
        and not bool(metric_spec["pairable"])
    ):
        raise ValueError(
            f"metric_type '{metric_type}' is descriptive-only until its "
            "task-specific fairness schema enables paired inference."
        )
    analysis = (claim or {}).get("analysis_settings") or {}
    analysis_mode = str(
        payload.get("analysis_mode")
        or (
            "single_candidate_diagnostic"
            if task_type == "ground_state"
            else "paired_candidate_baseline"
        )
    ).strip()
    energy_error_threshold = None
    if task_type == "ground_state":
        if metric_type != "ground_state_energy_error":
            raise ValueError(
                "Ground-state studies require metric_type "
                "'ground_state_energy_error'."
            )
        if baseline_policy != "none":
            raise ValueError(
                "Ground-state studies require baseline_policy='none': no "
                "registered comparison-eligible finite-shot quantum runner "
                "or registered classical solver runner is available."
            )
        if analysis_mode != "single_candidate_diagnostic":
            raise ValueError(
                "Ground-state studies require "
                "analysis_mode='single_candidate_diagnostic'."
            )
        if payload.get("queue_analogues") is True:
            raise ValueError(
                "Ground-state studies cannot queue analogue solvers: no "
                "registered comparison-eligible finite-shot quantum runner "
                "or registered classical solver runner is available."
            )
        contract_threshold = analysis.get(
            "diagnostic_tolerance",
            analysis.get("practical_equivalence_margin", 0.001),
        )
        requested_threshold = payload.get(
            "energy_error_threshold", contract_threshold
        )
        try:
            energy_error_threshold = float(requested_threshold)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "energy_error_threshold must be a finite non-negative number."
            ) from exc
        if not math.isfinite(energy_error_threshold) or energy_error_threshold < 0:
            raise ValueError(
                "energy_error_threshold must be a finite non-negative number."
            )
        if claim and not math.isclose(
            energy_error_threshold,
            float(contract_threshold),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError(
                "energy_error_threshold must match the claim's prespecified "
                f"diagnostic tolerance ({float(contract_threshold):g})."
            )
        requested_device = str(
            payload.get("device_target") or "cpu"
        ).strip().lower()
        if requested_device != "cpu":
            raise ValueError(
                "Ground-state studies require device_target='cpu'."
            )
        if control_ids:
            raise ValueError(
                "Ground-state control solvers remain disabled: no registered "
                "classical solver runner is available for matched evidence."
            )
        if payload.get("batch_size") not in (None, "") or payload.get(
            "seq_len"
        ) not in (None, ""):
            raise ValueError(
                "batch_size and seq_len do not apply to ground-state VQE studies."
            )
    return {
        "name": (payload.get("name") or "Untitled study").strip(),
        "research_question": (payload.get("research_question") or "").strip(),
        "task": (payload.get("task") or "").strip(),
        "task_type": task_type,
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
        "device_target": str(
            payload.get("device_target")
            or ("cpu" if task_type == "ground_state" else "auto")
        ).strip().lower(),
        "queue_now": bool(payload.get("queue_now", True)),
        "queue_analogues": (
            False
            if task_type == "ground_state"
            else bool(payload.get("queue_analogues", True))
        ),
        "sweep": {"qubits": qubits, "depths": depths},
        "metrics": payload.get("metrics") or (
            ["energy_error", "energy", "wall_seconds", "n_params"]
            if task_type == "ground_state"
            else ["val_ppl", "wall_seconds", "n_params"]
        ),
        "claim_id": claim_id,
        "claim": claim,
        "metric_type": metric_type,
        "seed_axes": payload.get("seed_axes"),
        "analysis_settings": analysis,
        "analysis_mode": analysis_mode,
        "energy_error_threshold": energy_error_threshold,
    }


def create_study(db: ResultsDB, queue: ExperimentQueue, payload: dict) -> dict:
    protocol = _study_protocol(payload)
    if not protocol["name"]:
        raise ValueError("Study name is required.")
    if not protocol["candidate_preset_id"]:
        raise ValueError("Candidate preset is required.")
    candidate_meta = preset_meta(protocol["candidate_preset_id"])
    candidate_task_type = str(
        (candidate_meta.get("config") or {}).get(
            "problem.task_type", SEQUENCE_TASK_TYPE
        )
    )
    if candidate_task_type != protocol["task_type"]:
        raise ValueError(
            "Candidate preset task_type does not match the study: "
            f"{candidate_task_type!r} != {protocol['task_type']!r}."
        )
    for control_id in protocol["control_preset_ids"]:
        control_meta = preset_meta(control_id)
        control_task_type = str(
            (control_meta.get("config") or {}).get(
                "problem.task_type", SEQUENCE_TASK_TYPE
            )
        )
        if control_task_type != protocol["task_type"]:
            raise ValueError(
                f"Control preset '{control_id}' task_type does not match the study: "
                f"{control_task_type!r} != {protocol['task_type']!r}."
            )
    for dataset in protocol["dataset_names"]:
        if protocol["task_type"] == "ground_state":
            from ..problems import get_ground_state_instance

            try:
                get_ground_state_instance(dataset)
            except KeyError as exc:
                raise ValueError(
                    f"Unknown ground-state problem instance '{dataset}'"
                ) from exc
            candidate_instance = (candidate_meta.get("config") or {}).get(
                "problem.instance_id"
            )
            if dataset != candidate_instance:
                raise ValueError(
                    "Ground-state study instance must match the candidate "
                    f"preset: {dataset!r} != {candidate_instance!r}."
                )
        elif get_dataset(db, dataset) is None:
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
    protocol = _resolved_study_protocol(study)
    candidate_meta = preset_meta(study["candidate_preset_id"])
    candidate_config = candidate_meta.get("config") or {}
    candidate_task_type = str(
        candidate_config.get("problem.task_type", SEQUENCE_TASK_TYPE)
    )
    if candidate_task_type != protocol["task_type"]:
        raise ValueError(
            "Candidate preset task_type does not match the stored study: "
            f"{candidate_task_type!r} != {protocol['task_type']!r}."
        )
    if protocol["task_type"] == "ground_state":
        if (
            protocol.get("analysis_mode")
            != "single_candidate_diagnostic"
            or protocol.get("baseline_policy") != "none"
            or protocol.get("metric_type")
            != "ground_state_energy_error"
            or protocol.get("device_target") != "cpu"
            or protocol.get("queue_analogues")
            or study.get("control_preset_ids")
        ):
            raise ValueError(
                "Stored ground-state study protocol violates the bounded "
                "single-candidate diagnostic contract."
            )
        candidate_instance = candidate_config.get("problem.instance_id")
        for instance_id in study.get("dataset_names") or []:
            if instance_id != candidate_instance:
                raise ValueError(
                    "Ground-state study instance must match the candidate "
                    f"preset: {instance_id!r} != {candidate_instance!r}."
                )
    if db.fetch_study_jobs(study_id):
        return study_payload(db, study_id)

    group_id = study["group_id"]
    queued = 0
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


def _ground_state_reference_ladder(
    claim: dict | None,
    instances: list,
    completed_runs: list[dict],
) -> dict:
    configured = list((claim or {}).get("analogue_ladder") or [])
    rungs = []
    for configured_rung in configured:
        rung_id = str(
            configured_rung.get("id")
            or configured_rung.get("rung_id")
            or "unknown"
        )
        if rung_id == "exact_diagonalization_reference":
            met = bool(completed_runs) and all(
                any(
                    reference.reference_id == "exact_diagonalization"
                    and reference.role == "oracle"
                    and reference.certified
                    for reference in instance.classical_references
                )
                for instance in instances
            )
        elif rung_id in {
            "best_product_state_reference",
            "strong_classical_challenger",
        }:
            met = bool(instances) and all(
                any(
                    reference.reference_id == "best_product_state"
                    and reference.role == "descriptive_challenger"
                    and reference.certified
                    for reference in instance.classical_references
                )
                for instance in instances
            )
        elif rung_id == "resource_accounting":
            met = bool(completed_runs) and all(
                isinstance(run.get("resources"), dict)
                and isinstance(
                    run["resources"].get(
                        "measured_hardware_circuit_executions"
                    ),
                    dict,
                )
                for run in completed_runs
            )
        else:
            met = False
        rungs.append(
            {
                "id": rung_id,
                "required": bool(configured_rung.get("required")),
                "status": "met" if met else "unknown",
                "limitation": configured_rung.get("limitation"),
                "comparison_enabled": False,
            }
        )
    missing = [
        rung["id"]
        for rung in rungs
        if rung["required"] and rung["status"] != "met"
    ]
    return {
        "required_complete": not missing,
        "missing_required": missing,
        "rungs": rungs,
        "comparison_enabled": False,
    }


def _ground_state_diagnostic_evidence(
    jobs: list[dict],
    protocol: dict,
) -> dict:
    """Summarize exact-reference diagnostics without paired inference."""
    from ..problems import get_ground_state_instance

    claim_id = protocol.get("claim_id")
    claim = get_claim(claim_id) if claim_id else None
    metric_type = protocol.get("metric_type")
    metric_spec = metric_type_spec(metric_type)
    metric_key = (
        str(metric_spec["extraction_key"]) if metric_spec is not None else None
    )
    threshold = float(protocol.get("energy_error_threshold") or 0.001)
    candidates = [
        job for job in jobs if job.get("study_role") == "candidate"
    ]
    observations = []
    instances_by_id = {}
    completed_runs = []
    mismatches = []
    cells: dict[tuple, list[dict]] = defaultdict(list)
    observed_units: set[str] = set()

    for job in candidates:
        instance_id = str(job.get("dataset_name") or "")
        try:
            instance = get_ground_state_instance(instance_id)
        except KeyError:
            mismatches.append(
                {
                    "job_id": job.get("id"),
                    "path": "job.dataset_name",
                    "candidate": instance_id,
                    "baseline": None,
                    "allowed": False,
                    "reason": "unknown registered ground-state instance",
                }
            )
            continue
        instances_by_id[instance_id] = instance
        observed_units.add(instance.energy_units)
        declared_metric = job.get("metric_type") or (
            job.get("config") or {}
        ).get("research.metric_type")
        if declared_metric != metric_type:
            mismatches.append(
                {
                    "job_id": job.get("id"),
                    "path": "research.metric_type",
                    "candidate": declared_metric,
                    "baseline": metric_type,
                    "allowed": False,
                    "reason": "job metric differs from the diagnostic protocol",
                }
            )
        run = job.get("final_run")
        value = (
            primary_metric_value(run, metric_key)
            if metric_key is not None
            else None
        )
        available = run is not None and value is not None
        if available:
            completed_runs.append(run)
        sweep = job.get("study_sweep") or {}
        cell_key = (
            instance_id,
            sweep.get("n_qubits"),
            sweep.get("n_circuit_layers"),
        )
        observation = {
            "job_id": job.get("id"),
            "instance_id": instance_id,
            "seed": int(job.get("seed", 0)),
            "sweep": sweep,
            "status": job.get("status"),
            "available": available,
            "metric_type": metric_type,
            "metric_key": metric_key,
            "metric_units": instance.energy_units,
            "energy_error": float(value) if value is not None else None,
            "within_tolerance": (
                bool(float(value) <= threshold)
                if value is not None
                else None
            ),
            "threshold": threshold,
            "seed_axes": job.get("seed_axes"),
            "run_uuid": job.get("run_uuid"),
        }
        observations.append(observation)
        cells[cell_key].append(observation)

    duplicate_seeds = []
    analyses = []
    for cell_key, rows in sorted(cells.items(), key=lambda item: str(item[0])):
        by_seed: dict[int, list[dict]] = defaultdict(list)
        for row in rows:
            by_seed[int(row["seed"])].append(row)
        duplicates = sorted(
            seed for seed, seed_rows in by_seed.items() if len(seed_rows) != 1
        )
        duplicate_seeds.extend(duplicates)
        unique_rows = [
            seed_rows[0]
            for _, seed_rows in sorted(by_seed.items())
            if len(seed_rows) == 1 and seed_rows[0]["available"]
        ]
        values = [float(row["energy_error"]) for row in unique_rows]
        outcome = (
            "incomplete"
            if not values or duplicates or len(unique_rows) != len(rows)
            else "within_tolerance"
            if all(value <= threshold for value in values)
            else "outside_tolerance"
        )
        analyses.append(
            {
                "analysis_mode": "single_candidate_diagnostic",
                "instance_id": cell_key[0],
                "sweep": {
                    "n_qubits": cell_key[1],
                    "n_circuit_layers": cell_key[2],
                },
                "metric_type": metric_type,
                "metric_key": metric_key,
                "metric_units": (
                    instances_by_id[cell_key[0]].energy_units
                    if cell_key[0] in instances_by_id
                    else None
                ),
                "threshold": threshold,
                "completed_initialization_seeds": len(unique_rows),
                "duplicate_seeds": duplicates,
                "mean_energy_error": mean(values) if values else None,
                "std_energy_error": (
                    pstdev(values) if len(values) > 1 else None
                ),
                "max_energy_error": max(values) if values else None,
                "outcome": outcome,
                "observations": unique_rows,
                "paired_stats": None,
                "comparative_inference_enabled": False,
            }
        )

    if len(observed_units) > 1:
        mismatches.append(
            {
                "path": "problem.energy_units",
                "candidate": sorted(observed_units),
                "baseline": None,
                "allowed": False,
                "reason": "mixed energy units cannot be aggregated",
            }
        )
    if duplicate_seeds:
        mismatches.append(
            {
                "path": "study.seed",
                "candidate": sorted(set(duplicate_seeds)),
                "baseline": None,
                "allowed": False,
                "reason": "duplicate initialization seeds in one instance cell",
            }
        )

    completed = sum(1 for row in observations if row["available"])
    outcome = (
        "incomplete"
        if not completed or completed != len(candidates) or mismatches
        else "within_tolerance"
        if all(row["within_tolerance"] for row in observations if row["available"])
        else "outside_tolerance"
    )
    reason = {
        "incomplete": (
            "No complete unique diagnostic observations are available, or "
            "the stored protocol is inconsistent."
        ),
        "within_tolerance": (
            f"All {completed} completed analytic simulator diagnostic(s) "
            f"are within the prespecified {threshold:g} energy-error tolerance."
        ),
        "outside_tolerance": (
            f"At least one completed analytic simulator diagnostic exceeds "
            f"the prespecified {threshold:g} energy-error tolerance."
        ),
    }[outcome]
    instances = list(instances_by_id.values())
    reference_ladder = _ground_state_reference_ladder(
        claim,
        instances,
        completed_runs,
    )
    solver_competition_readiness = ground_state_solver_competition_readiness(
        claim
    )
    warnings = [
        {
            "code": "simulator_diagnostic_only",
            "severity": "warning",
            "title": "Analytic simulator diagnostic only",
            "message": (
                "This study does not provide QPU evidence or an equal-budget "
                "classical solver comparison."
            ),
            "evidence": {
                "comparative_inference_enabled": False,
                "qpu_evidence": False,
            },
        }
    ]
    if completed == 1:
        warnings.extend(
            interpretation_warnings(
                single_seed=True,
                available=True,
                baseline_linked=None,
                candidate_uses_quantum=False,
            )
        )
    if mismatches:
        warnings.extend(
            interpretation_warnings(
                available=True,
                assessment_status="invalid",
                duplicate_seeds=sorted(set(duplicate_seeds)),
            )
        )

    evidence = {
        "label": "simulator diagnostic",
        "reason": reason,
        "outcome": outcome,
        "analysis_mode": "single_candidate_diagnostic",
        "candidate_count": len(candidates),
        "completed_diagnostics": completed,
        "complete_pairs": 0,
        "fair_pairs": 0,
        "eligible_pairs": 0,
        "independent_pairs": None,
        "analysis_cell_count": len(analyses),
        "rerun_required_pairs": 0,
        "wins": 0,
        "aggregate_available": False,
        "metric_type": metric_type,
        "metric_key": metric_key,
        "metric_units": (
            next(iter(observed_units)) if len(observed_units) == 1 else None
        ),
        "lower_is_better": (
            bool(metric_spec["lower_is_better"])
            if metric_spec is not None
            else None
        ),
        "energy_error_threshold": threshold,
        "mean_delta": None,
        "std_delta": None,
        "mean_delta_val_ppl": None,
        "std_delta_val_ppl": None,
        "comparisons": [],
        "descriptive_observations": observations,
        "analyses": analyses,
        "paired_stats": None,
        "equivalence": None,
        "power": None,
        "analogue_ladder": reference_ladder,
        "reference_ladder": reference_ladder,
        "claim_id": claim_id,
        "claim": claim,
        "claim_level": (claim or {}).get("level") or "untested",
        "replication_status": (claim or {}).get("replication_status") or "none",
        "task_type": "ground_state",
        "assessment_status": (
            "invalid" if mismatches else "descriptive"
        ),
        "comparative_inference_enabled": False,
        "solver_competition_readiness": solver_competition_readiness,
        "mixed_metric_types": False,
        "mixed_claim_ids": False,
        "fairness_mismatches": mismatches,
        "fairness_mismatch_count": len(mismatches),
        "seed_axes": {
            "requested": protocol.get("seed_axes"),
            "observed": [
                {
                    "job_id": row.get("job_id"),
                    "role": "candidate",
                    "seed": row.get("seed"),
                    "axes": row.get("seed_axes"),
                }
                for row in observations
            ],
        },
        "interpretation_warnings": warnings,
    }
    evidence["ladder"] = [
        {
            "key": "registered_metric",
            "label": "Registered diagnostic metric",
            "ok": metric_type == "ground_state_energy_error",
            "detail": f"{metric_type} extracts {metric_key}",
            "caution": "This metric is intentionally non-pairable.",
        },
        {
            "key": "exact_reference",
            "label": "Certified exact reference",
            "ok": any(
                rung["id"] == "exact_diagonalization_reference"
                and rung["status"] == "met"
                for rung in reference_ladder["rungs"]
            ),
            "detail": "Exact diagonalization is a metric oracle, not a competitor.",
            "caution": None,
        },
        {
            "key": "completed_diagnostics",
            "label": "Completed analytic diagnostics",
            "ok": completed > 0,
            "detail": f"{completed} completed initialization seed(s)",
            "caution": "Seeds are nested within each problem instance.",
        },
        {
            "key": "prespecified_tolerance",
            "label": "Prespecified energy tolerance",
            "ok": outcome == "within_tolerance",
            "detail": f"Outcome: {outcome}; threshold: {threshold:g}",
            "caution": "Tolerance is instance-specific and not chemical accuracy.",
        },
        {
            "key": "comparative_inference",
            "label": "Comparative solver inference",
            "ok": False,
            "detail": (
                "Blocked by the missing registered comparison-eligible "
                "finite-shot quantum runner, registered classical solver "
                "runner, and matched paired solver evidence."
            ),
            "caution": "No advantage or composite score is produced.",
        },
    ]
    return evidence


def _evidence_for_jobs(
    db: ResultsDB,
    jobs: list[dict],
    protocol: dict | None = None,
) -> dict:
    protocol = protocol or {}
    if protocol.get("task_type") == "ground_state":
        return _ground_state_diagnostic_evidence(jobs, protocol)
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
                "disallowed_fairness_mismatches": (
                    (payload.get("fairness") or {}).get("disallowed_mismatches") or []
                ),
            })
            continue
        flags = payload.get("fairness") or {}
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
        metric_spec = metric_type_spec(metric_type, require_pairable=True)
        metric_key = (
            str(metric_spec["extraction_key"]) if metric_spec is not None else None
        )
        candidate_score = (
            primary_metric_value(cfinal, metric_key) if metric_key else None
        )
        baseline_score = (
            primary_metric_value(bfinal, metric_key) if metric_key else None
        )
        delta = (
            candidate_score - baseline_score
            if candidate_score is not None and baseline_score is not None
            else None
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
            "metric_key": metric_key,
            "metric_units": metric_spec.get("units") if metric_spec else None,
            "lower_is_better": (
                bool(metric_spec["lower_is_better"]) if metric_spec else None
            ),
            "metric_delta": delta,
            "delta_val_ppl": delta if metric_key == "val_ppl" else None,
            "comparison_link": f"/comparisons/{job['id']}",
            "fairness_mismatches": payload.get("fairness_mismatches") or [],
            "disallowed_fairness_mismatches": flags.get("disallowed_mismatches") or [],
            "analysis_eligible": metric_spec is not None,
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
            and candidate_score is not None
            and baseline_score is not None
            and metric_spec is not None
        ):
            cells[cell].append({
                "seed": int(job["seed"]),
                "candidate_score": float(candidate_score),
                "baseline_score": float(baseline_score),
                "payload": payload,
                "study_sweep": sweep,
                "claim_id": payload_claim_id,
                "metric_type": metric_type,
            })

    analyses = []
    for cell, observations in sorted(cells.items(), key=lambda item: str(item[0])):
        cell_claim = get_claim(cell[0]) if cell[0] else None
        metric_spec = metric_type_spec(cell[1], require_pairable=True)
        if metric_spec is None:
            continue
        metric_key = str(metric_spec["extraction_key"])
        lower_is_better = bool(metric_spec["lower_is_better"])
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
                lower_is_better=lower_is_better,
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
            paired_improvements(
                candidate_scores,
                baseline_scores,
                lower_is_better=lower_is_better,
            )
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
        verdict = with_legacy_assessment_alias(
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
                metric_name=str(cell[1]).replace("_", " "),
            )
            if stats else {
                "label": "incomplete",
                "assessment_level": "incomplete",
                "assessment_status": "incomplete",
                "reason": "no unique fair paired observations",
            }
        )
        raw_deltas = [
            row["candidate_score"] - row["baseline_score"] for row in unique
        ]
        mean_delta = mean(raw_deltas) if raw_deltas else None
        std_delta = pstdev(raw_deltas) if raw_deltas else None
        analyses.append({
            "claim_id": cell[0],
            "claim": cell_claim,
            "metric_type": cell[1],
            "metric_key": metric_key,
            "metric_units": metric_spec["units"],
            "lower_is_better": lower_is_better,
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
                (
                    row["candidate_score"] < row["baseline_score"]
                    if lower_is_better
                    else row["candidate_score"] > row["baseline_score"]
                )
                for row in unique
            ),
            "mean_delta": mean_delta,
            "std_delta": std_delta,
            "mean_delta_val_ppl": mean_delta if metric_key == "val_ppl" else None,
            "std_delta_val_ppl": std_delta if metric_key == "val_ppl" else None,
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
    mean_delta = primary.get("mean_delta") if primary else None
    std_delta = primary.get("std_delta") if primary else None
    aggregate_available = bool(primary and primary.get("paired_stats"))
    aggregate_mismatches = [
        {"job_id": row.get("job_id"), **mismatch}
        for row in comparisons
        for mismatch in row.get("fairness_mismatches") or []
    ]
    aggregate_disallowed_mismatches = [
        {"job_id": row.get("job_id"), **mismatch}
        for row in comparisons
        for mismatch in row.get("disallowed_fairness_mismatches") or []
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
        "metric_key": primary.get("metric_key") if primary else None,
        "metric_units": primary.get("metric_units") if primary else None,
        "lower_is_better": primary.get("lower_is_better") if primary else None,
        "mean_delta": mean_delta,
        "std_delta": std_delta,
        "mean_delta_val_ppl": (
            primary.get("mean_delta_val_ppl") if primary else None
        ),
        "std_delta_val_ppl": (
            primary.get("std_delta_val_ppl") if primary else None
        ),
        "comparisons": comparisons,
        "analyses": analyses,
        "paired_stats": primary.get("paired_stats") if primary else None,
        "equivalence": primary.get("equivalence") if primary else None,
        "power": primary.get("power") if primary else None,
        "analogue_ladder": primary.get("analogue_ladder") if primary else None,
        "claim_id": claim_id,
        "claim": claim,
        "task_type": protocol.get("task_type"),
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
    duplicate_seeds = sorted({
        seed
        for analysis in analyses
        for seed in analysis.get("duplicate_seeds") or []
    })
    evidence["interpretation_warnings"] = interpretation_warnings(
        available=any(row.get("available") for row in comparisons),
        independent_pairs=evidence.get("independent_pairs"),
        baseline_linked=any(row.get("available") for row in comparisons),
        candidate_uses_quantum=any(job.get("uses_quantum") for job in candidate_jobs),
        analogue_ladder=evidence.get("analogue_ladder"),
        claim=claim,
        metric_type=evidence.get("metric_type"),
        fairness={
            "valid": not bool(aggregate_disallowed_mismatches),
            "mismatches": aggregate_mismatches,
            "disallowed_mismatches": aggregate_disallowed_mismatches,
        },
        duplicate_seeds=duplicate_seeds,
        assessment_status=evidence.get("assessment_status"),
        mixed_metric_types=mixed_metrics,
        mixed_claim_ids=mixed_claims,
    )
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
    if study.get("task_type") == "ground_state":
        limitations.append(
            "Analytic simulator diagnostics are not QPU evidence and do not "
            "support comparative solver or quantum-advantage inference."
        )
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
        metric_type = payload.get("metric_type") or metric_contract.get("metric_type")
        metric_spec = metric_type_spec(metric_type, require_pairable=True)
        metric_key = (
            str(metric_spec["extraction_key"]) if metric_spec is not None else None
        )
        candidate_metric = (
            primary_metric_value(crun, metric_key) if metric_key else None
        )
        baseline_metric = (
            primary_metric_value(brun, metric_key) if metric_key else None
        )
        metric_delta = (
            candidate_metric - baseline_metric
            if candidate_metric is not None and baseline_metric is not None
            else None
        )
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
            "metric_type": metric_type,
            "metric_key": metric_key,
            "metric_units": metric_spec.get("units") if metric_spec else None,
            "lower_is_better": (
                bool(metric_spec["lower_is_better"]) if metric_spec else None
            ),
            "claim_id": payload.get("claim_id"),
            "seed_axes": payload.get("seed_axes"),
            "fairness_mismatches": payload.get("fairness_mismatches") or [],
            "analogue_ladder": payload.get("analogue_ladder"),
            "verdict_label": (payload.get("verdict") or {}).get("label"),
            "metric_delta": metric_delta,
            "candidate_metric": candidate_metric,
            "baseline_metric": baseline_metric,
            "delta_val_ppl": metric_delta if metric_key == "val_ppl" else None,
            "delta_wall_seconds": (payload.get("deltas") or {}).get("wall_seconds"),
            "candidate_val_ppl": (
                candidate_metric if metric_key == "val_ppl" else None
            ),
            "baseline_val_ppl": (
                baseline_metric if metric_key == "val_ppl" else None
            ),
            "comparison_link": f"/comparisons/{job['id']}" if payload.get("available") else None,
            "reason": payload.get("reason") or (payload.get("verdict") or {}).get("reason"),
            "interpretation_warnings": payload.get("interpretation_warnings") or [],
        })
    return rows


def _report_markdown(report: dict) -> str:
    protocol = report["protocol"]
    verdict = report["verdict"]
    stats = report["statistics"]
    metric_label = stats.get("metric_type") or "metric"
    lines = [
        f"# Study Report: {report['name']}",
        "",
        f"Research question: {report['research_question'] or 'Multi-run quantum/classical study'}",
        "",
        "## Verdict",
        f"- Label: {verdict['label']}",
        f"- Reason: {verdict['reason']}",
        f"- Outcome: {verdict.get('outcome') or '-'}",
        (
            "- Comparative inference enabled: "
            f"{bool(verdict.get('comparative_inference_enabled'))}"
        ),
        f"- Fair pairs: {stats['fair_pairs']}",
        f"- Aggregate available: {stats.get('aggregate_available', False)}",
        f"- Rerun-required pairs: {stats.get('rerun_required_pairs', 0)}",
        f"- Candidate wins: {stats['wins']}",
        f"- Mean delta {metric_label}: {stats['mean_delta'] if stats['mean_delta'] is not None else '-'}",
        f"- Std delta {metric_label}: {stats['std_delta'] if stats['std_delta'] is not None else '-'}",
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
    stored_task_type = protocol.get("task_type")
    claim_task_type = (claim or {}).get("task_type")
    if stored_task_type is None:
        task_type = claim_task_type or SEQUENCE_TASK_TYPE
    else:
        task_type = stored_task_type
    if not isinstance(task_type, str) or task_type not in TASK_TYPES:
        raise ValueError(
            f"Stored study has unsupported task_type {task_type!r}."
        )
    if claim_task_type is not None and task_type != claim_task_type:
        raise ValueError(
            f"Stored study task_type must match claim '{claim_id}': "
            f"{claim_task_type}"
        )
    protocol["task_type"] = task_type
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
    if task_type == "ground_state":
        protocol.setdefault(
            "analysis_mode", "single_candidate_diagnostic"
        )
        if protocol.get("energy_error_threshold") is None:
            protocol["energy_error_threshold"] = protocol[
                "analysis_settings"
            ].get(
                "diagnostic_tolerance",
                protocol["analysis_settings"].get(
                    "practical_equivalence_margin", 0.001
                ),
            )
        protocol["queue_analogues"] = False
    else:
        protocol.setdefault("analysis_mode", "paired_candidate_baseline")
        protocol.setdefault("energy_error_threshold", None)
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
        durability = job_durability_payload(job)
        job["manifest"] = durability["manifest"]
        job["durability"] = durability
        job.update(run_resource_payload(final))
        job["interpretation_warnings"] = durability["interpretation_warnings"]
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
    analogue_ladder = evidence.get("analogue_ladder") or {}
    analogue_limitations = [
        row.get("limitation")
        for row in analogue_ladder.get("rungs") or []
        if row.get("limitation")
    ]
    payload = {
        "id": study["id"],
        "name": study["name"],
        "research_question": study.get("research_question"),
        "task": study.get("task"),
        "task_type": resolved_protocol["task_type"],
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
        "interpretation_warnings": evidence.get("interpretation_warnings") or [],
        "analogue_ladder": analogue_ladder or None,
        "analogue_limitations": analogue_limitations,
        "analysis_mode": evidence.get(
            "analysis_mode", resolved_protocol.get("analysis_mode")
        ),
        "diagnostic_outcome": evidence.get("outcome"),
        "descriptive_observations": (
            evidence.get("descriptive_observations") or []
        ),
        "comparative_inference_enabled": bool(
            evidence.get(
                "comparative_inference_enabled",
                resolved_protocol["task_type"] != "ground_state",
            )
        ),
        "claim_level": evidence.get("claim_level"),
        "replication_status": evidence.get("replication_status"),
        "reference_ladder": evidence.get("reference_ladder"),
    }
    if resolved_protocol["task_type"] == "ground_state":
        payload["solver_competition_readiness"] = (
            evidence.get("solver_competition_readiness")
            or ground_state_solver_competition_readiness(
                get_claim(resolved_protocol.get("claim_id"))
                if resolved_protocol.get("claim_id")
                else None
            )
        )
    return payload


def study_report_payload(db: ResultsDB, study_id: int) -> dict:
    payload = study_payload(db, study_id, include_jobs=True)
    candidate_meta = preset_meta(payload["candidate_preset_id"])
    controls_meta = [preset_meta(item) for item in payload.get("control_preset_ids") or []]
    jobs = payload["jobs"]
    evidence = payload["evidence"]
    pair_rows = (
        []
        if payload.get("task_type") == "ground_state"
        else _pair_report_rows(db, jobs)
    )
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
        "completed_diagnostics": evidence.get("completed_diagnostics", 0),
        "win_rate": (
            float(evidence.get("wins", 0)) / float(evidence.get("fair_pairs", 1))
            if evidence.get("aggregate_available") and evidence.get("fair_pairs")
            else None
        ),
        "metric_type": evidence.get("metric_type"),
        "metric_key": evidence.get("metric_key"),
        "metric_units": evidence.get("metric_units"),
        "lower_is_better": evidence.get("lower_is_better"),
        "mean_delta": evidence.get("mean_delta"),
        "std_delta": evidence.get("std_delta"),
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
            "task_type": payload.get("task_type"),
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
            "analysis_mode": payload.get("analysis_mode"),
            "energy_error_threshold": payload["resolved_protocol"].get(
                "energy_error_threshold"
            ),
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
            "outcome": evidence.get("outcome"),
            "assessment_status": evidence.get("assessment_status"),
            "claim_level": evidence.get("claim_level"),
            "replication_status": evidence.get("replication_status"),
            "comparative_inference_enabled": bool(
                evidence.get("comparative_inference_enabled", True)
            ),
            "paired_stats": evidence.get("paired_stats"),
            "ladder": evidence.get("ladder") or [],
        },
        "statistics": statistics,
        "resource_summary": resource_summary,
        "pair_rows": pair_rows,
        "limitations": _study_limitations(payload, jobs, evidence),
        "claim_id": payload.get("claim_id"),
        "claim": payload.get("claim"),
        "task_type": payload.get("task_type"),
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
        "interpretation_warnings": evidence.get("interpretation_warnings") or [],
        "analogue_limitations": payload.get("analogue_limitations") or [],
        "analysis_mode": payload.get("analysis_mode"),
        "diagnostic_outcome": payload.get("diagnostic_outcome"),
        "descriptive_observations": payload.get(
            "descriptive_observations"
        ) or [],
        "comparative_inference_enabled": payload.get(
            "comparative_inference_enabled"
        ),
        "claim_level": payload.get("claim_level"),
        "replication_status": payload.get("replication_status"),
        "reference_ladder": payload.get("reference_ladder"),
    }
    if payload.get("task_type") == "ground_state":
        report["solver_competition_readiness"] = payload.get(
            "solver_competition_readiness"
        )
    report["markdown"] = _report_markdown(report)
    return report
