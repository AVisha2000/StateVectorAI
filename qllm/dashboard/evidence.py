"""Cautious quantum-advantage evidence ladder payloads."""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ..registry import metric_type_spec
from ._shared import primary_metric_value
from .analogues import DEFAULT_FAIRNESS_REQUIREMENTS

REQUIRED_FAIRNESS = DEFAULT_FAIRNESS_REQUIREMENTS


def _warning(
    code: str,
    severity: str,
    title: str,
    message: str,
    evidence: Any,
) -> dict[str, Any]:
    """Return the stable warning shape consumed by every evidence view."""
    return {
        "code": code,
        "severity": severity,
        "title": title,
        "message": message,
        "evidence": evidence,
    }


def job_durability_payload(job: Mapping[str, Any] | None) -> dict[str, Any]:
    """Project additive immutable/recovery state without rejecting legacy rows."""
    row = dict(job or {})
    manifest = row.get("manifest")
    manifest_warning = None
    if manifest is None and row.get("manifest_json"):
        try:
            decoded = json.loads(str(row["manifest_json"]))
            if not isinstance(decoded, dict):
                raise TypeError("manifest JSON is not an object")
            manifest = decoded
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            manifest_warning = _warning(
                "invalid_protocol",
                "error",
                "Unreadable legacy manifest",
                "The stored manifest is malformed; the job remains readable but its immutable identity cannot be verified.",
                {"field": "manifest_json", "error": f"{type(exc).__name__}: {exc}"},
            )
            manifest = None
    if manifest is not None and not isinstance(manifest, Mapping):
        manifest_warning = _warning(
            "invalid_protocol",
            "error",
            "Unreadable legacy manifest",
            "The stored manifest is not an object; immutable identity cannot be verified.",
            {"field": "manifest", "value_type": type(manifest).__name__},
        )
        manifest = None
    manifest = dict(manifest) if isinstance(manifest, Mapping) else None
    identity_fields = (
        "experiment_uuid", "run_uuid", "manifest_hash", "config_hash",
        "code_hash", "data_hash", "environment_hash", "seed_axes_hash",
    )
    identity = {
        key: row.get(key) if row.get(key) is not None else (manifest or {}).get(key)
        for key in identity_fields
    }
    warnings = [manifest_warning] if manifest_warning else []
    return {
        "status": row.get("status"),
        "manifest": manifest,
        "immutable_identity": identity,
        "checkpoint": {
            "latest": row.get("checkpoint_path"),
            "best": row.get("best_checkpoint_path"),
            "resume_from": row.get("resume_from"),
            "completed_step": row.get("completed_step"),
        },
        "worker": {
            "id": row.get("worker_id"),
            "claimed_ts": row.get("claimed_ts"),
            "heartbeat_ts": row.get("heartbeat_ts"),
            "lease_expires_ts": row.get("lease_expires_ts"),
        },
        "recovery": {
            "attempt_count": row.get("attempt_count"),
            "recovery_count": row.get("recovery_count"),
            "parent_run_uuid": row.get("parent_run_uuid"),
        },
        "interpretation_warnings": warnings,
    }


def run_resource_payload(final_run: Mapping[str, Any] | None) -> dict[str, Any]:
    """Expose the complete resource ledger and recorded component capabilities."""
    run = dict(final_run or {})
    resources = run.get("resources")
    if not isinstance(resources, Mapping):
        resources = None
    ledger = dict(resources) if resources is not None else None
    quantum_backend = (ledger or {}).get("quantum_backend")
    if not isinstance(quantum_backend, Mapping):
        quantum_backend = {}
    components = quantum_backend.get("components")
    if not isinstance(components, Mapping):
        components = {}
    capabilities = {
        str(name): value.get("capabilities")
        for name, value in components.items()
        if isinstance(value, Mapping) and value.get("capabilities") is not None
    }
    if not components and isinstance(quantum_backend.get("capabilities"), Mapping):
        capabilities["quantum_backend"] = dict(quantum_backend)
    return {
        "resource_ledger": ledger,
        "backend_capabilities": capabilities or None,
    }


def interpretation_warnings(
    *,
    available: bool = True,
    independent_pairs: int | None = None,
    single_seed: bool = False,
    baseline_linked: bool | None = None,
    candidate_uses_quantum: bool = False,
    analogue_ladder: Mapping[str, Any] | None = None,
    resource_normalized: Mapping[str, Any] | None = None,
    claim: Mapping[str, Any] | None = None,
    metric_contract: Mapping[str, Any] | None = None,
    metric_type: str | None = None,
    fairness: Mapping[str, Any] | None = None,
    duplicate_seeds: list[Any] | None = None,
    assessment_status: str | None = None,
    mixed_metric_types: bool = False,
    mixed_claim_ids: bool = False,
) -> list[dict[str, Any]]:
    """Classify presentation warnings from recorded evidence only.

    No threshold is inferred: the cost warning is possible only when a claim
    supplies its predeclared practical-equivalence margin.
    """
    warnings: list[dict[str, Any]] = []
    if independent_pairs == 1 or single_seed:
        pair_evidence = independent_pairs == 1
        warnings.append(_warning(
            "single_seed", "warning", "Single-seed evidence",
            (
                "One independent pair is smoke evidence only; repeat matched seeds before interpreting an edge."
                if pair_evidence
                else "This standalone run uses one seed and is descriptive only; a matched multi-seed comparison is required for comparative interpretation."
            ),
            {"independent_pairs": 1} if pair_evidence else {"independent_seeds": 1, "comparison_linked": False},
        ))
    if baseline_linked is False or not available:
        warnings.append(_warning(
            "unmatched_comparison", "error", "Unmatched comparison",
            "A linked candidate and baseline are required before interpreting this comparison.",
            {"available": bool(available), "baseline_linked": baseline_linked},
        ))

    ladder = dict(analogue_ladder or {})
    missing = list(ladder.get("missing_required") or [])
    if candidate_uses_quantum and not ladder and claim:
        missing = [
            str(row.get("id") or row.get("rung_id"))
            for row in (claim.get("analogue_ladder") or [])
            if isinstance(row, Mapping)
            and row.get("required")
            and (row.get("id") or row.get("rung_id"))
        ]
    if candidate_uses_quantum and (baseline_linked is False or missing):
        warnings.append(_warning(
            "missing_control", "error", "Required control missing",
            "The candidate is missing a required classical analogue or control rung.",
            {"missing_required": missing or ["linked_component_analogue"]},
        ))

    normalized = dict(resource_normalized or {})
    settings = dict((claim or {}).get("analysis_settings") or {})
    margin = settings.get("practical_equivalence_margin")
    improvement = normalized.get("improvement")
    candidate_wall = normalized.get("candidate_wall_seconds")
    baseline_wall = normalized.get("baseline_wall_seconds")
    compatible_cost_metric = (
        metric_type_spec(metric_type, require_pairable=True) is not None
    )
    if compatible_cost_metric and all(
        value is not None
        for value in (margin, improvement, candidate_wall, baseline_wall)
    ):
        extra_wall = float(candidate_wall) - float(baseline_wall)
        if extra_wall > 0 and 0.0 < float(improvement) < float(margin):
            warnings.append(_warning(
                "negligible_gain_high_cost", "warning", "Gain below practical margin",
                "The candidate costs more wall time and its improvement does not clear the claim's predeclared practical margin.",
                {"improvement": float(improvement), "practical_equivalence_margin": float(margin), "extra_wall_seconds": extra_wall},
            ))

    invalid_reasons: list[str] = []
    if (metric_contract or {}).get("rerun_required"):
        invalid_reasons.append("rerun_required")
    if assessment_status in {"invalid", "unsupported", "rerun_required"}:
        invalid_reasons.append(str(assessment_status))
    if mixed_metric_types:
        invalid_reasons.append("mixed_metric_types")
    if mixed_claim_ids:
        invalid_reasons.append("mixed_claim_ids")
    if duplicate_seeds:
        invalid_reasons.append("duplicate_seeds")
    fairness_payload = dict(fairness or {})
    mismatches = list(fairness_payload.get("disallowed_mismatches") or [])
    if not mismatches and fairness_payload.get("valid") is False:
        mismatches = list(fairness_payload.get("mismatches") or [])
    if fairness_payload.get("valid") is False or mismatches:
        invalid_reasons.append("fairness_mismatches")
    if metric_type in {"teacher_forced_side_information", None} and metric_contract:
        if (metric_contract or {}).get("protocol_status") != "current":
            invalid_reasons.append("unsupported_metric_contract")
    if invalid_reasons:
        warnings.append(_warning(
            "invalid_protocol", "error", "Invalid research protocol",
            "This result cannot support comparative interpretation until the protocol issues are resolved.",
            {"reasons": sorted(set(invalid_reasons)), "fairness_mismatches": mismatches, "duplicate_seeds": list(duplicate_seeds or [])},
        ))
    return warnings


def _step(key: str, label: str, ok: bool, detail: str, caution: str | None = None) -> dict:
    return {
        "key": key,
        "label": label,
        "ok": bool(ok),
        "detail": detail,
        "caution": caution,
    }


def comparison_evidence_ladder(payload: dict[str, Any]) -> dict:
    """Return run-level evidence rungs without strengthening claim gates."""
    available = bool(payload.get("available"))
    verdict = payload.get("verdict") or {}
    flags = payload.get("fairness") or {}
    resource = payload.get("resource_normalized") or {}
    candidate = payload.get("candidate") or {}
    baseline = payload.get("baseline") or {}
    cjob = candidate.get("job") or {}
    bjob = baseline.get("job") or {}
    metric_contract = payload.get("metric_contract") or {}
    protocol_valid = not metric_contract.get("rerun_required", False)
    metric_type = payload.get("metric_type") or metric_contract.get("metric_type")
    metric_spec = metric_type_spec(metric_type, require_pairable=True)
    metric_key = (
        str(metric_spec["extraction_key"]) if metric_spec is not None else None
    )
    candidate_metric = primary_metric_value(
        candidate.get("final_run"), metric_key
    ) if metric_key else None
    baseline_metric = primary_metric_value(
        baseline.get("final_run"), metric_key
    ) if metric_key else None
    metric_delta = (
        candidate_metric - baseline_metric
        if candidate_metric is not None and baseline_metric is not None
        else None
    )
    parameter_ratio = flags.get("parameter_delta_ratio")
    analogue = payload.get("analogue_ladder") or {}
    analogue_by_id = {
        row.get("id"): row for row in analogue.get("rungs") or []
    }
    parameter_tolerance = float(analogue.get("parameter_tolerance", 0.10))
    parameter_matched = (
        parameter_ratio is not None
        and abs(float(parameter_ratio)) <= parameter_tolerance
    )
    fair = (
        available
        and protocol_valid
        and metric_spec is not None
        and bool(flags.get("complete"))
        and bool(flags.get("valid"))
    )
    improvement = (
        metric_delta is not None
        and (
            float(metric_delta) < 0
            if bool(metric_spec["lower_is_better"])
            else float(metric_delta) > 0
        )
        and fair
    )
    cost_reviewed = resource.get("improvement") is not None
    cost_justified = (
        fair
        and cost_reviewed
        and resource.get("improvement_per_extra_second") is not None
        and float(resource.get("improvement") or 0.0) > 0
    )
    data_kind = (
        (cjob.get("config") or {}).get("data.kind")
        or (bjob.get("config") or {}).get("data.kind")
        or ""
    )
    task_specific = data_kind in {
        "monitored_ising",
        "contextual_parity",
        "markov_control",
    }
    metric_detail = (
        f"{metric_type} extracts {metric_key} ({metric_spec['units']})"
        if metric_spec is not None
        else f"unsupported metric_type {metric_type!r}"
    )
    if not protocol_valid:
        fair_detail = (
            metric_contract.get("limitation") or "metric contract requires a rerun"
        )
    elif metric_spec is None:
        fair_detail = verdict.get("reason") or metric_detail
    else:
        fair_detail = (
            "dataset, seed, steps, eval cadence, device, roles, "
            "training budget, and preprocessing match"
        )

    steps = [
        _step(
            "matched_baseline",
            "Matched classical baseline",
            available,
            "linked candidate/baseline pair" if available else payload.get("reason", "no linked baseline"),
        ),
        _step(
            "metric_supported",
            "Registered metric contract",
            metric_spec is not None,
            metric_detail,
        ),
        _step(
            "fair_protocol",
            "Fair protocol",
            fair,
            fair_detail,
        ),
        _step(
            "run_level_improvement",
            "Run-level improvement",
            improvement,
            verdict.get("reason")
            or f"requires a favorable {metric_type or 'registered metric'} delta on a fair pair",
            "A single fair run is smoke evidence, not an advantage claim.",
        ),
        _step(
            "parameter_matched",
            "Parameter-matched improvement",
            parameter_matched and improvement,
            (
                "parameter delta ratio "
                + ("unknown" if parameter_ratio is None else f"{float(parameter_ratio):.3f}")
            ),
        ),
        _step(
            "ablation_supported",
            "Ablation-supported improvement",
            improvement
            and analogue_by_id.get("frozen_random_control", {}).get("status") == "met",
            "requires trainable quantum to beat frozen/random quantum controls",
        ),
        _step(
            "task_specific",
            "Task-specific evidence",
            task_specific and improvement,
            data_kind or "requires a quantum-structured task or explicit task prior",
        ),
        _step(
            "cost_aware",
            "Cost-aware advantage",
            cost_justified,
            (
                "resource-normalized improvement per extra second "
                + (
                    "unavailable"
                    if resource.get("improvement_per_extra_second") is None
                    else f"{float(resource['improvement_per_extra_second']):.4f}"
                )
            ),
        ),
        _step(
            "multi_seed",
            "Multi-seed study evidence",
            False,
            "open or create a Study to aggregate repeated paired seeds",
        ),
    ]

    if metric_contract.get("rerun_required"):
        label = "rerun required"
        reason = metric_contract.get("limitation") or "metric contract is obsolete"
    elif not available:
        label = "incomplete"
        reason = payload.get("reason", "comparison missing")
    elif metric_spec is None:
        label = "unsupported metric"
        reason = verdict.get("reason") or f"unsupported metric_type {metric_type!r}"
    elif not fair:
        label = "unfair comparison"
        reason = "one or more fairness gates failed"
    elif metric_delta is None:
        label = "incomplete"
        reason = verdict.get("reason") or f"{metric_key} is unavailable"
    elif not improvement:
        label = "negative"
        reason = "candidate does not beat the baseline on the fair run"
    elif payload.get("claim_id") is None:
        label = "unassigned smoke result"
        reason = "no unambiguous claim ID is attached; this run cannot be promoted"
    elif parameter_matched and cost_justified:
        label = "cost-aware promising run"
        reason = "single fair run improves while passing parameter and cost review"
    else:
        label = "promising run"
        reason = "single fair run improves, but study/control evidence is still required"

    assessment_level = (
        verdict.get("assessment_level")
        or verdict.get("claim_level")
        or "run"
    )
    return {
        "label": label,
        "assessment_level": assessment_level,
        "claim_level": assessment_level,
        "claim_id": payload.get("claim_id"),
        "assessment_status": payload.get("assessment_status") or verdict.get("assessment_status"),
        "reason": reason,
        "steps": steps,
        "met_count": sum(1 for step in steps if step["ok"]),
        "total_count": len(steps),
    }


def study_evidence_ladder(evidence: dict[str, Any]) -> list[dict]:
    """Study-level rungs shared by study detail and future reports."""
    fair_pairs = int(evidence.get("fair_pairs") or 0)
    wins = int(evidence.get("wins") or 0)
    metric_type = evidence.get("metric_type")
    metric_spec = metric_type_spec(metric_type, require_pairable=True)
    metric_key = (
        str(metric_spec["extraction_key"]) if metric_spec is not None else None
    )
    mean_delta = evidence.get("mean_delta")
    std_delta = evidence.get("std_delta")
    if metric_key == "val_ppl":
        if mean_delta is None:
            mean_delta = evidence.get("mean_delta_val_ppl")
        if std_delta is None:
            std_delta = evidence.get("std_delta_val_ppl")
    paired = evidence.get("paired_stats") or {}
    metric_supported = metric_spec is not None
    has_multi_seed = metric_supported and int(paired.get("n_pairs") or 0) >= 6
    favorable_mean = (
        mean_delta is not None
        and (
            float(mean_delta) < 0
            if bool(metric_spec["lower_is_better"])
            else float(mean_delta) > 0
        )
        if metric_spec is not None
        else False
    )
    candidate_wins = has_multi_seed and wins > fair_pairs / 2 and favorable_mean
    low_variance = metric_supported and std_delta is not None and fair_pairs >= 2
    return [
        _step("matched_baseline", "Matched baselines", fair_pairs > 0, f"{fair_pairs} fair pair(s)"),
        _step(
            "metric_supported",
            "Registered metric contract",
            metric_supported,
            (
                f"{metric_type} extracts {metric_key} ({metric_spec['units']})"
                if metric_spec is not None
                else f"unsupported metric_type {metric_type!r}"
            ),
        ),
        _step("multi_seed", "Repeated multi-seed evidence", has_multi_seed, f"{fair_pairs} fair completed seed(s)"),
        _step("candidate_better", "Candidate better on average", candidate_wins, f"{wins}/{fair_pairs} candidate win(s)"),
        _step("variance_reviewed", "Variance reviewed", low_variance, "-" if std_delta is None else f"std {float(std_delta):.3f}"),
        _step("parameter_matched", "Parameter-matched evidence", bool((evidence.get("analogue_ladder") or {}).get("required_complete")), "see structured analogue ladder"),
        _step("ablation_supported", "Ablation-supported evidence", any(row.get("id") == "frozen_random_control" and row.get("status") == "met" for row in ((evidence.get("analogue_ladder") or {}).get("rungs") or [])), "requires frozen/random quantum controls"),
        _step("task_specific", "Task-specific evidence", False, "requires explicit task prior or quantum-structured dataset"),
        _step("cost_aware", "Cost-aware advantage", False, "requires resource-normalized study-level benefit"),
    ]
