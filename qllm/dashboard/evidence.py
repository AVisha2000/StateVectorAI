"""Cautious quantum-advantage evidence ladder payloads."""
from __future__ import annotations

from typing import Any

from .analogues import DEFAULT_FAIRNESS_REQUIREMENTS

REQUIRED_FAIRNESS = DEFAULT_FAIRNESS_REQUIREMENTS


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
    deltas = payload.get("deltas") or {}
    candidate = payload.get("candidate") or {}
    baseline = payload.get("baseline") or {}
    cjob = candidate.get("job") or {}
    bjob = baseline.get("job") or {}
    metric_contract = payload.get("metric_contract") or {}
    protocol_valid = not metric_contract.get("rerun_required", False)
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
        and bool(flags.get("complete"))
        and bool(flags.get("valid"))
    )
    improvement = (
        deltas.get("val_ppl") is not None
        and float(deltas["val_ppl"]) < 0
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

    steps = [
        _step(
            "matched_baseline",
            "Matched classical baseline",
            available,
            "linked candidate/baseline pair" if available else payload.get("reason", "no linked baseline"),
        ),
        _step(
            "fair_protocol",
            "Fair protocol",
            fair,
            (
                metric_contract.get("limitation")
                if not protocol_valid
                else "dataset, seed, steps, eval cadence, device, roles, "
                "training budget, and preprocessing match"
            ),
        ),
        _step(
            "run_level_improvement",
            "Run-level improvement",
            improvement,
            verdict.get("reason") or "requires lower candidate validation perplexity on a fair pair",
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
            analogue_by_id.get("frozen_random_control", {}).get("status") == "met",
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
    elif not fair:
        label = "unfair comparison"
        reason = "one or more fairness gates failed"
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

    return {
        "label": label,
        "claim_level": verdict.get("claim_level") or "run",
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
    mean_delta = evidence.get("mean_delta_val_ppl")
    std_delta = evidence.get("std_delta_val_ppl")
    paired = evidence.get("paired_stats") or {}
    has_multi_seed = int(paired.get("n_pairs") or 0) >= 6
    candidate_wins = has_multi_seed and wins > fair_pairs / 2 and (mean_delta or 0) < 0
    low_variance = std_delta is not None and fair_pairs >= 2
    return [
        _step("matched_baseline", "Matched baselines", fair_pairs > 0, f"{fair_pairs} fair pair(s)"),
        _step("multi_seed", "Repeated multi-seed evidence", has_multi_seed, f"{fair_pairs} fair completed seed(s)"),
        _step("candidate_better", "Candidate better on average", candidate_wins, f"{wins}/{fair_pairs} candidate win(s)"),
        _step("variance_reviewed", "Variance reviewed", low_variance, "-" if std_delta is None else f"std {float(std_delta):.3f}"),
        _step("parameter_matched", "Parameter-matched evidence", bool((evidence.get("analogue_ladder") or {}).get("required_complete")), "see structured analogue ladder"),
        _step("ablation_supported", "Ablation-supported evidence", any(row.get("id") == "frozen_random_control" and row.get("status") == "met" for row in ((evidence.get("analogue_ladder") or {}).get("rungs") or [])), "requires frozen/random quantum controls"),
        _step("task_specific", "Task-specific evidence", False, "requires explicit task prior or quantum-structured dataset"),
        _step("cost_aware", "Cost-aware advantage", False, "requires resource-normalized study-level benefit"),
    ]
