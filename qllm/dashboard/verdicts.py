"""Durable, claim-ledger-bound verdict snapshot service models.

This module intentionally has no route registration or automatic persistence
hook.  Routes may use its Pydantic projections once they are explicitly wired.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

from pydantic import BaseModel

from ..claims import get_claim, list_claims
from ..resultsdb import ResultsDB


VERDICT_SNAPSHOT_SCHEMA_VERSION = 1
_FORBIDDEN_SCORE_KEYS = frozenset({"advantage_score", "composite_score", "composite_advantage_score"})
_RECONCILIATION_JOB_LIMIT = 200


class VerdictSnapshotSummary(BaseModel):
    id: int
    verdict_key: str
    revision: int
    content_hash: str
    source_kind: str
    source_id: str
    claim_id: str
    claim_level: str
    claim_status: str
    replication_status: str
    assessment_level: str | None = None
    assessment_status: str | None = None
    created_ts: str

    class Config:
        extra = "forbid"


class VerdictSnapshotDetail(VerdictSnapshotSummary):
    source_job_id: int | None = None
    source_study_id: int | None = None
    source_run_id: str | None = None
    scorecard: dict[str, Any]
    fairness: dict[str, Any]
    controls: dict[str, Any]
    caveats: list[Any]
    evidence: dict[str, Any]
    diagnostics: dict[str, Any]
    schema_version: int


class VerdictSnapshotListResponse(BaseModel):
    snapshots: list[VerdictSnapshotSummary]

    class Config:
        extra = "forbid"


class VerdictSnapshotHistoryResponse(BaseModel):
    snapshot: VerdictSnapshotDetail
    history: list[VerdictSnapshotSummary]

    class Config:
        extra = "forbid"


def _reject_forbidden_score_keys(value: Any, *, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if key_text.lower() in _FORBIDDEN_SCORE_KEYS:
                raise ValueError(f"{path}.{key_text} is not permitted in verdict snapshots")
            _reject_forbidden_score_keys(item, path=f"{path}.{key_text}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_forbidden_score_keys(item, path=f"{path}[{index}]")


def _canonical_claim(payload: Mapping[str, Any]) -> dict[str, Any]:
    claim_id = payload.get("claim_id")
    if not isinstance(claim_id, str) or not claim_id:
        raise ValueError("A canonical claim_id is required for a verdict snapshot")
    try:
        claim = get_claim(claim_id)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc
    expected = {
        "claim_level": claim["level"],
        "claim_status": claim["status"],
        "replication_status": claim["replication_status"],
    }
    supplied_claim = payload.get("claim")
    if supplied_claim is not None and not isinstance(supplied_claim, Mapping):
        raise ValueError("claim must be a mapping when supplied")
    for name, value in expected.items():
        supplied = payload.get(name)
        if supplied is None and isinstance(supplied_claim, Mapping):
            supplied = supplied_claim.get(name.removeprefix("claim_"))
            if name == "replication_status":
                supplied = supplied_claim.get(name)
        if supplied is not None and supplied != value:
            raise ValueError(f"{name} must match the canonical claim ledger")
    return claim


def build_verdict_snapshot(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Build a validated snapshot without allowing caller-side claim promotion."""
    _reject_forbidden_score_keys(payload)
    claim = _canonical_claim(payload)
    source_kind = payload.get("source_kind")
    source_id = payload.get("source_id")
    if not isinstance(source_kind, str) or not source_kind:
        raise ValueError("source_kind is required")
    if source_id is None or str(source_id) == "":
        raise ValueError("source_id is required")
    assessment = payload.get("assessment") or {}
    if not isinstance(assessment, Mapping):
        raise ValueError("assessment must be a mapping")
    for name in ("level", "status"):
        value = assessment.get(name)
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"assessment.{name} must be a non-empty string or null")
    named = {}
    for key, default in (
        ("scorecard", {}), ("fairness", {}), ("controls", {}),
        ("caveats", []), ("evidence", {}), ("diagnostics", {}),
    ):
        value = payload.get(key, default)
        if not isinstance(value, type(default)):
            raise ValueError(f"{key} must be a {type(default).__name__}")
        named[key] = value
    return {
        "verdict_key": str(payload.get("verdict_key") or f"{source_kind}:{source_id}"),
        "source_kind": source_kind,
        "source_id": str(source_id),
        "claim_id": claim["claim_id"],
        "claim_level": claim["level"],
        "claim_status": claim["status"],
        "replication_status": claim["replication_status"],
        "assessment_level": assessment.get("level"),
        "assessment_status": assessment.get("status"),
        "source_job_id": payload.get("source_job_id"),
        "source_study_id": payload.get("source_study_id"),
        "source_run_id": payload.get("source_run_id"),
        **named,
        "schema_version": VERDICT_SNAPSHOT_SCHEMA_VERSION,
    }


def persist_verdict_snapshot(db: ResultsDB, payload: Mapping[str, Any]) -> dict[str, Any]:
    return db.append_verdict_snapshot(build_verdict_snapshot(payload))


def advantage_report_scorecard(report: Mapping[str, Any] | Any) -> dict[str, dict[str, Any]]:
    """Project advantage-screen diagnostics without assigning any claim level."""
    value = asdict(report) if is_dataclass(report) else dict(report)
    _reject_forbidden_score_keys(value, path="advantage_report")
    return {
        "diagnostics": {
            "geometric_difference": {
                "minimum": value.get("g_min"),
                "per_classical": value.get("g_per_classical") or {},
            },
            "kernel_concentration": {
                "offdiag_mean": value.get("kq_offdiag_mean"),
                "offdiag_std": value.get("kq_offdiag_std"),
                "metrics": value.get("kernel_diagnostics") or {},
            },
        },
        "controls": {
            "engineered": value.get("r2_engineered") or {},
            "classical_natural": value.get("r2_classical_natural") or {},
            "dequantization": value.get("dequantization"),
        },
    }


def _job_provenance(job: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: job.get(key)
        for key in (
            "id",
            "run_uuid",
            "experiment_uuid",
            "manifest_hash",
            "preset_id",
            "dataset_name",
            "seed",
            "steps",
            "device_target",
        )
        if job.get(key) is not None
    }


def _run_provenance(run: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: run.get(key)
        for key in (
            "id",
            "run_uuid",
            "experiment_uuid",
            "manifest_hash",
            "suite",
            "variant",
            "dataset",
            "seed",
            "steps",
        )
        if run.get(key) is not None
    }


def comparison_verdict_snapshot(db: ResultsDB, payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """Persist available, claim-bound comparison evidence as an immutable revision."""
    claim_id = payload.get("claim_id")
    candidate = payload.get("candidate") or {}
    baseline = payload.get("baseline") or {}
    candidate_job = candidate.get("job") or {}
    baseline_job = baseline.get("job") or {}
    candidate_run = candidate.get("final_run") or {}
    baseline_run = baseline.get("final_run") or {}
    if not claim_id or not payload.get("available") or not candidate_run or not baseline_run:
        return None
    source_id = payload.get("job_id") or candidate_job.get("id")
    if source_id is None:
        return None
    verdict = payload.get("verdict") or {}
    warnings = payload.get("interpretation_warnings") or []
    scorecard = {
        "dimensions": {
            "metric_type": payload.get("metric_type"),
            "deltas": payload.get("deltas") or {},
            "resource_normalized": payload.get("resource_normalized"),
        }
    }
    service_payload = {
        "verdict_key": f"comparison:{source_id}",
        "source_kind": "comparison",
        "source_id": str(source_id),
        "claim_id": claim_id,
        "assessment": {
            "level": verdict.get("claim_level"),
            "status": payload.get("assessment_status") or verdict.get("assessment_status"),
        },
        "source_job_id": candidate_job.get("id"),
        "source_study_id": payload.get("study_id") or candidate_job.get("study_id"),
        "source_run_id": candidate_run.get("run_uuid") or candidate_run.get("id"),
        "scorecard": scorecard,
        "fairness": payload.get("fairness") or {},
        "controls": {
            "analogue_ladder": payload.get("analogue_ladder"),
            "metric_contract": payload.get("metric_contract"),
        },
        "caveats": warnings if isinstance(warnings, list) else [warnings],
        "evidence": {
            "candidate": {
                "job": _job_provenance(candidate_job),
                "run": _run_provenance(candidate_run),
            },
            "baseline": {
                "job": _job_provenance(baseline_job),
                "run": _run_provenance(baseline_run),
            },
            "evidence_ladder": payload.get("evidence_ladder"),
        },
        "diagnostics": {
            "paired_stats": payload.get("paired_stats"),
            "equivalence": payload.get("equivalence"),
            "power": payload.get("power"),
        },
    }
    return persist_verdict_snapshot(db, service_payload)


def reconcile_comparison_verdict_snapshots(
    db: ResultsDB, *, job_limit: int = _RECONCILIATION_JOB_LIMIT
) -> None:
    """Materialize eligible recent comparison verdicts without read-order coupling.

    The canonical comparison payload remains the sole owner of claim, fairness,
    final-run, and persistence decisions.  The bounded pending batch advances
    by excluding existing snapshots.  A malformed or unavailable job must not
    prevent the verdict list from remaining available.
    """
    from .lab import comparison_research_payload

    bounded = max(1, int(job_limit))
    claims = list_claims()
    claim_ids = [claim["claim_id"] for claim in claims]
    preset_ids = sorted({
        preset_id
        for claim in claims
        for preset_id in claim["analysis_settings"]["preset_ids"]
    })
    if not claim_ids and not preset_ids:
        return
    claim_placeholders = ", ".join("?" for _ in claim_ids)
    preset_placeholders = ", ".join("?" for _ in preset_ids)
    with db._conn() as con:
        rows = con.execute(
            "SELECT candidate.id FROM lab_jobs AS candidate "
            "JOIN lab_jobs AS baseline ON baseline.id=candidate.compare_to_job_id "
            "WHERE candidate.status='done' AND baseline.status='done' "
            "AND candidate.comparison_role != 'baseline' "
            "AND EXISTS (SELECT 1 FROM run_results AS candidate_run "
            "WHERE candidate_run.run_uuid=candidate.run_uuid "
            "AND candidate_run.suite='lab') "
            "AND EXISTS (SELECT 1 FROM run_results AS baseline_run "
            "WHERE baseline_run.run_uuid=baseline.run_uuid "
            "AND baseline_run.suite='lab') "
            "AND json_valid(candidate.config_json) "
            "AND (candidate.preset_id IN (" + preset_placeholders + ") "
            "OR CASE WHEN json_valid(candidate.config_json) "
            "THEN json_extract(candidate.config_json, '$.\"research.claim_id\"') "
            "IN (" + claim_placeholders + ") ELSE 0 END) "
            "AND NOT EXISTS ("
            "SELECT 1 FROM verdict_snapshots AS snapshot "
            "WHERE snapshot.verdict_key=('comparison:' || candidate.id)"
            ") ORDER BY candidate.id ASC LIMIT ?",
            (*preset_ids, *claim_ids, bounded),
        ).fetchall()

    for row in rows:
        try:
            comparison_research_payload(db, int(row["id"]))
        # This is an availability boundary for a list endpoint: malformed,
        # deleted, or otherwise unavailable historical rows are isolated.
        except Exception:
            continue


def verdict_snapshot_list_response(db: ResultsDB, *, limit: int = 100) -> VerdictSnapshotListResponse:
    reconcile_comparison_verdict_snapshots(db)
    return VerdictSnapshotListResponse(
        snapshots=[_summary(snapshot) for snapshot in db.list_latest_verdict_snapshots(limit)]
    )


def verdict_snapshot_detail_response(db: ResultsDB, snapshot_id: int) -> VerdictSnapshotHistoryResponse | None:
    snapshot = db.get_verdict_snapshot(snapshot_id)
    if snapshot is None:
        return None
    history = db.get_verdict_snapshot_history(snapshot["verdict_key"])
    return VerdictSnapshotHistoryResponse(
        snapshot=_detail(snapshot), history=[_summary(item) for item in history]
    )


def _summary(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    fields = set(
        getattr(
            VerdictSnapshotSummary,
            "model_fields",
            VerdictSnapshotSummary.__fields__,
        )
    )
    return {key: value for key, value in snapshot.items() if key in fields}


def _detail(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    fields = set(
        getattr(
            VerdictSnapshotDetail,
            "model_fields",
            VerdictSnapshotDetail.__fields__,
        )
    )
    return {key: value for key, value in snapshot.items() if key in fields}


__all__ = [
    "VERDICT_SNAPSHOT_SCHEMA_VERSION", "VerdictSnapshotDetail", "VerdictSnapshotHistoryResponse",
    "VerdictSnapshotListResponse", "VerdictSnapshotSummary", "advantage_report_scorecard",
    "build_verdict_snapshot", "comparison_verdict_snapshot", "persist_verdict_snapshot",
    "reconcile_comparison_verdict_snapshots", "verdict_snapshot_detail_response",
    "verdict_snapshot_list_response",
]
