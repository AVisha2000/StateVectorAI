"""Durable, claim-ledger-bound verdict snapshot service models.

This module intentionally has no route registration or automatic persistence
hook.  Routes may use its Pydantic projections once they are explicitly wired.
"""
from __future__ import annotations

import sqlite3
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

from pydantic import BaseModel

from ..claims import get_claim, list_claims
from ..resultsdb import ResultsDB
from .atlas import (
    ATLAS_VERDICT_KEY_PREFIX,
    ATLAS_VERDICT_SOURCE_KIND,
    atlas_verdict_key,
)


VERDICT_SNAPSHOT_SCHEMA_VERSION = 1
VERDICT_SNAPSHOT_LIST_LIMIT = 100
VERDICT_RECONCILIATION_JOB_LIMIT = 25
_FORBIDDEN_SCORE_KEYS = frozenset({"advantage_score", "composite_score", "composite_advantage_score"})
_RECONCILIATION_CURSOR_NAME = "comparison_verdict_snapshots"


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
    verdict_key = str(payload.get("verdict_key") or f"{source_kind}:{source_id}")
    expected_projection_key = atlas_verdict_key(claim["claim_id"])
    if source_kind == ATLAS_VERDICT_SOURCE_KIND:
        if str(source_id) != claim["claim_id"] or verdict_key != expected_projection_key:
            raise ValueError("Claim projection identity must match its canonical claim_id")
    elif verdict_key.startswith(ATLAS_VERDICT_KEY_PREFIX):
        raise ValueError("Claim projection verdict keys are reserved")
    return {
        "verdict_key": verdict_key,
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
    snapshot = build_verdict_snapshot(payload)
    if snapshot["source_kind"] == ATLAS_VERDICT_SOURCE_KIND:
        raise ValueError("Claim projections are internal persistence records")
    snapshot = db.append_verdict_snapshot(snapshot)
    _persist_claim_verdict_projection(db, snapshot)
    return snapshot


def _claim_contracts() -> list[tuple[str, str, str, str]]:
    return [
        (
            claim["claim_id"],
            claim["level"],
            claim["status"],
            claim["replication_status"],
        )
        for claim in list_claims()
    ]


def current_claim_verdict_sources(db: ResultsDB) -> list[dict]:
    """Return one current-ledger source snapshot per claim, excluding projections."""
    return db.list_current_verdict_snapshots_for_claims(
        _claim_contracts(),
        exclude_source_kind=ATLAS_VERDICT_SOURCE_KIND,
    )


def current_claim_verdict_projections(db: ResultsDB) -> list[dict]:
    """Return only projections that exactly reproduce their current source row."""
    projections = []
    for source in current_claim_verdict_sources(db):
        expected = _claim_verdict_projection(source)
        projection = next(
            (
                candidate
                for candidate in db.get_verdict_snapshot_history(
                    atlas_verdict_key(source["claim_id"])
                )
                if all(
                    candidate.get(key) == value
                    for key, value in expected.items()
                )
            ),
            None,
        )
        if projection is not None:
            projections.append(projection)
    return projections


def reconcile_claim_verdict_projections(db: ResultsDB) -> None:
    """Persist stable per-claim join keys over append-only source snapshots."""
    for source in current_claim_verdict_sources(db):
        try:
            _persist_claim_verdict_projection(db, source)
        except ValueError:
            # A malformed historical projection must not poison all verdict reads.
            continue


def _persist_claim_verdict_projection(db: ResultsDB, source: Mapping[str, Any]) -> dict:
    projection = _claim_verdict_projection(source)
    return db.append_verdict_snapshot(
        projection,
        projection_source_snapshot_id=int(source["id"]),
    )


def _claim_verdict_projection(source: Mapping[str, Any]) -> dict:
    evidence = dict(source.get("evidence") or {})
    evidence["claim_projection_source"] = {
        "snapshot_id": source["id"],
        "verdict_key": source["verdict_key"],
        "source_kind": source["source_kind"],
        "source_id": source["source_id"],
    }
    return build_verdict_snapshot(
        {
            "verdict_key": atlas_verdict_key(source["claim_id"]),
            "source_kind": ATLAS_VERDICT_SOURCE_KIND,
            "source_id": source["claim_id"],
            "claim_id": source["claim_id"],
            "assessment": {
                "level": source.get("assessment_level"),
                "status": source.get("assessment_status"),
            },
            "source_job_id": source.get("source_job_id"),
            "source_study_id": source.get("source_study_id"),
            "source_run_id": source.get("source_run_id"),
            "scorecard": source.get("scorecard") or {},
            "fairness": source.get("fairness") or {},
            "controls": source.get("controls") or {},
            "caveats": source.get("caveats") or [],
            "evidence": evidence,
            "diagnostics": source.get("diagnostics") or {},
        }
    )


def _is_claim_projection_record(snapshot: Mapping[str, Any]) -> bool:
    return (
        snapshot.get("source_kind") == ATLAS_VERDICT_SOURCE_KIND
        or str(snapshot.get("verdict_key") or "").startswith(
            ATLAS_VERDICT_KEY_PREFIX
        )
    )


def _valid_claim_projection(db: ResultsDB, projection: Mapping[str, Any]) -> bool:
    if not _is_claim_projection_record(projection):
        return False
    evidence = projection.get("evidence") or {}
    source_ref = (
        evidence.get("claim_projection_source")
        if isinstance(evidence, Mapping)
        else None
    )
    source_id = (
        source_ref.get("snapshot_id")
        if isinstance(source_ref, Mapping)
        else None
    )
    if (
        isinstance(source_id, bool)
        or not isinstance(source_id, int)
        or source_id <= 0
    ):
        return False
    source = db.get_verdict_snapshot(source_id)
    if source is None or _is_claim_projection_record(source):
        return False
    try:
        expected = _claim_verdict_projection(source)
    except (KeyError, TypeError, ValueError):
        return False
    return all(
        projection.get(key) == value for key, value in expected.items()
    )


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
            "level": verdict.get("assessment_level") or verdict.get("claim_level"),
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
    db: ResultsDB, *, job_limit: int = VERDICT_RECONCILIATION_JOB_LIMIT
) -> None:
    """Materialize eligible recent comparison verdicts without read-order coupling.

    The canonical comparison payload remains the sole owner of claim, fairness,
    final-run, and persistence decisions.  A durable high-water cursor advances
    after every deterministic attempt, so a malformed historical row cannot
    starve later evidence.  Exhausted passes wrap to allow later corrections.
    """
    from .lab import comparison_research_payload

    bounded = max(1, min(int(job_limit), VERDICT_RECONCILIATION_JOB_LIMIT))
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
    cursor = db.get_reconciliation_cursor(_RECONCILIATION_CURSOR_NAME)
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
            "AND candidate.id>? AND NOT EXISTS ("
            "SELECT 1 FROM verdict_snapshots AS snapshot "
            "WHERE snapshot.verdict_key=('comparison:' || candidate.id)"
            ") ORDER BY candidate.id ASC LIMIT ?",
            (*preset_ids, *claim_ids, cursor, bounded),
        ).fetchall()

    if not rows:
        db.reset_reconciliation_cursor(_RECONCILIATION_CURSOR_NAME)
        return

    for row in rows:
        candidate_job_id = int(row["id"])
        try:
            comparison_research_payload(
                db, candidate_job_id, include_curves=False
            )
        except sqlite3.Error:
            # Database/system failures are not deterministic row failures. Do
            # not advance the cursor, so a later call can safely retry it.
            return
        # Deterministically malformed historical rows are isolated and cannot
        # starve later evidence. Unexpected failures remain retryable.
        except (ArithmeticError, KeyError, TypeError, ValueError):
            pass
        except Exception:
            return
        db.advance_reconciliation_cursor(
            _RECONCILIATION_CURSOR_NAME, candidate_job_id
        )


def verdict_snapshot_list_response(
    db: ResultsDB,
    *,
    limit: int = VERDICT_SNAPSHOT_LIST_LIMIT,
) -> VerdictSnapshotListResponse:
    bounded = max(1, min(int(limit), VERDICT_SNAPSHOT_LIST_LIMIT))
    reconcile_comparison_verdict_snapshots(db)
    reconcile_claim_verdict_projections(db)
    projections = current_claim_verdict_projections(db)
    projected_claims = {snapshot["claim_id"] for snapshot in projections}
    regular = [
        snapshot
        for snapshot in db.list_latest_verdict_snapshots(bounded)
        if not _is_claim_projection_record(snapshot)
        and snapshot["claim_id"] not in projected_claims
    ]
    rows = (projections + regular)[:bounded]
    return VerdictSnapshotListResponse(
        snapshots=[_summary(snapshot) for snapshot in rows]
    )


def verdict_snapshot_detail_response(db: ResultsDB, snapshot_id: int) -> VerdictSnapshotHistoryResponse | None:
    snapshot = db.get_verdict_snapshot(snapshot_id)
    if snapshot is None or (
        _is_claim_projection_record(snapshot)
        and not _valid_claim_projection(db, snapshot)
    ):
        return None
    history = [
        item
        for item in db.get_verdict_snapshot_history(snapshot["verdict_key"])
        if not _is_claim_projection_record(item)
        or _valid_claim_projection(db, item)
    ]
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
    "VERDICT_RECONCILIATION_JOB_LIMIT", "VERDICT_SNAPSHOT_LIST_LIMIT",
    "VERDICT_SNAPSHOT_SCHEMA_VERSION",
    "VerdictSnapshotDetail", "VerdictSnapshotHistoryResponse",
    "VerdictSnapshotListResponse", "VerdictSnapshotSummary", "advantage_report_scorecard",
    "build_verdict_snapshot", "comparison_verdict_snapshot",
    "current_claim_verdict_projections", "current_claim_verdict_sources",
    "persist_verdict_snapshot",
    "reconcile_claim_verdict_projections", "reconcile_comparison_verdict_snapshots",
    "verdict_snapshot_detail_response", "verdict_snapshot_list_response",
]
