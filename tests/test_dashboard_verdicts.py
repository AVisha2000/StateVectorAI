import sqlite3

import pytest

from qllm.dashboard.verdicts import (
    advantage_report_scorecard,
    build_verdict_snapshot,
    comparison_verdict_snapshot,
    persist_verdict_snapshot,
    verdict_snapshot_detail_response,
    verdict_snapshot_list_response,
)
from qllm.claims import get_claim
from qllm.quantum.advantage import AdvantageReport
from qllm.resultsdb import ResultsDB


CLAIM_ID = "variational_component_swaps"


def _payload(**overrides):
    payload = {
        "source_kind": "comparison",
        "source_id": "41",
        "claim_id": CLAIM_ID,
        "assessment": {"level": "smoke", "status": "pilot_only"},
        "scorecard": {"dimensions": {"val_ppl_delta": -0.2}},
        "fairness": {"valid": True},
        "controls": {"matched_gru": "pending"},
        "caveats": ["single pair"],
        "evidence": {"candidate": "run-a"},
        "diagnostics": {"paired_stats": None},
    }
    payload.update(overrides)
    return payload


def test_legacy_and_repeatable_verdict_migration_preserves_existing_rows(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE runs (id INTEGER PRIMARY KEY, suite TEXT)")
        con.execute("INSERT INTO runs VALUES (1, 'legacy')")
    db = ResultsDB(path)
    ResultsDB(path)
    with db._conn() as con:
        assert con.execute("SELECT suite FROM runs WHERE id=1").fetchone()[0] == "legacy"
        assert con.execute("SELECT COUNT(*) FROM verdict_snapshots").fetchone()[0] == 0


def test_snapshot_idempotence_revision_history_and_list_detail(tmp_path):
    db = ResultsDB(tmp_path / "results.db")
    first = persist_verdict_snapshot(db, _payload())
    same = persist_verdict_snapshot(db, _payload())
    second = persist_verdict_snapshot(db, _payload(caveats=["two pairs"]))
    assert (first["id"], first["revision"]) == (same["id"], 1)
    assert second["revision"] == 2
    history = db.get_verdict_snapshot_history(first["verdict_key"])
    assert [item["revision"] for item in history] == [2, 1]
    response = verdict_snapshot_list_response(db, limit=999)
    assert len(response.snapshots) == 1
    detail = verdict_snapshot_detail_response(db, second["id"])
    assert detail is not None and len(detail.history) == 2
    assert verdict_snapshot_detail_response(db, 99999) is None


def test_canonical_claim_is_separate_from_assessment_and_promotion_is_rejected(tmp_path):
    db = ResultsDB(tmp_path / "results.db")
    snapshot = persist_verdict_snapshot(db, _payload())
    claim = get_claim(CLAIM_ID)
    assert snapshot["claim_level"] == claim["level"]
    assert snapshot["claim_status"] == claim["status"]
    assert snapshot["replication_status"] == claim["replication_status"]
    assert snapshot["assessment_level"] == "smoke"
    with pytest.raises(ValueError, match="canonical claim ledger"):
        build_verdict_snapshot(_payload(claim_level="hardware"))
    with pytest.raises(ValueError, match="canonical claim ledger"):
        build_verdict_snapshot(_payload(claim_status="established"))
    with pytest.raises(ValueError, match="unknown claim_id"):
        build_verdict_snapshot(_payload(claim_id="not-a-claim"))


def test_comparison_snapshot_requires_canonical_available_evidence_and_preserves_provenance(tmp_path):
    db = ResultsDB(tmp_path / "results.db")
    assert comparison_verdict_snapshot(db, {"claim_id": CLAIM_ID, "available": False}) is None
    payload = {
        "claim_id": CLAIM_ID,
        "available": True,
        "job_id": 9,
        "candidate": {"job": {"id": 9, "preset_id": "candidate"}, "final_run": {"run_uuid": "run-c"}},
        "baseline": {"job": {"id": 8, "preset_id": "baseline"}, "final_run": {"run_uuid": "run-b"}},
        "verdict": {"claim_level": "anecdote", "assessment_status": "smoke"},
        "fairness": {"valid": True},
        "interpretation_warnings": ["pilot"],
        "deltas": {"val_ppl": -0.1},
    }
    snapshot = comparison_verdict_snapshot(db, payload)
    assert snapshot is not None
    assert snapshot["evidence"]["baseline"]["run"]["run_uuid"] == "run-b"
    assert snapshot["assessment_level"] == "anecdote"
    assert snapshot["claim_level"] == get_claim(CLAIM_ID)["level"]


def test_advantage_report_scorecard_and_forbidden_composites(tmp_path):
    report = AdvantageReport(
        n_qubits=2, n_layers=1, n_samples=4, g_per_classical={"rbf": 0.2}, g_min=0.2,
        kq_offdiag_mean=0.1, kq_offdiag_std=0.01, kernel_diagnostics={"rank": 2.0},
        r2_engineered={"quantum": 0.8}, r2_classical_natural={"rbf": 0.7},
    )
    scorecard = advantage_report_scorecard(report)
    assert scorecard["diagnostics"]["geometric_difference"]["minimum"] == 0.2
    assert scorecard["diagnostics"]["kernel_concentration"]["offdiag_mean"] == 0.1
    assert scorecard["controls"]["engineered"]["quantum"] == 0.8
    assert "claim_level" not in scorecard
    with pytest.raises(ValueError, match="composite_score"):
        persist_verdict_snapshot(ResultsDB(tmp_path / "results.db"), _payload(diagnostics={"composite_score": 1}))
    valid = build_verdict_snapshot(_payload(diagnostics={"kernel_gap": 0.4}))
    assert valid["diagnostics"]["kernel_gap"] == 0.4
    with pytest.raises(ValueError, match="assessment.level"):
        build_verdict_snapshot(_payload(assessment={"level": {"bad": True}}))
