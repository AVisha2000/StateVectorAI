import sqlite3

import pytest

from qllm.dashboard.verdicts import (
    advantage_report_scorecard,
    build_verdict_snapshot,
    comparison_verdict_snapshot,
    persist_verdict_snapshot,
    reconcile_comparison_verdict_snapshots,
    verdict_snapshot_detail_response,
    verdict_snapshot_list_response,
)
from qllm.dashboard.runner import ExperimentQueue
from qllm.dashboard.lab import comparison_research_payload
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


def _complete_pair(db, pair, *, candidate_ppl=1.0, baseline_ppl=2.0):
    """Persist the minimum canonical evidence required by reconciliation."""
    for job, ppl in ((pair, candidate_ppl), (pair["comparison_job"], baseline_ppl)):
        db.update_lab_job(
            job["id"],
            status="done",
            run_key=(
                f"lab/{job['preset_id']}/{job['dataset_name']}/"
                f"{job['seed']}/{job['steps']}"
            ),
        )
        db.record(
            "lab", job["preset_id"], "default-text", job["seed"], 2,
            10, ppl / 2, ppl, ppl / 3, 1.0,
            config=job["config"], run_uuid=job["run_uuid"],
            experiment_uuid=job["experiment_uuid"],
        )


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


def test_verdict_list_reconciles_completed_pairs_without_comparison_read(tmp_path):
    db_path = tmp_path / "results.db"
    queue = ExperimentQueue(str(db_path), start_worker=False)
    candidate = queue.submit(
        "quantum-ffn-4q", "default-text", "completed", 5, 2, 1,
        queue_classical_comparison=True,
        claim_id=CLAIM_ID,
    )
    baseline = candidate["comparison_job"]
    incomplete = queue.submit(
        "quantum-ffn-4q", "default-text", "incomplete", 6, 2, 1,
        queue_classical_comparison=True,
        claim_id=CLAIM_ID,
    )
    db = ResultsDB(db_path)
    for job, ppl in ((candidate, 1.0), (baseline, 2.0)):
        db.update_lab_job(job["id"], status="done")
        db.record(
            "lab", job["preset_id"], "default-text", 5, 2,
            10, ppl / 2, ppl, ppl / 3, 1.0,
            config=job["config"],
            run_uuid=job["run_uuid"],
            experiment_uuid=job["experiment_uuid"],
        )

    first = verdict_snapshot_list_response(db)
    second = verdict_snapshot_list_response(db)

    assert incomplete["comparison_job"]["id"] != baseline["id"]
    assert [snapshot.source_id for snapshot in first.snapshots] == [str(candidate["id"])]
    assert [snapshot.id for snapshot in second.snapshots] == [first.snapshots[0].id]
    assert len(db.get_verdict_snapshot_history(first.snapshots[0].verdict_key)) == 1


def test_verdict_reconciliation_advances_through_pending_pairs(tmp_path):
    db_path = tmp_path / "results.db"
    queue = ExperimentQueue(str(db_path), start_worker=False)
    pairs = [
        queue.submit(
            "quantum-ffn-4q", "default-text", f"pair-{seed}", seed, 2, 1,
            queue_classical_comparison=True,
            claim_id=CLAIM_ID,
        )
        for seed in (7, 8)
    ]
    db = ResultsDB(db_path)
    for pair in pairs:
        for job, ppl in ((pair, 1.0), (pair["comparison_job"], 2.0)):
            db.update_lab_job(job["id"], status="done")
            db.record(
                "lab", job["preset_id"], "default-text", job["seed"], 2,
                10, ppl / 2, ppl, ppl / 3, 1.0,
                config=job["config"],
                run_uuid=job["run_uuid"],
                experiment_uuid=job["experiment_uuid"],
            )

    reconcile_comparison_verdict_snapshots(db, job_limit=1)
    first = db.list_latest_verdict_snapshots()
    reconcile_comparison_verdict_snapshots(db, job_limit=1)
    second = db.list_latest_verdict_snapshots()

    assert [snapshot["source_id"] for snapshot in first] == [str(pairs[0]["id"])]
    assert {snapshot["source_id"] for snapshot in second} == {
        str(pair["id"]) for pair in pairs
    }


def test_reconciliation_cursor_skips_invalid_oldest_pair_and_survives_reopen(tmp_path):
    db_path = tmp_path / "results.db"
    queue = ExperimentQueue(str(db_path), start_worker=False)
    invalid = queue.submit(
        "quantum-ffn-4q", "default-text", "invalid", 11, 2, 1,
        queue_classical_comparison=True, claim_id=CLAIM_ID,
    )
    valid = queue.submit(
        "quantum-ffn-4q", "default-text", "valid", 12, 2, 1,
        queue_classical_comparison=True, claim_id=CLAIM_ID,
    )
    db = ResultsDB(db_path)
    _complete_pair(db, invalid)
    _complete_pair(db, valid)
    invalid_config = dict(invalid["config"])
    invalid_config["research.claim_id"] = "unknown-claim"
    db.update_lab_job(invalid["id"], config=invalid_config)

    reconcile_comparison_verdict_snapshots(db, job_limit=1)
    assert db.list_latest_verdict_snapshots() == []
    assert db.get_reconciliation_cursor("comparison_verdict_snapshots") == invalid["id"]

    reopened = ResultsDB(db_path)
    reconcile_comparison_verdict_snapshots(reopened, job_limit=1)
    assert [row["source_id"] for row in reopened.list_latest_verdict_snapshots()] == [
        str(valid["id"])
    ]


def test_reconciliation_cursor_wrap_retries_an_earlier_corrected_pair(tmp_path):
    db_path = tmp_path / "results.db"
    queue = ExperimentQueue(str(db_path), start_worker=False)
    pair = queue.submit(
        "quantum-ffn-4q", "default-text", "retry", 13, 2, 1,
        queue_classical_comparison=True, claim_id=CLAIM_ID,
    )
    db = ResultsDB(db_path)
    _complete_pair(db, pair)
    invalid_config = dict(pair["config"])
    invalid_config["research.claim_id"] = "unknown-claim"
    db.update_lab_job(pair["id"], config=invalid_config)

    reconcile_comparison_verdict_snapshots(db, job_limit=1)
    reconcile_comparison_verdict_snapshots(db, job_limit=1)
    assert db.get_reconciliation_cursor("comparison_verdict_snapshots") == 0

    db.update_lab_job(pair["id"], config=pair["config"])
    reconcile_comparison_verdict_snapshots(db, job_limit=1)
    assert [row["source_id"] for row in db.list_latest_verdict_snapshots()] == [
        str(pair["id"])
    ]


def test_reconciliation_uses_metadata_only_but_comparison_defaults_to_curves(tmp_path, monkeypatch):
    db_path = tmp_path / "results.db"
    queue = ExperimentQueue(str(db_path), start_worker=False)
    pair = queue.submit(
        "quantum-ffn-4q", "default-text", "curves", 14, 2, 1,
        queue_classical_comparison=True, claim_id=CLAIM_ID,
    )
    db = ResultsDB(db_path)
    _complete_pair(db, pair)
    candidate_key = (
        f"lab/{pair['preset_id']}/{pair['dataset_name']}/"
        f"{pair['seed']}/{pair['steps']}"
    )
    db.log_step(candidate_key, 1, {"loss": 1.5}, run_uuid=pair["run_uuid"])

    default_payload = comparison_research_payload(db, pair["id"])
    assert default_payload["candidate"]["curve"]["loss"] == [
        {"step": 1, "value": 1.5}
    ]

    # Use a new pair because the default comparison intentionally persisted the
    # first pair's snapshot, excluding it from reconciliation selection.
    metadata_pair = queue.submit(
        "quantum-ffn-4q", "default-text", "metadata", 15, 2, 1,
        queue_classical_comparison=True, claim_id=CLAIM_ID,
    )
    _complete_pair(db, metadata_pair)
    monkeypatch.setattr(
        db, "fetch_steps", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("curves fetched"))
    )

    reconcile_comparison_verdict_snapshots(db, job_limit=25)
    assert {row["source_id"] for row in db.list_latest_verdict_snapshots()} == {
        str(pair["id"]), str(metadata_pair["id"])
    }


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


@pytest.mark.parametrize(
    "forbidden_payload",
    [
        {"nested": {"AdVaNtAgE_sCoRe": 1}},
        {"items": [{"composite_advantage_score": 1}]},
        {"composite_score": 1},
    ],
)
def test_resultsdb_rejects_nested_forbidden_score_keys_before_persisting(
    tmp_path, forbidden_payload
):
    db = ResultsDB(tmp_path / "results.db")
    snapshot = build_verdict_snapshot(_payload(diagnostics={"kernel_gap": 0.4}))
    snapshot["diagnostics"] = forbidden_payload
    with pytest.raises(ValueError, match="forbidden key"):
        db.append_verdict_snapshot(snapshot)
    assert db.list_latest_verdict_snapshots() == []
