"""Primary-metric persistence stays registry-driven and UUID-safe."""
from __future__ import annotations

import sqlite3
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from qllm.resultsdb import ResultsDB


def _alternate_registry(monkeypatch):
    def resolve(metric_type, *, require_pairable=False):
        if metric_type == "ground_state_energy_error":
            return {"extraction_key": "energy_error", "pairable": True}
        if metric_type == "strict_autoregressive_next_token":
            return {"extraction_key": "val_ppl", "pairable": True}
        return None

    monkeypatch.setattr("qllm.resultsdb.registry.metric_type_spec", resolve)


def _record(db, **kwargs):
    values = dict(
        suite="suite", variant="variant", dataset="data", seed=1, steps=2,
        n_params=3, val_loss=1.0, val_ppl=2.0, val_bpc=1.4, wall_seconds=5.0,
    )
    values.update(kwargs)
    if "manifest" not in values:
        values["manifest"] = {
            "run_uuid": str(uuid.uuid4()),
            "experiment_uuid": str(uuid.uuid4()),
            "manifest_hash": str(uuid.uuid4()),
        }
    # These persistence tests do not exercise artifact manifest validation;
    # avoid importing optional training dependencies into the SQLite contract.
    db.register_run_manifest = lambda manifest: None
    return db.record(**values)


def test_schema_migrates_backfills_repeatably_and_guards_minimal_sources(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as con:
        con.executescript(
            """
            CREATE TABLE runs (id INTEGER PRIMARY KEY, val_ppl REAL);
            CREATE TABLE run_results (run_uuid TEXT PRIMARY KEY, val_ppl REAL);
            CREATE TABLE live_runs (run_key TEXT PRIMARY KEY, last_val_ppl REAL);
            INSERT INTO runs VALUES (1, 2.5);
            INSERT INTO run_results VALUES ('run', 3.5);
            INSERT INTO live_runs VALUES ('key', 4.5);
            """
        )
    ResultsDB(path)
    with sqlite3.connect(path) as con:
        con.execute(
            "INSERT INTO runs(id, val_ppl, primary_metric_name, "
            "primary_metric_value) VALUES (2, 9.5, 'energy_error', NULL)"
        )
        con.execute(
            "INSERT INTO runs(id, val_ppl, primary_metric_name, "
            "primary_metric_value) VALUES (3, 8.5, NULL, 0.125)"
        )
    ResultsDB(path)
    with sqlite3.connect(path) as con:
        assert con.execute(
            "SELECT primary_metric_name, primary_metric_value FROM runs "
            "WHERE id=1"
        ).fetchone() == ("val_ppl", 2.5)
        assert con.execute(
            "SELECT primary_metric_name, primary_metric_value FROM runs "
            "WHERE id=2"
        ).fetchone() == ("energy_error", None)
        assert con.execute(
            "SELECT primary_metric_name, primary_metric_value FROM runs "
            "WHERE id=3"
        ).fetchone() == (None, 0.125)
        assert con.execute("SELECT primary_metric_name, primary_metric_value FROM run_results").fetchone() == ("val_ppl", 3.5)
        assert con.execute("SELECT primary_metric_name, last_primary_metric_value FROM live_runs").fetchone() == ("val_ppl", 4.5)

    minimal = tmp_path / "minimal.db"
    with sqlite3.connect(minimal) as con:
        con.executescript(
            """
            CREATE TABLE runs (id INTEGER PRIMARY KEY);
            CREATE TABLE run_results (run_uuid TEXT PRIMARY KEY);
            CREATE TABLE live_runs (run_key TEXT PRIMARY KEY);
            """
        )
    ResultsDB(minimal)
    with sqlite3.connect(minimal) as con:
        assert "primary_metric_value" in {r[1] for r in con.execute("PRAGMA table_info(runs)")}


def test_dynamic_primary_projection_rejects_specialized_conflicts():
    assert ResultsDB.decode_result_row(
        {
            "primary_metric_name": "energy_error",
            "primary_metric_value": 0.125,
        }
    )["energy_error"] == pytest.approx(0.125)
    with pytest.raises(ValueError, match="conflicts with its specialized"):
        ResultsDB.decode_result_row(
            {
                "val_ppl": 2.0,
                "primary_metric_name": "val_ppl",
                "primary_metric_value": 3.0,
            }
        )


def test_record_default_alt_dynamic_and_rejections_precede_mutation(tmp_path, monkeypatch):
    db = ResultsDB(tmp_path / "results.db")
    recorded = _record(db)
    assert recorded["primary_metric_name"] == "val_ppl"
    assert db.get_run("suite", "variant", "data", 1, 2)["primary_metric_value"] == pytest.approx(2.0)

    _alternate_registry(monkeypatch)
    alternate = _record(
        db, variant="energy", val_loss=None, val_ppl=None, val_bpc=None,
        primary_metric_type="ground_state_energy_error",
        metric_values={"energy_error": 0.125},
    )
    row = db.get_run("suite", "energy", "data", 1, 2, run_uuid=alternate["run_uuid"])
    assert row["primary_metric_name"] == "energy_error"
    assert row["energy_error"] == pytest.approx(0.125)

    before = (len(db.fetch("suite")), db.get_run_manifest(str(uuid.uuid4())))
    with pytest.raises(ValueError, match="Unknown primary metric"):
        _record(db, variant="unknown", primary_metric_type="unknown")
    with pytest.raises(ValueError, match="Missing primary metric"):
        _record(db, variant="missing", val_ppl=None)
    assert (len(db.fetch("suite")), db.get_run_manifest(str(uuid.uuid4()))) == before
    with pytest.raises(ValueError, match="Conflicting val_ppl"):
        _record(db, variant="conflict", metric_values={"val_ppl": 9.0})


def test_record_uuid_primary_evidence_is_immutable(tmp_path, monkeypatch):
    db = ResultsDB(tmp_path / "uuid-results.db")
    identity = _record(db)
    _record(db, manifest=identity["manifest"])
    with pytest.raises(ValueError, match="Conflicting final result evidence"):
        _record(db, val_ppl=3.0, manifest=identity["manifest"])
    _alternate_registry(monkeypatch)
    with pytest.raises(ValueError, match="Conflicting final result evidence"):
        _record(
            db, val_loss=None, val_ppl=None, val_bpc=None,
            primary_metric_type="ground_state_energy_error",
            metric_values={"energy_error": 0.1}, manifest=identity["manifest"],
        )


def test_live_primary_metric_monotonicity_uuid_name_and_recovery(tmp_path, monkeypatch):
    _alternate_registry(monkeypatch)
    db = ResultsDB(tmp_path / "live.db")
    key, run_a, run_b = "suite/energy/data/1/8", str(uuid.uuid4()), str(uuid.uuid4())
    db.start_run(key, "energy", "suite", "energy", "data", 1, 8,
                 run_uuid=run_a, primary_metric_type="ground_state_energy_error")
    db.log_step(
        key, 5, {"energy_error": 0.5}, train_loss=1.0, val_ppl=2.0,
        run_uuid=run_a,
    )
    db.log_step(key, 2, {"energy_error": 9.0}, run_uuid=run_a)
    live = db.fetch_live_runs()[0]
    assert live["current_step"] == 5
    assert live["last_primary_metric_value"] == pytest.approx(0.5)
    db.start_run(key, "energy", "suite", "energy", "data", 1, 8,
                 run_uuid=run_a, primary_metric_type="ground_state_energy_error")
    assert db.fetch_live_runs()[0]["current_step"] == 5
    with db._conn() as con:
        con.execute(
            "UPDATE live_runs SET primary_metric_name=NULL WHERE run_uuid=?",
            (run_a,),
        )
    db.start_run(key, "energy", "suite", "energy", "data", 1, 8,
                 run_uuid=run_a, primary_metric_type="ground_state_energy_error")
    assert db.fetch_live_runs()[0]["primary_metric_name"] == "energy_error"
    with pytest.raises(ValueError, match="cannot switch"):
        db.start_run(key, "energy", "suite", "energy", "data", 1, 8, run_uuid=run_a)

    db.start_run(key, "energy", "suite", "energy", "data", 1, 8,
                 run_uuid=run_b, primary_metric_type="ground_state_energy_error")
    live = db.fetch_live_runs()[0]
    assert live["current_step"] == 0
    assert live["last_train_loss"] is None
    assert live["last_val_ppl"] is None
    assert live["last_primary_metric_value"] is None
    db.log_step(key, 5, {"energy_error": 0.5}, run_uuid=run_b)
    job_id = db.create_lab_job(
        {
            "preset_id": "energy", "dataset_name": "data", "run_name": "energy",
            "seed": 1, "steps": 8, "eval_every": 1, "run_uuid": run_b,
        }
    )
    with db._conn() as con:
        ResultsDB._reconcile_job_run_recovery(
            con, job_id, status="queued", updated_ts="now", completed_step=1,
            checkpoint_path=None,
        )
    assert db.fetch_live_runs()[0]["last_primary_metric_value"] is None


def test_concurrent_same_uuid_primary_declarations_are_atomic(
    tmp_path, monkeypatch
):
    _alternate_registry(monkeypatch)
    path = tmp_path / "primary-race.db"
    ResultsDB(path)
    barrier = threading.Barrier(2)
    run_uuid = str(uuid.uuid4())
    key = "suite/race/data/0/1"

    def declare(metric_type: str) -> tuple[str, str]:
        store = ResultsDB(path)
        barrier.wait(timeout=10)
        try:
            store.start_run(
                key,
                "race",
                "suite",
                "race",
                "data",
                0,
                1,
                run_uuid=run_uuid,
                primary_metric_type=metric_type,
            )
        except ValueError as exc:
            return "rejected", str(exc)
        return "accepted", metric_type

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(
            pool.map(
                declare,
                (
                    "strict_autoregressive_next_token",
                    "ground_state_energy_error",
                ),
            )
        )

    assert [status for status, _detail in outcomes].count("accepted") == 1
    assert [status for status, _detail in outcomes].count("rejected") == 1
    assert "cannot switch" in next(
        detail for status, detail in outcomes if status == "rejected"
    )
