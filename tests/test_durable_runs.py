from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import serialization

from qllm.config import ModelConfig, to_flat_dict
from qllm.data.text import CharTokenizer
from qllm.dashboard.datasets import get_dataset
from qllm.dashboard.model_tests import model_test_payload, run_model_test
from qllm.dashboard.model_specs import create_spec, update_spec
from qllm.dashboard.presets import build_preset
from qllm.dashboard.runner import ExperimentQueue, _lease_heartbeat_loop
from qllm.models.model import build_model
from qllm.resultsdb import ResultsDB
from qllm.train.artifacts import (
    RunOptions,
    atomic_write_bytes,
    build_record_manifest,
    build_run_manifest,
    code_identity,
    read_checkpoint,
    write_checkpoint,
    write_immutable_manifest,
)
from qllm.train.loop import fit, generate_outcome


def _job(**overrides):
    payload = {
        "preset_id": "classical-small",
        "dataset_name": "default-text",
        "run_name": "durable-test",
        "seed": 0,
        "steps": 4,
        "eval_every": 2,
        "config": {
            "lab.gpu_reservation.state": "queued",
            "lab.gpu_reservation.job_id": 99,
        },
    }
    payload.update(overrides)
    return payload


def test_immutable_manifest_rejects_changed_identity(tmp_path):
    target = tmp_path / "manifest.json"
    first = build_record_manifest(
        suite="test",
        variant="one",
        dataset="data",
        seed=0,
        steps=1,
    )
    write_immutable_manifest(target, first)
    write_immutable_manifest(target, first)
    second = build_record_manifest(
        suite="test",
        variant="two",
        dataset="data",
        seed=0,
        steps=1,
    )
    with pytest.raises(ValueError, match="Immutable"):
        write_immutable_manifest(target, second)


def test_checkpoint_rejects_tampered_embedded_manifest(
    tiny_classical_cfg, tmp_path
):
    result = fit(
        tiny_classical_cfg,
        verbose=False,
        run_options=RunOptions(artifact_dir=tmp_path / "tamper-source"),
    )
    checkpoint = Path(result["summary"]["checkpoint_path"])
    payload = serialization.msgpack_restore(checkpoint.read_bytes())
    manifest = json.loads(payload["manifest_json"])
    manifest["config"]["train"]["lr"] = 999.0
    payload["manifest_json"] = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    )
    checksum_body = dict(payload)
    checksum_body.pop("payload_sha256", None)
    payload["payload_sha256"] = hashlib.sha256(
        serialization.msgpack_serialize(checksum_body)
    ).hexdigest()
    tampered = tmp_path / "tampered.msgpack"
    tampered.write_bytes(serialization.msgpack_serialize(payload))
    with pytest.raises(ValueError, match="manifest_hash"):
        read_checkpoint(tampered)


def test_code_identity_excludes_artifacts_tracks_source_and_reports_unavailable(
    monkeypatch, tmp_path
):
    class Result:
        def __init__(self, stdout=b""):
            self.stdout = stdout

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if "rev-parse" in command:
            return Result("abc123\n")
        if "ls-files" in command:
            return Result("")
        return Result(b"tracked diff")

    monkeypatch.setattr("qllm.train.artifacts.subprocess.run", fake_run)
    identity = code_identity()
    assert identity["status"] == "dirty"
    assert identity["commit"] == "abc123"
    assert all("status" not in command for command in calls)

    source = tmp_path / "qllm" / "new_runtime.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n")

    def untracked_source(command, **kwargs):
        if "rev-parse" in command:
            return Result("abc123\n")
        if "ls-files" in command:
            return Result("qllm/new_runtime.py\nresults/run/summary.json\n")
        return Result(b"")

    monkeypatch.setattr("qllm.train.artifacts.subprocess.run", untracked_source)
    source_identity = code_identity(tmp_path)
    assert source_identity["status"] == "dirty"
    assert source_identity["untracked_source_count"] == 1

    def unavailable(*args, **kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr("qllm.train.artifacts.subprocess.run", unavailable)
    assert code_identity()["status"] == "unavailable"


def test_data_hash_ignores_cache_access_provenance_but_records_it(
    tiny_classical_cfg,
):
    from types import SimpleNamespace

    common = {
        "config_hash": "a" * 64,
        "content_hash": "b" * 64,
        "shape": (2, 4),
        "sampler_policy": "within_trajectory",
    }
    generated = SimpleNamespace(
        **common,
        provenance={"source": "generated", "cache_path": "one.npz"},
        metadata={"provenance": {"source": "generated"}},
    )
    cached = SimpleNamespace(
        **common,
        provenance={"source": "cache", "cache_path": "one.npz"},
        metadata={"provenance": {"source": "cache"}},
    )
    options = RunOptions(
        experiment_uuid=str(uuid.uuid4()), run_uuid=str(uuid.uuid4())
    )
    left = build_run_manifest(
        tiny_classical_cfg, generated, options, run_name="stable", seed_axes={}
    )
    right = build_run_manifest(
        tiny_classical_cfg, cached, options, run_name="stable", seed_axes={}
    )
    assert left["data_hash"] == right["data_hash"]
    assert left["data"]["provenance"] != right["data"]["provenance"]


def test_manifest_only_retry_reuses_generated_manifest_after_cache_transition(
    tiny_classical_cfg, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    cfg = dataclasses.replace(
        tiny_classical_cfg,
        train=dataclasses.replace(
            tiny_classical_cfg.train,
            steps=1,
            eval_every=1,
            eval_batches=1,
        ),
        data=dataclasses.replace(
            tiny_classical_cfg.data,
            kind="seq_cancellation",
            gen_sequences=4,
            gen_len=32,
            ctx_observables=8,
            ctx_context_size=3,
            gen_seed=12345,
        ),
    )
    artifact_dir = tmp_path / "generated-cache-retry"
    options = RunOptions(
        experiment_uuid=str(uuid.uuid4()),
        run_uuid=str(uuid.uuid4()),
        artifact_dir=artifact_dir,
    )

    from qllm.train import loop as train_loop

    real_write_checkpoint = train_loop.write_checkpoint

    def crash_before_checkpoint(*args, **kwargs):
        raise RuntimeError("simulated generated-data checkpoint crash")

    monkeypatch.setattr(train_loop, "write_checkpoint", crash_before_checkpoint)
    with pytest.raises(RuntimeError, match="generated-data checkpoint crash"):
        fit(cfg, verbose=False, run_options=options)

    first_manifest = json.loads(
        (artifact_dir / "manifest.json").read_text("utf-8")
    )
    assert first_manifest["data"]["provenance"]["source"] == "generated"

    monkeypatch.setattr(train_loop, "write_checkpoint", real_write_checkpoint)
    retried = fit(cfg, verbose=False, run_options=options)

    assert retried["dataset"].provenance["source"] == "cache"
    assert retried["manifest"] == first_manifest
    assert retried["manifest"]["data_hash"] == first_manifest["data_hash"]


def test_exact_checkpoint_resume_restores_state_rng_and_history(
    tiny_classical_cfg, tmp_path
):
    cfg = dataclasses.replace(
        tiny_classical_cfg,
        train=dataclasses.replace(
            tiny_classical_cfg.train, steps=4, eval_every=2, eval_batches=1
        ),
    )
    interrupted_dir = tmp_path / "interrupted"
    stop = {"value": False}

    def progress(payload):
        if payload["completed_step"] >= 2:
            stop["value"] = True

    first = fit(
        cfg,
        verbose=False,
        should_cancel=lambda: stop["value"],
        run_options=RunOptions(
            experiment_uuid=str(uuid.uuid4()),
            run_uuid=str(uuid.uuid4()),
            artifact_dir=interrupted_dir,
            checkpoint_every=1,
        ),
        progress_callback=progress,
    )
    assert first["summary"]["completed_step"] == 2
    assert first["summary"]["cancelled"] is True
    latest = interrupted_dir / "checkpoints" / "latest.msgpack"
    assert latest.exists()

    resumed = fit(
        cfg,
        verbose=False,
        run_options=RunOptions(resume_from=latest, checkpoint_every=1),
    )
    uninterrupted = fit(
        cfg,
        verbose=False,
        run_options=RunOptions(
            artifact_dir=tmp_path / "uninterrupted",
            experiment_uuid=str(uuid.uuid4()),
            run_uuid=str(uuid.uuid4()),
            checkpoint_every=1,
        ),
    )
    assert resumed["summary"]["completed_step"] == 4
    assert resumed["summary"]["run_uuid"] == first["summary"]["run_uuid"]
    assert resumed["summary"]["history"] == uninterrupted["summary"]["history"]
    for left, right in zip(
        jax.tree_util.tree_leaves(resumed["state"]),
        jax.tree_util.tree_leaves(uninterrupted["state"]),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    payload = read_checkpoint(latest)
    assert payload["completed_step"] == 4
    assert payload["resume_lineage"]["source_checkpoint_sha256"]
    assert (interrupted_dir / "checkpoints" / "best.msgpack").exists()
    full_payload = read_checkpoint(
        tmp_path / "uninterrupted" / "checkpoints" / "latest.msgpack"
    )
    resumed_rng = np.random.default_rng()
    resumed_rng.bit_generator.state = payload["rng_state"]
    full_rng = np.random.default_rng()
    full_rng.bit_generator.state = full_payload["rng_state"]
    np.testing.assert_array_equal(
        resumed_rng.integers(0, 2**31, size=16),
        full_rng.integers(0, 2**31, size=16),
    )


def test_warm_start_manifest_is_honest_and_its_checkpoint_resumes(
    tiny_classical_cfg, tmp_path
):
    donor = fit(
        tiny_classical_cfg,
        verbose=False,
        run_options=RunOptions(artifact_dir=tmp_path / "warm-donor"),
    )
    warm = fit(
        tiny_classical_cfg,
        verbose=False,
        init_params=donor["state"].params,
        run_options=RunOptions(
            artifact_dir=tmp_path / "warm-run",
            caller_metadata={"warm_start_source": "test-donor"},
        ),
    )
    assert warm["manifest"]["initialization"]["mode"] == "warm_start"
    assert warm["manifest"]["initialization"]["source"] == "test-donor"
    assert warm["manifest"]["initialization"]["parameters_sha256"]
    assert warm["manifest"]["seed_axes"]["initialization"] is None
    assert warm["manifest"]["seed_axes"]["minibatch"] == tiny_classical_cfg.train.seed
    resumed = fit(
        tiny_classical_cfg,
        verbose=False,
        run_options=RunOptions(
            resume_from=warm["summary"]["checkpoint_path"]
        ),
    )
    assert resumed["summary"]["run_uuid"] == warm["summary"]["run_uuid"]
    assert resumed["manifest"]["initialization"] == warm["manifest"][
        "initialization"
    ]


def test_same_run_resume_rejects_older_checkpoint_when_latest_is_newer(
    tiny_classical_cfg, tmp_path
):
    cfg = dataclasses.replace(
        tiny_classical_cfg,
        train=dataclasses.replace(
            tiny_classical_cfg.train, steps=4, eval_every=2, eval_batches=1
        ),
    )
    artifact_dir = tmp_path / "rollback-guard"
    older = artifact_dir / "checkpoints" / "step-2-copy.msgpack"

    def capture_step_two(progress):
        if progress["completed_step"] == 2 and not older.exists():
            source = Path(progress["checkpoint_path"])
            payload = read_checkpoint(source)
            if payload["completed_step"] == 2:
                older.write_bytes(source.read_bytes())

    trained = fit(
        cfg,
        verbose=False,
        run_options=RunOptions(
            artifact_dir=artifact_dir,
            checkpoint_every=2,
        ),
        progress_callback=capture_step_two,
    )
    latest = Path(trained["summary"]["checkpoint_path"])
    assert read_checkpoint(older)["completed_step"] == 2
    assert read_checkpoint(latest)["completed_step"] == 4
    latest_before = latest.read_bytes()

    with pytest.raises(ValueError, match="Refusing to roll back"):
        fit(
            cfg,
            verbose=False,
            run_options=RunOptions(resume_from=older),
        )

    assert latest.read_bytes() == latest_before
    assert read_checkpoint(latest)["completed_step"] == 4


def test_fork_reuses_child_manifest_after_pre_checkpoint_crash(
    tiny_classical_cfg, tmp_path, monkeypatch
):
    source = fit(
        tiny_classical_cfg,
        verbose=False,
        run_options=RunOptions(artifact_dir=tmp_path / "fork-crash-source"),
    )
    source_checkpoint = source["summary"]["checkpoint_path"]
    child_dir = tmp_path / "fork-crash-child"
    child_run_uuid = str(uuid.uuid4())
    child_options = RunOptions(
        resume_from=source_checkpoint,
        run_uuid=child_run_uuid,
        artifact_dir=child_dir,
    )

    from qllm.train import loop as train_loop

    real_write_checkpoint = train_loop.write_checkpoint

    def crash_before_checkpoint(*args, **kwargs):
        raise RuntimeError("simulated child crash before first checkpoint")

    monkeypatch.setattr(train_loop, "write_checkpoint", crash_before_checkpoint)
    with pytest.raises(RuntimeError, match="simulated child crash"):
        fit(
            tiny_classical_cfg,
            verbose=False,
            run_options=child_options,
        )

    child_manifest_path = child_dir / "manifest.json"
    child_manifest = json.loads(child_manifest_path.read_text("utf-8"))
    assert child_manifest["run_uuid"] == child_run_uuid
    assert child_manifest["resume_lineage"]["mode"] == "fork"
    assert not (child_dir / "checkpoints" / "latest.msgpack").exists()

    monkeypatch.setattr(train_loop, "write_checkpoint", real_write_checkpoint)
    retried = fit(
        tiny_classical_cfg,
        verbose=False,
        run_options=child_options,
    )

    assert retried["manifest"] == child_manifest
    assert retried["summary"]["run_uuid"] == child_run_uuid
    assert Path(retried["summary"]["checkpoint_path"]).exists()


def test_resume_rejects_config_data_and_environment_identity_mismatches(
    tiny_classical_cfg, tmp_path, monkeypatch
):
    source_dir = tmp_path / "source"
    source = fit(
        tiny_classical_cfg,
        verbose=False,
        run_options=RunOptions(artifact_dir=source_dir, checkpoint_every=1),
    )
    latest = source["summary"]["checkpoint_path"]
    changed_config = dataclasses.replace(
        tiny_classical_cfg,
        train=dataclasses.replace(
            tiny_classical_cfg.train, lr=tiny_classical_cfg.train.lr * 2
        ),
    )
    with pytest.raises(ValueError, match="resume_compatibility_hash"):
        fit(
            changed_config,
            verbose=False,
            run_options=RunOptions(resume_from=latest),
        )

    operational_tracking = dataclasses.replace(
        tiny_classical_cfg,
        tracking=dataclasses.replace(
            tiny_classical_cfg.tracking,
            experiment="different-tracker",
            tracking_uri="sqlite:///different.db",
            log_quantum_diagnostics=(
                not tiny_classical_cfg.tracking.log_quantum_diagnostics
            ),
        ),
    )
    continued = fit(
        operational_tracking,
        verbose=False,
        run_options=RunOptions(resume_from=latest),
    )
    assert continued["summary"]["run_uuid"] == source["summary"]["run_uuid"]

    from qllm.train import artifacts

    real_environment_identity = artifacts.environment_identity

    def changed_environment():
        identity = real_environment_identity()
        return {**identity, "hash": "f" * 64}

    monkeypatch.setattr(artifacts, "environment_identity", changed_environment)
    with pytest.raises(ValueError, match="environment_hash"):
        fit(
            tiny_classical_cfg,
            verbose=False,
            run_options=RunOptions(resume_from=latest),
        )
    monkeypatch.setattr(artifacts, "environment_identity", real_environment_identity)

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abc " * 200)
    corpus_cfg = dataclasses.replace(
        tiny_classical_cfg,
        data=dataclasses.replace(tiny_classical_cfg.data, corpus_path=str(corpus)),
    )
    data_dir = tmp_path / "data-source"
    data_source = fit(
        corpus_cfg,
        verbose=False,
        run_options=RunOptions(artifact_dir=data_dir),
    )
    corpus.write_text("cba " * 200)
    with pytest.raises(ValueError, match="data_hash"):
        fit(
            corpus_cfg,
            verbose=False,
            run_options=RunOptions(
                resume_from=data_source["summary"]["checkpoint_path"]
            ),
        )


def test_failed_atomic_replace_leaves_prior_checkpoint_readable(
    tiny_classical_cfg, tmp_path, monkeypatch
):
    result = fit(
        tiny_classical_cfg,
        verbose=False,
        run_options=RunOptions(artifact_dir=tmp_path / "atomic"),
    )
    latest = Path(result["summary"]["checkpoint_path"])
    before = read_checkpoint(latest)

    nonfinite = tmp_path / "nonfinite-history.msgpack"
    rng = np.random.default_rng(0)
    write_checkpoint(
        nonfinite,
        result["state"],
        completed_step=int(np.asarray(result["state"].step)),
        rng_state=rng.bit_generator.state,
        history=[
            {
                "step": 1,
                "val_loss": float("nan"),
                "grad_norm": float("inf"),
            }
        ],
        best_metric=float("nan"),
        best_step=1,
        manifest=result["manifest"],
        resume_lineage={},
    )
    restored = read_checkpoint(nonfinite)
    assert math.isnan(restored["history"][0]["val_loss"])
    assert math.isinf(restored["history"][0]["grad_norm"])

    def fail_replace(*args, **kwargs):
        raise OSError("simulated replace failure")

    monkeypatch.setattr("qllm.train.artifacts.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        atomic_write_bytes(latest, b"incomplete replacement")
    after = read_checkpoint(latest)
    assert after["completed_step"] == before["completed_step"]
    assert after["manifest"]["manifest_hash"] == before["manifest"]["manifest_hash"]


def test_corrupt_checkpoint_and_nonempty_fresh_artifact_are_rejected(
    tiny_classical_cfg, tmp_path
):
    corrupt = tmp_path / "corrupt.msgpack"
    corrupt.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="Invalid checkpoint"):
        read_checkpoint(corrupt)

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "legacy.txt").write_text("preserve me")
    with pytest.raises(ValueError, match="non-empty artifact"):
        fit(
            tiny_classical_cfg,
            verbose=False,
            run_options=RunOptions(artifact_dir=occupied),
        )
    assert (occupied / "legacy.txt").read_text() == "preserve me"

    source = fit(
        tiny_classical_cfg,
        verbose=False,
        run_options=RunOptions(artifact_dir=tmp_path / "fork-source"),
    )
    fork_dir = tmp_path / "occupied-fork"
    fork_dir.mkdir()
    (fork_dir / "legacy.txt").write_text("keep")
    with pytest.raises(ValueError, match="non-empty artifact"):
        fit(
            tiny_classical_cfg,
            verbose=False,
            run_options=RunOptions(
                resume_from=source["summary"]["checkpoint_path"],
                run_uuid=str(uuid.uuid4()),
                artifact_dir=fork_dir,
            ),
        )
    assert (fork_dir / "legacy.txt").read_text() == "keep"


def test_default_fit_artifacts_are_uuid_scoped_across_same_name_reruns(
    tiny_classical_cfg, tmp_path
):
    first = fit(tiny_classical_cfg, verbose=False, out_dir=tmp_path)
    second = fit(tiny_classical_cfg, verbose=False, out_dir=tmp_path)

    first_dir = Path(first["summary"]["artifact_dir"])
    second_dir = Path(second["summary"]["artifact_dir"])
    assert first["summary"]["run_uuid"] != second["summary"]["run_uuid"]
    assert first_dir != second_dir
    assert first_dir == tmp_path / "runs" / first["summary"]["run_uuid"]
    assert second_dir == tmp_path / "runs" / second["summary"]["run_uuid"]
    assert (first_dir / "summary.json").exists()
    assert (second_dir / "summary.json").exists()

    result_db = ResultsDB(tmp_path / "fit-result.db")
    summary = first["summary"]
    result_db.record(
        "fit",
        "classical",
        "synthetic",
        tiny_classical_cfg.train.seed,
        tiny_classical_cfg.train.steps,
        summary["n_params"],
        summary["val_loss"],
        summary["val_ppl"],
        summary["val_bpc"],
        summary["wall_seconds"],
        manifest=first["manifest"],
    )
    recorded_manifest = result_db.get_run_manifest(summary["run_uuid"])
    assert recorded_manifest["status"] == "done"
    assert recorded_manifest["completed_step"] == tiny_classical_cfg.train.steps


def test_additive_step_migration_preserves_legacy_and_isolates_run_uuids(tmp_path):
    path = tmp_path / "results.db"
    db = ResultsDB(path)
    with db._conn() as con:
        con.executemany(
            "INSERT INTO steps(run_key, step, name, value) VALUES (?,?,?,?)",
            [
                ("same", 2, "loss", 4.0),
                ("same", 1, "loss", 3.0),
                ("same", 1, "loss", 2.0),
            ],
        )
    ResultsDB(path)
    with db._conn() as con:
        assert con.execute("SELECT COUNT(*) FROM steps").fetchone()[0] == 3
    assert db.fetch_steps("same") == [
        {"step": 1, "name": "loss", "value": 3.0},
        {"step": 1, "name": "loss", "value": 2.0},
        {"step": 2, "name": "loss", "value": 4.0},
    ]

    run_a, run_b = str(uuid.uuid4()), str(uuid.uuid4())
    db.log_step("same", 2, {"loss": 1.0}, run_uuid=run_a)
    db.log_step("same", 2, {"loss": 1.0}, run_uuid=run_a)
    db.log_step("same", 3, {"nan_metric": float("nan")}, run_uuid=run_a)
    db.log_step("same", 3, {"nan_metric": float("nan")}, run_uuid=run_a)
    with pytest.raises(ValueError, match="Conflicting retry"):
        db.log_step("same", 2, {"loss": 9.0}, run_uuid=run_a)
    db.log_step("same", 2, {"loss": 4.0}, run_uuid=run_b)
    assert db.fetch_steps("same", run_uuid=run_a) == [
        {"step": 2, "name": "loss", "value": 1.0},
        {"step": 3, "name": "nan_metric", "value": None},
    ]
    assert db.fetch_steps("same", run_uuid=run_b) == [
        {"step": 2, "name": "loss", "value": 4.0}
    ]


@pytest.mark.parametrize("legacy", [False, True])
def test_concurrent_resultsdb_initialization_converges_atomically(tmp_path, legacy):
    path = tmp_path / ("legacy-race.db" if legacy else "fresh-race.db")
    if legacy:
        with sqlite3.connect(path) as con:
            con.executescript(
                """
                CREATE TABLE runs (
                    id INTEGER PRIMARY KEY,
                    suite TEXT NOT NULL
                );
                CREATE TABLE lab_jobs (
                    id INTEGER PRIMARY KEY,
                    status TEXT NOT NULL
                );
                CREATE TABLE lab_datasets (
                    name TEXT PRIMARY KEY
                );
                CREATE TABLE live_runs (
                    run_key TEXT PRIMARY KEY
                );
                CREATE TABLE run_results (
                    run_uuid TEXT PRIMARY KEY
                );
                INSERT INTO runs(id, suite) VALUES (7, 'preserved');
                INSERT INTO lab_jobs(id, status) VALUES (9, 'queued');
                """
            )

    workers = 8
    barrier = threading.Barrier(workers)

    def initialize(_worker):
        barrier.wait(timeout=10)
        ResultsDB(path)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(initialize, range(workers)))

    with sqlite3.connect(path) as con:
        job_columns = {row[1] for row in con.execute("PRAGMA table_info(lab_jobs)")}
        run_columns = {row[1] for row in con.execute("PRAGMA table_info(runs)")}
        indexes = {
            row[1]
            for table in ("lab_jobs", "runs")
            for row in con.execute(f"PRAGMA index_list({table})")
        }
        assert {
            "run_uuid",
            "worker_id",
            "lease_expires_ts",
            "checkpoint_path",
        } <= job_columns
        assert {"experiment_uuid", "run_uuid", "manifest_hash"} <= run_columns
        assert {
            "idx_lab_jobs_run_uuid",
            "idx_runs_run_uuid",
            "idx_lab_jobs_status_id",
            "idx_lab_jobs_status_lease",
        } <= indexes
        if legacy:
            assert con.execute("SELECT id, suite FROM runs").fetchall() == [
                (7, "preserved")
            ]
            assert con.execute("SELECT id, status FROM lab_jobs").fetchall() == [
                (9, "queued")
            ]


def test_concurrent_step_retries_are_idempotent_and_progress_is_monotonic(tmp_path):
    path = tmp_path / "step-race.db"
    db = ResultsDB(path)
    manifest = build_record_manifest(
        suite="race", variant="v", dataset="d", seed=0, steps=5
    )
    run_uuid = manifest["run_uuid"]
    db.start_run(
        "race/v/d/0/5",
        "race",
        "race",
        "v",
        "d",
        0,
        5,
        run_uuid=run_uuid,
        experiment_uuid=manifest["experiment_uuid"],
        manifest=manifest,
    )

    def retry(_):
        ResultsDB(path).log_step(
            "race/v/d/0/5", 5, {"loss": 1.25}, run_uuid=run_uuid
        )

    with ThreadPoolExecutor(max_workers=10) as pool:
        list(pool.map(retry, range(10)))
    assert db.fetch_steps("race/v/d/0/5", run_uuid=run_uuid) == [
        {"step": 5, "name": "loss", "value": 1.25}
    ]
    db.log_step(
        "race/v/d/0/5", 2, {"older": 9.0}, run_uuid=run_uuid
    )
    live = next(
        row for row in db.fetch_live_runs() if row["run_uuid"] == run_uuid
    )
    assert live["current_step"] == 5


def test_uuid_step_logging_repairs_null_legacy_live_progress(tmp_path):
    db = ResultsDB(tmp_path / "legacy-null-progress.db")
    manifest = build_record_manifest(
        suite="legacy", variant="v", dataset="d", seed=0, steps=3
    )
    run_uuid = manifest["run_uuid"]
    db.start_run(
        "legacy/v/d/0/3",
        "legacy-null",
        "legacy",
        "v",
        "d",
        0,
        3,
        run_uuid=run_uuid,
        experiment_uuid=manifest["experiment_uuid"],
        manifest=manifest,
    )
    with db._conn() as con:
        con.execute(
            "UPDATE live_runs SET current_step=NULL WHERE run_uuid=?", (run_uuid,)
        )

    db.log_step(
        "legacy/v/d/0/3",
        1,
        {"train_loss": 2.0},
        train_loss=2.0,
        run_uuid=run_uuid,
    )

    live = next(
        row for row in db.fetch_live_runs() if row["run_uuid"] == run_uuid
    )
    assert live["current_step"] == 1
    assert live["last_train_loss"] == pytest.approx(2.0)


def test_hand_built_pre_m05_schema_migrates_twice_without_rewriting_rows(tmp_path):
    path = tmp_path / "legacy.db"
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE runs (
          id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,
          suite TEXT NOT NULL, variant TEXT NOT NULL, dataset TEXT NOT NULL,
          seed INTEGER NOT NULL, steps INTEGER NOT NULL, n_params INTEGER NOT NULL,
          val_loss REAL, val_ppl REAL, val_bpc REAL, wall_seconds REAL,
          config_json TEXT, UNIQUE(suite, variant, dataset, seed, steps)
        );
        CREATE TABLE steps (
          id INTEGER PRIMARY KEY AUTOINCREMENT, run_key TEXT NOT NULL,
          step INTEGER NOT NULL, name TEXT NOT NULL, value REAL
        );
        CREATE TABLE lab_jobs (
          id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,
          updated_ts TEXT NOT NULL, status TEXT NOT NULL, preset_id TEXT NOT NULL,
          dataset_name TEXT NOT NULL, run_name TEXT NOT NULL, seed INTEGER NOT NULL,
          steps INTEGER NOT NULL, eval_every INTEGER NOT NULL, run_key TEXT,
          error TEXT, config_json TEXT, group_id TEXT, parent_job_id INTEGER,
          compare_to_job_id INTEGER, device_target TEXT DEFAULT 'auto',
          comparison_role TEXT DEFAULT 'primary'
        );
        INSERT INTO runs VALUES
          (7, 'old', 'suite', 'variant', 'data', 1, 2, 3, 1.0, 2.0, 1.4, 5.0, '{"old":true}');
        INSERT INTO steps(run_key, step, name, value) VALUES
          ('legacy/key', 1, 'loss', 4.0), ('legacy/key', 1, 'loss', 3.0);
        INSERT INTO lab_jobs
          (id, ts, updated_ts, status, preset_id, dataset_name, run_name,
           seed, steps, eval_every, config_json)
          VALUES (9, 'old', 'old', 'queued', 'p', 'd', 'r', 1, 2, 1, '{"raw":1}');
        """
    )
    con.commit()
    con.close()

    db = ResultsDB(path)
    ResultsDB(path)
    identity = db.record(
        "suite",
        "variant",
        "data",
        1,
        2,
        99,
        9.0,
        10.0,
        11.0,
        12.0,
        config={"new": True},
    )
    legacy_row = db.get_run("suite", "variant", "data", 1, 2)
    assert legacy_row["id"] == 7
    assert legacy_row["n_params"] == 3
    assert legacy_row["val_loss"] == pytest.approx(1.0)
    assert legacy_row["config_json"] == '{"old":true}'
    assert legacy_row["run_uuid"] is None
    canonical = db.get_run(
        "suite", "variant", "data", 1, 2, run_uuid=identity["run_uuid"]
    )
    assert canonical["n_params"] == 99
    assert db.get_run_manifest(identity["run_uuid"])["status"] == "done"
    con = sqlite3.connect(path)
    assert con.execute("SELECT id, config_json FROM runs").fetchone() == (
        7,
        '{"old":true}',
    )
    assert con.execute("SELECT COUNT(*) FROM steps").fetchone()[0] == 2
    assert con.execute("SELECT id, config_json FROM lab_jobs").fetchone() == (
        9,
        '{"raw":1}',
    )
    job_columns = {row[1] for row in con.execute("PRAGMA table_info(lab_jobs)")}
    assert {"run_uuid", "worker_id", "lease_expires_ts", "checkpoint_path"} <= job_columns
    con.close()

    claimed = db.claim_lab_job(9, "legacy-claim-worker", lease_seconds=1)
    assert claimed is not None
    assert str(uuid.UUID(claimed["experiment_uuid"])) == claimed["experiment_uuid"]
    assert str(uuid.UUID(claimed["run_uuid"])) == claimed["run_uuid"]
    db.update_lab_job(9, lease_expires_ts=time.time() - 1)
    recovery = db.recover_stale_lab_jobs(now=time.time())
    assert 9 in recovery["requeued"]
    recovered = db.get_lab_job(9)
    assert recovered["run_uuid"] == claimed["run_uuid"]
    assert recovered["experiment_uuid"] == claimed["experiment_uuid"]


def test_claim_is_single_owner_heartbeat_and_completion_are_fenced(tmp_path):
    path = tmp_path / "claims.db"
    db = ResultsDB(path)
    job_id = db.create_lab_job(_job())

    def claim(worker):
        return ResultsDB(path).claim_next_lab_job(worker, lease_seconds=5)

    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(claim, ("one", "two")))
    winners = [row for row in claims if row is not None]
    assert len(winners) == 1
    owner = winners[0]["worker_id"]
    claimed = winners[0]
    manifest = build_record_manifest(
        suite="lab",
        variant="durable-test",
        dataset="default-text",
        seed=0,
        steps=4,
        experiment_uuid=claimed["experiment_uuid"],
        run_uuid=claimed["run_uuid"],
    )
    conflicting_manifest = build_record_manifest(
        suite="lab",
        variant="different",
        dataset="default-text",
        seed=0,
        steps=4,
        experiment_uuid=claimed["experiment_uuid"],
        run_uuid=claimed["run_uuid"],
    )
    assert db.heartbeat_lab_job(job_id, "wrong", lease_seconds=5) is False
    assert db.heartbeat_lab_job(
        job_id,
        owner,
        lease_seconds=5,
        completed_step=1,
        manifest=manifest,
    )
    with pytest.raises(ValueError, match="Immutable lab job manifest"):
        db.heartbeat_lab_job(
            job_id,
            owner,
            lease_seconds=5,
            manifest=conflicting_manifest,
        )
    with pytest.raises(ValueError, match="Immutable lab job field"):
        db.update_lab_job(job_id, run_uuid=None)
    assert db.finish_claimed_lab_job(job_id, "wrong", status="done") is False
    assert db.finish_claimed_lab_job(job_id, owner, status="done") is True
    assert db.finish_claimed_lab_job(job_id, owner, status="done") is True


def test_heartbeat_loop_retries_transient_sqlite_errors_without_losing_ownership():
    stop = threading.Event()
    ownership_lost = threading.Event()

    class TransientDB:
        calls = 0

        def heartbeat_lab_job(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise sqlite3.OperationalError("database is locked")
            stop.set()
            return True

    db = TransientDB()
    thread = threading.Thread(
        target=_lease_heartbeat_loop,
        args=(db, 1, "worker", 0.15, stop, ownership_lost),
    )
    thread.start()
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert db.calls == 2
    assert not ownership_lost.is_set()


@pytest.mark.parametrize(
    "outcome",
    [
        False,
        RuntimeError("unexpected"),
        sqlite3.OperationalError("disk I/O error"),
    ],
)
def test_heartbeat_loop_fences_false_and_unexpected_renewals(outcome, caplog):
    stop = threading.Event()
    ownership_lost = threading.Event()

    class FailingDB:
        def heartbeat_lab_job(self, *args, **kwargs):
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    thread = threading.Thread(
        target=_lease_heartbeat_loop,
        args=(FailingDB(), 1, "worker", 0.15, stop, ownership_lost),
    )
    with caplog.at_level("ERROR"):
        thread.start()
        thread.join(timeout=1)
    assert not thread.is_alive()
    assert ownership_lost.is_set()
    if isinstance(outcome, Exception):
        assert "Unexpected" in caplog.text
        assert "renewing job 1 lease" in caplog.text


def test_heartbeat_loop_stops_promptly_when_requested():
    stop = threading.Event()
    ownership_lost = threading.Event()

    class DB:
        def heartbeat_lab_job(self, *args, **kwargs):
            raise AssertionError("heartbeat should not run after prompt stop")

    thread = threading.Thread(
        target=_lease_heartbeat_loop,
        args=(DB(), 1, "worker", 30, stop, ownership_lost),
    )
    thread.start()
    stop.set()
    thread.join(timeout=0.2)
    assert not thread.is_alive()
    assert not ownership_lost.is_set()


def test_exclusive_lane_blocks_exclusive_claims_but_not_cpu_work(tmp_path):
    path = tmp_path / "exclusive-claims.db"
    db = ResultsDB(path)
    running = db.create_lab_job(
        _job(run_name="running-exclusive", config={"lab.gpu_reservation.required": True})
    )
    assert db.claim_lab_job(running, "owner", lease_seconds=5) is not None
    blocked = [
        db.create_lab_job(_job(run_name="gpu-malformed", device_target="gpu")),
        db.create_lab_job(_job(run_name="high-memory", config={"lab.resource.high_memory": True})),
        db.create_lab_job(_job(run_name="high-band", config={"lab.resource.band": "high"})),
        db.create_lab_job(_job(run_name="extreme-band", config={"lab.resource.band": "extreme"})),
    ]
    db.update_lab_job(blocked[0], config_json="{malformed")
    malformed_standard = db.create_lab_job(
        _job(run_name="malformed-standard", device_target="cpu", config={})
    )
    db.update_lab_job(malformed_standard, config_json="{malformed")
    cpu = db.create_lab_job(_job(run_name="cpu", device_target="cpu", config={}))
    second_connection = ResultsDB(path)
    assert (
        second_connection.claim_next_lab_job("malformed-worker", lease_seconds=5)["id"]
        == malformed_standard
    )
    assert second_connection.claim_next_lab_job("cpu-worker", lease_seconds=5)["id"] == cpu
    assert all(
        ResultsDB(path).claim_lab_job(job_id, "other", lease_seconds=5) is None
        for job_id in blocked
    )


def test_duplicate_run_uuid_submission_is_idempotent_for_single_and_pair_jobs(tmp_path):
    path = tmp_path / "duplicate-lab-jobs.db"
    run_uuid = str(uuid.uuid4())
    job = _job(
        run_name="single-race",
        experiment_uuid=str(uuid.uuid4()),
        run_uuid=run_uuid,
    )

    def submit_single(_):
        return ResultsDB(path).create_lab_job(
            {**job, "experiment_uuid": str(uuid.uuid4())}
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        ids = list(pool.map(submit_single, range(2)))
    assert ids[0] == ids[1]
    assert len(ResultsDB(path).fetch_lab_jobs()) == 1
    retry = {
        **job,
        "experiment_uuid": str(uuid.uuid4()),
        "group_id": "retry-generated-group",
    }
    runtime_config = {
        **job["config"],
        "lab.analogue.job_id": 99,
        "lab.gpu_reservation.owner_job_id": ids[0],
        "lab.gpu_reservation.state": "released",
        "tracking.dashboard_db": "runtime-only.db",
    }
    ResultsDB(path).update_lab_job(ids[0], config_json=json.dumps(runtime_config))
    assert ResultsDB(path).create_lab_job(retry) == ids[0]
    conflicting_jobs = [
        {**retry, "run_name": "conflicting-name"},
        {**retry, "eval_every": int(retry["eval_every"]) + 1},
        {**retry, "artifact_dir": str(tmp_path / "different-artifact")},
        {
            **retry,
            "config": {**job["config"], "research.metric_type": "changed"},
        },
    ]
    for conflicting in conflicting_jobs:
        with pytest.raises(ValueError, match="conflicts with submitted run_uuid"):
            ResultsDB(path).create_lab_job(conflicting)

    db = ResultsDB(path)
    experiment_uuid = str(uuid.uuid4())
    primary = _job(
        run_name="pair-primary",
        experiment_uuid=experiment_uuid,
        run_uuid=str(uuid.uuid4()),
    )
    comparison = _job(
        run_name="pair-comparison",
        comparison_role="baseline",
        experiment_uuid=experiment_uuid,
        run_uuid=str(uuid.uuid4()),
    )
    first_pair = db.create_lab_job_pair(primary, comparison)
    second_pair = ResultsDB(path).create_lab_job_pair(
        {**primary, "experiment_uuid": str(uuid.uuid4())},
        {
            **comparison,
            "experiment_uuid": str(uuid.uuid4()),
            "run_uuid": str(uuid.uuid4()),
        },
    )
    assert second_pair == first_pair
    assert len(ResultsDB(path).fetch_lab_jobs()) == 3


def test_queue_run_uuid_retry_requires_identical_immutable_submission(tmp_path):
    results_dir = tmp_path / "idempotent-results"
    queue = ExperimentQueue(
        str(tmp_path / "idempotent-queue.db"),
        start_worker=False,
        results_dir=results_dir,
    )
    run_uuid = str(uuid.uuid4())
    submission = {
        "preset_id": "classical-small",
        "dataset_name": "default-text",
        "run_name": "idempotent-run",
        "seed": 0,
        "steps": 2,
        "eval_every": 1,
        "device_target": "cpu",
        "batch_size": 1,
        "seq_len": 8,
        "run_uuid": run_uuid,
    }
    first = queue.submit(**submission)
    assert queue.submit(**submission)["id"] == first["id"]

    conflicts = [
        {"run_name": "changed-name"},
        {"eval_every": 2},
        {"batch_size": 2},
        {"artifact_dir": str(results_dir / "different-artifact")},
    ]
    for override in conflicts:
        with pytest.raises(ValueError, match="conflicts with submitted run_uuid"):
            queue.submit(**{**submission, **override})


def test_worker_executes_persisted_model_snapshot_and_uuid_artifact_root(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "snapshot.db"
    db = ResultsDB(db_path)
    original_cfg = build_preset("classical-small")
    spec = create_spec(
        db,
        {"name": "snapshot-source", "config": dataclasses.asdict(original_cfg)},
    )
    queue = ExperimentQueue(
        str(db_path),
        start_worker=False,
        worker_id="snapshot-worker",
        results_dir=tmp_path / "snapshot-results",
    )
    job = queue.submit_model_spec(
        spec["id"], "default-text", "snapshot-run", 0, 1, 1,
        batch_size=1, seq_len=8, device_target="cpu",
    )
    mutated_cfg = dataclasses.replace(
        original_cfg,
        model=dataclasses.replace(
            original_cfg.model,
            d_model=32,
            n_heads=4,
            d_ff=64,
        ),
    )
    update_spec(db, spec["id"], {"config": dataclasses.asdict(mutated_cfg)})

    captured = {}

    def stop_after_capture(cfg, **kwargs):
        captured["cfg"] = cfg
        captured["options"] = kwargs["run_options"]
        raise RuntimeError("stop after persisted snapshot capture")

    monkeypatch.setattr("qllm.train.loop.fit", stop_after_capture)
    queue._run_one(job["id"])

    assert captured["cfg"].model.d_model == original_cfg.model.d_model
    assert captured["cfg"].model.d_model != mutated_cfg.model.d_model
    expected_dir = (
        tmp_path / "snapshot-results" / "runs" / job["run_uuid"]
    ).resolve()
    assert Path(captured["options"].artifact_dir) == expected_dir
    persisted = queue.get(job["id"])
    assert Path(persisted["artifact_dir"]) == expected_dir
    assert persisted["config"]["model.d_model"] == original_cfg.model.d_model
    assert persisted["status"] == "error"


def test_worker_preserves_problem_task_and_rejects_before_sequence_work(
    tmp_path, monkeypatch
):
    queue = ExperimentQueue(
        str(tmp_path / "task-snapshot.db"),
        start_worker=False,
        worker_id="task-snapshot-worker",
        results_dir=tmp_path / "task-snapshot-results",
    )
    job = queue.submit(
        "classical-small",
        "default-text",
        "task-snapshot",
        0,
        1,
        1,
        device_target="cpu",
    )
    config = dict(job["config"])
    config["problem.task_type"] = "ground_state"
    config["problem.instance_id"] = "tfim-open-n4-j1-h1-v1"
    config.pop("research.task_type", None)
    queue.db().update_lab_job(job["id"], config_json=json.dumps(config))

    def unexpected(*_args, **_kwargs):
        raise AssertionError("sequence-specific worker preparation must not start")

    monkeypatch.setattr(queue, "_confined_data_path", unexpected)
    monkeypatch.setattr("qllm.train.loop.fit", unexpected)
    queue._run_one(job["id"])

    failed = queue.get(job["id"])
    assert failed["status"] == "error"
    assert "immutable preset task_type" in failed["error"]


def test_recovery_cancel_and_legacy_stale_preserve_integrity(tmp_path):
    db = ResultsDB(tmp_path / "recovery.db")
    zero = db.create_lab_job(_job(run_name="zero"))
    db.claim_lab_job(zero, "worker", lease_seconds=5)
    db.update_lab_job(zero, lease_expires_ts=time.time() - 1)

    progressed = db.create_lab_job(_job(run_name="progressed"))
    db.claim_lab_job(progressed, "worker", lease_seconds=5)
    progressed_row = db.get_lab_job(progressed)
    progressed_manifest = build_record_manifest(
        suite="lab",
        variant="progressed",
        dataset="default-text",
        seed=0,
        steps=4,
        experiment_uuid=progressed_row["experiment_uuid"],
        run_uuid=progressed_row["run_uuid"],
    )
    db.start_run(
        "lab/progressed/default-text/0/4",
        "progressed",
        "lab",
        "progressed",
        "default-text",
        0,
        4,
        run_uuid=progressed_row["run_uuid"],
        experiment_uuid=progressed_row["experiment_uuid"],
        manifest=progressed_manifest,
    )
    db.update_lab_job(
        progressed, completed_step=2, lease_expires_ts=time.time() - 1
    )

    legacy = db.create_lab_job(_job(run_name="legacy"))
    db.update_lab_job(
        legacy,
        status="running",
        worker_id=None,
        claimed_ts=None,
        lease_expires_ts=None,
    )
    recovered = db.recover_stale_lab_jobs(now=time.time())
    assert zero in recovered["requeued"]
    assert progressed in recovered["errored"]
    assert legacy in recovered["errored"]
    assert db.finish_claimed_lab_job(zero, "worker", status="done") is False
    assert db.get_run_manifest(progressed_row["run_uuid"])["status"] == "error"
    assert next(
        row
        for row in db.fetch_live_runs()
        if row["run_uuid"] == progressed_row["run_uuid"]
    )["status"] == "error"

    restartable = db.create_lab_job(_job(run_name="restart-status"))
    restartable_row = db.get_lab_job(restartable)
    restartable_manifest = build_record_manifest(
        suite="lab",
        variant="restart-status",
        dataset="default-text",
        seed=0,
        steps=4,
        experiment_uuid=restartable_row["experiment_uuid"],
        run_uuid=restartable_row["run_uuid"],
    )
    restart_key = "lab/restart-status/default-text/0/4"
    db.start_run(
        restart_key,
        "restart-status",
        "lab",
        "restart-status",
        "default-text",
        0,
        4,
        run_uuid=restartable_row["run_uuid"],
        experiment_uuid=restartable_row["experiment_uuid"],
        manifest=restartable_manifest,
    )
    db.claim_lab_job(restartable, "old-owner", lease_seconds=5)
    db.update_lab_job(restartable, lease_expires_ts=time.time() - 1)
    db.recover_stale_lab_jobs(now=time.time())
    assert db.get_run_manifest(restartable_row["run_uuid"])["status"] == "queued"
    db.claim_lab_job(restartable, "new-owner", lease_seconds=5)
    assert db.get_run_manifest(restartable_row["run_uuid"])["status"] == "running"
    assert next(
        row
        for row in db.fetch_live_runs()
        if row["run_uuid"] == restartable_row["run_uuid"]
    )["status"] == "running"

    behind = db.create_lab_job(_job(run_name="checkpoint-behind", steps=10))
    behind_row = db.get_lab_job(behind)
    behind_manifest = build_record_manifest(
        suite="lab",
        variant="checkpoint-behind",
        dataset="default-text",
        seed=0,
        steps=10,
        experiment_uuid=behind_row["experiment_uuid"],
        run_uuid=behind_row["run_uuid"],
    )
    behind_key = "lab/checkpoint-behind/default-text/0/10"
    db.start_run(
        behind_key,
        "checkpoint-behind",
        "lab",
        "checkpoint-behind",
        "default-text",
        0,
        10,
        run_uuid=behind_row["run_uuid"],
        experiment_uuid=behind_row["experiment_uuid"],
        manifest=behind_manifest,
    )
    db.claim_lab_job(behind, "behind-worker", lease_seconds=5)
    assert db.heartbeat_lab_job(
        behind,
        "behind-worker",
        lease_seconds=5,
        completed_step=9,
        checkpoint_path=str(tmp_path / "checkpoint-step-1.msgpack"),
        manifest=behind_manifest,
    )
    db.log_step(
        behind_key,
        9,
        {"train_loss": 9.0},
        train_loss=9.0,
        run_uuid=behind_row["run_uuid"],
    )
    db.update_lab_job(behind, lease_expires_ts=time.time() - 1)
    reconciled = db.recover_stale_lab_jobs(
        now=time.time(),
        checkpoint_resolver=lambda job: {
            "path": str(tmp_path / "checkpoint-step-1.msgpack"),
            "completed_step": 1,
            "fresh": False,
        },
    )
    assert behind in reconciled["requeued"]
    assert db.get_lab_job(behind)["completed_step"] == 1
    assert db.get_run_manifest(behind_row["run_uuid"])["completed_step"] == 1
    behind_live = next(
        row
        for row in db.fetch_live_runs()
        if row["run_uuid"] == behind_row["run_uuid"]
    )
    assert behind_live["current_step"] == 1
    assert behind_live["last_train_loss"] is None

    queued = db.create_lab_job(_job(run_name="cancel"))
    cancelled = db.request_cancel_lab_job(queued)
    assert cancelled["status"] == "cancelled"
    config = json.loads(cancelled["config_json"])
    assert config["lab.gpu_reservation.state"] == "released"
    assert config["lab.gpu_reservation.job_id"] is None

    malformed = db.create_lab_job(_job(run_name="raw"))
    db.update_lab_job(malformed, config_json="{raw evidence")
    raw = db.request_cancel_lab_job(malformed)
    assert raw["config_json"] == "{raw evidence"

    checkpoint = tmp_path / "valid.msgpack"
    checkpoint.write_bytes(b"validated elsewhere")
    resumable = db.create_lab_job(_job(run_name="resumable"))
    db.claim_lab_job(resumable, "resume-worker", lease_seconds=5)
    db.update_lab_job(
        resumable,
        completed_step=2,
        checkpoint_path=str(checkpoint),
        lease_expires_ts=time.time() - 1,
    )
    result = db.recover_stale_lab_jobs(
        now=time.time(), checkpoint_validator=lambda path: path == str(checkpoint)
    )
    assert resumable in result["requeued"]
    assert db.get_lab_job(resumable)["resume_from"] == str(checkpoint)

    running = db.create_lab_job(_job(run_name="running-cancel"))
    db.claim_lab_job(running, "cancel-worker", lease_seconds=30)
    requested = db.request_cancel_lab_job(running)
    assert requested["status"] == "running"
    assert db.lab_job_cancel_requested(running) is True
    assert db.finish_claimed_lab_job(
        running, "cancel-worker", status="cancelled"
    )
    assert json.loads(db.get_lab_job(running)["config_json"])[
        "lab.gpu_reservation.state"
    ] == "released"


@pytest.mark.parametrize("status", ["done", "error", "cancelled"])
def test_all_claimed_terminal_statuses_release_reservations(tmp_path, status):
    db = ResultsDB(tmp_path / f"terminal-{status}.db")
    job_id = db.create_lab_job(_job(run_name=status))
    db.claim_lab_job(job_id, "worker", lease_seconds=30)
    assert db.finish_claimed_lab_job(job_id, "worker", status=status)
    config = json.loads(db.get_lab_job(job_id)["config_json"])
    assert config["lab.gpu_reservation.state"] == "released"
    assert config["lab.gpu_reservation.job_id"] is None


def test_canonical_results_keep_uuid_identity_and_legacy_row_id(tmp_path):
    db = ResultsDB(tmp_path / "results-identity.db")
    args = ("suite", "variant", "data", 1, 2, 3, 1.0, 2.0, 1.4, 5.0)
    first_identity = db.record(*args)
    legacy = db.get_run("suite", "variant", "data", 1, 2)
    legacy_id = legacy["id"]
    db.record(*args[:-1], 99.0, manifest=first_identity["manifest"])
    assert db.get_run("suite", "variant", "data", 1, 2)[
        "wall_seconds"
    ] == pytest.approx(5.0)
    assert db.get_run(
        "suite",
        "variant",
        "data",
        1,
        2,
        run_uuid=first_identity["run_uuid"],
    )["id"] == legacy_id
    db.record(*args[:-4], 0.9, 1.9, 1.3, 4.0)
    assert db.get_run("suite", "variant", "data", 1, 2)["id"] == legacy_id

    experiment = str(uuid.uuid4())
    run_a = str(uuid.uuid4())
    manifest_a = build_record_manifest(
        suite="suite",
        variant="variant",
        dataset="data",
        seed=1,
        steps=2,
        experiment_uuid=experiment,
        run_uuid=run_a,
    )
    db.record(*args, manifest=manifest_a)
    assert db.get_run("suite", "variant", "data", 1, 2)["id"] == legacy_id
    run_a_result = db.get_run(
        "suite", "variant", "data", 1, 2, run_uuid=run_a
    )
    assert run_a_result["run_uuid"] == run_a
    assert "id" in run_a_result
    assert run_a_result["id"] is None
    assert db.get_run(
        "suite", "wrong-variant", "data", 1, 2, run_uuid=run_a
    ) is None
    db.record(*args, manifest=manifest_a)
    db.record(*args[:-1], 99.0, manifest=manifest_a)
    assert db.get_run(
        "suite", "variant", "data", 1, 2, run_uuid=run_a
    )["wall_seconds"] == pytest.approx(5.0)
    with pytest.raises(ValueError, match="Conflicting final result evidence"):
        db.record(
            "suite",
            "variant",
            "data",
            1,
            2,
            3,
            8.0,
            2.0,
            1.4,
            5.0,
            manifest=manifest_a,
        )
    with pytest.raises(ValueError, match="Conflicting immutable result identity"):
        db.record(
            "suite",
            "other",
            "data",
            1,
            2,
            3,
            1.0,
            2.0,
            1.4,
            5.0,
            manifest=manifest_a,
        )
    run_b = str(uuid.uuid4())
    manifest_b = build_record_manifest(
        suite="suite",
        variant="variant",
        dataset="data",
        seed=1,
        steps=2,
        experiment_uuid=experiment,
        run_uuid=run_b,
    )
    db.record(*args, manifest=manifest_b)
    assert db.get_run("suite", "variant", "data", 1, 2, run_uuid=run_b)[
        "run_uuid"
    ] == run_b
    assert db.get_run("suite", "variant", "data", 1, 2)["run_uuid"] == (
        first_identity["run_uuid"]
    )

    pending = str(uuid.uuid4())
    assert db.get_run(
        "suite", "variant", "data", 1, 2, run_uuid=pending
    ) is None


def test_pair_relationships_and_analogue_links_commit_together(tmp_path):
    db = ResultsDB(tmp_path / "pair.db")
    experiment_uuid = str(uuid.uuid4())
    primary = _job(
        run_name="candidate",
        experiment_uuid=experiment_uuid,
        config={"lab.analogue.kind": "classical", "lab.analogue.role": "candidate"},
    )
    baseline = _job(
        run_name="baseline",
        experiment_uuid=experiment_uuid,
        config={"lab.analogue.kind": "classical", "lab.analogue.role": "baseline"},
    )
    candidate_id, baseline_id = db.create_lab_job_pair(primary, baseline)
    candidate = db.get_lab_job(candidate_id)
    control = db.get_lab_job(baseline_id)
    assert candidate["compare_to_job_id"] == baseline_id
    assert control["parent_job_id"] == candidate_id
    assert control["compare_to_job_id"] == candidate_id
    assert candidate["experiment_uuid"] == control["experiment_uuid"]
    assert candidate["run_uuid"] != control["run_uuid"]
    assert json.loads(candidate["config_json"])["lab.analogue.job_id"] == baseline_id
    assert json.loads(control["config_json"])["lab.analogue.job_id"] == candidate_id


def test_posthoc_analogue_link_rolls_back_as_one_transaction(
    tmp_path, monkeypatch
):
    db = ResultsDB(tmp_path / "posthoc-pair.db")
    primary_id = db.create_lab_job(_job(run_name="posthoc-candidate"))
    original_insert = ResultsDB._insert_lab_job

    def insert_then_fail(con, job, now):
        original_insert(con, job, now)
        raise RuntimeError("simulated link failure")

    monkeypatch.setattr(
        ResultsDB, "_insert_lab_job", staticmethod(insert_then_fail)
    )
    with pytest.raises(RuntimeError, match="simulated link failure"):
        db.create_linked_lab_job(
            primary_id,
            _job(
                run_name="posthoc-baseline",
                comparison_role="baseline",
            ),
            primary_config={"lab.analogue.role": "candidate"},
            group_id="posthoc-group",
        )
    assert len(db.fetch_lab_jobs()) == 1
    assert db.get_lab_job(primary_id)["compare_to_job_id"] is None


def test_sqlite_authoritative_queue_runs_one_cpu_step_and_persists_checkpoint(tmp_path):
    db_path = tmp_path / "queue.db"
    results_root = tmp_path / "queue-results"
    queue = ExperimentQueue(
        str(db_path),
        start_worker=True,
        lease_seconds=30,
        poll_seconds=0.05,
        results_dir=results_root,
    )
    try:
        job = queue.submit(
            "classical-small",
            "default-text",
            "durable-cpu-smoke",
            0,
            1,
            1,
            device_target="cpu",
            batch_size=1,
            seq_len=8,
            checkpoint_every=1,
            artifact_dir=str(results_root / "custom-artifacts"),
        )
        deadline = time.monotonic() + 60
        current = job
        while current["status"] not in {"done", "error", "cancelled"}:
            assert time.monotonic() < deadline
            time.sleep(0.05)
            current = queue.get(job["id"])
        assert current["status"] == "done", current.get("error")
        assert current["completed_step"] == 1
        assert Path(current["checkpoint_path"]).exists()
        assert current["worker_id"] == queue.worker_id
        assert current["attempt_count"] == 1
        summary = json.loads(
            (Path(current["artifact_dir"]) / "summary.json").read_text("utf-8")
        )
        assert summary["resources"]["execution_device"]["requested"] == "cpu"
        assert summary["resources"]["execution_device"]["resolved"]["platform"] == "cpu"
    finally:
        queue.close()


def test_queue_restart_discovers_identity_matched_step_zero_checkpoint(tmp_path):
    db_path = tmp_path / "restart.db"
    results_root = tmp_path / "restart-results"
    artifact_dir = results_root / "custom-artifacts"
    first_queue = ExperimentQueue(
        str(db_path), start_worker=False, results_dir=results_root
    )
    job = first_queue.submit(
        "classical-small",
        "default-text",
        "restart-zero",
        0,
        1,
        1,
        device_target="cpu",
        batch_size=1,
        seq_len=8,
        checkpoint_every=1,
        artifact_dir=str(artifact_dir),
    )
    db = ResultsDB(db_path)
    claimed = db.claim_lab_job(job["id"], "crashed-worker", lease_seconds=30)
    assert claimed is not None
    cfg = first_queue._config_for_job(
        first_queue._config_for_source(job["preset_id"]),
        job["config"]["data.corpus_path"],
        job["run_name"],
        int(job["seed"]),
        int(job["steps"]),
        int(job["eval_every"]),
        1,
        8,
    )
    partial = fit(
        cfg,
        verbose=False,
        should_cancel=lambda: True,
        run_options=RunOptions(
            experiment_uuid=job["experiment_uuid"],
            run_uuid=job["run_uuid"],
            artifact_dir=artifact_dir,
            checkpoint_every=1,
            seed_axes=job["config"]["research.seed_axes"],
            device_target=job["device_target"],
        ),
    )
    assert partial["summary"]["completed_step"] == 0
    db.update_lab_job(
        job["id"],
        lease_expires_ts=time.time() - 1,
        checkpoint_path=None,
        completed_step=0,
    )

    restarted = ExperimentQueue(
        str(db_path),
        start_worker=False,
        worker_id="restarted-worker",
        lease_seconds=30,
        results_dir=results_root,
    )
    recovered = restarted.get(job["id"])
    assert recovered["status"] == "queued"
    assert recovered["recovery_count"] == 1
    assert Path(recovered["resume_from"]) == Path(
        partial["summary"]["checkpoint_path"]
    )
    restarted._run_one(job["id"])
    done = restarted.get(job["id"])
    assert done["status"] == "done", done.get("error")
    assert done["completed_step"] == 1


def test_fork_crash_before_first_checkpoint_preserves_bootstrap_lineage(tmp_path):
    db_path = tmp_path / "fork-bootstrap.db"
    results_root = tmp_path / "fork-results"
    queue = ExperimentQueue(
        str(db_path),
        start_worker=False,
        worker_id="fork-bootstrap-worker",
        results_dir=results_root,
    )
    source_cfg = queue._config_for_job(
        queue._config_for_source("classical-small"),
        "data/input.txt",
        "fork-source",
        0,
        1,
        1,
        1,
        8,
    )
    source = fit(
        source_cfg,
        verbose=False,
        run_options=RunOptions(artifact_dir=results_root / "fork-source"),
    )
    source_checkpoint = source["summary"]["checkpoint_path"]
    fork_run_uuid = str(uuid.uuid4())
    fork_experiment_uuid = str(uuid.uuid4())
    fork_dir = results_root / "fork-child"
    job = queue.submit(
        "classical-small",
        "default-text",
        "fork-child",
        0,
        1,
        1,
        device_target="cpu",
        batch_size=1,
        seq_len=8,
        resume_from=source_checkpoint,
        run_uuid=fork_run_uuid,
        experiment_uuid=fork_experiment_uuid,
        artifact_dir=str(fork_dir),
    )
    assert job["parent_run_uuid"] == source["summary"]["run_uuid"]
    claimed = queue.db().claim_lab_job(
        job["id"], "crashed-fork-worker", lease_seconds=1
    )
    assert claimed is not None
    queue.db().update_lab_job(job["id"], lease_expires_ts=time.time() - 1)

    recovery = queue.db().recover_stale_lab_jobs(
        now=time.time(), checkpoint_resolver=queue._recoverable_checkpoint
    )

    assert job["id"] in recovery["requeued"]
    recovered = queue.get(job["id"])
    assert recovered["run_uuid"] == fork_run_uuid
    assert recovered["parent_run_uuid"] == source["summary"]["run_uuid"]
    assert Path(recovered["resume_from"]) == Path(source_checkpoint).resolve()
    assert Path(recovered["checkpoint_path"]) == Path(source_checkpoint).resolve()
    assert recovered["completed_step"] == source["summary"]["completed_step"]

    corrupt_bootstrap = results_root / "fork-bootstrap-copy.msgpack"
    corrupt_bootstrap.write_bytes(Path(source_checkpoint).read_bytes())
    corrupt_job = queue.submit(
        "classical-small",
        "default-text",
        "fork-corrupt-child",
        0,
        1,
        1,
        device_target="cpu",
        batch_size=1,
        seq_len=8,
        resume_from=str(corrupt_bootstrap),
        run_uuid=str(uuid.uuid4()),
        experiment_uuid=str(uuid.uuid4()),
        artifact_dir=str(results_root / "fork-corrupt-child"),
    )
    corrupt_bootstrap.write_bytes(b"corrupt after submission")
    assert queue.db().claim_lab_job(
        corrupt_job["id"], "crashed-corrupt-fork", lease_seconds=1
    )
    queue.db().update_lab_job(
        corrupt_job["id"], lease_expires_ts=time.time() - 1
    )
    failed_recovery = queue.db().recover_stale_lab_jobs(
        now=time.time(), checkpoint_resolver=queue._recoverable_checkpoint
    )
    assert corrupt_job["id"] in failed_recovery["errored"]
    failed = queue.get(corrupt_job["id"])
    assert failed["status"] == "error"
    assert Path(failed["resume_from"]) == corrupt_bootstrap

    lost = queue.submit(
        "classical-small",
        "default-text",
        "lost-progress",
        0,
        1,
        1,
        device_target="cpu",
        batch_size=1,
        seq_len=8,
    )
    assert queue.db().claim_lab_job(
        lost["id"], "lost-artifact-worker", lease_seconds=1
    )
    queue.db().update_lab_job(
        lost["id"], completed_step=1, lease_expires_ts=time.time() - 1
    )
    lost_recovery = queue.db().recover_stale_lab_jobs(
        now=time.time(), checkpoint_resolver=queue._recoverable_checkpoint
    )
    assert lost["id"] in lost_recovery["errored"]
    assert queue.get(lost["id"])["status"] == "error"


def test_same_name_dashboard_jobs_receive_distinct_uuid_artifact_dirs(tmp_path):
    queue = ExperimentQueue(
        str(tmp_path / "same-name.db"),
        start_worker=False,
        results_dir=tmp_path / "same-name-artifacts",
    )
    first = queue.submit(
        "classical-small", "default-text", "same-name", 0, 1, 1,
        device_target="cpu",
    )
    second = queue.submit(
        "classical-small", "default-text", "same-name", 1, 1, 1,
        device_target="cpu",
    )
    assert first["run_uuid"] != second["run_uuid"]
    assert first["artifact_dir"] != second["artifact_dir"]
    assert Path(first["artifact_dir"]).name == first["run_uuid"]
    assert Path(second["artifact_dir"]).name == second["run_uuid"]


@pytest.mark.parametrize("raw_config", ["[]", "42", '"text"'])
def test_worker_rejects_non_object_job_config_without_stranding_job(
    tmp_path, raw_config
):
    queue = ExperimentQueue(
        str(tmp_path / f"invalid-config-{len(raw_config)}.db"),
        start_worker=False,
        worker_id="config-worker",
        results_dir=tmp_path / "invalid-config-artifacts",
    )
    job = queue.submit(
        "classical-small", "default-text", "invalid-config", 0, 1, 1,
        device_target="cpu",
    )
    queue.db().update_lab_job(job["id"], config_json=raw_config)
    queue._run_one(job["id"])
    failed = queue.get(job["id"])
    assert failed["status"] == "error"
    assert failed["config_json"] == raw_config


@pytest.mark.parametrize(
    "cfg",
    [
        ModelConfig(d_model=8, n_heads=2, n_blocks=1, d_ff=16, max_seq_len=8),
        ModelConfig(arch="gru", rnn_hidden=8, max_seq_len=8),
        ModelConfig(
            arch="two_stream",
            encoder_kind="classical",
            condition="token",
            d_sent=4,
            sent_hidden=4,
            d_model=8,
            n_heads=2,
            n_blocks=1,
            d_ff=16,
            max_seq_len=8,
        ),
    ],
    ids=["transformer", "gru", "two-stream"],
)
def test_generation_outcome_is_architecture_neutral(cfg):
    tokenizer = CharTokenizer("abc ")
    model, final_cfg = build_model(cfg, tokenizer.vocab_size)
    tokens = jnp.zeros((1, 3), dtype=jnp.int32)
    params = model.init(jax.random.PRNGKey(0), tokens)["params"]
    outcome = generate_outcome(
        model,
        params,
        tokenizer,
        model_cfg=final_cfg,
        prompt="a",
        max_new_tokens=2,
        seed=0,
    )
    assert outcome["ok"] is True
    assert outcome["supported"] is True
    assert outcome["status"] == "supported"
    assert outcome["architecture"] == cfg.arch
    assert outcome["generated_text"].startswith("a")


def test_generation_invalid_settings_return_unsupported_without_throwing():
    tokenizer = CharTokenizer("ab")
    model, cfg = build_model(ModelConfig(max_seq_len=4), tokenizer.vocab_size)
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 2), dtype=jnp.int32))[
        "params"
    ]
    outcome = generate_outcome(
        model,
        params,
        tokenizer,
        model_cfg=cfg,
        max_new_tokens="not-a-number",
        temperature="also-bad",
        seed="bad",
    )
    assert outcome["ok"] is False
    assert outcome["supported"] is False
    assert outcome["status"] == "unsupported"
    assert outcome["generated_text"] is None


def test_dashboard_generation_uses_persisted_artifact_within_results_root(
    tiny_classical_cfg, tmp_path
):
    results_root = tmp_path / "trusted-results"
    data_root = tmp_path / "trusted-data"
    cfg = dataclasses.replace(
        tiny_classical_cfg,
        data=dataclasses.replace(
            tiny_classical_cfg.data,
            corpus_path=str(data_root / "missing-corpus.txt"),
        ),
    )
    artifact_dir = results_root / "custom-artifacts"
    trained = fit(
        cfg,
        verbose=False,
        run_options=RunOptions(artifact_dir=artifact_dir),
    )
    db = ResultsDB(tmp_path / "model-test.db")
    job_id = db.create_lab_job(
        _job(
            status="done",
            run_name=trained["summary"]["run_name"],
            steps=cfg.train.steps,
            eval_every=cfg.train.eval_every,
            config=to_flat_dict(cfg),
            artifact_dir=str(artifact_dir),
            checkpoint_path=trained["summary"]["checkpoint_path"],
        )
    )
    info = model_test_payload(
        db, job_id, results_dir=results_root, data_dir=data_root
    )
    assert Path(info["artifacts"]["directory"]) == artifact_dir
    outcome = run_model_test(
        db,
        job_id,
        {"prompt": "a", "max_new_tokens": 1, "temperature": 0.8},
        results_dir=results_root,
        data_dir=data_root,
    )
    assert outcome["ok"] is True
    assert outcome["generated_text"]
