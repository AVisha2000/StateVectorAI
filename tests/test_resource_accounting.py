from __future__ import annotations

import dataclasses
import sqlite3

import jax
import pytest

from qllm.resources import (
    device_memory_evidence,
    resolve_execution_device,
    static_resource_plan,
)
from qllm.dashboard.queries import all_runs, run_detail
from qllm.resultsdb import ResultsDB
from qllm.train.artifacts import RunOptions, build_record_manifest
from qllm.train.loop import fit


class _FakeDevice:
    platform = "cpu"
    device_kind = "test-cpu"
    id = 0

    def __init__(self, stats=None):
        self._stats = stats

    def memory_stats(self):
        return self._stats


def test_run_options_and_device_resolver_are_explicit(monkeypatch):
    cpu = _FakeDevice()
    calls = []

    def fake_devices(backend=None):
        calls.append(backend)
        if backend == "gpu":
            raise RuntimeError("no gpu backend")
        return [cpu]

    monkeypatch.setattr("qllm.resources.jax.devices", fake_devices)
    assert RunOptions(device_target=" CPU ").normalized().device_target == "cpu"
    assert resolve_execution_device("cpu") is cpu
    assert resolve_execution_device("auto") is cpu
    with pytest.raises(RuntimeError, match="gpu.*unavailable"):
        resolve_execution_device("gpu")
    assert calls == ["cpu", None, "gpu"]
    with pytest.raises(ValueError, match="auto, cpu, gpu"):
        RunOptions(device_target="tpu").normalized()


def test_static_quantum_plan_distinguishes_logical_from_backend_calls(
    tiny_quantum_cfg,
):
    device = jax.devices("cpu")[0]
    plan = static_resource_plan(
        tiny_quantum_cfg,
        n_params=123,
        requested_device="cpu",
        resolved_device=device,
    )
    assert plan["parameters"] == {"value": 123, "status": "exact", "unit": "count"}
    assert plan["state_dimension"]["value"] == 4
    logical = plan["logical_circuit_forward_instances_per_train_step"]
    assert logical["value"] > 0
    assert logical["status"] == "derived_logical_forward_instances"
    assert plan["backend_execution_calls"]["value"] is None
    assert plan["backend_execution_calls"]["status"] == "unsupported"
    configured_component = plan["quantum_backend"]["components"][
        "block_0.feed_forward"
    ]
    assert configured_component["capabilities"]["exactness"] == "exact"
    assert configured_component["capabilities"]["approximation"] is None

    sampled_cfg = dataclasses.replace(
        tiny_quantum_cfg,
        model=dataclasses.replace(
            tiny_quantum_cfg.model,
            quantum=dataclasses.replace(
                tiny_quantum_cfg.model.quantum,
                diff_method="parameter-shift",
                shots=64,
            ),
        ),
    )
    sampled = static_resource_plan(
        sampled_cfg,
        n_params=123,
        requested_device="cpu",
        resolved_device=device,
    )
    sampled_capabilities = sampled["quantum_backend"]["components"][
        "block_0.feed_forward"
    ]["capabilities"]
    assert sampled_capabilities["exactness"] == "sampled"
    assert sampled_capabilities["approximation"]["shots"] == 64

    parallel_cfg = dataclasses.replace(
        tiny_quantum_cfg,
        model=dataclasses.replace(
            tiny_quantum_cfg.model,
            quantum=dataclasses.replace(
                tiny_quantum_cfg.model.quantum, n_circuits=3
            ),
        ),
    )
    parallel = static_resource_plan(
        parallel_cfg,
        n_params=123,
        requested_device="cpu",
        resolved_device=device,
    )
    assert (
        parallel["logical_circuit_forward_instances_per_train_step"]["value"]
        == logical["value"] * 3
    )

    native_cfg = dataclasses.replace(
        tiny_quantum_cfg,
        model=dataclasses.replace(
            tiny_quantum_cfg.model,
            ffn_type="quantum_linear",
            quantum=dataclasses.replace(tiny_quantum_cfg.model.quantum, shots=100),
        ),
    )
    native = static_resource_plan(
        native_cfg,
        n_params=123,
        requested_device="cpu",
        resolved_device=device,
    )
    native_component = native["quantum_backend"]["components"][
        "block_0.feed_forward"
    ]
    assert native_component["implementation"] == "jax_native_statevector"
    assert (
        native_component["configured_backend_status"]
        == "not_applicable_to_native_jax_component"
    )
    assert native_component["shots"] is None
    assert native_component["configured_shots"] == 100
    assert native_component["shots_status"] == "not_applicable_to_native_jax_component"
    assert native_component["capabilities"]["backend"] == "jax_native_statevector"
    assert native_component["capabilities"]["exactness"] == "exact"
    assert native_component["capabilities"]["approximation"] is None
    assert native_component["capabilities"]["capabilities"]["gradients"][
        "status"
    ] == "supported"
    assert native["backend"] == "jax_native_statevector"
    assert native["backend_status"] == "actual_implementation"


def test_memory_evidence_never_turns_unsupported_into_zero():
    evidence = device_memory_evidence(_FakeDevice(None))
    assert evidence["status"] == "unsupported"
    assert evidence["peak_bytes"] is None
    assert evidence["available_bytes"] is None

    measured = device_memory_evidence(
        _FakeDevice({"peak_bytes_in_use": 12, "bytes_limit": 100, "bytes_in_use": 20})
    )
    assert measured["status"] == "measured"
    assert measured["peak_bytes"] == 12
    assert measured["available_bytes"] == 80
    assert measured["capacity_bytes"] == 100


def test_fit_records_blocked_timing_precision_and_static_manifest(
    tiny_classical_cfg, tmp_path
):
    cfg = dataclasses.replace(
        tiny_classical_cfg,
        train=dataclasses.replace(tiny_classical_cfg.train, steps=2, eval_every=2),
    )
    result = fit(
        cfg,
        verbose=False,
        run_options=RunOptions(
            artifact_dir=tmp_path / "resource-fit",
            device_target="cpu",
        ),
    )
    resources = result["summary"]["resources"]
    timing = resources["timing"]
    assert timing["completion_barrier"] == "jax.block_until_ready(loss)"
    assert timing["compile_plus_first_executed_train_step_seconds"] >= 0
    assert timing["steady_state_train_steps"] == 1
    assert timing["loop_wall_seconds"] >= sum(
        value or 0
        for value in (
            timing["compile_plus_first_executed_train_step_seconds"],
            timing["steady_state_train_step_seconds_total"],
        )
    )
    assert timing["fit_wall_seconds"] >= timing["attempt_wall_seconds"]
    assert resources["precision"]["parameter_dtypes"]
    assert resources["execution_device"]["requested"] == "cpu"
    assert resources["execution_device"]["resolved"]["platform"] == "cpu"
    assert resources["logical_circuit_forward_instances"]["value"] == 0
    assert resources["circuit_calls"] == 0
    assert resources["circuit_calls_kind"] == "exact"
    assert resources["peak_memory_bytes"] is None or resources["peak_memory_bytes"] >= 0
    assert result["manifest"]["resource_plan"]["parameters"]["value"] == result["summary"]["n_params"]
    assert result["manifest"]["environment"]["jax_runtime"]["devices"]

    with pytest.raises(ValueError, match="resource_plan.execution_device"):
        fit(
            cfg,
            verbose=False,
            run_options=RunOptions(
                resume_from=result["summary"]["checkpoint_path"],
                device_target="auto",
            ),
        )


def test_resultsdb_resource_migration_decode_and_first_result_preservation(tmp_path):
    db_path = tmp_path / "resources.db"
    with sqlite3.connect(db_path) as con:
        con.executescript(
            """
            CREATE TABLE runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,
                suite TEXT NOT NULL, variant TEXT NOT NULL, dataset TEXT NOT NULL,
                seed INTEGER NOT NULL, steps INTEGER NOT NULL, n_params INTEGER NOT NULL,
                val_loss REAL, val_ppl REAL, val_bpc REAL, wall_seconds REAL,
                config_json TEXT,
                UNIQUE(suite, variant, dataset, seed, steps)
            );
            CREATE TABLE run_results (
                run_uuid TEXT PRIMARY KEY, experiment_uuid TEXT NOT NULL, ts TEXT NOT NULL,
                suite TEXT NOT NULL, variant TEXT NOT NULL, dataset TEXT NOT NULL,
                seed INTEGER NOT NULL, steps INTEGER NOT NULL, n_params INTEGER NOT NULL,
                val_loss REAL, val_ppl REAL, val_bpc REAL, wall_seconds REAL,
                config_json TEXT, manifest_hash TEXT
            );
            """
        )
    db = ResultsDB(db_path)
    # Reopening proves the additive migration is repeatable.
    db = ResultsDB(db_path)
    with sqlite3.connect(db_path) as con:
        for table in ("runs", "run_results"):
            columns = {row[1] for row in con.execute(f"PRAGMA table_info({table})")}
            assert "resources_json" in columns

    manifest = build_record_manifest(
        suite="resource-test", variant="classical", dataset="tiny", seed=1, steps=2
    )
    common = dict(
        suite="resource-test",
        variant="classical",
        dataset="tiny",
        seed=1,
        steps=2,
        n_params=9,
        val_loss=1.0,
        val_ppl=2.0,
        val_bpc=1.4,
        wall_seconds=0.2,
        manifest=manifest,
    )
    db.record(**common, resources={"timing": {"attempt_wall_seconds": 0.2}})
    db.record(**common, resources={"timing": {"attempt_wall_seconds": 99.0}})
    row = db.get_run(
        "resource-test", "classical", "tiny", 1, 2, run_uuid=manifest["run_uuid"]
    )
    assert row is not None
    assert row["resources"]["timing"]["attempt_wall_seconds"] == 0.2
    assert db.fetch("resource-test")[0]["resources"] is not None
    api_row = all_runs(db)[0]
    assert api_row["resources"]["timing"]["attempt_wall_seconds"] == 0.2
    detail = run_detail(db, api_row["id"])
    assert detail["resources"]["timing"]["attempt_wall_seconds"] == 0.2
