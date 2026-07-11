from __future__ import annotations

import dataclasses
import sqlite3

import jax
import pytest

from qllm.config import (
    BlockConfig,
    ExperimentConfig,
    ModelConfig,
    QuantumConfig,
    TrainConfig,
)
from qllm.dashboard.resources import quantum_resource_estimate
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


def test_manifest_environment_records_optional_backend_distributions(
    monkeypatch, tmp_path
):
    absent = {"tensorcircuit-ng", "tensornetwork-ng"}

    def fake_version(name):
        if name in absent:
            from importlib.metadata import PackageNotFoundError

            raise PackageNotFoundError(name)
        return f"test-{name}"

    monkeypatch.setattr(
        "qllm.train.artifacts.importlib.metadata.version", fake_version
    )
    manifest = build_record_manifest(
        suite="environment-test",
        variant="classical",
        dataset="tiny",
        seed=1,
        steps=2,
        repo_root=tmp_path,
    )
    packages = manifest["environment"]["packages"]
    assert packages["pennylane-lightning"] == "test-pennylane-lightning"
    assert packages["PyYAML"] == "test-PyYAML"
    assert packages["tensorcircuit-ng"] is None
    assert packages["tensornetwork-ng"] is None
    assert manifest["environment_hash"] == manifest["environment"]["hash"]


def test_static_plan_passes_global_and_per_block_mps_capability_options(
    monkeypatch,
):
    calls = []

    def fake_capabilities(
        backend,
        device,
        diff_method,
        shots,
        *,
        mps_max_bond_dimension=None,
        mps_max_truncation_error=None,
        mps_relative_truncation=False,
    ):
        calls.append(
            {
                "backend": backend,
                "device": device,
                "diff_method": diff_method,
                "shots": shots,
                "bond": mps_max_bond_dimension,
                "threshold": mps_max_truncation_error,
                "relative": mps_relative_truncation,
            }
        )
        return {
            "backend": backend,
            "exactness": "approximate",
            "representation": "matrix_product_state",
            "approximation": {"method": "svd_truncation"},
            "capabilities": {},
        }

    monkeypatch.setattr(
        "qllm.resources.backend_capabilities_payload", fake_capabilities
    )
    global_q = QuantumConfig(
        n_qubits=3,
        n_circuit_layers=1,
        backend="tensorcircuit_mps",
        device="mps",
        mps_max_bond_dimension=4,
        mps_max_truncation_error=None,
        mps_relative_truncation=False,
    )
    block_q = dataclasses.replace(
        global_q,
        n_qubits=5,
        n_circuits=2,
        mps_max_bond_dimension=7,
        mps_max_truncation_error=None,
        mps_relative_truncation=False,
    )
    cfg = ExperimentConfig(
        model=ModelConfig(
            embed_type="quantum",
            quantum=global_q,
            blocks=(
                BlockConfig(
                    attn_type="classical", ffn_type="quantum", quantum=block_q
                ),
            ),
        ),
        train=TrainConfig(batch_size=1, seq_len=2),
    )
    plan = static_resource_plan(
        cfg,
        n_params=12,
        requested_device="cpu",
        resolved_device=jax.devices("cpu")[0],
    )

    assert {(call["bond"], call["threshold"], call["relative"]) for call in calls} == {
        (4, None, False),
        (7, None, False),
    }
    assert set(plan["quantum_backend"]["components"]) == {
        "embedding",
        "block_0.feed_forward",
    }


def test_mps_resource_evidence_separates_logical_dimension_from_storage_bound():
    qcfg = QuantumConfig(
        n_qubits=5,
        n_circuit_layers=3,
        n_circuits=2,
        backend="tensorcircuit_mps",
        device="mps",
        mps_max_bond_dimension=3,
        mps_max_truncation_error=None,
        mps_relative_truncation=False,
    )
    cfg = ExperimentConfig(
        model=ModelConfig(ffn_type="quantum", quantum=qcfg, n_blocks=1),
        train=TrainConfig(batch_size=2, seq_len=4),
    )
    plan = static_resource_plan(
        cfg,
        n_params=12,
        requested_device="cpu",
        resolved_device=jax.devices("cpu")[0],
    )
    component = plan["quantum_backend"]["components"]["block_0.feed_forward"]
    storage = component["storage"]

    assert plan["state_dimension"]["value"] == 2**5
    assert plan["state_dimension"]["allocation_status"] == (
        "not_an_allocation_measurement"
    )
    assert "not evidence" in plan["state_dimension"]["basis"]
    assert component["representation"] == "matrix_product_state"
    assert storage["stored_state_tensor_elements_per_instance"] == 2 * 5 * 3**2
    assert storage["stored_state_tensor_elements_per_instance_status"] == (
        "configured_conservative_upper_bound"
    )
    assert storage["observed_bond_dimension"] is None
    assert storage["observed_bond_dimension_status"] == "unmeasured"
    assert storage["peak_memory_bytes"] is None
    assert storage["peak_memory_status"] == "unmeasured"
    assert "automatic-differentiation intermediates" in storage["excludes"]
    approximation = component["approximation_evidence"]
    assert approximation["truncation_mode"] == "fixed_bond_dimension_only"
    assert approximation["threshold_support"] == (
        "unsupported_for_jit_vmap_training"
    )
    assert approximation["configured_svd_split_threshold"] is None
    assert approximation["configured_svd_split_threshold_status"] == (
        "unsupported_for_jit_vmap_training"
    )
    assert approximation["relative_truncation"] is False
    assert approximation["relative_truncation_status"] == (
        "unsupported_for_jit_vmap_training"
    )
    assert approximation["realized_truncation_error"] is None
    assert approximation["realized_truncation_error_status"] == "unmeasured"
    assert approximation["discarded_weight_status"] == "unmeasured"
    assert approximation["convergence_status"] == "unmeasured"


def test_dashboard_estimate_covers_global_per_block_mps_and_native_components():
    global_q = QuantumConfig(n_qubits=2, n_circuit_layers=2)
    mps_q = QuantumConfig(
        n_qubits=5,
        n_circuit_layers=3,
        n_circuits=2,
        backend="tensorcircuit_mps",
        device="mps",
        mps_max_bond_dimension=3,
        mps_max_truncation_error=None,
        mps_relative_truncation=False,
    )
    native_q = QuantumConfig(n_qubits=3, n_circuit_layers=4)
    cfg = ExperimentConfig(
        model=ModelConfig(
            embed_type="quantum",
            quantum=global_q,
            blocks=(
                BlockConfig(
                    attn_type="classical", ffn_type="quantum", quantum=mps_q
                ),
                BlockConfig(
                    attn_type="classical",
                    ffn_type="quantum_linear",
                    quantum=native_q,
                ),
            ),
        ),
        train=TrainConfig(batch_size=2, seq_len=4),
    )

    estimate = quantum_resource_estimate(cfg)
    components = estimate["components"]
    assert set(components) == {
        "embedding",
        "block_0.feed_forward",
        "block_1.feed_forward",
    }
    assert components["embedding"]["representation"] == "dense_statevector"
    assert components["embedding"]["storage"][
        "stored_state_tensor_elements_per_instance"
    ] == 2**2
    mps = components["block_0.feed_forward"]
    assert mps["actual_backend"] == "tensorcircuit_mps"
    assert mps["storage"]["stored_state_tensor_elements_per_instance"] == (
        2 * 5 * 3**2
    )
    native = components["block_1.feed_forward"]
    assert native["actual_backend"] == "jax_native_statevector"
    assert native["representation"] == "dense_statevector"
    assert native["storage"]["stored_state_tensor_elements_per_instance"] == 2**3
    assert estimate["state_dim"] == 2**5
    assert estimate["component_multiplier"] == 4
    assert estimate["score_status"].startswith("coarse_configured")
    assert estimate["band_status"].startswith("coarse_threshold")
    assert estimate["peak_memory"]["bytes"] is None
    assert estimate["peak_memory"]["status"] == "unmeasured"
    assert "intermediates are excluded" in estimate["peak_memory"]["caveat"]
    assert any(
        "Threshold and relative truncation are unsupported" in item
        for item in estimate["advice"]
    )
