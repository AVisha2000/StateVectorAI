from __future__ import annotations

import json
import uuid
from dataclasses import replace

import numpy as np
import pytest

from qllm.config import ExperimentConfig, ProblemConfig, QuantumConfig, TrainConfig
from qllm.problems import PauliTerm, get_ground_state_instance
from qllm.quantum.backends import PennyLaneBackend
from qllm.resultsdb import ResultsDB
from qllm.train.artifacts import RunOptions
from qllm.train.vqe import (
    GROUND_STATE_PRIMARY_METRIC_TYPE,
    hamiltonian_matrix,
    run_vqe,
)


def _config(
    *,
    steps: int = 2,
    eval_every: int = 1,
    seed: int = 7,
    dashboard_db: str | None = None,
) -> ExperimentConfig:
    cfg = ExperimentConfig(
        model=replace(
            ExperimentConfig().model,
            quantum=QuantumConfig(
                n_qubits=2,
                n_circuit_layers=1,
                ansatz="hardware_efficient",
                init_scale=0.1,
            ),
        ),
        train=TrainConfig(
            seed=seed,
            steps=steps,
            eval_every=eval_every,
            lr=0.05,
            weight_decay=0.0,
        ),
        problem=ProblemConfig(
            task_type="ground_state",
            instance_id="tfim-2q-open-j1-h1",
        ),
    )
    return replace(
        cfg,
        tracking=replace(cfg.tracking, dashboard_db=dashboard_db),
    )


def _cpu_options(**kwargs) -> RunOptions:
    return RunOptions(device_target="cpu", **kwargs)


def test_registered_toy_hamiltonian_has_reproducible_reference_certificates():
    instance = get_ground_state_instance("tfim-2q-open-j1-h1")
    assert instance.shape == (4, 4)
    assert np.linalg.eigvalsh(hamiltonian_matrix(instance))[0] == pytest.approx(
        -np.sqrt(5)
    )
    assert instance.references[0].role == "oracle"
    assert instance.references[0].certificate["eigenvalue_expression"] == "-sqrt(5)"
    assert instance.references[1].role == "descriptive_challenger"
    assert instance.references[1].energy == -2.0
    assert instance.references[1].certificate["state"] == "|+> tensor |+>"
    assert len(instance.config_hash) == len(instance.content_hash) == 64
    assert instance.sampler_policy.startswith("not_applicable")


def test_registered_problem_payloads_cannot_mutate_registry_certificates():
    instance = get_ground_state_instance("tfim-2q-open-j1-h1")
    original_hash = instance.content_hash
    payload = instance.to_payload()
    payload["classical_references"][1]["certificate"]["bloch_vectors"][0][
        "x"
    ] = 0.0

    assert instance.content_hash == original_hash
    assert instance.references[1].certificate["bloch_vectors"][0]["x"] == 1.0
    with pytest.raises(TypeError):
        instance.references[1].certificate["bloch_vectors"][0]["x"] = 0.0


def test_vqe_state_qnode_enforces_declared_backprop_method():
    circuit = PennyLaneBackend(diff_method="backprop").state_circuit(
        2, 1, "hardware_efficient"
    )
    assert circuit.diff_method == "backprop"


@pytest.mark.parametrize(
    "term",
    [
        lambda: PauliTerm(float("nan"), "X", (0,)),
        lambda: PauliTerm(1.0, "A", (0,)),
        lambda: PauliTerm(1.0, "XX", (0,)),
        lambda: PauliTerm(1.0, "XX", (0, 0)),
    ],
)
def test_pauli_terms_fail_closed_on_malformed_definitions(term):
    with pytest.raises(ValueError):
        term()


def test_cpu_vqe_writes_durable_artifacts_and_dashboard_row(tmp_path):
    db_path = tmp_path / "results.db"
    result = run_vqe(
        _config(dashboard_db=str(db_path)),
        verbose=False,
        out_dir=tmp_path,
        run_options=_cpu_options(),
    )
    summary = result["summary"]
    assert summary["schema_version"] == 1
    assert summary["task_type"] == "ground_state"
    assert summary["solver_kind"] == "vqe"
    assert summary["evidence_kind"] == "analytic_simulator_diagnostic"
    assert summary["claim_eligible"] is False
    assert summary["comparative_inference_enabled"] is False
    assert summary["paired_stats"] is None
    assert summary["completed_step"] == 2
    assert summary["final_energy"] == summary["energy"]
    assert np.isfinite(summary["energy"])
    assert summary["energy_error"] >= 0.0
    resources = summary["resources"]
    assert resources["attempt_optimizer_gradient_steps"]["value"] == 2
    assert resources["attempt_energy_evaluations"]["value"] == 3
    assert resources["attempt_logical_analytic_objective_invocations"]["value"] == 5
    assert resources["measured_hardware_circuit_executions"]["value"] is None
    artifact_dir = tmp_path / "runs" / result["run_options"].run_uuid
    assert (artifact_dir / "manifest.json").is_file()
    assert (artifact_dir / "checkpoints" / "latest.msgpack").is_file()
    assert (artifact_dir / "checkpoints" / "best.msgpack").is_file()
    assert (artifact_dir / "summary.json").is_file()
    assert result["manifest"]["data"]["metadata"]["instance_id"] == (
        "tfim-2q-open-j1-h1"
    )
    assert result["manifest"]["resource_plan"]["primary_metric"] == {
        "metric_type": GROUND_STATE_PRIMARY_METRIC_TYPE,
        "extraction_key": "energy_error",
    }
    row = ResultsDB(db_path).get_run(
        "adhoc",
        "tfim-2q-open-j1-h1",
        "tfim-2q-open-j1-h1",
        7,
        2,
        result["run_options"].run_uuid,
    )
    assert row["primary_metric_name"] == "energy_error"
    assert row["primary_metric_value"] == pytest.approx(summary["energy_error"])
    assert json.loads(row["config_json"])["research.metric_type"] == (
        GROUND_STATE_PRIMARY_METRIC_TYPE
    )
    assert row["val_loss"] is row["val_ppl"] is row["val_bpc"] is None


def test_resume_keeps_identity_history_and_completed_step_monotonic(tmp_path):
    cfg = _config(steps=3)
    calls = 0

    def cancel_after_one_step():
        nonlocal calls
        calls += 1
        return calls > 1

    cancelled = run_vqe(
        cfg,
        verbose=False,
        out_dir=tmp_path,
        should_cancel=cancel_after_one_step,
        run_options=_cpu_options(),
    )
    assert cancelled["summary"]["completed_step"] == 1
    checkpoint = cancelled["summary"]["checkpoint_path"]
    resumed = run_vqe(
        cfg,
        verbose=False,
        out_dir=tmp_path,
        run_options=_cpu_options(resume_from=checkpoint),
    )
    assert resumed["manifest"]["run_uuid"] == cancelled["manifest"]["run_uuid"]
    assert resumed["summary"]["completed_step"] == 3
    assert [row["step"] for row in resumed["summary"]["history"]] == [1, 2, 3]
    assert resumed["manifest"]["initialization_hash"] == (
        cancelled["manifest"]["initialization_hash"]
    )
    timing = resumed["summary"]["resources"]["timing"]
    assert timing["prior_attempt_count"] == 1
    assert timing["attempt_count"] == 2
    assert timing["prior_wall_seconds"] == pytest.approx(
        cancelled["summary"]["wall_seconds"], abs=1e-6
    )
    assert timing["cumulative_wall_seconds"] == pytest.approx(
        timing["prior_wall_seconds"] + timing["attempt_wall_seconds"]
    )
    assert resumed["summary"]["wall_seconds"] == pytest.approx(
        timing["cumulative_wall_seconds"], abs=1e-6
    )


def test_checkpoint_fork_gets_new_identity_and_parent_lineage(tmp_path):
    cfg = _config(steps=2)
    calls = 0

    def cancel_after_one_step():
        nonlocal calls
        calls += 1
        return calls > 1

    source = run_vqe(
        cfg,
        verbose=False,
        out_dir=tmp_path,
        should_cancel=cancel_after_one_step,
        run_options=_cpu_options(),
    )
    fork_uuid = str(uuid.uuid4())
    with pytest.raises(ValueError, match="parent_run_uuid"):
        run_vqe(
            cfg,
            verbose=False,
            out_dir=tmp_path,
            run_options=_cpu_options(
                run_uuid=str(uuid.uuid4()),
                parent_run_uuid=str(uuid.uuid4()),
                resume_from=source["summary"]["checkpoint_path"],
                artifact_dir=tmp_path / "false-parent",
            ),
        )
    fork = run_vqe(
        cfg,
        verbose=False,
        out_dir=tmp_path,
        run_options=_cpu_options(
            run_uuid=fork_uuid,
            resume_from=source["summary"]["checkpoint_path"],
            artifact_dir=tmp_path / "fork",
        ),
    )
    lineage = fork["summary"]["resume_lineage"]
    assert fork["manifest"]["run_uuid"] == fork_uuid
    assert fork["manifest"]["run_uuid"] != source["manifest"]["run_uuid"]
    assert lineage["mode"] == "fork"
    assert lineage["source_run_uuid"] == source["manifest"]["run_uuid"]
    assert lineage["parent_run_uuid"] == source["manifest"]["run_uuid"]
    assert fork["summary"]["completed_step"] == 2


def test_parameter_initialization_is_seeded_and_reproducible(tmp_path):
    first = run_vqe(
        _config(seed=3, steps=1),
        verbose=False,
        out_dir=tmp_path / "first",
        run_options=_cpu_options(),
    )
    repeat = run_vqe(
        _config(seed=3, steps=1),
        verbose=False,
        out_dir=tmp_path / "repeat",
        run_options=_cpu_options(),
    )
    different = run_vqe(
        _config(seed=4, steps=1),
        verbose=False,
        out_dir=tmp_path / "different",
        run_options=_cpu_options(),
    )
    assert first["manifest"]["initialization_hash"] == (
        repeat["manifest"]["initialization_hash"]
    )
    assert first["manifest"]["initialization_hash"] != (
        different["manifest"]["initialization_hash"]
    )
    assert first["summary"]["energy"] == pytest.approx(repeat["summary"]["energy"])


def test_registered_preset_slice_converges_within_diagnostic_tolerance(tmp_path):
    result = run_vqe(
        _config(steps=100, eval_every=10, seed=0),
        verbose=False,
        out_dir=tmp_path,
        run_options=_cpu_options(),
    )
    assert result["summary"]["energy_error"] <= 0.001


@pytest.mark.parametrize(
    ("cfg_update", "metric"),
    [
        (
            lambda cfg: replace(
                cfg, problem=ProblemConfig(task_type="sequence_modeling")
            ),
            GROUND_STATE_PRIMARY_METRIC_TYPE,
        ),
        (
            lambda cfg: replace(
                cfg, problem=replace(cfg.problem, instance_id="missing")
            ),
            GROUND_STATE_PRIMARY_METRIC_TYPE,
        ),
        (
            lambda cfg: replace(
                cfg,
                model=replace(
                    cfg.model,
                    quantum=replace(cfg.model.quantum, n_qubits=3),
                ),
            ),
            GROUND_STATE_PRIMARY_METRIC_TYPE,
        ),
        (
            lambda cfg: replace(
                cfg,
                model=replace(
                    cfg.model,
                    quantum=replace(cfg.model.quantum, shots=10),
                ),
            ),
            GROUND_STATE_PRIMARY_METRIC_TYPE,
        ),
        (
            lambda cfg: replace(
                cfg,
                model=replace(
                    cfg.model,
                    quantum=replace(
                        cfg.model.quantum,
                        diff_method="parameter-shift",
                    ),
                ),
            ),
            GROUND_STATE_PRIMARY_METRIC_TYPE,
        ),
        (lambda cfg: cfg, "validation_perplexity"),
        (
            lambda cfg: replace(
                cfg,
                model=replace(
                    cfg.model,
                    quantum=replace(
                        cfg.model.quantum,
                        backend="tensorcircuit_mps",
                        device="mps",
                        mps_max_bond_dimension=2,
                    ),
                ),
            ),
            GROUND_STATE_PRIMARY_METRIC_TYPE,
        ),
    ],
)
def test_invalid_vqe_requests_fail_before_circuit_construction(
    tmp_path, monkeypatch, cfg_update, metric
):
    monkeypatch.setattr(
        "qllm.train.vqe.get_state_circuit",
        lambda *args, **kwargs: pytest.fail("circuit constructed"),
    )
    with pytest.raises(ValueError):
        run_vqe(
            cfg_update(_config()),
            verbose=False,
            out_dir=tmp_path,
            run_options=_cpu_options(),
            primary_metric_type=metric,
        )


@pytest.mark.parametrize("device_target", ["auto", "gpu"])
def test_vqe_requires_explicit_cpu_before_circuit_construction(
    tmp_path, monkeypatch, device_target
):
    monkeypatch.setattr(
        "qllm.train.vqe.get_state_circuit",
        lambda *args, **kwargs: pytest.fail("circuit constructed"),
    )
    with pytest.raises(ValueError, match="device_target='cpu'"):
        run_vqe(
            _config(),
            verbose=False,
            out_dir=tmp_path,
            run_options=RunOptions(device_target=device_target),
        )


def test_reference_mismatch_is_rejected_before_circuit_construction(
    tmp_path, monkeypatch
):
    instance = get_ground_state_instance("tfim-2q-open-j1-h1")
    bad_exact = replace(instance.references[0], energy=-1.0)
    bad_instance = replace(
        instance,
        classical_references=(bad_exact, instance.references[1]),
    )
    monkeypatch.setattr(
        "qllm.train.vqe.get_ground_state_instance",
        lambda _: bad_instance,
    )
    monkeypatch.setattr(
        "qllm.train.vqe.get_state_circuit",
        lambda *args, **kwargs: pytest.fail("circuit constructed"),
    )
    with pytest.raises(ValueError, match="exact-reference energy"):
        run_vqe(
            _config(),
            verbose=False,
            out_dir=tmp_path,
            run_options=_cpu_options(),
        )
