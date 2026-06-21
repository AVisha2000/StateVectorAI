from __future__ import annotations

import sqlite3
import types

import pytest

from qllm.config import BlockConfig, ExperimentConfig, ModelConfig, QuantumConfig, to_flat_dict, validate_config
from qllm.dashboard.datasets import import_hf_text_dataset, list_datasets
from qllm.dashboard.lab import (
    comparison_research_payload,
    lab_overview,
    scaling_test_payload,
    scaling_tests_overview,
)
from qllm.dashboard.model_graph import model_graph_from_config
from qllm.dashboard.model_specs import create_spec, spec_diff, update_spec, validation_payload
from qllm.dashboard.presets import build_preset, list_presets
from qllm.dashboard.runner import ExperimentQueue
from qllm.dashboard.workspace import comparison_payload, workspace_payload
from qllm.resultsdb import ResultsDB


def test_presets_build_configs():
    presets = list_presets()
    ids = {p["id"] for p in presets}
    assert {"classical-small", "quantum-ffn-4q", "gru-small"} <= ids
    for preset in presets:
        cfg = build_preset(preset["id"])
        assert cfg.train.steps > 0
        assert cfg.tracking.run_name


def test_presets_expose_run_workspace_metadata():
    presets = list_presets()
    by_id = {p["id"]: p for p in presets}
    required = {
        "description", "architecture", "quantum_role", "recommended_use",
        "risks", "comparison_policy",
    }
    for preset in presets:
        assert required <= set(preset)
        assert preset["description"]
        assert preset["architecture"]
        if preset["kind"] in {"quantum", "hybrid"}:
            twin_id = preset["classical_twin_id"]
            assert twin_id in by_id
            assert by_id[twin_id]["kind"] == "classical"


def test_model_graph_marks_quantum_components():
    graph = model_graph_from_config(build_preset("quantum-ffn-4q"))
    assert graph["summary"]["uses_quantum"] is True
    assert graph["summary"]["model_family"] == "quantum-ffn"
    quantum_nodes = [node for node in graph["nodes"] if node["kind"] == "quantum"]
    assert quantum_nodes
    assert graph["quantum"]["n_qubits"] == 4

    classical = model_graph_from_config(build_preset("classical-small"))
    assert classical["summary"]["uses_quantum"] is False
    assert classical["quantum"] is None


def test_per_layer_config_validates_and_serializes():
    cfg = ExperimentConfig(
        model=ModelConfig(
            n_blocks=2,
            blocks=(
                BlockConfig("classical", "classical"),
                BlockConfig("quantum_proj", "quantum", QuantumConfig(n_qubits=5)),
            ),
        )
    )
    assert validate_config(cfg) == []
    flat = to_flat_dict(cfg)
    assert flat["model.blocks.1.attn_type"] == "quantum_proj"
    assert flat["model.blocks.1.quantum.n_qubits"] == 5
    graph = model_graph_from_config(cfg)
    assert "block_1_attn" in graph["summary"]["quantum_components"]


def test_per_layer_config_rejects_bad_block_count():
    cfg = ExperimentConfig(model=ModelConfig(n_blocks=2, blocks=(BlockConfig(),)))
    assert validate_config(cfg)


def test_quantum_presets_expose_tuning_controls():
    presets = {p["id"]: p for p in list_presets()}
    quantum = presets["quantum-ffn-4q"]["quantum_controls"]
    classical = presets["classical-small"]["quantum_controls"]
    assert quantum["enabled"] is True
    assert {field["key"] for field in quantum["fields"]} == {
        "n_qubits", "n_circuit_layers",
    }
    assert classical["enabled"] is False


def test_model_spec_crud_validate_and_diff(tmp_path):
    db = ResultsDB(tmp_path / "results.db")
    config = {
        "model": {
            "d_model": 64,
            "n_heads": 4,
            "n_blocks": 1,
            "d_ff": 256,
            "max_seq_len": 128,
            "attn_type": "classical",
            "ffn_type": "classical",
            "quantum": {"n_qubits": 4, "n_circuit_layers": 2},
        },
        "train": {},
        "data": {},
        "tracking": {},
    }
    validation = validation_payload(config)
    assert validation["ok"] is True
    spec = create_spec(db, {"name": "editable", "source": "preset:classical-small", "config": config})
    updated = update_spec(db, spec["id"], {"notes": "baseline"})
    assert updated["notes"] == "baseline"
    updated_config = dict(config)
    updated_config["model"] = dict(config["model"], ffn_type="quantum")
    child = create_spec(db, {
        "name": "editable v2",
        "parent_id": spec["id"],
        "version": 2,
        "config": updated_config,
    })
    diff = spec_diff(db, child["id"], spec["id"])
    assert any(change["path"] == "model.ffn_type" for change in diff["changes"])


def test_hf_import_validation_requires_source(tmp_path):
    db = ResultsDB(tmp_path / "results.db")
    with pytest.raises(ValueError, match="required"):
        import_hf_text_dataset(db, "", "train", "text")


def test_hf_import_validation_rejects_bad_row_limits(tmp_path):
    db = ResultsDB(tmp_path / "results.db")
    with pytest.raises(ValueError, match="at least 1"):
        import_hf_text_dataset(db, "fake/stories", "train", "text", row_limit=0)
    with pytest.raises(ValueError, match="200000"):
        import_hf_text_dataset(db, "fake/stories", "train", "text", row_limit=200001)


def test_hf_import_with_mock_loader(monkeypatch, tmp_path):
    import sys

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(
        load_dataset=lambda source, split, **kwargs: [
            {"text": "alpha"}, {"text": "beta"}, {"text": ""}
        ],
    ))
    db = ResultsDB(tmp_path / "results.db")
    item = import_hf_text_dataset(
        db, "fake/stories", "train", "text", "stories", 10,
        cache_dir=tmp_path / "imported",
    )
    assert item["name"] == "stories"
    assert item["n_rows"] == 2
    assert "alpha" in (tmp_path / "imported" / "stories.txt").read_text()
    assert list_datasets(db)[1]["name"] == "stories"


def test_hf_import_validation_missing_column_and_empty_text(monkeypatch, tmp_path):
    import sys

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(
        load_dataset=lambda source, split, **kwargs: [{"body": "alpha"}],
    ))
    db = ResultsDB(tmp_path / "results.db")
    with pytest.raises(ValueError, match="Available columns"):
        import_hf_text_dataset(db, "fake/stories", "train", "text", cache_dir=tmp_path)

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(
        load_dataset=lambda source, split, **kwargs: [{"text": ""}, {"text": None}],
    ))
    with pytest.raises(ValueError, match="no non-empty text"):
        import_hf_text_dataset(db, "fake/stories", "train", "text", cache_dir=tmp_path)


def test_queue_transitions_queued_and_cancelled(tmp_path):
    db_path = tmp_path / "results.db"
    q = ExperimentQueue(str(db_path), start_worker=False)
    job = q.submit("classical-small", "default-text", "smoke", 0, 2, 1)
    assert job["status"] == "queued"
    cancelled = q.cancel(job["id"])
    assert cancelled["status"] in {"queued", "cancelled", "running"}

    db = sqlite3.connect(db_path)
    try:
        rows = db.execute("SELECT COUNT(*) FROM lab_jobs").fetchone()[0]
    finally:
        db.close()
    assert rows == 1


def test_queue_can_run_saved_model_spec(tmp_path):
    db_path = tmp_path / "results.db"
    db = ResultsDB(db_path)
    config = {
        "model": {
            "d_model": 64,
            "n_heads": 4,
            "n_blocks": 1,
            "d_ff": 256,
            "max_seq_len": 128,
            "attn_type": "classical",
            "ffn_type": "classical",
            "quantum": {"n_qubits": 4, "n_circuit_layers": 2},
        },
        "train": {},
        "data": {},
        "tracking": {},
    }
    spec = create_spec(db, {"name": "spec-run", "config": config})
    q = ExperimentQueue(str(db_path), start_worker=False)
    job = q.submit_model_spec(spec["id"], "default-text", "spec-run", 0, 2, 1)
    assert job["preset_id"] == f"model-spec:{spec['id']}"
    assert job["config"]["model.n_blocks"] == 1


def test_queue_creates_linked_classical_comparison(tmp_path):
    db_path = tmp_path / "results.db"
    q = ExperimentQueue(str(db_path), start_worker=False)
    job = q.submit(
        "quantum-ffn-4q", "default-text", "pair", 4, 3, 1,
        queue_classical_comparison=True,
    )
    twin = job["comparison_job"]
    primary = q.get(job["id"])
    baseline = q.get(twin["id"])
    assert primary["comparison_role"] == "candidate"
    assert baseline["comparison_role"] == "baseline"
    assert baseline["preset_id"] == "classical-small"
    assert primary["group_id"] == baseline["group_id"]
    assert primary["compare_to_job_id"] == baseline["id"]
    assert baseline["compare_to_job_id"] == primary["id"]
    assert baseline["parent_job_id"] == primary["id"]


def test_queue_applies_quantum_overrides_to_primary_job(tmp_path):
    q = ExperimentQueue(str(tmp_path / "results.db"), start_worker=False)
    job = q.submit(
        "quantum-ffn-4q", "default-text", "custom-quantum", 0, 2, 1,
        quantum_overrides={"n_qubits": 6, "n_circuit_layers": 5},
    )
    config = job["config"]
    assert config["model.quantum.n_qubits"] == 6
    assert config["model.quantum.n_circuit_layers"] == 5
    assert config["lab.quantum_override.n_qubits"] == 6
    assert config["lab.quantum_override.n_circuit_layers"] == 5


def test_queue_rejects_out_of_range_quantum_overrides(tmp_path):
    q = ExperimentQueue(str(tmp_path / "results.db"), start_worker=False)
    with pytest.raises(ValueError, match="Qubit count must stay between"):
        q.submit(
            "quantum-ffn-4q", "default-text", "bad-quantum", 0, 2, 1,
            quantum_overrides={"n_qubits": 99},
        )


def test_gpu_target_allows_larger_quantum_overrides(monkeypatch, tmp_path):
    monkeypatch.setattr(ExperimentQueue, "gpu_ready", staticmethod(lambda: True))
    q = ExperimentQueue(str(tmp_path / "results.db"), start_worker=False)
    job = q.submit(
        "quantum-ffn-4q", "default-text", "gpu-quantum", 0, 2, 1,
        device_target="gpu",
        quantum_overrides={"n_qubits": 12, "n_circuit_layers": 12},
    )
    assert job["config"]["model.quantum.n_qubits"] == 12
    assert job["config"]["model.quantum.n_circuit_layers"] == 12


def test_quantum_attention_extreme_memory_shape_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(ExperimentQueue, "gpu_ready", staticmethod(lambda: True))
    q = ExperimentQueue(str(tmp_path / "results.db"), start_worker=False)
    with pytest.raises(ValueError, match="likely to exhaust GPU memory"):
        q.submit(
            "quantum-attn-4q", "default-text", "too-large-attn", 0, 2, 1,
            device_target="gpu",
            quantum_overrides={"n_qubits": 10, "n_circuit_layers": 12},
        )


def test_quantum_attention_memory_shape_can_be_reduced(monkeypatch, tmp_path):
    monkeypatch.setattr(ExperimentQueue, "gpu_ready", staticmethod(lambda: True))
    q = ExperimentQueue(str(tmp_path / "results.db"), start_worker=False)
    job = q.submit(
        "quantum-attn-4q", "default-text", "small-attn", 0, 2, 1,
        device_target="gpu",
        quantum_overrides={"n_qubits": 10, "n_circuit_layers": 12},
        batch_size=1,
        seq_len=16,
    )
    assert job["config"]["train.batch_size"] == 1
    assert job["config"]["train.seq_len"] == 16
    assert job["config"]["lab.train_override.batch_size"] == 1
    assert job["config"]["lab.resource.band"] != "extreme"


def test_scaling_sweep_queues_grid_with_one_group(monkeypatch, tmp_path):
    monkeypatch.setattr(ExperimentQueue, "gpu_ready", staticmethod(lambda: True))
    q = ExperimentQueue(str(tmp_path / "results.db"), start_worker=False)
    payload = q.submit_scaling_sweep(
        "quantum-ffn-4q", "default-text", "scale", 0, 2, 1,
        "gpu", qubits=[4, 12], depths=[2, 8],
    )
    assert payload["count"] == 4
    groups = {job["group_id"] for job in payload["jobs"]}
    assert groups == {payload["group_id"]}
    run_names = {job["run_name"] for job in payload["jobs"]}
    assert "scale-q12-d8" in run_names


def test_scaling_test_payload_collects_scale_results(monkeypatch, tmp_path):
    monkeypatch.setattr(ExperimentQueue, "gpu_ready", staticmethod(lambda: True))
    db_path = tmp_path / "results.db"
    q = ExperimentQueue(str(db_path), start_worker=False)
    sweep = q.submit_scaling_sweep(
        "quantum-ffn-4q", "default-text", "scale", 3, 2, 1,
        "gpu", qubits=[4, 8], depths=[2],
    )
    db = ResultsDB(db_path)
    first = sweep["jobs"][0]
    db.update_lab_job(
        first["id"],
        status="done",
        run_key="lab/quantum-ffn-4q-q4-d2/default-text/3/2",
    )
    db.record(
        "lab", "quantum-ffn-4q-q4-d2", "default-text", 3, 2,
        20, 1.1, 3.0, 1.5, 12.0,
    )

    overview = scaling_tests_overview(db)
    assert overview[0]["group_id"] == sweep["group_id"]
    assert overview[0]["qubits"] == [4, 8]

    payload = scaling_test_payload(db, sweep["group_id"])
    assert payload["available"] is True
    assert payload["complete_count"] == 1
    assert payload["best"]["val_ppl"] == pytest.approx(3.0)
    assert {point["n_qubits"] for point in payload["points"]} == {4, 8}


def test_gpu_requested_job_is_rejected_when_jax_has_no_gpu(monkeypatch, tmp_path):
    monkeypatch.setattr(ExperimentQueue, "gpu_ready", staticmethod(lambda: False))
    q = ExperimentQueue(str(tmp_path / "results.db"), start_worker=False)
    with pytest.raises(ValueError, match="JAX does not currently see a GPU"):
        q.submit(
            "classical-small", "default-text", "gpu-smoke", 0, 2, 1,
            device_target="gpu",
        )


def test_workspace_payload_includes_job_metadata_and_comparison_deltas(tmp_path):
    db_path = tmp_path / "results.db"
    q = ExperimentQueue(str(db_path), start_worker=False)
    job = q.submit(
        "quantum-ffn-4q", "default-text", "pair", 5, 2, 1,
        queue_classical_comparison=True,
    )
    db = ResultsDB(db_path)
    twin = job["comparison_job"]

    queued = workspace_payload(db, job["id"])
    assert queued["job"]["status"] == "queued"
    assert queued["preset"]["classical_twin_id"] == "classical-small"
    assert queued["dataset"]["name"] == "default-text"
    assert queued["comparison"]["available"] is True
    assert queued["comparison"]["deltas"] is None

    db.update_lab_job(job["id"], status="done")
    db.update_lab_job(twin["id"], status="done")
    db.record("lab", "quantum-ffn-4q", "default-text", 5, 2, 10, 1.0, 2.5, 1.4, 4.0)
    db.record("lab", "classical-small", "default-text", 5, 2, 12, 1.2, 3.0, 1.7, 2.5)

    done = workspace_payload(db, job["id"])
    assert done["final_run"]["val_ppl"] == pytest.approx(2.5)
    assert done["comparison"]["deltas"]["val_ppl"] == pytest.approx(-0.5)
    assert done["comparison"]["deltas"]["wall_seconds"] == pytest.approx(1.5)

    comparison = comparison_research_payload(db, job["id"])
    assert comparison["fairness"]["same_dataset"] is True
    assert comparison["fairness"]["same_seed"] is True
    assert comparison["verdict"]["label"] == "candidate better on this run"


def test_lab_overview_summarizes_jobs_and_comparisons(tmp_path):
    db_path = tmp_path / "results.db"
    q = ExperimentQueue(str(db_path), start_worker=False)
    job = q.submit(
        "quantum-ffn-4q", "default-text", "pair", 2, 2, 1,
        queue_classical_comparison=True,
    )
    db = ResultsDB(db_path)
    payload = lab_overview(db, {"gpu": {"ready": False, "jax_backend": "cpu"}})
    assert payload["counts"]["queued"] == 2
    assert len(payload["active_jobs"]) == 2
    assert payload["active_jobs"][0]["model_family"] in {"quantum-ffn", "transformer"}
    assert payload["recent_comparisons"][0]["job_id"] == job["id"]


def test_comparison_payload_handles_missing_baseline(tmp_path):
    q = ExperimentQueue(str(tmp_path / "results.db"), start_worker=False)
    job = q.submit("classical-small", "default-text", "solo", 0, 2, 1)
    payload = comparison_payload(ResultsDB(tmp_path / "results.db"), job["id"])
    assert payload["available"] is False
    assert payload["reason"] == "no linked classical comparison"
