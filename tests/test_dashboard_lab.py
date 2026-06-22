from __future__ import annotations

import sqlite3
import types

import pytest

from qllm.config import BlockConfig, ExperimentConfig, ModelConfig, QuantumConfig, to_flat_dict, validate_config
from qllm.dashboard.datasets import import_hf_text_dataset, list_datasets
from qllm.dashboard.analogues import classical_analogue_for_config, classical_analogue_for_preset
from qllm.dashboard.explore import (
    domain_payload,
    explore_payload,
    infer_research_context,
    result_dashboard_payload,
)
from qllm.dashboard.gpu_reservation import gpu_reservation_status
from qllm.dashboard.lab import (
    comparison_research_payload,
    enrich_job,
    lab_overview,
    scaling_test_payload,
    scaling_tests_overview,
)
from qllm.dashboard.model_graph import model_graph_from_config
from qllm.dashboard.model_specs import create_spec, spec_diff, update_spec, validation_payload
from qllm.dashboard.model_tests import model_test_payload
from qllm.dashboard.presets import build_preset, list_presets
from qllm.dashboard.runner import ExperimentQueue
from qllm.dashboard.studies import create_study, list_studies, study_payload
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
    assert {level["id"] for level in graph["levels"]} == {
        "overview", "blocks", "components", "quantum",
    }
    quantum_nodes = [node for node in graph["nodes"] if node["kind"] == "quantum"]
    assert quantum_nodes
    assert quantum_nodes[0]["meta"]["component_type"] == "ffn"
    assert quantum_nodes[0]["meta"]["config_path"] == "model.ffn_type"
    assert quantum_nodes[0]["meta"]["resource"]["n_qubits"] == 4
    assert any(component["id"] == quantum_nodes[0]["id"] for component in graph["components"])
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
    quantum_level = next(level for level in graph["levels"] if level["id"] == "quantum")
    assert "block_1_attn" in quantum_level["node_ids"]


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


def test_classical_analogue_resolver_uses_curated_twins_and_component_swaps():
    preset = classical_analogue_for_preset("quantum-ffn-4q")
    assert preset is not None
    assert preset.resolver == "curated_twin"
    assert preset.analogue_preset_id == "classical-small"

    cfg = ExperimentConfig(
        model=ModelConfig(
            n_blocks=2,
            blocks=(
                BlockConfig("classical", "classical"),
                BlockConfig("quantum_proj", "quantum", QuantumConfig(n_qubits=5)),
            ),
        )
    )
    analogue = classical_analogue_for_config(cfg)
    assert analogue is not None
    assert analogue.resolver == "automatic_component_swap"
    assert analogue.config.model.blocks[1].attn_type == "classical"
    assert analogue.config.model.blocks[1].ffn_type == "classical"
    assert "same_dataset" in analogue.fairness_requirements


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


def test_model_spec_validation_surfaces_layer_resource_and_fairness_review():
    config = {
        "model": {
            "arch": "transformer",
            "d_model": 64,
            "n_heads": 4,
            "n_blocks": 2,
            "d_ff": 256,
            "max_seq_len": 128,
            "attn_type": "classical",
            "ffn_type": "classical",
            "quantum": {"n_qubits": 4, "n_circuit_layers": 2, "trainable": True},
            "blocks": [
                {"attn_type": "classical", "ffn_type": "classical", "quantum": {"n_qubits": 4, "n_circuit_layers": 2, "trainable": True}},
                {"attn_type": "classical", "ffn_type": "quantum", "quantum": {"n_qubits": 4, "n_circuit_layers": 2, "trainable": False}},
            ],
        },
        "train": {},
        "data": {},
        "tracking": {},
    }
    validation = validation_payload(config)
    assert validation["ok"] is True
    assert validation["layer_summary"]["count"] == 2
    assert validation["layer_summary"]["quantum_layers"] == 1
    assert validation["layer_summary"]["frozen_quantum_layers"] == 1
    assert validation["resource_review"]["band"] in {"low", "medium", "high", "extreme"}
    assert validation["fairness_review"]["analogue_available"] is True
    assert validation["fairness_review"]["claim_readiness"] == "paired-ready"
    assert "same_dataset" in validation["fairness_review"]["requirements"]


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


def test_model_test_payload_reports_artifact_capabilities(tmp_path):
    db_path = tmp_path / "results.db"
    q = ExperimentQueue(str(db_path), start_worker=False)
    job = q.submit("classical-small", "default-text", "testable", 0, 2, 1)
    db = ResultsDB(db_path)
    db.update_lab_job(job["id"], status="done")

    out_dir = tmp_path / "results" / job["run_name"]
    out_dir.mkdir(parents=True)
    (out_dir / "summary.json").write_text(
        '{"val_ppl": 2.5, "steps": 2, "n_params": 128, "wall_seconds": 1.2}'
    )

    payload = model_test_payload(db, job["id"], tmp_path / "results")
    assert payload["supported_tests"]["summary_review"] is True
    assert payload["supported_tests"]["prompt_generation"] is False
    assert payload["artifacts"]["summary_exists"] is True
    assert payload["artifacts"]["params_exists"] is False
    assert payload["summary"]["val_ppl"] == pytest.approx(2.5)
    assert "params.msgpack artifact is missing" in payload["unsupported_reasons"]


def test_queue_generates_classical_analogue_for_quantum_model_spec(tmp_path):
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
            "ffn_type": "quantum",
            "quantum": {"n_qubits": 4, "n_circuit_layers": 2},
        },
        "train": {},
        "data": {},
        "tracking": {},
    }
    spec = create_spec(db, {"name": "quantum-spec", "config": config})
    q = ExperimentQueue(str(db_path), start_worker=False)
    job = q.submit_model_spec(
        spec["id"], "default-text", "spec-run", 3, 4, 1,
        queue_classical_comparison=True,
        batch_size=2,
        seq_len=16,
    )
    twin = job["comparison_job"]
    assert twin["preset_id"].startswith("model-spec:")
    assert twin["comparison_role"] == "baseline"
    assert twin["config"]["model.ffn_type"] == "classical"
    assert twin["config"]["train.batch_size"] == 2
    assert twin["config"]["train.seq_len"] == 16
    assert twin["config"]["lab.analogue.resolver"] == "automatic_component_swap"
    assert q.get(job["id"])["compare_to_job_id"] == twin["id"]

    payload = workspace_payload(ResultsDB(db_path), job["id"])
    assert payload["comparison"]["available"] is True
    assert payload["comparison"]["baseline"]["preset"]["kind"] == "classical"


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


def test_queue_missing_classical_analogue_after_quantum_job(tmp_path):
    db_path = tmp_path / "results.db"
    q = ExperimentQueue(str(db_path), start_worker=False)
    job = q.submit("quantum-ffn-4q", "default-text", "solo-quantum", 4, 3, 1)
    db = ResultsDB(db_path)

    enriched = enrich_job(q.get(job["id"]), db)
    assert enriched["analogue_state"] == "missing"
    missing = comparison_research_payload(db, job["id"])
    assert missing["available"] is False
    assert missing["verdict"]["label"] == "incomplete"

    queued = q.queue_classical_analogue(job["id"])
    twin = queued["comparison_job"]
    primary = q.get(job["id"])
    assert twin["preset_id"] == "classical-small"
    assert primary["comparison_role"] == "candidate"
    assert primary["compare_to_job_id"] == twin["id"]
    assert primary["config"]["lab.analogue.state"] == "queued"
    assert enrich_job(primary, db)["analogue_state"] == "queued"


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


def test_gpu_jobs_reserve_exclusive_lane_and_surface_memory_warning(monkeypatch, tmp_path):
    monkeypatch.setattr(ExperimentQueue, "gpu_ready", staticmethod(lambda: True))
    db_path = tmp_path / "results.db"
    q = ExperimentQueue(str(db_path), start_worker=False)
    job = q.submit(
        "quantum-ffn-4q", "default-text", "reserved", 0, 2, 1,
        device_target="gpu",
        quantum_overrides={"n_qubits": 12, "n_circuit_layers": 12},
    )
    db = ResultsDB(db_path)

    assert job["config"]["lab.gpu_reservation.required"] is True
    assert job["config"]["lab.gpu_reservation.lane"] == "exclusive-gpu"
    assert job["config"]["lab.resource.high_memory"] is True

    waiting = gpu_reservation_status(db)
    assert waiting["state"] == "waiting"
    assert waiting["waiting_count"] == 1
    assert waiting["high_memory_count"] == 1

    enriched = enrich_job(q.get(job["id"]), db)
    assert enriched["gpu_reservation"]["required"] is True
    assert enriched["gpu_reservation"]["high_memory"] is True
    assert enriched["gpu_reservation"]["state"] == "queued"

    db.update_lab_job(job["id"], status="running")
    active = gpu_reservation_status(db)
    assert active["state"] == "active"
    assert active["owner"]["id"] == job["id"]


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


def test_study_creation_queues_candidates_baselines_and_controls(tmp_path):
    db_path = tmp_path / "results.db"
    db = ResultsDB(db_path)
    q = ExperimentQueue(str(db_path), start_worker=False)
    study = create_study(db, q, {
        "name": "ffn-study",
        "research_question": "Does quantum FFN win across seeds?",
        "task": "Language modelling",
        "dataset_names": ["default-text"],
        "candidate_preset_id": "quantum-ffn-4q",
        "control_preset_ids": ["classical-small"],
        "seeds": [0, 1],
        "steps": 2,
        "eval_every": 1,
        "sweep": {"qubits": [4], "depths": [2]},
    })
    assert study["status"] == "queued"
    assert study["job_count"] == 6
    assert study["role_counts"]["candidate"] == 2
    assert study["role_counts"]["baseline"] == 2
    assert study["role_counts"]["control"] == 2
    assert list_studies(db)[0]["name"] == "ffn-study"
    assert all(job["group_id"] == study["group_id"] for job in study["jobs"])


def test_study_payload_summarizes_multiseed_evidence(tmp_path):
    db_path = tmp_path / "results.db"
    db = ResultsDB(db_path)
    q = ExperimentQueue(str(db_path), start_worker=False)
    study = create_study(db, q, {
        "name": "evidence-study",
        "dataset_names": ["default-text"],
        "candidate_preset_id": "quantum-ffn-4q",
        "seeds": [0, 1, 2],
        "steps": 2,
        "eval_every": 1,
        "sweep": {"qubits": [4], "depths": [2]},
    })
    for row in db.fetch_study_jobs(study["id"]):
        db.update_lab_job(row["id"], status="done")
        val_ppl = 2.0 if row["role"] == "candidate" else 2.5
        db.record(
            "lab",
            row["preset_id"],
            row["dataset_name"],
            int(row["seed"]),
            int(row["steps"]),
            100,
            1.0,
            val_ppl,
            1.2,
            5.0 if row["role"] == "candidate" else 3.0,
            config=row.get("config") or {},
        )

    payload = study_payload(db, study["id"])
    assert payload["evidence"]["label"] == "promising study"
    assert payload["evidence"]["fair_pairs"] == 3
    assert payload["evidence"]["wins"] == 3
    assert payload["evidence"]["mean_delta_val_ppl"] == pytest.approx(-0.5)
    assert payload["evidence"]["std_delta_val_ppl"] == pytest.approx(0.0)
    ladder = {item["key"]: item for item in payload["evidence"]["ladder"]}
    assert ladder["multi_seed"]["ok"] is True
    assert ladder["candidate_better"]["ok"] is True
    assert ladder["ablation_supported"]["ok"] is False


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
    assert comparison["fairness"]["same_training_budget"] is True
    assert comparison["fairness"]["same_preprocessing"] is True
    assert comparison["fairness"]["matched_config_fields"]["train.batch_size"] is True
    assert comparison["verdict"]["label"] == "single-run candidate better"
    assert comparison["verdict"]["claim_level"] == "anecdote"
    assert comparison["resource_normalized"]["improvement"] == pytest.approx(0.5)
    assert comparison["resource_normalized"]["improvement_per_extra_second"] == pytest.approx(1 / 3)
    ladder = comparison["evidence_ladder"]
    assert ladder["label"] in {"promising run", "cost-aware promising run"}
    by_key = {step["key"]: step for step in ladder["steps"]}
    assert by_key["matched_baseline"]["ok"] is True
    assert by_key["fair_protocol"]["ok"] is True
    assert by_key["run_level_improvement"]["ok"] is True
    assert by_key["multi_seed"]["ok"] is False


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


def test_explore_payload_maps_runs_and_jobs_to_research_context(tmp_path):
    db_path = tmp_path / "results.db"
    q = ExperimentQueue(str(db_path), start_worker=False)
    job = q.submit(
        "quantum-ffn-4q", "default-text", "pair", 7, 2, 1,
        queue_classical_comparison=True,
    )
    db = ResultsDB(db_path)
    db.record(
        "qnlp-v1", "quantum-ffn-4q", "default-text", 7, 2,
        120, 1.0, 2.4, 1.3, 12.0,
        config=job["config"],
    )

    payload = explore_payload(db)
    domains = {domain["name"]: domain for domain in payload["domains"]}
    assert "QNLP" in domains
    assert "Language modelling" in domains["QNLP"]["tasks"]
    datasets = {dataset["name"]: dataset for dataset in payload["datasets"]}
    assert datasets["default-text"]["best_val_ppl"] == pytest.approx(2.4)
    assert any(run["link"].startswith("/run/") for run in payload["runs"])
    assert any(item["id"] == job["id"] for item in payload["jobs"])

    qnlp = domain_payload(db, "qnlp")
    assert qnlp["available"] is True
    assert qnlp["domain"]["name"] == "QNLP"
    assert qnlp["runs"][0]["resource"]["n_qubits"] == 4


def test_infer_research_context_handles_synthetic_quantum_data():
    context = infer_research_context(
        suite="qnlp-v1",
        variant="qrnn-small",
        dataset="ising",
        config={"data.kind": "monitored_ising", "model.arch": "qrnn"},
    )
    assert context["domain"] == "Synthetic quantum data"
    assert context["task"] == "Quantum-generated sequence prediction"
    assert context["confidence"] > 0.8


def test_result_dashboard_payload_surfaces_cards_cost_and_verdicts(tmp_path):
    db_path = tmp_path / "results.db"
    q = ExperimentQueue(str(db_path), start_worker=False)
    job = q.submit(
        "quantum-ffn-4q", "default-text", "pair", 8, 2, 1,
        queue_classical_comparison=True,
    )
    db = ResultsDB(db_path)
    twin = job["comparison_job"]
    db.update_lab_job(job["id"], status="done")
    db.update_lab_job(twin["id"], status="done")
    db.record("lab", "quantum-ffn-4q", "default-text", 8, 2, 100, 1.0, 2.2, 1.2, 5.0, config=job["config"])
    db.record("lab", "classical-small", "default-text", 8, 2, 120, 1.1, 2.6, 1.4, 3.0, config=twin["config"])

    payload = result_dashboard_payload(db, dataset="default-text")
    assert payload["available"] is True
    assert any(card["label"] == "Champion model overall" and card["model"] == "quantum-ffn-4q" for card in payload["summaries"])
    qrow = next(row for row in payload["rows"] if row["model"] == "quantum-ffn-4q")
    assert qrow["resource"]["n_qubits"] == 4
    assert qrow["resource"]["resource_band"] in {"low", "medium", "high", "extreme", "classical"}
    assert qrow["verdict_label"] == "single-run candidate better"
    assert qrow["claim_level"] == "anecdote"
    assert qrow["comparison_link"] == f"/comparisons/{job['id']}"

    task_payload = result_dashboard_payload(db, task_slug="language-modelling", domain_slug="qnlp")
    assert task_payload["available"] is True
    assert "Language modelling" in task_payload["tasks"]
