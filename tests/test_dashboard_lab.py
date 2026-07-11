from __future__ import annotations

import dataclasses
import hashlib
import importlib
import sqlite3
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from qllm.config import BlockConfig, ExperimentConfig, ModelConfig, QuantumConfig, to_flat_dict, validate_config
from qllm.dashboard.datasets import import_hf_text_dataset, list_datasets
from qllm.dashboard.analogues import (
    AnalogueSpec,
    DEFAULT_FAIRNESS_REQUIREMENTS,
    classical_analogue_for_config,
    classical_analogue_for_preset,
    config_from_flat_payload,
)
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
from qllm.dashboard.queries import run_detail, suite_detail, suites_overview
from qllm.dashboard.runner import ExperimentQueue
from qllm.dashboard.studies import (
    create_study,
    list_studies,
    study_payload,
    study_report_payload,
)
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


@pytest.mark.parametrize(
    ("arch", "label"),
    [
        ("contextual_qrnn", "Contextual Quantum Memory Cell"),
        ("routed_contextual", "Routed Contextual Quantum Memory Cell"),
    ],
)
def test_contextual_recurrent_graphs_are_quantum_and_architecture_honest(
    arch, label
):
    graph = model_graph_from_config(ModelConfig(arch=arch))
    memory = next(node for node in graph["nodes"] if node["id"] == "q_memory")
    assert memory["label"] == label
    assert memory["kind"] == "quantum"
    assert memory["meta"]["architecture"] == arch
    assert graph["summary"]["uses_quantum"] is True
    assert graph["summary"]["model_family"] == arch


@pytest.mark.parametrize(
    ("config", "family"),
    [
        (ModelConfig(attn_type="quantum_qkv"), "quantum-attention"),
        (ModelConfig(ffn_type="quantum_linear"), "quantum-ffn"),
        (ModelConfig(arch="two_stream"), "two-stream"),
    ],
)
def test_model_family_covers_every_registered_component_variant(config, family):
    assert model_graph_from_config(config)["summary"]["model_family"] == family


def test_two_stream_graph_exposes_the_causal_prefix_protocol():
    graph = model_graph_from_config(
        ModelConfig(arch="two_stream", encoder_kind="classical", condition="bias")
    )
    encoder = next(node for node in graph["nodes"] if node["id"] == "sentence_encoder")
    assert encoder["label"] == "Classical Causal Prefix Encoder"
    assert encoder["meta"]["protocol"] == "causal_prefix_v2"


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


def test_flat_config_adapter_preserves_sections_and_ignores_lab_metadata():
    source = ExperimentConfig(model=ModelConfig(d_model=32, n_heads=4))
    flat = to_flat_dict(source)
    flat.update({
        "lab.resource.band": "low",
        "lab.analogue.state": "queued",
    })
    restored = config_from_flat_payload(flat)
    assert restored.model.d_model == 32
    assert restored.model.n_heads == 4
    assert restored.train == source.train


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


def test_classical_model_spec_supports_omitted_quantum_config(tmp_path):
    db = ResultsDB(tmp_path / "results.db")
    payload = dataclasses.asdict(
        ExperimentConfig(model=ModelConfig(quantum=None))
    )

    review = validation_payload(payload)
    spec = create_spec(db, {"name": "classical-no-q", "config": payload})

    assert review["ok"] is True
    assert review["resource"]["band"] == "classical"
    assert review["resource"]["state_dim"] == 0
    assert spec["name"] == "classical-no-q"


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


def test_model_spec_validation_returns_controlled_errors_before_derived_views():
    validation = validation_payload({
        "model": {"quantum": {"n_qubits": "many"}},
    })
    assert validation["ok"] is False
    assert validation["resource"] is None
    assert validation["graph"] is None
    assert "model.quantum.n_qubits" in "\n".join(validation["errors"])


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
    expected = b"alpha\n\nbeta"
    assert (tmp_path / "imported" / "stories.txt").read_bytes() == expected
    assert item["n_bytes"] == len(expected)
    assert item["sha256"] == hashlib.sha256(expected).hexdigest()
    assert item["rows_examined"] == 3
    assert item["truncated"] is False
    stored = list_datasets(db)[1]
    assert stored["name"] == "stories"
    assert stored["sha256"] == item["sha256"]
    assert stored["warnings"] == item["warnings"]


def test_hf_import_revision_row_limit_and_fingerprint(monkeypatch, tmp_path):
    calls = {}
    pulls = []

    class FakeDataset:
        _fingerprint = "resolved-fingerprint-123"

        def __iter__(self):
            for index, row in enumerate([
                {"text": None}, {"text": "alpha"}, {"text": "beta"},
                {"text": "unreached"},
            ]):
                pulls.append(index)
                yield row

    def fake_load(source, split, **kwargs):
        calls.update({"source": source, "split": split, **kwargs})
        return FakeDataset()

    monkeypatch.setitem(
        sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load)
    )
    db = ResultsDB(tmp_path / "results.db")
    item = import_hf_text_dataset(
        db,
        "fake/stories",
        "train",
        "text",
        "revisioned",
        row_limit=2,
        cache_dir=tmp_path / "imported",
        revision="commit-abc",
    )
    assert calls["revision"] == "commit-abc"
    assert calls["streaming"] is True
    assert pulls == [0, 1, 2]  # two processed rows plus one bounded lookahead
    assert item["n_rows"] == 1  # None consumed one row-limit slot
    assert item["rows_examined"] == 3
    assert item["requested_revision"] == "commit-abc"
    assert item["resolved_fingerprint"] == "resolved-fingerprint-123"
    assert item["revision_applicable"] is True
    assert item["truncated"] is True
    assert item["truncation_reason"] == "row_limit"
    assert item["warnings"]


def test_hf_import_enforces_utf8_character_and_byte_limits(monkeypatch, tmp_path):
    class FakeDataset(list):
        _fingerprint = "resolved"

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(
        load_dataset=lambda source, split, **kwargs: FakeDataset([
            {"text": "ééé"}, {"text": "later"},
        ]),
    ))
    db = ResultsDB(tmp_path / "results.db")
    byte_limited = import_hf_text_dataset(
        db, "fake/utf8", "train", "text", "utf8", 10,
        cache_dir=tmp_path / "imported", char_limit=100, byte_limit=5,
    )
    assert byte_limited["truncation_reason"] == "byte_limit"
    assert byte_limited["n_chars"] == 2
    assert byte_limited["n_bytes"] == 4
    assert (tmp_path / "imported" / "utf8.txt").read_text(encoding="utf-8") == "éé"

    character_limited = import_hf_text_dataset(
        db, "fake/chars", "train", "text", "chars", 10,
        cache_dir=tmp_path / "imported", char_limit=2, byte_limit=100,
    )
    assert character_limited["truncation_reason"] == "character_limit"
    assert character_limited["n_chars"] == 2


def test_url_import_rejects_inapplicable_revision(tmp_path):
    db = ResultsDB(tmp_path / "results.db")
    with pytest.raises(ValueError, match="not applicable"):
        import_hf_text_dataset(
            db,
            "https://example.invalid/data.txt",
            "train",
            "text",
            revision="main",
            cache_dir=tmp_path / "imported",
        )


def test_url_import_marks_revision_not_applicable(monkeypatch, tmp_path):
    captured = {}

    class FakeDataset(list):
        _fingerprint = "url-loader-fingerprint"

    def fake_load(source, split, **kwargs):
        captured.update({"source": source, "split": split, **kwargs})
        return FakeDataset([{"text": "url text"}])

    monkeypatch.setitem(
        sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load)
    )
    db = ResultsDB(tmp_path / "results.db")
    url = "https://example.invalid/data.txt"
    item = import_hf_text_dataset(
        db, url, "train", "text", "url-data", 10,
        cache_dir=tmp_path / "imported",
    )
    assert captured["source"] == "text"
    assert captured["data_files"] == url
    assert captured["streaming"] is True
    assert "revision" not in captured
    assert item["requested_revision"] is None
    assert item["revision_applicable"] is False
    assert item["resolved_fingerprint"] == "url-loader-fingerprint"


def test_import_avoids_overwriting_unregistered_corpus(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(
        load_dataset=lambda source, split, **kwargs: [{"text": "new content"}],
    ))
    imported = tmp_path / "imported"
    imported.mkdir()
    existing = imported / "stories.txt"
    existing.write_text("user artifact", encoding="utf-8")
    db = ResultsDB(tmp_path / "results.db")
    item = import_hf_text_dataset(
        db, "fake/stories", "train", "text", "stories", 10,
        cache_dir=imported,
    )
    assert existing.read_text(encoding="utf-8") == "user artifact"
    assert item["name"] == "stories-2"
    assert (imported / "stories-2.txt").read_text(encoding="utf-8") == "new content"


def test_import_reserves_builtin_dataset_name(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(
        load_dataset=lambda source, split, **kwargs: [{"text": "new content"}],
    ))
    db = ResultsDB(tmp_path / "results.db")

    item = import_hf_text_dataset(
        db, "fake/stories", "train", "text", "default-text", 10,
        cache_dir=tmp_path / "imported",
    )

    assert item["name"] == "default-text-2"
    assert list_datasets(db)[0]["name"] == "default-text"
    assert list_datasets(db)[1]["name"] == "default-text-2"


def test_concurrent_imports_claim_distinct_corpus_paths(monkeypatch, tmp_path):
    barrier = threading.Barrier(2)

    def fake_load(source, split, **kwargs):
        barrier.wait(timeout=5)
        return [{"text": f"content from {source}"}]

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(load_dataset=fake_load),
    )
    db = ResultsDB(tmp_path / "results.db")

    def run(source):
        return import_hf_text_dataset(
            db, source, "train", "text", "shared", 10,
            cache_dir=tmp_path / "imported",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        items = list(pool.map(run, ("fake/one", "fake/two")))

    assert len({item["name"] for item in items}) == 2
    assert len({item["corpus_path"] for item in items}) == 2
    for item in items:
        assert Path(item["corpus_path"]).read_text(encoding="utf-8") == (
            f"content from {item['source']}"
        )


def test_lab_dataset_migration_is_additive_and_repeatable(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as con:
        con.execute(
            "CREATE TABLE lab_datasets ("
            "name TEXT PRIMARY KEY, source_type TEXT NOT NULL, source TEXT NOT NULL, "
            "split TEXT, text_column TEXT, corpus_path TEXT NOT NULL, n_rows INTEGER, "
            "n_chars INTEGER, preview TEXT, ts TEXT NOT NULL)"
        )
        con.execute(
            "INSERT INTO lab_datasets VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                "legacy", "huggingface", "fake/legacy", "train", "text",
                "data/imported/legacy.txt", 3, 12, "old", "2026-01-01T00:00:00",
            ),
        )

    store = ResultsDB(path)
    ResultsDB(path)  # repeatable migration
    with sqlite3.connect(path) as con:
        columns = {row[1] for row in con.execute("PRAGMA table_info(lab_datasets)")}
        count = con.execute("SELECT COUNT(*) FROM lab_datasets").fetchone()[0]
    assert {
        "requested_revision", "resolved_fingerprint", "row_limit",
        "char_limit", "byte_limit", "rows_examined", "n_bytes", "sha256",
        "truncated", "truncation_reason", "warnings_json",
    } <= columns
    assert count == 1
    legacy = store.get_lab_dataset("legacy")
    assert legacy["n_chars"] == 12
    assert legacy["sha256"] is None
    assert legacy["truncated"] is False
    assert legacy["warnings"] == []


def test_dataset_import_api_forwards_provenance_controls(monkeypatch, tmp_path):
    monkeypatch.setenv("QLLM_DB", str(tmp_path / "api.db"))
    monkeypatch.setenv("QLLM_DATA", str(tmp_path / "data"))
    monkeypatch.delitem(sys.modules, "qllm.dashboard.server", raising=False)
    server = importlib.import_module("qllm.dashboard.server")
    captured = {}

    def fake_import(store, **kwargs):
        captured.update(kwargs)
        return {"name": "api-import", **kwargs}

    monkeypatch.setattr(server, "import_hf_text_dataset", fake_import)
    response = server.api_import_hf({
        "source": "fake/stories",
        "split": "validation",
        "text_column": "body",
        "display_name": "api-import",
        "row_limit": "12",
        "revision": "commit-1",
        "char_limit": "3456",
        "byte_limit": "7890",
    })
    assert response["name"] == "api-import"
    assert captured["revision"] == "commit-1"
    assert captured["row_limit"] == "12"
    assert captured["char_limit"] == "3456"
    assert captured["byte_limit"] == "7890"
    assert captured["cache_dir"] == tmp_path / "data" / "imported"
    choices = server.api_config_choices()
    assert choices["attention"] == [
        "classical", "quantum_proj", "quantum_qkv",
    ]
    assert choices["backend"] == ["pennylane", "tensorcircuit"]
    assert choices["quantum_architecture"] == [
        "qrnn", "contextual_qrnn", "routed_contextual",
    ]
    assert choices["quantum_default"]["ansatz"] == "reuploading"


def test_hf_import_validation_missing_column_and_empty_text(monkeypatch, tmp_path):
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


def test_queue_rejects_invalid_candidate_before_job_insert(tmp_path):
    db_path = tmp_path / "results.db"
    q = ExperimentQueue(str(db_path), start_worker=False)
    with pytest.raises(ValueError, match="train.seq_len must be <="):
        q.submit(
            "classical-small", "default-text", "invalid", 0, 2, 1,
            seq_len=256,
        )
    assert ResultsDB(db_path).fetch_lab_jobs() == []


def test_queue_rejects_invalid_analogue_before_job_insert(monkeypatch, tmp_path):
    db_path = tmp_path / "results.db"
    q = ExperimentQueue(str(db_path), start_worker=False)
    invalid = ExperimentConfig(
        model=ModelConfig(max_seq_len=8),
        train=dataclasses.replace(ExperimentConfig().train, seq_len=64),
    )
    analogue = AnalogueSpec(
        kind="classical_analogue",
        analogue_type="component_swap",
        resolver="test",
        label="Invalid analogue",
        reason="validation fixture",
        config=invalid,
    )
    monkeypatch.setattr(q, "_analogue_for_source", lambda *_args, **_kwargs: analogue)
    with pytest.raises(ValueError, match="Invalid classical analogue config"):
        q.submit(
            "quantum-ffn-4q", "default-text", "invalid-pair", 0, 2, 1,
            queue_classical_comparison=True,
        )
    assert ResultsDB(db_path).fetch_lab_jobs() == []


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
    q = ExperimentQueue(
        str(db_path), start_worker=False, results_dir=tmp_path / "artifacts"
    )
    job = q.submit("classical-small", "default-text", "testable", 0, 2, 1)
    db = ResultsDB(db_path)
    db.update_lab_job(job["id"], status="done")

    out_dir = Path(job["artifact_dir"])
    out_dir.mkdir(parents=True)
    (out_dir / "summary.json").write_text(
        '{"val_ppl": 2.5, "steps": 2, "n_params": 128, "wall_seconds": 1.2}'
    )

    payload = model_test_payload(db, job["id"], tmp_path / "artifacts")
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

    db.update_lab_job(job["id"], status="done")
    db.update_lab_job(twin["id"], status="done")
    db.record(
        "lab", job["preset_id"], "default-text", 3, 4,
        101, 1.0, 2.0, 1.2, 2.0, config=job["config"],
        run_uuid=job["run_uuid"], experiment_uuid=job["experiment_uuid"],
    )
    db.record(
        "lab", twin["preset_id"], "default-text", 3, 4,
        100, 1.1, 2.2, 1.3, 1.0, config=twin["config"],
        run_uuid=twin["run_uuid"], experiment_uuid=twin["experiment_uuid"],
    )
    comparison = comparison_research_payload(db, job["id"])
    assert comparison["claim_id"] is None
    assert comparison["fairness"]["valid"] is True
    assert comparison["evidence_ladder"]["label"] == "unassigned smoke result"


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
        run_uuid=first["run_uuid"], experiment_uuid=first["experiment_uuid"],
    )

    overview = scaling_tests_overview(db)
    assert overview[0]["group_id"] == sweep["group_id"]
    assert overview[0]["qubits"] == [4, 8]

    payload = scaling_test_payload(db, sweep["group_id"])
    assert payload["available"] is True
    assert payload["complete_count"] == 1
    assert payload["best"]["val_ppl"] == pytest.approx(3.0)
    assert {point["n_qubits"] for point in payload["points"]} == {4, 8}


def test_scaling_test_does_not_select_historical_two_stream_points(tmp_path):
    db_path = tmp_path / "results.db"
    queue = ExperimentQueue(str(db_path), start_worker=False)
    sweep = queue.submit_scaling_sweep(
        "two-stream-quantum-bias", "default-text", "historical-scale",
        0, 2, 1, "cpu", qubits=[2, 3], depths=[1],
    )
    db = ResultsDB(db_path)
    first = sweep["jobs"][0]
    config = dict(first["config"])
    config.pop("lab.two_stream_protocol")
    variant = f"two-stream-quantum-bias-q{config['lab.quantum_override.n_qubits']}-d1"
    db.update_lab_job(
        first["id"], status="done",
        run_key=f"lab/{variant}/default-text/0/2",
        config=config,
    )
    db.record(
        "lab", variant, "default-text", 0, 2,
        20, 0.5, 1.0, 0.7, 1.0, config=config,
        run_uuid=first["run_uuid"], experiment_uuid=first["experiment_uuid"],
    )

    payload = scaling_test_payload(db, sweep["group_id"])
    point = next(item for item in payload["points"] if item["job"]["id"] == first["id"])
    assert point["rerun_required"] is True
    assert payload["protocol_warnings"]
    assert payload["best"] is None
    assert payload["complete_count"] == 0


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
                run_uuid=row["run_uuid"],
                experiment_uuid=row["experiment_uuid"],
        )

    payload = study_payload(db, study["id"])
    assert payload["evidence"]["label"] == "paired smoke only"
    assert payload["evidence"]["assessment_status"] == "pilot_only"
    assert payload["evidence"]["fair_pairs"] == 3
    assert payload["evidence"]["wins"] == 3
    assert payload["evidence"]["mean_delta_val_ppl"] == pytest.approx(-0.5)
    assert payload["evidence"]["std_delta_val_ppl"] == pytest.approx(0.0)
    ladder = {item["key"]: item for item in payload["evidence"]["ladder"]}
    assert ladder["multi_seed"]["ok"] is False
    assert ladder["candidate_better"]["ok"] is False
    assert ladder["ablation_supported"]["ok"] is False


def test_study_report_payload_surfaces_verdict_cost_and_limitations(tmp_path):
    db_path = tmp_path / "results.db"
    db = ResultsDB(db_path)
    q = ExperimentQueue(str(db_path), start_worker=False)
    study = create_study(db, q, {
        "name": "report-study",
        "research_question": "Does the quantum FFN retain a repeated edge across seeds?",
        "task": "Language modelling",
        "dataset_names": ["default-text"],
        "candidate_preset_id": "quantum-ffn-4q",
        "seeds": [0, 1, 2],
        "steps": 2,
        "eval_every": 1,
        "sweep": {"qubits": [4], "depths": [2]},
    })
    for row in db.fetch_study_jobs(study["id"]):
        db.update_lab_job(row["id"], status="done")
        val_ppl = 2.0 if row["role"] == "candidate" else 2.6
        wall = 6.0 if row["role"] == "candidate" else 3.0
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
                wall,
                config=row.get("config") or {},
                run_uuid=row["run_uuid"],
                experiment_uuid=row["experiment_uuid"],
        )

    report = study_report_payload(db, study["id"])
    assert report["verdict"]["label"] == "paired smoke only"
    assert report["statistics"]["fair_pairs"] == 3
    assert report["statistics"]["wins"] == 3
    assert report["statistics"]["aggregate_available"] is True
    assert report["resource_summary"]["candidate"]["completed_jobs"] == 3
    assert report["candidate"]["label"]
    assert report["pair_rows"][0]["comparison_link"].startswith("/comparisons/")
    assert report["pair_rows"][0]["metric_type"] == "validation_perplexity"
    assert "## Verdict" in report["markdown"]
    assert any("Ablation-supported evidence" in item for item in report["limitations"])


def test_study_excludes_historical_two_stream_pairs_from_evidence(tmp_path):
    db_path = tmp_path / "results.db"
    db = ResultsDB(db_path)
    queue = ExperimentQueue(str(db_path), start_worker=False)
    study = create_study(db, queue, {
        "name": "historical-two-stream",
        "task": "Language modelling",
        "dataset_names": ["default-text"],
        "candidate_preset_id": "two-stream-quantum-bias",
        "seeds": [0],
        "steps": 2,
        "eval_every": 1,
    })
    for row in db.fetch_study_jobs(study["id"]):
        config = dict(enrich_job(row, db).get("config") or {})
        config.pop("lab.two_stream_protocol")
        db.update_lab_job(row["id"], status="done", config=config)
        val_ppl = 1.0 if row["role"] == "candidate" else 2.0
        db.record(
            "lab", row["preset_id"], row["dataset_name"],
            int(row["seed"]), int(row["steps"]), 100, 0.5,
            val_ppl, 0.7, 1.0, config=config,
            run_uuid=row["run_uuid"], experiment_uuid=row["experiment_uuid"],
        )

    payload = study_payload(db, study["id"])
    assert payload["evidence"]["label"] == "rerun required"
    assert payload["evidence"]["fair_pairs"] == 0
    assert payload["evidence"]["wins"] == 0
    assert payload["evidence"]["rerun_required_pairs"] == 1
    assert payload["assessment_status"] == "rerun_required"
    assert payload["evidence"]["comparisons"][0]["fair"] is False

    report = study_report_payload(db, study["id"])
    assert report["verdict"]["label"] == "rerun required"
    assert report["statistics"]["rerun_required_pairs"] == 1
    assert report["pair_rows"][0]["rerun_required"] is True
    assert report["pair_rows"][0]["fair"] is False
    assert report["assessment_status"] == "rerun_required"
    assert any("causal rerun" in item for item in report["limitations"])


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
    db.record(
        "lab", "quantum-ffn-4q", "default-text", 5, 2,
        10, 1.0, 2.5, 1.4, 4.0,
        run_uuid=job["run_uuid"], experiment_uuid=job["experiment_uuid"],
    )
    db.record(
        "lab", "classical-small", "default-text", 5, 2,
        12, 1.2, 3.0, 1.7, 2.5,
        run_uuid=twin["run_uuid"], experiment_uuid=twin["experiment_uuid"],
    )

    done = workspace_payload(db, job["id"])
    assert done["final_run"]["val_ppl"] == pytest.approx(2.5)
    assert "id" in done["final_run"]
    assert "id" in done["comparison"]["candidate"]["final_run"]
    assert "id" in done["comparison"]["baseline"]["final_run"]
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
    claimed_run = next(run for run in payload["runs"] if run["variant"] == "quantum-ffn-4q")
    assert claimed_run["claim_id"] == "variational_component_swaps"
    assert claimed_run["metric_type"] == "validation_perplexity"
    assert claimed_run["seed_axes"]["initialization"] == 7

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
    db.record(
        "lab", "quantum-ffn-4q", "default-text", 8, 2,
        100, 1.0, 2.2, 1.2, 5.0, config=job["config"],
        run_uuid=job["run_uuid"], experiment_uuid=job["experiment_uuid"],
    )
    db.record(
        "lab", "classical-small", "default-text", 8, 2,
        120, 1.1, 2.6, 1.4, 3.0, config=twin["config"],
        run_uuid=twin["run_uuid"], experiment_uuid=twin["experiment_uuid"],
    )

    payload = result_dashboard_payload(db, dataset="default-text")
    assert payload["available"] is True
    assert any(card["label"] == "Champion model overall" and card["model"] == "quantum-ffn-4q" for card in payload["summaries"])
    qrow = next(row for row in payload["rows"] if row["model"] == "quantum-ffn-4q")
    assert qrow["resource"]["n_qubits"] == 4
    assert qrow["resource"]["resource_band"] in {"low", "medium", "high", "extreme", "classical"}
    assert qrow["verdict_label"] == "single-run candidate better"
    assert qrow["claim_level"] == "anecdote"
    assert qrow["claim_id"] == "variational_component_swaps"
    assert qrow["metric_type"] == "validation_perplexity"
    assert qrow["seed_axes"]["initialization"] == 8
    assert qrow["comparison_link"] == f"/comparisons/{job['id']}"

    task_payload = result_dashboard_payload(db, task_slug="language-modelling", domain_slug="qnlp")
    assert task_payload["available"] is True
    assert "Language modelling" in task_payload["tasks"]


def test_historical_two_stream_rows_are_visible_but_never_promoted(tmp_path):
    db = ResultsDB(tmp_path / "results.db")
    db.record(
        "two-stream-v1", "quantum-bias", "default-text", 0, 100,
        10, 0.5, 1.1, 0.7, 1.0,
    )
    db.record(
        "strict-current", "classical-control", "default-text", 0, 100,
        12, 0.8, 2.2, 1.1, 0.5,
    )

    historical_suite = next(
        row for row in suites_overview(db) if row["suite"] == "two-stream-v1"
    )
    assert historical_suite["best_ppl"] is None
    assert historical_suite["historical_best_ppl"] == pytest.approx(1.1)
    assert historical_suite["metric_contract"]["rerun_required"] is True

    detail = suite_detail(db, "two-stream-v1")
    assert detail["leaderboard"][0]["val_ppl_mean"] == pytest.approx(1.1)
    assert detail["metric_contract"]["metric_type"] == "teacher_forced_side_information"

    historical_run = db.fetch("two-stream-v1")[0]
    run = run_detail(db, historical_run["id"])
    assert run["metric_contract"]["protocol_status"] == "rerun_required"

    payload = result_dashboard_payload(db, dataset="default-text")
    historical_row = next(
        row for row in payload["rows"] if row["model"] == "quantum-bias"
    )
    assert historical_row["rerun_required"] is True
    assert historical_row["metric_type"] == "teacher_forced_side_information"
    assert payload["protocol_warnings"]
    champion = next(
        card for card in payload["summaries"]
        if card["label"] == "Champion model overall"
    )
    assert champion["model"] == "classical-control"

    explored = explore_payload(db)
    dataset = next(row for row in explored["datasets"] if row["name"] == "default-text")
    assert dataset["best_val_ppl"] == pytest.approx(2.2)

    overview = lab_overview(db, {"gpu": {"ready": False}})
    assert [row["variant"] for row in overview["leaderboard_highlights"]] == [
        "classical-control"
    ]


def test_new_two_stream_dashboard_jobs_record_the_causal_protocol(tmp_path):
    queue = ExperimentQueue(str(tmp_path / "results.db"), start_worker=False)
    job = queue.submit(
        "two-stream-classical-bias", "default-text", "causal", 0, 2, 1,
    )
    assert job["config"]["lab.two_stream_protocol"] == "causal_prefix_v2"


def test_unmarked_historical_two_stream_comparison_cannot_gain_a_verdict(tmp_path):
    db_path = tmp_path / "results.db"
    queue = ExperimentQueue(str(db_path), start_worker=False)
    candidate = queue.submit(
        "two-stream-quantum-bias", "default-text", "historical", 0, 2, 1,
        queue_classical_comparison=True,
    )
    baseline = candidate["comparison_job"]
    db = ResultsDB(db_path)

    for job, ppl in ((candidate, 1.0), (baseline, 2.0)):
        config = dict(job["config"])
        config.pop("lab.two_stream_protocol")
        db.update_lab_job(job["id"], status="done", config=config)
        db.record(
            "lab", job["preset_id"], "default-text", 0, 2,
            10, ppl / 2, ppl, ppl / 3, 1.0,
            config=config,
            run_uuid=job["run_uuid"], experiment_uuid=job["experiment_uuid"],
        )

    raw = comparison_payload(db, candidate["id"])
    assert raw["metric_contract"]["rerun_required"] is True

    payload = comparison_research_payload(db, candidate["id"])
    assert payload["available"] is True
    assert payload["deltas"]["val_ppl"] == pytest.approx(-1.0)
    assert payload["metric_contract"]["rerun_required"] is True
    assert payload["verdict"]["label"] == "rerun required"
    assert payload["verdict"]["claim_level"] == "invalid"
    assert payload["evidence_ladder"]["label"] == "rerun required"
    assert next(
        step for step in payload["evidence_ladder"]["steps"]
        if step["key"] == "fair_protocol"
    )["ok"] is False


def _complete_study_jobs(db: ResultsDB, study_id: int) -> None:
    for row in db.fetch_study_jobs(study_id):
        variant = ExperimentQueue._variant_for_job(row)
        run_key = (
            f"lab/{variant}/{row['dataset_name']}/{row['seed']}/{row['steps']}"
        )
        db.update_lab_job(row["id"], status="done", run_key=run_key)
        role = row["role"]
        val_ppl = 2.0 if role == "candidate" else 2.5 if role == "baseline" else 2.4
        wall = 5.0 if role == "candidate" else 3.0
        db.record(
            "lab", variant, row["dataset_name"], int(row["seed"]),
            int(row["steps"]), 100, 1.0, val_ppl, 1.2, wall,
            config=row.get("config") or {},
            run_uuid=row["run_uuid"], experiment_uuid=row["experiment_uuid"],
        )


def test_legacy_study_protocol_infers_additive_claim_and_metric_fields(tmp_path):
    db_path = tmp_path / "results.db"
    db = ResultsDB(db_path)
    queue = ExperimentQueue(str(db_path), start_worker=False)
    study = create_study(db, queue, {
        "name": "legacy-protocol",
        "candidate_preset_id": "quantum-ffn-4q",
        "dataset_names": ["default-text"],
        "seeds": [0, 1, 2],
        "steps": 2,
        "eval_every": 1,
    })
    stored = dict(db.get_study(study["id"])["protocol"])
    for key in ("claim_id", "claim", "metric_type", "analysis_settings", "seed_axes"):
        stored.pop(key, None)
    db.update_study(study["id"], protocol=stored)
    _complete_study_jobs(db, study["id"])

    payload = study_payload(db, study["id"])
    assert "claim_id" not in payload["protocol"]
    assert payload["resolved_protocol"]["claim_id"] == "variational_component_swaps"
    assert payload["claim_id"] == "variational_component_swaps"
    assert payload["metric_type"] == "validation_perplexity"
    assert payload["evidence"]["label"] == "paired smoke only"
    assert payload["evidence"]["mixed_claim_ids"] is False
    assert payload["evidence"]["mixed_metric_types"] is False
    report = study_report_payload(db, study["id"])
    assert all(
        row["metric_type"] == "validation_perplexity"
        for row in report["pair_rows"]
    )


def test_claim_api_and_preset_analogue_use_canonical_contract(monkeypatch, tmp_path):
    monkeypatch.setenv("QLLM_DB", str(tmp_path / "api.db"))
    server = importlib.import_module("qllm.dashboard.server")
    claims = server.api_claims()
    assert len(claims) == 19
    assert server.api_claim("variational_component_swaps")["status"] == "contradicted"
    with pytest.raises(Exception) as exc_info:
        server.api_claim("missing-claim")
    assert getattr(exc_info.value, "status_code", None) == 404

    qffn = next(row for row in list_presets() if row["id"] == "quantum-ffn-4q")
    assert qffn["classical_analogue"]["fairness_requirements"] == list(
        DEFAULT_FAIRNESS_REQUIREMENTS
    )


def test_claim_metric_and_seed_axes_propagate_to_pairs_controls_and_sweeps(tmp_path):
    db_path = tmp_path / "results.db"
    db = ResultsDB(db_path)
    queue = ExperimentQueue(str(db_path), start_worker=False)
    study = create_study(db, queue, {
        "name": "lineage",
        "candidate_preset_id": "quantum-ffn-4q",
        "control_preset_ids": ["classical-small"],
        "dataset_names": ["default-text"],
        "seeds": [5],
        "steps": 2,
        "eval_every": 1,
        "seed_axes": {"initialization": 5},
    })
    jobs = study["jobs"]
    assert {job["claim_id"] for job in jobs} == {"variational_component_swaps"}
    assert {job["metric_type"] for job in jobs} == {"validation_perplexity"}
    for job in jobs:
        axes = job["seed_axes"]
        assert axes["requested"]["initialization"] == 5
        assert axes["initialization"] == axes["minibatch"] == 5
        if job["study_role"] == "candidate":
            assert axes["circuit"] == 5
        else:
            assert axes["circuit"] is None
    observed = study["seed_axes"]["observed"]
    assert len(observed) == 3
    assert {row["role"] for row in observed} == {"candidate", "baseline", "control"}

    sweep = queue.submit_scaling_sweep(
        "quantum-ffn-4q", "default-text", "claim-scale", 5, 2, 1,
        "cpu", qubits=[3, 4], depths=[1],
        claim_id="variational_component_swaps",
        seed_axes={"initialization": 5},
        metric_type="validation_perplexity",
    )
    assert all(
        job["config"]["research.claim_id"] == "variational_component_swaps"
        and job["config"]["research.metric_type"] == "validation_perplexity"
        and job["config"]["research.seed_axes"]["requested"]["initialization"] == 5
        for job in sweep["jobs"]
    )


def test_supported_explicit_seed_request_keeps_pair_fair(tmp_path):
    db_path = tmp_path / "results.db"
    queue = ExperimentQueue(str(db_path), start_worker=False)
    candidate = queue.submit(
        "quantum-ffn-4q", "default-text", "explicit-seed", 5, 2, 1,
        queue_classical_comparison=True,
        seed_axes={"initialization": 5},
    )
    baseline = candidate["comparison_job"]
    db = ResultsDB(db_path)
    for job, ppl in ((candidate, 2.0), (baseline, 2.5)):
        db.update_lab_job(job["id"], status="done")
        db.record(
            "lab", job["preset_id"], "default-text", 5, 2,
            100, 1.0, ppl, 1.2, 2.0, config=job["config"],
            run_uuid=job["run_uuid"], experiment_uuid=job["experiment_uuid"],
        )
    payload = comparison_research_payload(db, candidate["id"])
    assert payload["fairness"]["valid"] is True
    assert payload["fairness"]["seed_axes"]["candidate"]["requested"] == {
        "initialization": 5
    }
    assert payload["fairness"]["seed_axes"]["baseline"]["requested"] == {
        "initialization": 5
    }

    db.record(
        "legacy", "classical-legacy", "default-text", 9, 2,
        10, 1.0, 2.0, 1.2, 1.0,
        config={"model.ffn_type": "classical", "data.kind": "text"},
    )
    legacy = run_detail(db, db.fetch("legacy")[0]["id"])
    assert legacy["seed_axes"]["circuit"] is None


def test_unsupported_claim_metric_is_never_relabelled_as_perplexity(tmp_path):
    db_path = tmp_path / "results.db"
    queue = ExperimentQueue(str(db_path), start_worker=False)
    candidate = queue.submit(
        "qrnn-small", "default-text", "qrnn-metric", 0, 2, 1,
        queue_classical_comparison=True,
    )
    baseline = candidate["comparison_job"]
    db = ResultsDB(db_path)
    for job, ppl in ((candidate, 2.0), (baseline, 2.5)):
        db.update_lab_job(job["id"], status="done")
        db.record(
            "lab", job["preset_id"], "default-text", 0, 2,
            100, 1.0, ppl, 1.2, 2.0, config=job["config"],
            run_uuid=job["run_uuid"], experiment_uuid=job["experiment_uuid"],
        )
    payload = comparison_research_payload(db, candidate["id"])
    assert payload["metric_type"] == "time_to_target"
    assert payload["verdict"]["label"] == "unsupported metric"
    assert payload["assessment_status"] == "unsupported"
    with pytest.raises(ValueError, match="does not support metric_type 'time_to_target'"):
        create_study(db, queue, {
            "name": "unsupported-qrnn",
            "candidate_preset_id": "qrnn-small",
            "dataset_names": ["default-text"],
        })


def test_sweep_cells_are_independent_and_never_pooled(tmp_path):
    db_path = tmp_path / "results.db"
    db = ResultsDB(db_path)
    queue = ExperimentQueue(str(db_path), start_worker=False)
    study = create_study(db, queue, {
        "name": "three-cells-one-seed",
        "candidate_preset_id": "quantum-ffn-4q",
        "dataset_names": ["default-text"],
        "seeds": [0],
        "steps": 2,
        "eval_every": 1,
        "sweep": {"qubits": [3, 4, 5], "depths": [1]},
    })
    _complete_study_jobs(db, study["id"])
    payload = study_payload(db, study["id"])
    assert payload["evidence"]["label"] == "multiple analysis cells"
    assert payload["evidence"]["fair_pairs"] == 3
    assert payload["evidence"]["aggregate_available"] is False
    assert payload["evidence"]["wins"] == 0
    assert payload["evidence"]["mean_delta_val_ppl"] is None
    assert len(payload["analyses"]) == 3
    assert all(row["independent_pairs"] == 1 for row in payload["analyses"])
    assert all(row["paired_stats"]["n_pairs"] == 1 for row in payload["analyses"])
    assert len({job["run_key"] for job in payload["jobs"]}) == 6

    report = study_report_payload(db, study["id"])
    assert report["statistics"]["aggregate_available"] is False
    assert report["statistics"]["win_rate"] is None


def test_study_controls_are_cell_matched_and_missing_baseline_is_reported(tmp_path):
    db_path = tmp_path / "results.db"
    db = ResultsDB(db_path)
    queue = ExperimentQueue(str(db_path), start_worker=False)
    study = create_study(db, queue, {
        "name": "matched-control",
        "candidate_preset_id": "quantum-ffn-4q",
        "control_preset_ids": ["classical-small"],
        "dataset_names": ["default-text"],
        "seeds": [0],
        "steps": 2,
        "eval_every": 1,
    })
    _complete_study_jobs(db, study["id"])
    payload = study_payload(db, study["id"])
    rungs = {
        row["id"]: row
        for row in payload["analyses"][0]["analogue_ladder"]["rungs"]
    }
    assert rungs["strong_classical_challenger"]["status"] == "met"
    assert rungs["frozen_random_control"]["status"] == "unknown"
    assert rungs["resource_accounting"]["status"] == "unknown"

    missing = create_study(db, queue, {
        "name": "missing-baseline",
        "candidate_preset_id": "quantum-ffn-4q",
        "baseline_policy": "none",
        "dataset_names": ["default-text"],
        "seeds": [1],
        "steps": 2,
        "eval_every": 1,
    })
    missing_payload = study_payload(db, missing["id"])
    assert missing_payload["evidence"]["label"] == "incomplete"
    assert missing_payload["fairness_mismatch_count"] > 0
    assert any(
        row["path"] == "comparison.pair"
        for row in missing_payload["fairness_mismatches"]
    )
