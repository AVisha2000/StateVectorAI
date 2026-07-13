from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from qllm.dashboard.atlas import AtlasOntologyError, atlas_ontology_response


ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY_PATH = ROOT / "docs" / "ATLAS_ONTOLOGY.yaml"
RESEARCH_MAP_PATH = ROOT / "docs" / "RESEARCH_MAP.yaml"


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_yaml(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _cells(payload: dict) -> list[dict]:
    return [cell for domain in payload["domains"] for cell in domain["cells"]]


def _contains_forbidden_score_key(value) -> bool:
    forbidden = {
        "advantage_score",
        "composite_score",
        "composite_advantage_score",
    }
    if isinstance(value, dict):
        return any(
            str(key).casefold() in forbidden
            or _contains_forbidden_score_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_score_key(item) for item in value)
    return False


def test_canonical_atlas_covers_the_research_map_without_reclassifying_it():
    payload = atlas_ontology_response().model_dump()
    research_map = _yaml(RESEARCH_MAP_PATH)
    map_areas = {area["id"]: area for area in research_map["areas"]}
    cells = _cells(payload)

    assert payload["source"] == "backend-canonical"
    assert len(payload["domains"]) == 6
    assert len(cells) == 19
    assert len({cell["id"] for cell in cells}) == 19
    assert {cell["area_id"] for cell in cells} == set(map_areas)
    assert payload["claim_levels"] == research_map["claim_levels"]
    assert payload["replication_statuses"] == research_map["replication_statuses"]
    assert _contains_forbidden_score_key(payload) is False

    by_area = {cell["area_id"]: cell for cell in cells}
    for area_id, area in map_areas.items():
        cell = by_area[area_id]
        assert cell["label"] == area["label"]
        assert cell["seed_status"] == area["status"]
        assert cell["seed_claim_level"] == area["claim_level"]
        assert cell["seed_replication_status"] == area["replication_status"]
        assert cell["pipeline_stage"] in area["pipeline_stages"]
        assert cell["quantum_resource"] in area["quantum_resources"]
        assert cell["advantage_target"] in area["advantage_targets"]

    expected_relations = {
        (
            by_area[relation["from"]]["id"],
            by_area[relation["to"]]["id"],
            relation["type"],
        )
        for relation in research_map["relations"]
    }
    actual_relations = {
        (relation["from_cell"], relation["to_cell"], relation["type"])
        for relation in payload["relations"]
    }
    assert actual_relations == expected_relations


@pytest.mark.parametrize(
    "forbidden_field",
    ["status", "claim_level", "replication_status", "seed_status", "label"],
)
def test_grouping_metadata_cannot_override_map_owned_fields(
    tmp_path,
    forbidden_field,
):
    ontology = _yaml(ONTOLOGY_PATH)
    ontology["domains"][0]["cells"][0][forbidden_field] = "promoted"
    path = _write_yaml(tmp_path / "ontology.yaml", ontology)

    with pytest.raises(AtlasOntologyError, match="unsupported fields"):
        atlas_ontology_response(path, RESEARCH_MAP_PATH)


def test_atlas_rejects_missing_duplicate_and_unknown_area_coverage(tmp_path):
    missing = _yaml(ONTOLOGY_PATH)
    missing["domains"][-1]["cells"].pop()
    with pytest.raises(AtlasOntologyError, match="coverage must be exact"):
        atlas_ontology_response(
            _write_yaml(tmp_path / "missing.yaml", missing),
            RESEARCH_MAP_PATH,
        )

    duplicate = _yaml(ONTOLOGY_PATH)
    duplicate["domains"][-1]["cells"][-1] = {
        **duplicate["domains"][0]["cells"][0],
        "id": "c_duplicate_area",
    }
    with pytest.raises(AtlasOntologyError, match="more than one cell"):
        atlas_ontology_response(
            _write_yaml(tmp_path / "duplicate.yaml", duplicate),
            RESEARCH_MAP_PATH,
        )

    unknown = _yaml(ONTOLOGY_PATH)
    unknown["domains"][0]["cells"][0]["area_id"] = "unknown_area"
    with pytest.raises(AtlasOntologyError, match="unknown area"):
        atlas_ontology_response(
            _write_yaml(tmp_path / "unknown.yaml", unknown),
            RESEARCH_MAP_PATH,
        )


def test_atlas_primary_dimensions_must_be_declared_by_the_map_area(tmp_path):
    ontology = _yaml(ONTOLOGY_PATH)
    ontology["domains"][0]["cells"][0]["pipeline_stage"] = "generation"

    with pytest.raises(AtlasOntologyError, match="not declared"):
        atlas_ontology_response(
            _write_yaml(tmp_path / "dimension.yaml", ontology),
            RESEARCH_MAP_PATH,
        )


def test_atlas_rejects_unvalidated_kinds_and_verdict_bindings(tmp_path):
    invalid_kind = _yaml(ONTOLOGY_PATH)
    invalid_kind["domains"][0]["cells"][0]["kind"] = "advantage"
    with pytest.raises(AtlasOntologyError, match="kind must be one of"):
        atlas_ontology_response(
            _write_yaml(tmp_path / "invalid-kind.yaml", invalid_kind),
            RESEARCH_MAP_PATH,
        )

    unbound_verdict = _yaml(ONTOLOGY_PATH)
    unbound_verdict["domains"][0]["cells"][0]["verdict_ref"] = {
        "verdict_key": "comparison:unverified"
    }
    with pytest.raises(AtlasOntologyError, match="not supported"):
        atlas_ontology_response(
            _write_yaml(tmp_path / "unbound-verdict.yaml", unbound_verdict),
            RESEARCH_MAP_PATH,
        )


def test_map_evidence_fields_and_relations_remain_live_sources(tmp_path):
    research_map = _yaml(RESEARCH_MAP_PATH)
    first = research_map["areas"][0]
    first["status"] = "partial"
    first["claim_level"] = "mechanism"
    first["replication_status"] = "none"
    map_path = _write_yaml(tmp_path / "research-map.yaml", research_map)

    payload = atlas_ontology_response(ONTOLOGY_PATH, map_path).model_dump()
    cell = next(
        item for item in _cells(payload) if item["area_id"] == first["id"]
    )
    assert cell["seed_status"] == "partial"
    assert cell["seed_claim_level"] == "mechanism"
    assert cell["seed_replication_status"] == "none"

    research_map["relations"][0]["from"] = "missing_area"
    broken_map = _write_yaml(tmp_path / "broken-map.yaml", research_map)
    with pytest.raises(AtlasOntologyError, match="Relation endpoint"):
        atlas_ontology_response(ONTOLOGY_PATH, broken_map)


def test_atlas_http_contract_is_typed_and_sanitizes_config_errors(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("QLLM_DB", str(tmp_path / "atlas.db"))
    monkeypatch.setenv("QLLM_RESULTS", str(tmp_path / "results"))
    monkeypatch.setenv("QLLM_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("QLLM_DISABLE_WORKER", "1")
    monkeypatch.delitem(sys.modules, "qllm.dashboard.server", raising=False)
    server = importlib.import_module("qllm.dashboard.server")
    client = TestClient(server.app)

    response = client.get("/api/atlas/ontology")
    assert response.status_code == 200
    assert len(_cells(response.json())) == 19
    assert response.json()["source"] == "backend-canonical"

    def invalid_ontology():
        raise AtlasOntologyError("private filesystem detail")

    monkeypatch.setattr(server, "atlas_ontology_response", invalid_ontology)
    failure = client.get("/api/atlas/ontology")
    assert failure.status_code == 500
    assert failure.json() == {
        "detail": "Atlas ontology configuration is invalid."
    }
    assert "private filesystem detail" not in failure.text
    server.QUEUE.close()
