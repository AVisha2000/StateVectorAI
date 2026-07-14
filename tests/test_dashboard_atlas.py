from __future__ import annotations

import importlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from qllm.claims import get_claim
from qllm.dashboard.atlas import (
    ATLAS_VERDICT_SOURCE_KIND,
    AtlasOntologyError,
    atlas_verdict_key,
    atlas_ontology_response,
    bind_atlas_verdict_refs,
)
from qllm.dashboard.verdicts import (
    build_verdict_snapshot,
    current_claim_verdict_projections,
    persist_verdict_snapshot,
    verdict_snapshot_list_response,
)
from qllm.resultsdb import ResultsDB


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


def _snapshot(claim_id: str, snapshot_id: int, verdict_key: str) -> dict:
    claim = get_claim(claim_id)
    return {
        "id": snapshot_id,
        "verdict_key": verdict_key,
        "source_kind": "comparison",
        "source_id": str(snapshot_id),
        "claim_id": claim_id,
        "claim_level": claim["level"],
        "claim_status": claim["status"],
        "replication_status": claim["replication_status"],
    }


def test_canonical_atlas_covers_the_research_map_without_reclassifying_it():
    payload = atlas_ontology_response().model_dump()
    research_map = _yaml(RESEARCH_MAP_PATH)
    map_areas = {area["id"]: area for area in research_map["areas"]}
    cells = _cells(payload)

    assert payload["source"] == "backend-canonical"
    assert len(payload["domains"]) == 6
    assert len(cells) == 20
    assert len({cell["id"] for cell in cells}) == 20
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


def test_atlas_binds_only_latest_current_claim_snapshots():
    claim_id = "variational_component_swaps"
    older = _snapshot(claim_id, 4, "comparison:older")
    newest = _snapshot(claim_id, 9, "comparison:newest")
    stale = _snapshot(claim_id, 12, "comparison:stale")
    stale["claim_level"] = (
        "formal" if stale["claim_level"] != "formal" else "untested"
    )
    unknown = _snapshot(claim_id, 13, "comparison:unknown")
    unknown["claim_id"] = "unknown_claim"

    payload = bind_atlas_verdict_refs(
        atlas_ontology_response(),
        [newest, stale, older, unknown, {"id": 20}],
    ).model_dump()

    target = next(
        cell for cell in _cells(payload) if cell["area_id"] == claim_id
    )
    assert target["verdict_ref"] == {
        "verdict_key": atlas_verdict_key(claim_id),
        "source_kind": ATLAS_VERDICT_SOURCE_KIND,
        "source_id": claim_id,
    }
    claim = get_claim(claim_id)
    assert target["seed_claim_level"] == claim["level"]
    assert target["seed_replication_status"] == claim["replication_status"]
    assert sum(cell["verdict_ref"] is not None for cell in _cells(payload)) == 1
    assert _contains_forbidden_score_key(payload) is False


def test_atlas_projection_ref_survives_repeated_reconciliation_windows(tmp_path):
    db = ResultsDB(tmp_path / "atlas-projection.db")
    target_claim = "variational_component_swaps"
    filler_claim = "barren_plateau_scaling"

    def append(index: int, claim_id: str) -> None:
        persist_verdict_snapshot(
            db,
            {
                "verdict_key": f"window:{index}",
                "source_kind": "atlas-window",
                "source_id": str(index),
                "claim_id": claim_id,
            },
        )

    append(1, target_claim)
    for index in range(2, 102):
        append(index, filler_claim)

    bound = bind_atlas_verdict_refs(
        atlas_ontology_response(),
        current_claim_verdict_projections(db),
    ).model_dump()
    emitted = {
        cell["verdict_ref"]["verdict_key"]
        for cell in _cells(bound)
        if cell["verdict_ref"] is not None
    }
    assert atlas_verdict_key(target_claim) in emitted

    for start in (102, 127, 152):
        for index in range(start, start + 25):
            append(index, filler_claim)
        frontend_keys = {
            snapshot.verdict_key
            for snapshot in verdict_snapshot_list_response(db).snapshots
        }
        assert emitted <= frontend_keys


def test_atlas_omits_unmaterialized_legacy_ref_until_repaired(tmp_path):
    db = ResultsDB(tmp_path / "legacy-projection.db")
    claim_id = "variational_component_swaps"
    db.append_verdict_snapshot(
        build_verdict_snapshot(
            {
                "verdict_key": "legacy:source",
                "source_kind": "legacy",
                "source_id": "source",
                "claim_id": claim_id,
            }
        )
    )

    before = bind_atlas_verdict_refs(
        atlas_ontology_response(),
        current_claim_verdict_projections(db),
    ).model_dump()
    target_before = next(
        cell for cell in _cells(before) if cell["area_id"] == claim_id
    )
    assert target_before["verdict_ref"] is None

    verdicts = verdict_snapshot_list_response(db)
    after = bind_atlas_verdict_refs(
        atlas_ontology_response(),
        current_claim_verdict_projections(db),
    ).model_dump()
    target_after = next(
        cell for cell in _cells(after) if cell["area_id"] == claim_id
    )
    assert target_after["verdict_ref"]["verdict_key"] == atlas_verdict_key(
        claim_id
    )
    assert target_after["verdict_ref"]["verdict_key"] in {
        snapshot.verdict_key for snapshot in verdicts.snapshots
    }


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

    persisted = persist_verdict_snapshot(
        server.db(),
        {
            "source_kind": "comparison",
            "source_id": "atlas-http",
            "claim_id": "variational_component_swaps",
        },
    )

    response = client.get("/api/atlas/ontology")
    assert response.status_code == 200
    assert len(_cells(response.json())) == 20
    assert response.json()["source"] == "backend-canonical"
    bound = next(
        cell
        for cell in _cells(response.json())
        if cell["area_id"] == "variational_component_swaps"
    )
    assert bound["verdict_ref"] == {
        "verdict_key": atlas_verdict_key("variational_component_swaps"),
        "source_kind": ATLAS_VERDICT_SOURCE_KIND,
        "source_id": "variational_component_swaps",
    }
    verdicts = client.get("/api/verdicts")
    assert verdicts.status_code == 200
    assert bound["verdict_ref"]["verdict_key"] in {
        snapshot["verdict_key"] for snapshot in verdicts.json()["snapshots"]
    }
    assert persisted["id"] != verdicts.json()["snapshots"][0]["id"]

    projection_id = verdicts.json()["snapshots"][0]["id"]
    projection = server.db().get_verdict_snapshot(projection_id)
    with server.db()._conn() as con:
        columns = [
            row["name"]
            for row in con.execute("PRAGMA table_info(verdict_snapshots)")
            if row["name"] != "id"
        ]
        poisoned = {column: projection[column] for column in columns}
        evidence = json.loads(poisoned["evidence_json"])
        evidence["claim_projection_source"]["snapshot_id"] = 999999
        poisoned.update(
            {
                "revision": int(projection["revision"]) + 1,
                "content_hash": "d" * 64,
                "evidence_json": json.dumps(evidence),
                "scorecard_json": json.dumps({"forged": True}),
            }
        )
        forged_id = con.execute(
            f"INSERT INTO verdict_snapshots ({', '.join(columns)}) "
            f"VALUES ({', '.join('?' for _ in columns)})",
            [poisoned[column] for column in columns],
        ).lastrowid

    assert client.get(f"/api/verdicts/{forged_id}").status_code == 404
    valid_detail = client.get(f"/api/verdicts/{projection_id}")
    assert valid_detail.status_code == 200
    assert [row["id"] for row in valid_detail.json()["history"]] == [
        projection_id
    ]
    post_poison_list = client.get("/api/verdicts")
    assert post_poison_list.status_code == 200
    assert [
        row["id"] for row in post_poison_list.json()["snapshots"]
    ] == [projection_id]

    real_db = server.db

    class UnavailableDB:
        def list_current_verdict_snapshots_for_claims(self, *args, **kwargs):
            raise sqlite3.OperationalError("database unavailable")

    monkeypatch.setattr(server, "db", lambda: UnavailableDB())
    degraded = client.get("/api/atlas/ontology")
    assert degraded.status_code == 200
    assert all(cell["verdict_ref"] is None for cell in _cells(degraded.json()))
    monkeypatch.setattr(server, "db", real_db)

    def invalid_binding(*args, **kwargs):
        raise ValueError("claim registry unavailable")

    monkeypatch.setattr(server, "bind_atlas_verdict_refs", invalid_binding)
    invalid_binding_response = client.get("/api/atlas/ontology")
    assert invalid_binding_response.status_code == 200
    assert all(
        cell["verdict_ref"] is None
        for cell in _cells(invalid_binding_response.json())
    )

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
