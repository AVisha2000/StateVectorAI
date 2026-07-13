from __future__ import annotations

import sqlite3

import pytest

from qllm.research_ledger import LiteratureObservation
from qllm.resultsdb import ResultsDB


def _observation(
    *,
    version: int | None = 2,
    title: str = "A quantum learning paper",
    updated: str = "2024-01-03T00:00:00Z",
    discovery_topic: str = "qml",
) -> LiteratureObservation:
    version_suffix = f"v{version}" if version is not None else ""
    return LiteratureObservation(
        source="arxiv",
        external_id="2401.12345",
        discovery_topic=discovery_topic,
        version=version,
        title=title,
        abstract="A bounded metadata-only abstract.",
        authors=("Ada Lovelace", "Grace Hopper"),
        categories=("quant-ph", "cs.LG"),
        published="2024-01-02T00:00:00Z",
        updated=updated,
        source_url=f"https://arxiv.org/abs/2401.12345{version_suffix}",
    )


def test_observation_normalization_hash_and_validation() -> None:
    observation = _observation()
    normalized = LiteratureObservation(
        source=" ARXIV ",
        external_id=" 2401.12345 ",
        discovery_topic=" qml ",
        version=2,
        title="A  quantum\nlearning paper",
        abstract="A bounded metadata-only abstract.",
        authors=(" Ada Lovelace ", "Grace Hopper"),
        categories=("quant-ph", "cs.LG"),
        published="2024-01-02T00:00:00Z",
        updated="2024-01-03T00:00:00Z",
        source_url="https://arxiv.org/abs/2401.12345v2",
    )

    assert normalized.content_hash == observation.content_hash
    assert "discovery_topic" not in observation.metadata
    with pytest.raises(ValueError, match="source"):
        _observation().__class__(
            source="doi",
            external_id="10.1000/example",
            discovery_topic="qml",
            version=1,
            title="Example",
            abstract="Example metadata.",
            authors=(),
            categories=(),
            published="2024-01-01T00:00:00Z",
            updated="2024-01-01T00:00:00Z",
            source_url="https://example.invalid",
        )
    with pytest.raises(ValueError, match="version"):
        _observation(version=True)  # type: ignore[arg-type]


def test_literature_ledger_is_idempotent_version_aware_and_restart_safe(tmp_path) -> None:
    path = tmp_path / "research.db"
    store = ResultsDB(path)
    version_two = _observation(version=2)

    assert store.upsert_literature_observations([version_two]).inserted_papers == 1
    duplicate = ResultsDB(path).upsert_literature_observations([version_two])
    assert duplicate.inserted_papers == 0
    assert duplicate.inserted_observations == 0
    assert duplicate.existing_observations == 1

    version_three = _observation(
        version=3,
        title="Version three",
        updated="2024-01-04T00:00:00Z",
    )
    assert store.upsert_literature_observations([version_three]).inserted_observations == 1
    assert store.list_literature_papers()[0]["version"] == 3

    lower_version = _observation(
        version=1,
        title="Older source version",
        updated="2025-01-01T00:00:00Z",
    )
    unversioned = _observation(
        version=None,
        title="Unversioned source metadata",
        updated="2025-02-01T00:00:00Z",
    )
    store.upsert_literature_observations([lower_version, unversioned])
    current = store.list_literature_papers()[0]
    assert current["version"] == 3
    assert current["title"] == "Version three"

    same_version_newer_metadata = _observation(
        version=3,
        title="Version three corrected metadata",
        updated="2024-01-05T00:00:00Z",
    )
    same_metadata_other_topic = _observation(
        version=3,
        title="Version three corrected metadata",
        updated="2024-01-05T00:00:00Z",
        discovery_topic="quant-ph",
    )
    store.upsert_literature_observations(
        [same_version_newer_metadata, same_metadata_other_topic]
    )

    reopened = ResultsDB(path)
    papers = reopened.list_literature_papers(limit=1)
    assert reopened.count_literature_papers() == 1
    assert papers[0]["version"] == 3
    assert papers[0]["title"] == "Version three corrected metadata"
    assert papers[0]["observation_count"] == 6
    with sqlite3.connect(path) as con:
        rows = con.execute(
            "SELECT discovery_topic FROM literature_observations ORDER BY id"
        ).fetchall()
    assert [row[0] for row in rows] == [
        "qml",
        "qml",
        "qml",
        "qml",
        "qml",
        "quant-ph",
    ]


def test_same_version_same_timestamp_metadata_is_history_not_freshness(tmp_path) -> None:
    store = ResultsDB(tmp_path / "research.db")
    current = _observation(
        version=3,
        title="Current source metadata",
        updated="2024-01-05T00:00:00Z",
    )
    ambiguous = _observation(
        version=3,
        title="Different metadata with no newer source timestamp",
        updated="2024-01-05T00:00:00Z",
    )
    store.upsert_literature_observations([current, ambiguous])

    paper = store.list_literature_papers(limit=1)[0]
    assert paper["title"] == "Current source metadata"
    assert paper["observation_count"] == 2


def test_literature_schema_is_additive_to_an_existing_database(tmp_path) -> None:
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as con:
        con.execute(
            "CREATE TABLE research_scan_usage ("
            "source TEXT NOT NULL, day_utc TEXT NOT NULL, reserved_items INTEGER NOT NULL, "
            "updated_ts TEXT NOT NULL, PRIMARY KEY(source, day_utc))"
        )
        con.execute(
            "INSERT INTO research_scan_usage VALUES ('arxiv', '2026-07-13', 7, 'old')"
        )

    ResultsDB(path)
    with sqlite3.connect(path) as con:
        tables = {
            row[0]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        retained = con.execute(
            "SELECT reserved_items FROM research_scan_usage"
        ).fetchone()[0]
    assert {"literature_papers", "literature_observations"} <= tables
    assert retained == 7


def test_ledger_validates_all_items_before_writing_a_transaction(tmp_path) -> None:
    store = ResultsDB(tmp_path / "research.db")
    with pytest.raises(ValueError, match="LiteratureObservation"):
        store.upsert_literature_observations([_observation(), object()])  # type: ignore[list-item]
    assert store.count_literature_papers() == 0
