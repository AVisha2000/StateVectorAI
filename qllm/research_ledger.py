"""Deterministic metadata primitives for the local literature ledger.

These records intentionally describe only source metadata.  They are not paper
reviews, evidence classifications, claim updates, or an LLM-facing knowledge
graph.  The SQLite layer owns retrieval timestamps and immutable persistence.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json


ARXIV_SOURCE = "arxiv"
INBOX_REVIEW_STATE = "inbox"
METADATA_ONLY_EVIDENCE_STATUS = "metadata_only"
METADATA_SCHEMA_VERSION = 1


def canonical_json(value: object) -> str:
    """Return a stable JSON representation suitable for content addressing."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _normalized_text(field: str, value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f"literature {field} must be a string")
    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError(f"literature {field} must be non-empty")
    return normalized


def _normalized_items(field: str, value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError(f"literature {field} must be a list or tuple of strings")
    return tuple(_normalized_text(field, item) for item in value)


@dataclass(frozen=True)
class LiteratureObservation:
    """A validated, content-addressed observation of public paper metadata."""

    source: str
    external_id: str
    discovery_topic: str
    version: int | None
    title: str
    abstract: str
    authors: tuple[str, ...]
    categories: tuple[str, ...]
    published: str
    updated: str
    source_url: str

    def __post_init__(self) -> None:
        source = _normalized_text("source", self.source).casefold()
        if source != ARXIV_SOURCE:
            raise ValueError("literature source must be 'arxiv'")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "external_id", _normalized_text("external_id", self.external_id))
        object.__setattr__(self, "discovery_topic", _normalized_text("discovery_topic", self.discovery_topic))
        if self.version is not None and (
            isinstance(self.version, bool)
            or not isinstance(self.version, int)
            or self.version <= 0
        ):
            raise ValueError("literature version must be a positive integer or null")
        object.__setattr__(self, "title", _normalized_text("title", self.title))
        object.__setattr__(self, "abstract", _normalized_text("abstract", self.abstract))
        object.__setattr__(self, "authors", _normalized_items("authors", self.authors))
        object.__setattr__(self, "categories", _normalized_items("categories", self.categories))
        object.__setattr__(self, "published", _normalized_text("published", self.published))
        object.__setattr__(self, "updated", _normalized_text("updated", self.updated))
        object.__setattr__(self, "source_url", _normalized_text("source_url", self.source_url))

    @property
    def metadata(self) -> dict[str, object]:
        """Canonical source payload; excludes database IDs and retrieval time."""
        return {
            "schema_version": METADATA_SCHEMA_VERSION,
            "source": self.source,
            "external_id": self.external_id,
            "version": self.version,
            "title": self.title,
            "abstract": self.abstract,
            "authors": list(self.authors),
            "categories": list(self.categories),
            "published": self.published,
            "updated": self.updated,
            "source_url": self.source_url,
        }

    @property
    def metadata_json(self) -> str:
        return canonical_json(self.metadata)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.metadata_json.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LiteratureIngestionResult:
    """Counts from one atomic metadata-ingestion transaction."""

    inserted_papers: int
    inserted_observations: int
    existing_observations: int


__all__ = [
    "ARXIV_SOURCE",
    "INBOX_REVIEW_STATE",
    "LiteratureIngestionResult",
    "LiteratureObservation",
    "METADATA_ONLY_EVIDENCE_STATUS",
    "METADATA_SCHEMA_VERSION",
    "canonical_json",
]
