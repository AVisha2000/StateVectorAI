"""Dashboard projections for the bounded metadata-only research service."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Callable, Literal

from pydantic import BaseModel, Field

from ..research_ledger import LiteratureObservation
from ..research_service import (
    DAILY_QUOTA_LIMIT,
    ArxivResearchService,
    ResearchPaper,
    ResearchQuotaExceeded,
    ResearchService,
    ScanRequest,
)
from ..resultsdb import ResultsDB


class ArxivScanRequest(BaseModel):
    topic: Literal["quant-ph", "qml"] = "qml"
    max_results: int = Field(default=10, ge=1, le=25)

    class Config:
        extra = "forbid"


class ResearchPaperResponse(BaseModel):
    arxiv_id: str
    version: int | None
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    published: str
    updated: str
    abs_url: str

    class Config:
        extra = "forbid"


class ResearchIngestionResponse(BaseModel):
    inserted_papers: int = Field(ge=0)
    inserted_observations: int = Field(ge=0)
    existing_observations: int = Field(ge=0)

    class Config:
        extra = "forbid"


class ResearchLibraryPaperResponse(BaseModel):
    id: int
    source: str
    external_id: str
    version: int | None
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    published: str
    updated: str
    source_url: str
    metadata_hash: str
    observation_count: int = Field(ge=1)
    review_state: str
    evidence_status: str
    first_seen_ts: str
    last_seen_ts: str

    class Config:
        extra = "forbid"


class ResearchLibraryResponse(BaseModel):
    papers: list[ResearchLibraryPaperResponse]
    total: int = Field(ge=0)

    class Config:
        extra = "forbid"


class ResearchCapabilitiesResponse(BaseModel):
    metadata_only: bool
    full_text: bool
    unreviewed_preprints: bool
    claim_evidence_classification: bool
    human_review_required: bool
    paid_services_enabled: bool
    daily_cost_budget: float | None
    llm_provider: str | None
    embedding_provider: str | None
    vector_store_provider: str | None
    graph_store_provider: str | None
    d4_human_gate_open: bool

    class Config:
        extra = "forbid"


class ArxivScanResponse(BaseModel):
    request: ArxivScanRequest
    papers: list[ResearchPaperResponse]
    quota_used: int
    quota_remaining: int
    quota_limit: int = DAILY_QUOTA_LIMIT
    capabilities: ResearchCapabilitiesResponse
    ingestion: ResearchIngestionResponse | None = Field(
        default=None,
        description=(
            "Present for dashboard scans after local persistence; null is reserved "
            "for direct in-process helper use without a ResultsDB."
        ),
    )

    class Config:
        extra = "forbid"


def _utc_day() -> str:
    return datetime.now(timezone.utc).date().isoformat()


class ResultsDBDailyScanQuota:
    """Persist the arXiv item cap across requests and dashboard restarts."""

    def __init__(
        self,
        db_factory: Callable[[], ResultsDB],
        *,
        day_factory: Callable[[], str] = _utc_day,
    ) -> None:
        self._db_factory = db_factory
        self._day_factory = day_factory

    def reserve(self, requested_items: int) -> tuple[int, int]:
        try:
            return self._db_factory().reserve_research_scan_quota(
                source="arxiv",
                day_utc=self._day_factory(),
                requested_items=requested_items,
                daily_limit=DAILY_QUOTA_LIMIT,
            )
        except ValueError as exc:
            raise ResearchQuotaExceeded(str(exc)) from exc


def build_research_service(
    db_factory: Callable[[], ResultsDB], **kwargs
) -> ArxivResearchService:
    return ArxivResearchService(ResultsDBDailyScanQuota(db_factory), **kwargs)


def capabilities_response(
    service: ResearchService,
) -> ResearchCapabilitiesResponse:
    return ResearchCapabilitiesResponse(**asdict(service.capabilities))


def _arxiv_observation(
    paper: ResearchPaper, *, discovery_topic: str
) -> LiteratureObservation:
    return LiteratureObservation(
        source="arxiv",
        external_id=paper.arxiv_id,
        discovery_topic=discovery_topic,
        version=paper.version,
        title=paper.title,
        abstract=paper.abstract,
        authors=paper.authors,
        categories=paper.categories,
        published=paper.published,
        updated=paper.updated,
        source_url=paper.abs_url,
    )


def scan_response(
    service: ResearchService,
    request: ArxivScanRequest,
    *,
    database: ResultsDB | None = None,
) -> ArxivScanResponse:
    result = service.scan(
        ScanRequest(topic=request.topic, max_results=request.max_results)
    )
    ingestion = None
    if database is not None:
        persisted = database.upsert_literature_observations(
            _arxiv_observation(paper, discovery_topic=request.topic)
            for paper in result.papers
        )
        ingestion = ResearchIngestionResponse(**asdict(persisted))
    return ArxivScanResponse(
        request=request,
        papers=[ResearchPaperResponse(**asdict(paper)) for paper in result.papers],
        quota_used=result.quota_used,
        quota_remaining=result.quota_remaining,
        capabilities=ResearchCapabilitiesResponse(**asdict(result.capabilities)),
        ingestion=ingestion,
    )


def library_response(database: ResultsDB, *, limit: int = 50) -> ResearchLibraryResponse:
    return ResearchLibraryResponse(
        papers=[
            ResearchLibraryPaperResponse(**paper)
            for paper in database.list_literature_papers(limit=limit)
        ],
        total=database.count_literature_papers(),
    )


__all__ = [
    "ArxivScanRequest",
    "ArxivScanResponse",
    "ResearchCapabilitiesResponse",
    "ResearchIngestionResponse",
    "ResearchLibraryPaperResponse",
    "ResearchLibraryResponse",
    "ResearchPaperResponse",
    "ResultsDBDailyScanQuota",
    "build_research_service",
    "capabilities_response",
    "library_response",
    "scan_response",
]
