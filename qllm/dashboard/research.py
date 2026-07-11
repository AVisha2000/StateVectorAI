"""Dashboard projections for the bounded metadata-only research service."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Callable, Literal

from pydantic import BaseModel, Field

from ..research_service import (
    DAILY_QUOTA_LIMIT,
    ArxivResearchService,
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


def scan_response(
    service: ResearchService, request: ArxivScanRequest
) -> ArxivScanResponse:
    result = service.scan(
        ScanRequest(topic=request.topic, max_results=request.max_results)
    )
    return ArxivScanResponse(
        request=request,
        papers=[ResearchPaperResponse(**asdict(paper)) for paper in result.papers],
        quota_used=result.quota_used,
        quota_remaining=result.quota_remaining,
        capabilities=ResearchCapabilitiesResponse(**asdict(result.capabilities)),
    )


__all__ = [
    "ArxivScanRequest",
    "ArxivScanResponse",
    "ResearchCapabilitiesResponse",
    "ResearchPaperResponse",
    "ResultsDBDailyScanQuota",
    "build_research_service",
    "capabilities_response",
    "scan_response",
]
