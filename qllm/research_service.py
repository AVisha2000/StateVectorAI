"""Bounded, dependency-free metadata scanning for the research discovery loop.

This module deliberately stops at D4: it fetches public arXiv Atom metadata,
but does not persist papers, inspect full text, classify evidence, or select an
LLM, embedding, vector, or graph provider.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from threading import Lock
import time
from typing import Callable, Protocol
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree


ARXIV_API_ENDPOINT = "https://export.arxiv.org/api/query"
MAX_RESULTS = 25
DAILY_QUOTA_LIMIT = 50
REQUEST_TIMEOUT_SECONDS = 10.0
COURTESY_INTERVAL_SECONDS = 3.0
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
USER_AGENT = "QLLM-ResearchService/0.1 (+metadata-only; contact: local)"

_ATOM_NS = "http://www.w3.org/2005/Atom"
_TOPIC_QUERIES = {
    "quant-ph": "cat:quant-ph",
    "qml": (
        "cat:cs.LG AND "
        '(all:"quantum machine learning" OR all:"quantum neural network" '
        'OR all:"variational quantum" OR all:"quantum kernel")'
    ),
}


class ResearchServiceError(RuntimeError):
    """Raised when a bounded research-metadata scan cannot be completed."""


class ResearchQuotaExceeded(ResearchServiceError):
    """Raised before network access when the persistent UTC-day cap is full."""


@dataclass(frozen=True)
class ScanRequest:
    """A deliberately small scanner request with no raw query escape hatch."""

    topic: str
    max_results: int = 10


@dataclass(frozen=True)
class ResearchPaper:
    """Normalized arXiv metadata only; this never includes a PDF or full text."""

    arxiv_id: str
    version: int | None
    title: str
    abstract: str
    authors: tuple[str, ...]
    categories: tuple[str, ...]
    published: str
    updated: str
    abs_url: str


@dataclass(frozen=True)
class ResearchCapabilities:
    """Explicitly conservative boundary for the D4 metadata scanner."""

    metadata_only: bool = True
    full_text: bool = False
    unreviewed_preprints: bool = True
    claim_evidence_classification: bool = False
    human_review_required: bool = True
    paid_services_enabled: bool = False
    daily_cost_budget: float | None = None
    llm_provider: str | None = None
    embedding_provider: str | None = None
    vector_store_provider: str | None = None
    graph_store_provider: str | None = None
    d4_human_gate_open: bool = True


@dataclass(frozen=True)
class ScanResult:
    request: ScanRequest
    papers: tuple[ResearchPaper, ...]
    quota_used: int
    quota_remaining: int
    capabilities: ResearchCapabilities


class DailyScanQuota(Protocol):
    """Atomically reserve scan capacity and return ``(used, remaining)``.

    Implementations own UTC-day accounting and must reject a reservation that
    would exceed the fixed 50-item limit by raising ``ResearchServiceError``.
    """

    def reserve(self, requested_items: int) -> tuple[int, int]: ...


class ResearchService(Protocol):
    """Provider-neutral read-only research metadata interface."""

    @property
    def capabilities(self) -> ResearchCapabilities: ...

    def scan(self, request: ScanRequest) -> ScanResult: ...


class ArxivResearchService:
    """Scan the fixed arXiv Atom endpoint within conservative local bounds."""

    def __init__(
        self,
        quota: DailyScanQuota,
        *,
        opener: Callable[..., object] = urlopen,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._quota = quota
        self._opener = opener
        self._monotonic = monotonic
        self._sleep = sleep
        self._last_request_at: float | None = None
        self._courtesy_lock = Lock()
        self._capabilities = ResearchCapabilities()

    @property
    def capabilities(self) -> ResearchCapabilities:
        return self._capabilities

    def scan(self, request: ScanRequest) -> ScanResult:
        self._validate_request(request)
        quota_used, quota_remaining = self._reserve_quota(request.max_results)
        response_bytes = self._fetch(self._build_url(request))
        papers = self._parse_atom(response_bytes)[: request.max_results]
        return ScanResult(
            request=request,
            papers=papers,
            quota_used=quota_used,
            quota_remaining=quota_remaining,
            capabilities=self.capabilities,
        )

    @staticmethod
    def _validate_request(request: ScanRequest) -> None:
        if not isinstance(request, ScanRequest):
            raise ValueError("request must be a ScanRequest")
        if request.topic not in _TOPIC_QUERIES:
            raise ValueError("topic must be one of: quant-ph, qml")
        if (
            isinstance(request.max_results, bool)
            or not isinstance(request.max_results, int)
            or not 1 <= request.max_results <= MAX_RESULTS
        ):
            raise ValueError("max_results must be an integer between 1 and 25")

    @staticmethod
    def _build_url(request: ScanRequest) -> str:
        query = urlencode(
            [
                ("search_query", _TOPIC_QUERIES[request.topic]),
                ("start", 0),
                ("max_results", request.max_results),
                ("sort_by", "submittedDate"),
                ("sort_order", "descending"),
            ]
        )
        return f"{ARXIV_API_ENDPOINT}?{query}"

    def _reserve_quota(self, requested_items: int) -> tuple[int, int]:
        try:
            used, remaining = self._quota.reserve(requested_items)
        except ResearchServiceError:
            raise
        except Exception as exc:
            raise ResearchServiceError(f"daily scan quota reservation failed: {exc}") from exc
        if (
            isinstance(used, bool)
            or isinstance(remaining, bool)
            or not isinstance(used, int)
            or not isinstance(remaining, int)
            or used < 0
            or remaining < 0
            or used > DAILY_QUOTA_LIMIT
            or remaining > DAILY_QUOTA_LIMIT
            or used + remaining != DAILY_QUOTA_LIMIT
        ):
            raise ResearchServiceError("daily scan quota returned an invalid reservation")
        return used, remaining

    def _fetch(self, url: str) -> bytes:
        with self._courtesy_lock:
            if self._last_request_at is not None:
                delay = COURTESY_INTERVAL_SECONDS - (
                    self._monotonic() - self._last_request_at
                )
                if delay > 0:
                    self._sleep(delay)
            self._last_request_at = self._monotonic()
            request = Request(url, headers={"User-Agent": USER_AGENT}, method="GET")
            try:
                response = self._opener(request, timeout=REQUEST_TIMEOUT_SECONDS)
                with response:  # type: ignore[union-attr]
                    body = response.read(MAX_RESPONSE_BYTES + 1)  # type: ignore[union-attr]
            except ResearchServiceError:
                raise
            except Exception as exc:
                raise ResearchServiceError(f"arXiv metadata request failed: {exc}") from exc
        if not isinstance(body, bytes):
            raise ResearchServiceError("arXiv metadata response was not bytes")
        if len(body) > MAX_RESPONSE_BYTES:
            raise ResearchServiceError("arXiv metadata response exceeded 2 MiB limit")
        return body

    @staticmethod
    def _parse_atom(payload: bytes) -> tuple[ResearchPaper, ...]:
        try:
            root = ElementTree.fromstring(payload)
        except (ElementTree.ParseError, ValueError) as exc:
            raise ResearchServiceError(f"malformed arXiv Atom response: {exc}") from exc

        papers: list[ResearchPaper] = []
        seen_ids: set[str] = set()
        for entry in root.findall(f"{{{_ATOM_NS}}}entry"):
            paper = ArxivResearchService._parse_entry(entry)
            if paper.arxiv_id not in seen_ids:
                papers.append(paper)
                seen_ids.add(paper.arxiv_id)
        return tuple(papers)

    @staticmethod
    def _parse_entry(entry: ElementTree.Element) -> ResearchPaper:
        def text(name: str, *, required: bool = True) -> str:
            value = entry.findtext(f"{{{_ATOM_NS}}}{name}")
            normalized = " ".join((value or "").split())
            if required and not normalized:
                raise ResearchServiceError(f"arXiv Atom entry is missing {name}")
            return normalized

        atom_id = text("id")
        marker = "/abs/"
        if marker not in atom_id:
            raise ResearchServiceError("arXiv Atom entry has an invalid id")
        versioned_id = atom_id.rsplit(marker, 1)[1]
        if not versioned_id:
            raise ResearchServiceError("arXiv Atom entry has an empty id")
        matched = re.fullmatch(r"(.+?)(?:v([1-9][0-9]*))?", versioned_id)
        if matched is None:
            raise ResearchServiceError("arXiv Atom entry has an invalid versioned id")
        arxiv_id = matched.group(1)
        version = int(matched.group(2)) if matched.group(2) else None
        authors = tuple(
            value
            for author in entry.findall(f"{{{_ATOM_NS}}}author")
            if (value := " ".join((author.findtext(f"{{{_ATOM_NS}}}name") or "").split()))
        )
        categories = tuple(
            value
            for category in entry.findall(f"{{{_ATOM_NS}}}category")
            if (value := " ".join((category.get("term") or "").split()))
        )
        return ResearchPaper(
            arxiv_id=arxiv_id,
            version=version,
            title=text("title"),
            abstract=text("summary"),
            authors=authors,
            categories=categories,
            published=text("published"),
            updated=text("updated"),
            abs_url=f"https://arxiv.org/abs/{versioned_id}",
        )


__all__ = [
    "ARXIV_API_ENDPOINT",
    "ArxivResearchService",
    "COURTESY_INTERVAL_SECONDS",
    "DAILY_QUOTA_LIMIT",
    "DailyScanQuota",
    "MAX_RESPONSE_BYTES",
    "MAX_RESULTS",
    "REQUEST_TIMEOUT_SECONDS",
    "ResearchCapabilities",
    "ResearchPaper",
    "ResearchQuotaExceeded",
    "ResearchService",
    "ResearchServiceError",
    "ScanRequest",
    "ScanResult",
    "USER_AGENT",
]
