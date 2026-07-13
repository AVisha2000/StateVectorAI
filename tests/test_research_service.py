from __future__ import annotations

from dataclasses import FrozenInstanceError
import importlib
import sys
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.testclient import TestClient

from qllm.dashboard.research import (
    ArxivScanRequest,
    ResultsDBDailyScanQuota,
    build_research_service,
    capabilities_response,
    library_response,
    scan_response,
)
from qllm.research_service import (
    ARXIV_API_ENDPOINT,
    ArxivResearchService,
    DAILY_QUOTA_LIMIT,
    MAX_RESPONSE_BYTES,
    REQUEST_TIMEOUT_SECONDS,
    ResearchQuotaExceeded,
    ResearchServiceError,
    ScanRequest,
    USER_AGENT,
)
from qllm.resultsdb import ResultsDB


ATOM = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v2</id>
    <title> A   quantum\n learning paper </title>
    <summary> First   abstract\n line. </summary>
    <author><name> Ada   Lovelace </name></author>
    <author><name> Grace Hopper </name></author>
    <category term="quant-ph"/><category term="cs.LG"/>
    <published>2024-01-02T00:00:00Z</published>
    <updated>2024-01-03T00:00:00Z</updated>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2401.12345v1</id>
    <title> Duplicate </title><summary> Duplicate </summary>
    <published>2024-01-02T00:00:00Z</published>
    <updated>2024-01-03T00:00:00Z</updated>
  </entry>
</feed>"""


class Quota:
    def __init__(self, used: int = 0) -> None:
        self.used = used
        self.calls: list[int] = []

    def reserve(self, requested_items: int) -> tuple[int, int]:
        self.calls.append(requested_items)
        if self.used + requested_items > DAILY_QUOTA_LIMIT:
            raise ResearchServiceError("daily scan quota exhausted")
        self.used += requested_items
        return self.used, DAILY_QUOTA_LIMIT - self.used


class Response:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.read_sizes: list[int] = []

    def __enter__(self) -> Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int) -> bytes:
        self.read_sizes.append(size)
        return self.payload[:size]


def test_quant_ph_request_is_bounded_and_metadata_is_normalized_and_deduplicated() -> None:
    quota = Quota()
    response = Response(ATOM)
    calls: list[tuple[object, float]] = []

    def opener(request: object, *, timeout: float) -> Response:
        calls.append((request, timeout))
        return response

    service = ArxivResearchService(quota, opener=opener)
    result = service.scan(ScanRequest(topic="quant-ph", max_results=2))

    assert len(calls) == 1
    request, timeout = calls[0]
    assert timeout == REQUEST_TIMEOUT_SECONDS
    assert request.get_method() == "GET"
    assert request.get_header("User-agent") == USER_AGENT
    assert request.full_url == (
        "https://export.arxiv.org/api/query?search_query=cat%3Aquant-ph&start=0"
        "&max_results=2&sort_by=submittedDate&sort_order=descending"
    )
    split = urlsplit(request.full_url)
    assert f"{split.scheme}://{split.netloc}{split.path}" == ARXIV_API_ENDPOINT
    assert parse_qs(split.query) == {
        "search_query": ["cat:quant-ph"],
        "start": ["0"],
        "max_results": ["2"],
        "sort_by": ["submittedDate"],
        "sort_order": ["descending"],
    }
    assert response.read_sizes == [MAX_RESPONSE_BYTES + 1]
    assert result.quota_used == 2
    assert result.quota_remaining == 48
    assert len(result.papers) == 1
    assert result.papers[0].arxiv_id == "2401.12345"
    assert result.papers[0].version == 2
    assert result.papers[0].title == "A quantum learning paper"
    assert result.papers[0].abstract == "First abstract line."
    assert result.papers[0].authors == ("Ada Lovelace", "Grace Hopper")
    assert result.papers[0].categories == ("quant-ph", "cs.LG")
    assert result.papers[0].abs_url == "https://arxiv.org/abs/2401.12345v2"


def test_qml_query_requires_cs_lg_and_fixed_quantum_ml_keywords() -> None:
    quota = Quota()
    observed: list[str] = []

    def opener(request: object, *, timeout: float) -> Response:
        observed.append(request.full_url)
        return Response(ATOM)

    ArxivResearchService(quota, opener=opener).scan(ScanRequest(topic="qml", max_results=1))
    search_query = parse_qs(urlsplit(observed[0]).query)["search_query"][0]
    assert search_query.startswith("cat:cs.LG AND ")
    assert 'all:"quantum machine learning"' in search_query
    assert "cat:quant-ph" not in search_query


@pytest.mark.parametrize("scan_request", [
    ScanRequest("other", 1), ScanRequest("quant-ph", 0), ScanRequest("quant-ph", 26),
    ScanRequest("quant-ph", True),
])
def test_invalid_topics_and_result_limits_are_rejected_before_quota(scan_request: ScanRequest) -> None:
    quota = Quota()
    with pytest.raises(ValueError):
        ArxivResearchService(quota, opener=lambda *_args, **_kwargs: pytest.fail("opened")).scan(scan_request)
    assert quota.calls == []


def test_exhausted_quota_prevents_opener_call() -> None:
    quota = Quota(used=50)
    opened = False

    def opener(*args: object, **kwargs: object) -> Response:
        nonlocal opened
        opened = True
        return Response(ATOM)

    with pytest.raises(ResearchServiceError, match="exhausted"):
        ArxivResearchService(quota, opener=opener).scan(ScanRequest("quant-ph", 1))
    assert not opened


def test_malformed_and_oversized_responses_are_explicit_errors() -> None:
    with pytest.raises(ResearchServiceError, match="malformed"):
        ArxivResearchService(Quota(), opener=lambda *_args, **_kwargs: Response(b"<feed")).scan(ScanRequest("quant-ph", 1))
    with pytest.raises(ResearchServiceError, match="exceeded"):
        ArxivResearchService(
            Quota(), opener=lambda *_args, **_kwargs: Response(b"x" * (MAX_RESPONSE_BYTES + 1))
        ).scan(ScanRequest("quant-ph", 1))


def test_network_failures_are_wrapped() -> None:
    def opener(*args: object, **kwargs: object) -> Response:
        raise OSError("offline")

    with pytest.raises(ResearchServiceError, match="metadata request failed"):
        ArxivResearchService(Quota(), opener=opener).scan(ScanRequest("quant-ph", 1))


def test_courtesy_interval_is_deterministic_and_shared_by_service_instance() -> None:
    clock = [100.0]
    sleeps: list[float] = []

    def monotonic() -> float:
        return clock[0]

    def sleep(delay: float) -> None:
        sleeps.append(delay)
        clock[0] += delay

    service = ArxivResearchService(
        Quota(),
        opener=lambda *_args, **_kwargs: Response(ATOM),
        monotonic=monotonic,
        sleep=sleep,
    )
    service.scan(ScanRequest("quant-ph", 1))
    clock[0] += 1.25
    service.scan(ScanRequest("quant-ph", 1))
    assert sleeps == [pytest.approx(1.75)]


def test_capabilities_mark_the_d4_boundary_and_dataclasses_are_immutable() -> None:
    result = ArxivResearchService(Quota(), opener=lambda *_args, **_kwargs: Response(ATOM)).scan(
        ScanRequest("quant-ph", 1)
    )
    capabilities = result.capabilities
    assert capabilities.metadata_only
    assert not capabilities.full_text
    assert capabilities.unreviewed_preprints
    assert not capabilities.claim_evidence_classification
    assert capabilities.human_review_required
    assert not capabilities.paid_services_enabled
    assert capabilities.daily_cost_budget is None
    assert capabilities.llm_provider is None
    assert capabilities.embedding_provider is None
    assert capabilities.vector_store_provider is None
    assert capabilities.graph_store_provider is None
    assert capabilities.d4_human_gate_open
    with pytest.raises(FrozenInstanceError):
        result.request.topic = "qml"  # type: ignore[misc]


def test_parsed_results_are_capped_even_if_upstream_returns_extra_entries() -> None:
    extra = b"""
  <entry>
    <id>http://arxiv.org/abs/2401.54321v1</id>
    <title>Second paper</title><summary>Second abstract</summary>
    <published>2024-01-04T00:00:00Z</published>
    <updated>2024-01-04T00:00:00Z</updated>
  </entry>
"""
    payload = ATOM.replace(b"</feed>", extra + b"</feed>")
    result = ArxivResearchService(
        Quota(), opener=lambda *_args, **_kwargs: Response(payload)
    ).scan(ScanRequest("quant-ph", 1))
    assert len(result.papers) == 1


def test_sqlite_daily_quota_persists_across_adapters_and_resets_by_day(
    tmp_path,
) -> None:
    path = tmp_path / "research.db"

    def factory() -> ResultsDB:
        return ResultsDB(path)

    first = ResultsDBDailyScanQuota(factory, day_factory=lambda: "2026-07-11")
    second = ResultsDBDailyScanQuota(factory, day_factory=lambda: "2026-07-11")

    assert first.reserve(25) == (25, 25)
    assert second.reserve(25) == (50, 0)
    with pytest.raises(ResearchQuotaExceeded, match="quota exhausted"):
        first.reserve(1)

    next_day = ResultsDBDailyScanQuota(
        factory, day_factory=lambda: "2026-07-12"
    )
    assert next_day.reserve(1) == (1, 49)


def test_dashboard_research_projection_is_typed_and_offline(tmp_path) -> None:
    path = tmp_path / "research.db"
    service = build_research_service(
        lambda: ResultsDB(path),
        opener=lambda *_args, **_kwargs: Response(ATOM),
    )

    capabilities = capabilities_response(service)
    response = scan_response(
        service,
        ArxivScanRequest(topic="qml", max_results=1),
        database=ResultsDB(path),
    )

    assert capabilities.d4_human_gate_open
    assert response.request.topic == "qml"
    assert response.quota_used == 1
    assert response.quota_remaining == 49
    assert response.papers[0].arxiv_id == "2401.12345"
    assert response.ingestion is not None
    assert response.ingestion.inserted_papers == 1
    assert response.ingestion.inserted_observations == 1

    library = library_response(ResultsDB(path), limit=10)
    assert library.total == 1
    assert library.papers[0].external_id == "2401.12345"
    assert library.papers[0].review_state == "inbox"
    assert library.papers[0].evidence_status == "metadata_only"
    assert library.papers[0].observation_count == 1


def test_dashboard_research_routes_persist_and_read_the_local_vault(
    monkeypatch, tmp_path
) -> None:
    path = tmp_path / "server.db"
    monkeypatch.setenv("QLLM_DB", str(path))
    monkeypatch.setenv("QLLM_RESULTS", str(tmp_path / "results"))
    monkeypatch.setenv("QLLM_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("QLLM_DISABLE_WORKER", "1")
    monkeypatch.delitem(sys.modules, "qllm.dashboard.server", raising=False)
    server = importlib.import_module("qllm.dashboard.server")
    opened: list[object] = []

    def opener(request: object, *, timeout: float) -> Response:
        opened.append(request)
        return Response(ATOM)

    server.RESEARCH_SERVICE = build_research_service(
        lambda: ResultsDB(path), opener=opener
    )
    try:
        with TestClient(server.app) as client:
            scanned = client.post(
                "/api/discover/arxiv/scan",
                json={"topic": "qml", "max_results": 1},
            )
            assert scanned.status_code == 200
            assert scanned.json()["ingestion"] == {
                "inserted_papers": 1,
                "inserted_observations": 1,
                "existing_observations": 0,
            }

            library = client.get("/api/research/papers?limit=1")
            assert library.status_code == 200
            assert library.json()["total"] == 1
            assert library.json()["papers"][0]["external_id"] == "2401.12345"
            assert len(opened) == 1
    finally:
        server.QUEUE.close()
