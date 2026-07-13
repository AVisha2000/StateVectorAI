"""Hermetic end-to-end route inventory smoke coverage for the dashboard API."""
from __future__ import annotations

import importlib
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient


@dataclass(frozen=True)
class RouteCase:
    method: str
    route_path: str
    request_path: str
    status_code: int
    payload: dict | None = None


ROUTE_CASES = (
    RouteCase("GET", "/api/health", "/api/health", 200),
    RouteCase("GET", "/api/status", "/api/status", 200),
    RouteCase("GET", "/api/stream/jobs", "/api/stream/jobs", 200),
    RouteCase("GET", "/api/presets", "/api/presets", 200),
    RouteCase("GET", "/api/config/choices", "/api/config/choices", 200),
    RouteCase("GET", "/api/atlas/ontology", "/api/atlas/ontology", 200),
    RouteCase("GET", "/api/designer/circuit", "/api/designer/circuit", 200),
    RouteCase("POST", "/api/designer/circuit", "/api/designer/circuit", 200, {
        "ansatz": "hardware_efficient", "n_qubits": 2, "n_circuit_layers": 1,
        "backend": "pennylane", "readout": "z",
    }),
    RouteCase("GET", "/api/claims", "/api/claims", 200),
    RouteCase("GET", "/api/claims/{claim_id}", "/api/claims/variational_component_swaps", 200),
    RouteCase("GET", "/api/verdicts", "/api/verdicts", 200),
    RouteCase("GET", "/api/verdicts/{verdict_id}", "/api/verdicts/1", 200),
    RouteCase("GET", "/api/presets/{preset_id}/model-graph", "/api/presets/classical-small/model-graph", 200),
    RouteCase("GET", "/api/presets/{preset_id}/classical-analogue", "/api/presets/classical-small/classical-analogue", 200),
    RouteCase("GET", "/api/model-specs", "/api/model-specs", 200),
    RouteCase("POST", "/api/model-specs", "/api/model-specs", 200, {"config": {}}),
    RouteCase("GET", "/api/model-specs/{spec_id}", "/api/model-specs/1", 200),
    RouteCase("PATCH", "/api/model-specs/{spec_id}", "/api/model-specs/1", 200, {"config": {}}),
    RouteCase("POST", "/api/model-specs/validate", "/api/model-specs/validate", 200, {"config": {}}),
    RouteCase("POST", "/api/model-specs/{spec_id}/validate", "/api/model-specs/1/validate", 200),
    RouteCase("GET", "/api/model-specs/{spec_id}/diff", "/api/model-specs/1/diff", 200),
    RouteCase("POST", "/api/model-specs/{spec_id}/jobs", "/api/model-specs/1/jobs", 200, {}),
    RouteCase("GET", "/api/lab/overview", "/api/lab/overview", 200),
    RouteCase("GET", "/api/explore", "/api/explore", 200),
    RouteCase("GET", "/api/explore/domain/{domain_slug}", "/api/explore/domain/qnlp", 200),
    RouteCase("GET", "/api/explore/dataset/{dataset_name}", "/api/explore/dataset/default-text", 200),
    RouteCase("GET", "/api/explore/task/{task_slug}", "/api/explore/task/language-modelling", 200),
    RouteCase("GET", "/api/research/capabilities", "/api/research/capabilities", 200),
    RouteCase("POST", "/api/discover/arxiv/scan", "/api/discover/arxiv/scan", 200, {"topic": "qml", "max_results": 1}),
    RouteCase("GET", "/api/scaling-tests", "/api/scaling-tests", 200),
    RouteCase("GET", "/api/scaling-tests/{group_id}", "/api/scaling-tests/group-1", 200),
    RouteCase("GET", "/api/studies", "/api/studies", 200),
    RouteCase("POST", "/api/studies", "/api/studies", 200, {}),
    RouteCase("GET", "/api/studies/{study_id}", "/api/studies/1", 200),
    RouteCase("GET", "/api/studies/{study_id}/report", "/api/studies/1/report", 200),
    RouteCase("POST", "/api/studies/{study_id}/queue", "/api/studies/1/queue", 200),
    RouteCase("GET", "/api/datasets", "/api/datasets", 200),
    RouteCase("POST", "/api/datasets/hf/import", "/api/datasets/hf/import", 200, {"source": "org/dataset", "text_column": "text"}),
    RouteCase("GET", "/api/jobs", "/api/jobs", 200),
    RouteCase("GET", "/api/jobs/{job_id}", "/api/jobs/1", 200),
    RouteCase("GET", "/api/jobs/{job_id}/diagnostics", "/api/jobs/1/diagnostics", 200),
    RouteCase("POST", "/api/jobs", "/api/jobs", 200, {"preset_id": "classical-small"}),
    RouteCase("POST", "/api/jobs/sweep", "/api/jobs/sweep", 200, {"preset_id": "classical-small"}),
    RouteCase("POST", "/api/jobs/{job_id}/cancel", "/api/jobs/1/cancel", 200),
    RouteCase("GET", "/api/jobs/{job_id}/classical-analogue", "/api/jobs/1/classical-analogue", 200),
    RouteCase("POST", "/api/jobs/{job_id}/classical-analogue", "/api/jobs/1/classical-analogue", 200, {}),
    RouteCase("POST", "/api/groups/{group_id}/classical-analogues", "/api/groups/group-1/classical-analogues", 200),
    RouteCase("GET", "/api/jobs/{job_id}/workspace", "/api/jobs/1/workspace", 200),
    RouteCase("GET", "/api/jobs/{job_id}/comparison", "/api/jobs/1/comparison", 200),
    RouteCase("GET", "/api/jobs/{job_id}/model-graph", "/api/jobs/1/model-graph", 200),
    RouteCase("GET", "/api/jobs/{job_id}/model-tests", "/api/jobs/1/model-tests", 200),
    RouteCase("POST", "/api/jobs/{job_id}/model-tests", "/api/jobs/1/model-tests", 200, {}),
    RouteCase("GET", "/api/suites", "/api/suites", 200),
    RouteCase("GET", "/api/suite/{suite}", "/api/suite/lab", 200),
    RouteCase("GET", "/api/runs", "/api/runs", 200),
    RouteCase("GET", "/api/run/{run_id}", "/api/run/1", 200),
    RouteCase("GET", "/api/live", "/api/live", 200),
    RouteCase("GET", "/api/live/{run_key:path}/curve", "/api/live/test-run/curve", 200),
    RouteCase("GET", "/api/plots", "/api/plots", 200),
    RouteCase("GET", "/api/plot/{name}", "/api/plot/missing.png", 404),
)

EXPECTED_API_ROUTES = {(case.method, case.route_path) for case in ROUTE_CASES}


def _summary() -> dict:
    return {
        "id": 1, "verdict_key": "comparison:1", "revision": 1, "content_hash": "a" * 64,
        "source_kind": "comparison", "source_id": "1", "claim_id": "variational_component_swaps",
        "claim_level": "mechanism", "claim_status": "contradicted", "replication_status": "none",
        "assessment_level": None, "assessment_status": None, "created_ts": "2026-01-01T00:00:00Z",
    }


def _diagnostics() -> dict:
    dimension = {"status": "unavailable", "value": None, "source": "test", "reason": "not persisted", "provenance": {}}
    return {
        "job": {"id": 1, "run_name": "test", "status": "done", "group_id": None},
        "diagnostics": {name: dimension for name in (
            "gradient_variance", "parameter_shift_gradient_snr", "expressibility_kl", "meyer_wallach_q", "scaling_fit"
        )},
        "interpretation_warnings": [{"code": "test", "severity": "info", "title": "Test", "message": "Stub", "evidence": {}}],
    }


def _capabilities(server):
    return server.ResearchCapabilitiesResponse(
        metadata_only=True, full_text=False, unreviewed_preprints=True,
        claim_evidence_classification=False, human_review_required=True,
        paid_services_enabled=False, daily_cost_budget=None, llm_provider=None,
        embedding_provider=None, vector_store_provider=None, graph_store_provider=None,
        d4_human_gate_open=False,
    )


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """Import one isolated dashboard module and close its disabled queue on exit."""
    monkeypatch = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("dashboard-routes")
    monkeypatch.setenv("QLLM_DB", str(root / "qllm.db"))
    monkeypatch.setenv("QLLM_RESULTS", str(root / "results"))
    monkeypatch.setenv("QLLM_DATA", str(root / "data"))
    monkeypatch.setenv("QLLM_DISABLE_WORKER", "1")
    previous = sys.modules.pop("qllm.dashboard.server", None)
    module = importlib.import_module("qllm.dashboard.server")

    async def finite_sse(*_args, **_kwargs) -> AsyncIterator[str]:
        yield "event: jobs\ndata: {}\n\n"

    summary = _summary()
    detail = {**summary, "source_job_id": 1, "source_study_id": None, "source_run_id": None,
              "scorecard": {}, "fairness": {}, "controls": {}, "caveats": [], "evidence": {}, "diagnostics": {}, "schema_version": 1}
    capabilities = _capabilities(module)
    monkeypatch.setattr(module, "environment_status", lambda *_: {"gpu": {"ready": False}})
    monkeypatch.setattr(module, "stream_client_allowed", lambda *_: True)
    monkeypatch.setattr(module, "jobs_event_stream", finite_sse)
    monkeypatch.setattr(module, "verdict_snapshot_list_response", lambda *_args, **_kwargs: module.VerdictSnapshotListResponse(snapshots=[summary]))
    monkeypatch.setattr(module, "verdict_snapshot_detail_response", lambda *_: module.VerdictSnapshotHistoryResponse(snapshot=detail, history=[summary]))
    monkeypatch.setattr(module, "diagnostics_payload", lambda *_args, **_kwargs: _diagnostics())
    monkeypatch.setattr(module, "capabilities_response", lambda *_: capabilities)
    monkeypatch.setattr(module, "scan_response", lambda *_: module.ArxivScanResponse(request=module.ArxivScanRequest(topic="qml", max_results=1), papers=[], quota_used=1, quota_remaining=49, capabilities=capabilities))
    monkeypatch.setattr(module, "list_specs", lambda *_: [{"id": 1}])
    monkeypatch.setattr(module, "get_spec", lambda *_: {"id": 1, "config": {}})
    monkeypatch.setattr(module, "create_spec", lambda *_: {"id": 1})
    monkeypatch.setattr(module, "update_spec", lambda *_: {"id": 1})
    monkeypatch.setattr(module, "validation_payload", lambda *_: {"valid": True})
    monkeypatch.setattr(module, "spec_diff", lambda *_: {"changed": []})
    monkeypatch.setattr(module, "lab_overview", lambda *_: {"ok": True})
    monkeypatch.setattr(module, "explore_payload", lambda *_: {"ok": True})
    monkeypatch.setattr(module, "domain_payload", lambda *_: {"available": True})
    monkeypatch.setattr(module, "result_dashboard_payload", lambda *_args, **_kwargs: {"available": True})
    monkeypatch.setattr(module, "scaling_tests_overview", lambda *_: [])
    monkeypatch.setattr(module, "scaling_test_payload", lambda *_: {"available": True})
    monkeypatch.setattr(module, "list_studies", lambda *_: [])
    monkeypatch.setattr(module, "create_study", lambda *_: {"id": 1})
    monkeypatch.setattr(module, "study_payload", lambda *_: {"id": 1})
    monkeypatch.setattr(module, "study_report_payload", lambda *_: {"id": 1})
    monkeypatch.setattr(module, "queue_study", lambda *_: {"queued": True})
    monkeypatch.setattr(module, "list_datasets", lambda *_: [])
    monkeypatch.setattr(module, "import_hf_text_dataset", lambda *_args, **_kwargs: {"name": "stub"})
    monkeypatch.setattr(module, "workspace_payload", lambda *_: {"id": 1})
    monkeypatch.setattr(module, "comparison_research_payload", lambda *_: {"available": True})
    monkeypatch.setattr(module, "model_graph_from_config", lambda *_: {"nodes": [], "edges": []})
    monkeypatch.setattr(module, "model_test_payload", lambda *_: {"available": True})
    monkeypatch.setattr(module, "run_model_test", lambda *_: {"queued": False})
    monkeypatch.setattr(module.Q, "suites_overview", lambda *_: [])
    monkeypatch.setattr(module.Q, "suite_detail", lambda *_: {"suite": "lab"})
    monkeypatch.setattr(module.Q, "all_runs", lambda *_: [])
    monkeypatch.setattr(module.Q, "run_detail", lambda *_: {"id": 1})
    monkeypatch.setattr(module.Q, "live_runs", lambda *_: [])
    monkeypatch.setattr(module.Q, "live_curve", lambda *_: {"points": []})
    monkeypatch.setattr(module, "enrich_job", lambda job, *_: job)
    monkeypatch.setattr(module.QUEUE, "list", lambda: [])
    monkeypatch.setattr(module.QUEUE, "get", lambda *_: {"id": 1, "config": {}})
    monkeypatch.setattr(module.QUEUE, "worker_status", lambda: "disabled")
    for name, value in {
        "submit": {"id": 1}, "submit_scaling_sweep": {"jobs": []}, "submit_model_spec": {"id": 1},
        "cancel": {"id": 1, "status": "cancelled"}, "classical_analogue_for_job": {"available": True},
        "queue_classical_analogue": {"id": 1}, "queue_classical_analogues_for_group": {"jobs": []},
    }.items():
        monkeypatch.setattr(module.QUEUE, name, lambda *args, _value=value, **kwargs: _value)
    try:
        yield module
    finally:
        module.QUEUE.close()
        sys.modules.pop("qllm.dashboard.server", None)
        if previous is not None:
            sys.modules["qllm.dashboard.server"] = previous
        monkeypatch.undo()


def test_literal_inventory_matches_every_api_route(server):
    actual = {
        (method, route.path)
        for route in server.app.routes
        if route.path.startswith("/api/")
        for method in route.methods - {"HEAD", "OPTIONS"}
    }
    assert actual == EXPECTED_API_ROUTES
    assert len(actual) == 60


@pytest.mark.parametrize("case", ROUTE_CASES, ids=lambda case: f"{case.method} {case.route_path}")
def test_every_api_route_has_an_exact_http_status(server, case):
    with TestClient(server.app) as client:
        response = client.request(case.method, case.request_path, json=case.payload)

    assert response.status_code == case.status_code
    if case.route_path == "/api/status":
        assert response.json() == {"worker": "disabled", "gpu_available": False, "queued": 0, "running": 0, "runs": 0}
    elif case.route_path == "/api/verdicts":
        assert response.json()["snapshots"][0]["id"] == 1
    elif case.route_path == "/api/verdicts/{verdict_id}":
        assert response.json()["snapshot"]["schema_version"] == 1
    elif case.route_path == "/api/jobs/{job_id}/diagnostics":
        assert response.json()["diagnostics"]["gradient_variance"]["status"] == "unavailable"
    elif case.route_path == "/api/research/capabilities":
        assert response.json()["metadata_only"] is True
    elif case.route_path == "/api/discover/arxiv/scan":
        assert response.json()["request"] == {"topic": "qml", "max_results": 1}
