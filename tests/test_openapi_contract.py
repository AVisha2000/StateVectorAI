from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "qllm" / "dashboard" / "openapi.json"


def test_openapi_snapshot_is_current_and_contains_core_api():
    for _ in range(2):
        completed = subprocess.run(
            [sys.executable, "scripts/dump_openapi.py", "--check"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr

    document = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert document["info"]["title"] == "QLLM Dashboard"
    assert "/api/status" in document["paths"]
    assert "/api/stream/jobs" in document["paths"]
    assert "/api/jobs/{job_id}/diagnostics" in document["paths"]
    assert "/api/verdicts" in document["paths"]
    assert "/api/verdicts/{verdict_id}" in document["paths"]
    assert "/api/research/capabilities" in document["paths"]
    assert "/api/discover/arxiv/scan" in document["paths"]
    assert set(document["paths"]["/api/atlas/ontology"]) == {"get"}
    assert set(document["paths"]["/api/designer/circuit"]) == {"get", "post"}
    assert "get" in document["paths"]["/api/jobs/{job_id}"]
    assert "/{full_path}" not in document["paths"]

    status_schema = document["components"]["schemas"]["StatusResponse"]
    assert status_schema["additionalProperties"] is False
    assert set(status_schema["required"]) == {
        "worker",
        "gpu_available",
        "queued",
        "running",
        "runs",
    }
    assert status_schema["properties"]["worker"]["type"] == "string"
    assert status_schema["properties"]["gpu_available"]["type"] == "boolean"
    for key in ("queued", "running", "runs"):
        assert status_schema["properties"][key]["type"] == "integer"

    designer_request = document["components"]["schemas"][
        "DesignerCircuitRequest"
    ]
    assert designer_request["additionalProperties"] is False
    assert set(designer_request["required"]) == {
        "ansatz",
        "n_qubits",
        "n_circuit_layers",
        "backend",
        "readout",
    }

    atlas_cell = document["components"]["schemas"]["AtlasCell"]
    assert atlas_cell["additionalProperties"] is False
    assert {
        "seed_status",
        "seed_claim_level",
        "seed_replication_status",
    } <= set(atlas_cell["required"])
    assert not {
        "advantage_score",
        "composite_score",
        "composite_advantage_score",
    } & set(atlas_cell["properties"])

    stream_response = document["paths"]["/api/stream/jobs"]["get"]["responses"]["200"]
    assert "text/event-stream" in stream_response["content"]

    dimensions = document["components"]["schemas"]["DiagnosticsDimensions"]
    assert dimensions["additionalProperties"] is False
    assert set(dimensions["required"]) == {
        "gradient_variance",
        "parameter_shift_gradient_snr",
        "expressibility_kl",
        "meyer_wallach_q",
        "scaling_fit",
    }

    verdict = document["components"]["schemas"]["VerdictSnapshotSummary"]
    assert verdict["additionalProperties"] is False
    assert {
        "claim_level",
        "claim_status",
        "replication_status",
        "assessment_level",
        "assessment_status",
    }.issubset(verdict["properties"])

    capabilities = document["components"]["schemas"][
        "ResearchCapabilitiesResponse"
    ]
    assert capabilities["additionalProperties"] is False
    assert {
        "metadata_only",
        "full_text",
        "paid_services_enabled",
        "daily_cost_budget",
        "llm_provider",
        "embedding_provider",
        "vector_store_provider",
        "graph_store_provider",
        "d4_human_gate_open",
    }.issubset(capabilities["properties"])
