from __future__ import annotations

import importlib
import sys

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from qllm.dashboard.designer import (
    DesignerCircuitRequest,
    designer_circuit_capabilities,
    validate_designer_circuit,
)
from qllm.registry import (
    BACKEND_TYPES,
    CIRCUIT_ANSATZ_TYPES,
    QRNN_ONLY_ANSATZ_TYPES,
    READOUT_TYPES,
)


def _request(**overrides) -> DesignerCircuitRequest:
    payload = {
        "ansatz": "hardware_efficient",
        "n_qubits": 4,
        "n_circuit_layers": 2,
        "backend": "pennylane",
        "readout": "zz",
    }
    payload.update(overrides)
    return DesignerCircuitRequest(**payload)


def test_capabilities_are_registry_backed_and_side_effect_free():
    payload = designer_circuit_capabilities().model_dump()

    assert payload["validation_only"] is True
    assert payload["side_effect_free"] is True
    assert payload["client_estimates_authoritative"] is False
    assert payload["choices"] == {
        "architecture": ["qrnn"],
        "circuit_ansatz": list(CIRCUIT_ANSATZ_TYPES),
        "qrnn_only_ansatz": list(QRNN_ONLY_ANSATZ_TYPES),
        "backend": list(BACKEND_TYPES),
        "readout": list(READOUT_TYPES),
    }
    assert "all" not in payload["choices"]["readout"]
    assert payload["constraints"]["tensorcircuit_mps_requires"] == [
        "mps_max_bond_dimension"
    ]


def test_validation_recomputes_properties_and_ignores_client_estimates():
    response = validate_designer_circuit(
        _request(trainable_params=8, entangling_gates=6)
    ).model_dump()

    assert response["valid"] is True
    assert response["spec"]["device"] == "default.qubit"
    assert response["derived"]["circuit_weight_shape"] == [2, 4, 3]
    assert response["derived"]["trainable_circuit_parameters"]["value"] == 24
    assert response["derived"]["readout_features"]["value"] == 10
    assert response["derived"]["entangling_gates"]["status"] == "unavailable"
    assert response["client_estimates"]["trainable_params"] == {
        "supplied": 8,
        "authoritative": False,
        "matches_derived": False,
    }
    assert any("ignored" in warning for warning in response["warnings"])
    assert any(
        "not evidence of quantum advantage" in warning
        for warning in response["warnings"]
    )


def test_qrnn_only_ansatz_requires_explicit_architecture_context():
    with pytest.raises(ValueError, match="requires architecture='qrnn'"):
        validate_designer_circuit(_request(ansatz="ising"))

    response = validate_designer_circuit(
        _request(ansatz="ising", architecture="qrnn", readout="z")
    ).model_dump()
    assert response["spec"]["architecture"] == "qrnn"
    assert response["derived"]["trainable_circuit_parameters"]["value"] == 26
    assert response["derived"]["readout_features"]["status"] == "unavailable"
    assert response["derived"]["readout_features"]["value"] is None
    assert {"backend", "readout"} <= set(response["ignored_fields"])
    assert any(
        "not execution selectors" in warning
        for warning in response["warnings"]
    )


def test_non_ising_qrnn_includes_phase_parameters_and_has_no_readout_width():
    response = validate_designer_circuit(
        _request(
            ansatz="hardware_efficient",
            architecture="qrnn",
            readout="z",
        )
    ).model_dump()

    assert response["derived"]["trainable_circuit_parameters"]["value"] == 26
    assert response["derived"]["readout_features"] == {
        "status": "unavailable",
        "value": None,
        "scope": "QRNN token-probability emission.",
        "reason": (
            "QRNN emits token probabilities from amplitude blocks and does not use "
            "the configured expectation-value readout."
        ),
    }
    assert any(
        "both select CNOT-ring evolution" in warning
        for warning in response["warnings"]
    )


@pytest.mark.parametrize(
    "overrides, message",
    [
        (
            {"architecture": "qrnn", "backend": "tensorcircuit"},
            "does not dispatch through a selectable circuit backend",
        ),
        (
            {
                "architecture": "qrnn",
                "backend": "tensorcircuit_mps",
                "mps_max_bond_dimension": 8,
            },
            "does not dispatch through a selectable circuit backend",
        ),
        (
            {"architecture": "qrnn", "readout": "zz"},
            "emits token probabilities",
        ),
    ],
)
def test_qrnn_rejects_execution_selectors_it_cannot_honor(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_designer_circuit(_request(**overrides))


def test_mps_requires_an_explicit_bounded_representation():
    with pytest.raises(ValueError, match="mps_max_bond_dimension is required"):
        validate_designer_circuit(_request(backend="tensorcircuit_mps"))

    response = validate_designer_circuit(
        _request(
            backend="tensorcircuit_mps",
            mps_max_bond_dimension=16,
        )
    ).model_dump()
    assert response["spec"]["device"] == "mps"
    assert response["spec"]["mps_max_bond_dimension"] == 16
    assert any("approximate" in warning for warning in response["warnings"])


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"ansatz": "unknown"}, "ansatz must be one of"),
        ({"backend": "unknown"}, "backend must be one of"),
        ({"readout": "all"}, "readout must be one of"),
        ({"architecture": "unknown"}, "architecture context must be one of"),
        ({"architecture": "transformer"}, "architecture context must be one of"),
        ({"mps_max_bond_dimension": 8}, "supported only for tensorcircuit_mps"),
    ],
)
def test_registry_and_backend_mismatches_fail_closed(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_designer_circuit(_request(**overrides))


@pytest.mark.parametrize(
    "field, value",
    [
        ("n_qubits", 0),
        ("n_qubits", 13),
        ("n_circuit_layers", 0),
        ("n_circuit_layers", 9),
    ],
)
def test_request_bounds_are_typed_and_strict(field, value):
    with pytest.raises(ValidationError):
        _request(**{field: value})


def test_designer_http_contract_serializes_and_maps_errors(monkeypatch, tmp_path):
    monkeypatch.setenv("QLLM_DB", str(tmp_path / "designer.db"))
    monkeypatch.setenv("QLLM_RESULTS", str(tmp_path / "results"))
    monkeypatch.setenv("QLLM_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("QLLM_DISABLE_WORKER", "1")
    monkeypatch.delitem(sys.modules, "qllm.dashboard.server", raising=False)
    server = importlib.import_module("qllm.dashboard.server")
    client = TestClient(server.app)

    capabilities = client.get("/api/designer/circuit")
    assert capabilities.status_code == 200
    assert capabilities.json()["choices"]["backend"] == list(BACKEND_TYPES)

    valid = client.post(
        "/api/designer/circuit",
        json=_request().model_dump(),
    )
    assert valid.status_code == 200
    assert valid.json()["derived"]["trainable_circuit_parameters"]["value"] == 24

    unsupported = client.post(
        "/api/designer/circuit",
        json={**_request().model_dump(), "readout": "all"},
    )
    assert unsupported.status_code == 400
    assert "readout must be one of" in unsupported.json()["detail"]

    extra = client.post(
        "/api/designer/circuit",
        json={**_request().model_dump(), "composite_score": 1},
    )
    assert extra.status_code == 422
    server.QUEUE.close()
