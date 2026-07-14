"""Typed dashboard projection of dependency-free registry choices."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class MetricTypeSpecResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lower_is_better: bool
    units: str
    pairable: bool
    extraction_key: str
    comparator_class: str


class QuantumConfigDefaultsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_qubits: int
    n_circuit_layers: int
    ansatz: str
    backend: str
    device: str
    diff_method: str
    shots: int | None
    trainable: bool
    readout: str
    dressing: str
    init_scale: float
    n_circuits: int
    mps_max_bond_dimension: int | None
    mps_max_truncation_error: float | None
    mps_relative_truncation: bool


class PauliTermResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    coefficient: float
    pauli: str
    qubits: list[int]


class ClassicalReferenceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference_id: str
    role: str
    label: str
    energy: float
    method: str
    certified: bool
    certificate: dict[str, Any]
    limitation: str


class GroundStateInstanceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instance_id: str
    label: str
    family: str
    n_qubits: int
    energy_units: str
    shape: list[int]
    terms: list[PauliTermResponse]
    classical_references: list[ClassicalReferenceResponse]
    provenance: dict[str, Any]


class ConfigChoicesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    architecture: list[str]
    quantum_architecture: list[str]
    dataset: list[str]
    task_type: list[str]
    ansatz: list[str]
    circuit_ansatz: list[str]
    backend: list[str]
    readout: list[str]
    dressing: list[str]
    attention: list[str]
    embedding: list[str]
    feed_forward: list[str]
    head: list[str]
    encoder: list[str]
    condition: list[str]
    metric_types: dict[str, MetricTypeSpecResponse]
    ground_state_instances: list[GroundStateInstanceResponse]
    quantum_default: QuantumConfigDefaultsResponse


def validate_config_choices(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and return the JSON-safe API projection."""
    return ConfigChoicesResponse.model_validate(payload).model_dump()
