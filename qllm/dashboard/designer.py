"""Registry-backed, side-effect-free circuit validation for the Designer."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..config import ExperimentConfig, ModelConfig, QuantumConfig, validate_config
from ..registry import (
    ANSATZ_TYPES,
    ARCH_TYPES,
    BACKEND_TYPES,
    CIRCUIT_ANSATZ_TYPES,
    QRNN_ONLY_ANSATZ_TYPES,
    READOUT_TYPES,
    choices_text,
)


DESIGNER_SCHEMA_VERSION = 1
MIN_QUBITS = 1
MAX_QUBITS = 12
MIN_CIRCUIT_LAYERS = 1
MAX_CIRCUIT_LAYERS = 8
DESIGNER_ARCHITECTURE_CONTEXTS = tuple(
    value for value in ARCH_TYPES if value == "qrnn"
)

_BACKEND_DEVICES = {
    "pennylane": "default.qubit",
    "tensorcircuit": "statevector",
    "tensorcircuit_mps": "mps",
}


class DesignerCircuitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ansatz: str
    n_qubits: int = Field(ge=MIN_QUBITS, le=MAX_QUBITS)
    n_circuit_layers: int = Field(
        ge=MIN_CIRCUIT_LAYERS,
        le=MAX_CIRCUIT_LAYERS,
    )
    backend: str
    readout: str
    architecture: str | None = None
    mps_max_bond_dimension: int | None = Field(default=None, ge=1)
    trainable_params: int | None = Field(default=None, ge=0)
    entangling_gates: int | None = Field(default=None, ge=0)


class DesignerCircuitSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ansatz: str
    n_qubits: int
    n_circuit_layers: int
    backend: str
    readout: str
    architecture: str | None
    device: str
    diff_method: str
    shots: int | None
    mps_max_bond_dimension: int | None


class DesignerDerivedMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["derived", "unavailable"]
    value: int | None = None
    scope: str
    reason: str | None = None


class DesignerDerivedProperties(BaseModel):
    model_config = ConfigDict(extra="forbid")

    circuit_weight_shape: list[int]
    trainable_circuit_parameters: DesignerDerivedMetric
    readout_features: DesignerDerivedMetric
    entangling_gates: DesignerDerivedMetric


class DesignerClientEstimateReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    supplied: int | None
    authoritative: Literal[False] = False
    matches_derived: bool | None = None


class DesignerCircuitValidationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    valid: Literal[True] = True
    validation_only: Literal[True] = True
    spec: DesignerCircuitSpec
    derived: DesignerDerivedProperties
    ignored_fields: list[str]
    client_estimates: dict[str, DesignerClientEstimateReview]
    warnings: list[str]


class DesignerCircuitChoices(BaseModel):
    model_config = ConfigDict(extra="forbid")

    architecture: list[str]
    circuit_ansatz: list[str]
    qrnn_only_ansatz: list[str]
    backend: list[str]
    readout: list[str]


class DesignerBounds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    minimum: int
    maximum: int


class DesignerCircuitConstraints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_qubits: DesignerBounds
    n_circuit_layers: DesignerBounds
    qrnn_only_ansatz_requires_architecture: str
    tensorcircuit_mps_requires: list[str]


class DesignerCircuitCapabilitiesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    validation_only: Literal[True] = True
    side_effect_free: Literal[True] = True
    client_estimates_authoritative: Literal[False] = False
    choices: DesignerCircuitChoices
    defaults: DesignerCircuitSpec
    constraints: DesignerCircuitConstraints
    warnings: list[str]


def designer_circuit_capabilities() -> DesignerCircuitCapabilitiesResponse:
    default = QuantumConfig()
    return DesignerCircuitCapabilitiesResponse(
        schema_version=DESIGNER_SCHEMA_VERSION,
        choices=DesignerCircuitChoices(
            architecture=list(DESIGNER_ARCHITECTURE_CONTEXTS),
            circuit_ansatz=list(CIRCUIT_ANSATZ_TYPES),
            qrnn_only_ansatz=list(QRNN_ONLY_ANSATZ_TYPES),
            backend=list(BACKEND_TYPES),
            readout=list(READOUT_TYPES),
        ),
        defaults=DesignerCircuitSpec(
            ansatz=default.ansatz,
            n_qubits=default.n_qubits,
            n_circuit_layers=default.n_circuit_layers,
            backend=default.backend,
            readout=default.readout,
            architecture=None,
            device=default.device,
            diff_method=default.diff_method,
            shots=default.shots,
            mps_max_bond_dimension=default.mps_max_bond_dimension,
        ),
        constraints=DesignerCircuitConstraints(
            n_qubits=DesignerBounds(
                minimum=MIN_QUBITS,
                maximum=MAX_QUBITS,
            ),
            n_circuit_layers=DesignerBounds(
                minimum=MIN_CIRCUIT_LAYERS,
                maximum=MAX_CIRCUIT_LAYERS,
            ),
            qrnn_only_ansatz_requires_architecture="qrnn",
            tensorcircuit_mps_requires=["mps_max_bond_dimension"],
        ),
        warnings=[
            "Validation never constructs a circuit, model, backend, job, or device.",
            "Circuit properties and diagnostics are not evidence of quantum advantage.",
        ],
    )


def validate_designer_circuit(
    request: DesignerCircuitRequest,
) -> DesignerCircuitValidationResponse:
    _require_choice("ansatz", request.ansatz, ANSATZ_TYPES)
    _require_choice("backend", request.backend, BACKEND_TYPES)
    _require_choice("readout", request.readout, READOUT_TYPES)
    if request.architecture is not None:
        _require_choice(
            "architecture context",
            request.architecture,
            DESIGNER_ARCHITECTURE_CONTEXTS,
        )
    if (
        request.ansatz in QRNN_ONLY_ANSATZ_TYPES
        and request.architecture != "qrnn"
    ):
        raise ValueError(
            f"ansatz '{request.ansatz}' requires architecture='qrnn'; "
            "the architecture context cannot be inferred by the Designer API."
        )
    is_qrnn = request.architecture == "qrnn"
    if is_qrnn and request.backend != "pennylane":
        raise ValueError(
            "The current QRNN runtime does not dispatch through a selectable "
            "circuit backend; use backend='pennylane' as the compatibility value."
        )
    if is_qrnn and request.readout != "z":
        raise ValueError(
            "The current QRNN runtime emits token probabilities rather than a "
            "selectable expectation-value readout; use readout='z' as the "
            "compatibility value."
        )
    if request.backend == "tensorcircuit_mps":
        if request.mps_max_bond_dimension is None:
            raise ValueError(
                "mps_max_bond_dimension is required for tensorcircuit_mps."
            )
    elif request.mps_max_bond_dimension is not None:
        raise ValueError(
            "mps_max_bond_dimension is supported only for tensorcircuit_mps."
        )

    device = _BACKEND_DEVICES[request.backend]
    qcfg = QuantumConfig(
        n_qubits=request.n_qubits,
        n_circuit_layers=request.n_circuit_layers,
        ansatz=request.ansatz,
        backend=request.backend,
        device=device,
        diff_method="backprop",
        shots=None,
        readout=request.readout,
        mps_max_bond_dimension=request.mps_max_bond_dimension,
    )
    validation_architecture = request.architecture or "transformer"
    errors = validate_config(
        ExperimentConfig(
            model=ModelConfig(
                arch=validation_architecture,
                quantum=qcfg,
            )
        )
    )
    if errors:
        raise ValueError(" ".join(errors))

    layers = request.n_circuit_layers
    qubits = request.n_qubits
    variational_parameters = layers * qubits * 3
    if is_qrnn:
        variational_parameters += layers
    readout_features = None if is_qrnn else (
        qubits if request.readout == "z"
        else qubits + qubits * (qubits - 1) // 2
    )

    warnings = [
        "Validation only: no circuit, model, backend, job, or device was constructed.",
        "Circuit properties and diagnostics are not evidence of quantum advantage.",
    ]
    trainable_review = DesignerClientEstimateReview(
        supplied=request.trainable_params,
        matches_derived=(
            request.trainable_params == variational_parameters
            if request.trainable_params is not None
            else None
        ),
    )
    if trainable_review.matches_derived is False:
        warnings.append(
            "The client trainable_params estimate was ignored because it does not "
            "match the registry-backed circuit parameter shape."
        )
    if request.entangling_gates is not None:
        warnings.append(
            "The client entangling_gates estimate is advisory only; backend gate "
            "decomposition is not represented as an authoritative count."
        )
    ignored_fields: list[str] = []
    if is_qrnn:
        ignored_fields = ["backend", "readout", "device", "diff_method", "shots"]
        warnings.append(
            "The current QRNN runtime distinguishes Ising from CNOT-ring evolution; "
            "hardware_efficient and reuploading both select CNOT-ring evolution."
        )
        warnings.append(
            "QRNN backend, readout, device, differentiation, and shot fields are "
            "compatibility-only and are not execution selectors."
        )
    if request.backend == "tensorcircuit_mps":
        warnings.append(
            "tensorcircuit_mps is approximate and bounded by the supplied maximum "
            "bond dimension; it is not silently treated as an exact statevector."
        )

    return DesignerCircuitValidationResponse(
        schema_version=DESIGNER_SCHEMA_VERSION,
        spec=DesignerCircuitSpec(
            ansatz=request.ansatz,
            n_qubits=qubits,
            n_circuit_layers=layers,
            backend=request.backend,
            readout=request.readout,
            architecture=request.architecture,
            device=device,
            diff_method="backprop",
            shots=None,
            mps_max_bond_dimension=request.mps_max_bond_dimension,
        ),
        derived=DesignerDerivedProperties(
            circuit_weight_shape=[layers, qubits, 3],
            trainable_circuit_parameters=DesignerDerivedMetric(
                status="derived",
                value=variational_parameters,
                scope="Variational circuit parameters only; surrounding model excluded.",
            ),
            readout_features=DesignerDerivedMetric(
                status="unavailable" if is_qrnn else "derived",
                value=readout_features,
                scope=(
                    "QRNN token-probability emission."
                    if is_qrnn
                    else "Per-circuit expectation-value feature width."
                ),
                reason=(
                    "QRNN emits token probabilities from amplitude blocks and does "
                    "not use the configured expectation-value readout."
                    if is_qrnn
                    else None
                ),
            ),
            entangling_gates=DesignerDerivedMetric(
                status="unavailable",
                scope="Backend-level circuit decomposition.",
                reason=(
                    "The canonical config selects an ansatz family, not a stable "
                    "compiled gate list."
                ),
            ),
        ),
        ignored_fields=ignored_fields,
        client_estimates={
            "trainable_params": trainable_review,
            "entangling_gates": DesignerClientEstimateReview(
                supplied=request.entangling_gates,
            ),
        },
        warnings=warnings,
    )


def _require_choice(name: str, value: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(
            f"{name} must be one of: {choices_text(choices)}; got {value!r}."
        )


__all__ = [
    "DesignerCircuitCapabilitiesResponse",
    "DesignerCircuitRequest",
    "DesignerCircuitValidationResponse",
    "designer_circuit_capabilities",
    "validate_designer_circuit",
]
