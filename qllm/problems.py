"""Dependency-free registered ground-state problem definitions."""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


_REGISTRY_KEY = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_REFERENCE_ROLES = frozenset({"oracle", "descriptive_challenger"})


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _freeze_payload(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_payload(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_payload(item) for item in value)
    return value


def _mutable_payload(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _mutable_payload(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_mutable_payload(item) for item in value]
    return value


@dataclass(frozen=True)
class PauliTerm:
    coefficient: float
    pauli: str
    qubits: tuple[int, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(self.coefficient):
            raise ValueError("Pauli-term coefficients must be finite.")
        if not self.pauli or any(operator not in "IXYZ" for operator in self.pauli):
            raise ValueError("Pauli terms may contain only I, X, Y, and Z.")
        if len(self.pauli) != len(self.qubits):
            raise ValueError("Pauli operators and qubit indices must have equal length.")
        if any(isinstance(qubit, bool) or not isinstance(qubit, int) for qubit in self.qubits):
            raise ValueError("Pauli-term qubit indices must be integers.")
        if len(set(self.qubits)) != len(self.qubits):
            raise ValueError("Pauli terms may not repeat a qubit index.")

    def to_payload(self) -> dict[str, object]:
        return {
            "coefficient": self.coefficient,
            "pauli": self.pauli,
            "qubits": list(self.qubits),
        }

    payload = to_payload


@dataclass(frozen=True)
class ClassicalReference:
    reference_id: str
    role: str
    label: str
    energy: float
    method: str
    certified: bool
    certificate: Mapping[str, object]
    limitation: str

    def __post_init__(self) -> None:
        if not _REGISTRY_KEY.fullmatch(self.reference_id):
            raise ValueError("Classical-reference IDs must be registry keys.")
        if self.role not in _REFERENCE_ROLES:
            raise ValueError(
                "Classical-reference role must be 'oracle' or "
                "'descriptive_challenger'."
            )
        if not math.isfinite(self.energy):
            raise ValueError("Classical-reference energies must be finite.")
        if not self.method.strip() or not self.limitation.strip():
            raise ValueError("Classical references require method and limitation text.")
        if not isinstance(self.certificate, Mapping):
            raise ValueError("Classical-reference certificates must be mappings.")
        if self.certified and not self.certificate:
            raise ValueError(
                "Certified classical references require reproducible certificate metadata."
            )
        frozen_certificate = _freeze_payload(self.certificate)
        _hash(_mutable_payload(frozen_certificate))
        object.__setattr__(self, "certificate", frozen_certificate)

    def to_payload(self) -> dict[str, object]:
        return {
            "reference_id": self.reference_id,
            "role": self.role,
            "label": self.label,
            "energy": self.energy,
            "method": self.method,
            "certified": self.certified,
            "certificate": _mutable_payload(self.certificate),
            "limitation": self.limitation,
        }

    payload = to_payload

    @property
    def name(self) -> str:
        return self.reference_id

    @property
    def limitations(self) -> str:
        return self.limitation


@dataclass(frozen=True)
class GroundStateInstance:
    instance_id: str
    label: str
    family: str
    n_qubits: int
    terms: tuple[PauliTerm, ...]
    energy_units: str
    classical_references: tuple[ClassicalReference, ...]
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        if not _REGISTRY_KEY.fullmatch(self.instance_id):
            raise ValueError("Ground-state instance IDs must be registry keys.")
        if isinstance(self.n_qubits, bool) or not isinstance(self.n_qubits, int):
            raise ValueError("Ground-state n_qubits must be an integer.")
        if self.n_qubits < 1:
            raise ValueError("Ground-state n_qubits must be positive.")
        if not self.terms:
            raise ValueError("Ground-state instances require Hamiltonian terms.")
        for term in self.terms:
            if any(qubit < 0 or qubit >= self.n_qubits for qubit in term.qubits):
                raise ValueError(
                    "Ground-state Hamiltonian terms must reference in-range qubits."
                )
        reference_ids = [reference.reference_id for reference in self.classical_references]
        if len(reference_ids) != len(set(reference_ids)):
            raise ValueError("Ground-state classical-reference IDs must be unique.")
        exact = [
            reference
            for reference in self.classical_references
            if reference.reference_id == "exact_diagonalization"
            and reference.role == "oracle"
        ]
        if len(exact) != 1 or not exact[0].certified:
            raise ValueError(
                "Ground-state instances require one certified exact-diagonalization "
                "oracle."
            )
        if not isinstance(self.provenance, Mapping):
            raise ValueError("Ground-state provenance must be a mapping.")
        if not self.energy_units.strip() or not self.provenance:
            raise ValueError(
                "Ground-state instances require energy units and provenance."
            )
        frozen_provenance = _freeze_payload(self.provenance)
        _hash(_mutable_payload(frozen_provenance))
        object.__setattr__(self, "provenance", frozen_provenance)

    @property
    def shape(self) -> tuple[int, int]:
        dimension = 2 ** self.n_qubits
        return (dimension, dimension)

    @property
    def sampler_policy(self) -> str:
        return "not_applicable_exact_registered_hamiltonian"

    @property
    def metadata(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "instance_id": self.instance_id,
                "label": self.label,
                "family": self.family,
                "n_qubits": self.n_qubits,
                "energy_units": self.energy_units,
                "canonical_terms": [term.to_payload() for term in self.terms],
                "classical_references": [
                    reference.to_payload()
                    for reference in self.classical_references
                ],
            }
        )

    @property
    def references(self) -> tuple[ClassicalReference, ...]:
        return self.classical_references

    @property
    def config_hash(self) -> str:
        return _hash(
            {
                "instance_id": self.instance_id,
                "n_qubits": self.n_qubits,
                "energy_units": self.energy_units,
            }
        )

    @property
    def content_hash(self) -> str:
        return _hash(dict(self.metadata))

    def to_payload(self) -> dict[str, object]:
        return {
            "instance_id": self.instance_id,
            "label": self.label,
            "family": self.family,
            "n_qubits": self.n_qubits,
            "energy_units": self.energy_units,
            "shape": list(self.shape),
            "terms": [term.to_payload() for term in self.terms],
            "classical_references": [
                reference.to_payload()
                for reference in self.classical_references
            ],
            "provenance": _mutable_payload(self.provenance),
        }


_TFIM_2Q = GroundStateInstance(
    instance_id="tfim-2q-open-j1-h1",
    label="Open TFIM 2q (J=1, h=1)",
    family="transverse_field_ising",
    n_qubits=2,
    terms=(
        PauliTerm(-1.0, "ZZ", (0, 1)),
        PauliTerm(-1.0, "X", (0,)),
        PauliTerm(-1.0, "X", (1,)),
    ),
    energy_units="dimensionless",
    classical_references=(
        ClassicalReference(
            reference_id="exact_diagonalization",
            role="oracle",
            label="Certified exact diagonalization",
            energy=-(5.0 ** 0.5),
            method="dense exact diagonalization of the checked-in 4x4 Hamiltonian",
            certified=True,
            certificate=MappingProxyType({
                "kind": "dense_matrix_eigendecomposition",
                "matrix_dimension": 4,
                "eigenvalue_expression": "-sqrt(5)",
                "verification": (
                    "Rebuild the matrix from canonical Pauli terms and run "
                    "numpy.linalg.eigvalsh."
                ),
            }),
            limitation="A tiny classically tractable simulator diagnostic; it is not QPU evidence or a scaling result.",
        ),
        ClassicalReference(
            reference_id="best_product_state",
            role="descriptive_challenger",
            label="Best product-state reference",
            energy=-2.0,
            method="best product-state reference for this two-qubit Hamiltonian",
            certified=True,
            certificate=MappingProxyType({
                "kind": "analytic_product_state_bound",
                "state": "|+> tensor |+>",
                "bloch_vectors": [
                    {"x": 1.0, "y": 0.0, "z": 0.0},
                    {"x": 1.0, "y": 0.0, "z": 0.0},
                ],
                "energy_derivation": "-<ZZ>-<X0>-<X1> = -2",
                "optimality_bound": (
                    "Minimizing over two unit Bloch vectors gives energy >= -2."
                ),
            }),
            limitation="A product-state reference is a limited classical ladder rung, not a general solver comparison.",
        ),
    ),
    provenance=MappingProxyType({
        "source": "checked_in_registered_toy_instance",
        "status": "checked_in",
        "limitations": "Toy open-boundary transverse-field Ising instance for analytic CPU diagnostics only.",
    }),
)

GROUND_STATE_INSTANCES: Mapping[str, GroundStateInstance] = MappingProxyType({_TFIM_2Q.instance_id: _TFIM_2Q})


def get_ground_state_instance(instance_id: str) -> GroundStateInstance:
    try:
        return GROUND_STATE_INSTANCES[instance_id]
    except KeyError as exc:
        raise KeyError(f"Unknown ground-state instance {instance_id!r}.") from exc


def ground_state_instances_payload() -> list[dict[str, object]]:
    return [
        GROUND_STATE_INSTANCES[instance_id].to_payload()
        for instance_id in sorted(GROUND_STATE_INSTANCES)
    ]
