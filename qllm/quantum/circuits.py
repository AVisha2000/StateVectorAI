"""Circuit ansatz definitions (encodings + variational layers).

Design choices follow the planning doc's barren-plateau guidance:
- shallow circuits, angle/data-reuploading encodings (not amplitude encoding)
- local Pauli-Z measurements (local cost functions)
- unified trainable-weight shape ``(n_layers, n_qubits, 3)`` across ansatz
  and backends so layers/metrics are ansatz-agnostic.

PennyLane implementations live here; the TensorCircuit backend mirrors the
same gate sequences in :mod:`qllm.quantum.backends`.
"""
from __future__ import annotations

import pennylane as qml


def weight_shape(n_layers: int, n_qubits: int) -> tuple[int, int, int]:
    """Trainable parameter shape shared by every registered ansatz."""
    return (n_layers, n_qubits, 3)


def angle_encode(inputs, n_qubits: int, rotation: str = "Y") -> None:
    """One feature per qubit via single-qubit rotations. O(n) depth."""
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation=rotation)


def hardware_efficient(inputs, weights, n_qubits: int) -> None:
    """Encode once, then stacked strongly-entangling variational layers."""
    angle_encode(inputs, n_qubits)
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))


def data_reuploading(inputs, weights, n_qubits: int) -> None:
    """Re-encode the classical data before every variational layer.

    Perez-Salinas et al., Quantum 4, 226 (2020): re-uploading makes even a
    single qubit a universal function approximator; best
    expressivity-per-qubit among NISQ-friendly encodings.
    """
    n_layers = weights.shape[0]
    for layer in range(n_layers):
        angle_encode(inputs, n_qubits)
        qml.StronglyEntanglingLayers(weights[layer][None], wires=range(n_qubits))


ANSATZ_REGISTRY = {
    "hardware_efficient": hardware_efficient,
    "reuploading": data_reuploading,
}
