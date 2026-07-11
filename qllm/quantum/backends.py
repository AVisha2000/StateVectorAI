"""Quantum simulation backends behind a common protocol.

A backend turns an (ansatz, n_qubits, n_layers) spec into plain JAX-callable
functions, so every layer and metric is backend-agnostic. Swapping
``default.qubit`` (exact, <=~25 qubits) for a tensor-network simulator
(100+ qubits, bounded entanglement) is a one-line config change.

Returned callables:
- ``expval_circuit``: (inputs(n,), weights) -> (n,) local Pauli-Z expvals
- ``state_circuit``:  (inputs(n,), weights) -> (2**n,) complex statevector
  (used by diagnostics; exact backends only)
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable, Protocol

import jax.numpy as jnp

from ..registry import BACKEND_TYPES, CIRCUIT_ANSATZ_TYPES, READOUT_TYPES


class CircuitBackend(Protocol):
    name: str

    def expval_circuit(
        self, n_qubits: int, n_layers: int, ansatz: str, readout: str = "z"
    ) -> Callable: ...

    def state_circuit(
        self, n_qubits: int, n_layers: int, ansatz: str
    ) -> Callable: ...


class PennyLaneBackend:
    """Primary backend: PennyLane with the JAX interface.

    ``default.qubit`` + ``diff_method='backprop'`` is pure JAX end to end:
    jit-able, vmap-able, and differentiable inside a Flax model.
    ``parameter-shift`` is exposed for validation/hardware realism.
    """

    name = "pennylane"

    def __init__(
        self,
        device: str = "default.qubit",
        diff_method: str = "backprop",
        shots: int | None = None,
    ):
        self.device = device
        self.diff_method = diff_method
        self.shots = shots

    def _qnode(self, n_qubits: int, ansatz: str, measurement: str):
        import pennylane as qml

        from .circuits import ANSATZ_REGISTRY

        if ansatz not in CIRCUIT_ANSATZ_TYPES:
            raise ValueError(
                f"Unknown circuit ansatz '{ansatz}'. Options: "
                f"{CIRCUIT_ANSATZ_TYPES}"
            )
        dev = qml.device(self.device, wires=n_qubits, shots=self.shots)
        ansatz_fn = ANSATZ_REGISTRY[ansatz]

        if measurement in ("expval_z", "expval_zz"):
            observables = [qml.PauliZ(i) for i in range(n_qubits)]
            if measurement == "expval_zz":
                # weight-2 correlators: O(n^2) features while every
                # observable stays low-weight (local-cost trainable regime)
                observables += [
                    qml.PauliZ(i) @ qml.PauliZ(j)
                    for i in range(n_qubits)
                    for j in range(i + 1, n_qubits)
                ]

            @qml.qnode(dev, interface="jax", diff_method=self.diff_method)
            def circuit(inputs, weights):
                ansatz_fn(inputs, weights, n_qubits)
                return [qml.expval(o) for o in observables]

            def wrapped(inputs, weights):
                return jnp.stack(circuit(inputs, weights), axis=-1)

            return wrapped

        if measurement == "state":

            @qml.qnode(dev, interface="jax")
            def circuit(inputs, weights):
                ansatz_fn(inputs, weights, n_qubits)
                return qml.state()

            return circuit

        raise ValueError(f"Unknown measurement '{measurement}'")

    def expval_circuit(self, n_qubits, n_layers, ansatz, readout="z"):
        del n_layers  # encoded in the weights array shape
        return self._qnode(n_qubits, ansatz, f"expval_{readout}")

    def state_circuit(self, n_qubits, n_layers, ansatz):
        del n_layers
        return self._qnode(n_qubits, ansatz, "state")


class TensorCircuitBackend:
    """EXPERIMENTAL: TensorCircuit-NG (JAX backend) for large-qubit scaling.

    Mirrors the PennyLane gate sequence (Rot = RZ·RY·RZ + ring of CNOTs,
    matching ``qml.StronglyEntanglingLayers`` with range 1). Use
    ``tc.MPSCircuit`` here when pushing past state-vector limits; the
    exact/MPS API parity is what enables overlap-region benchmarking.
    Requires ``pip install tensorcircuit-ng``.
    """

    name = "tensorcircuit"

    def __init__(self, device: str = "statevector", **_):
        try:
            import tensorcircuit as tc
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "TensorCircuitBackend requires `pip install tensorcircuit-ng`."
            ) from exc
        tc.set_backend("jax")
        self._tc = tc
        self.device = device

    def _apply(self, c, inputs, weights, n_qubits, ansatz):
        n_layers = weights.shape[0]

        def encode():
            for i in range(n_qubits):
                c.ry(i, theta=inputs[i])

        def entangling_layer(layer_weights):
            for i in range(n_qubits):
                c.rz(i, theta=layer_weights[i, 0])
                c.ry(i, theta=layer_weights[i, 1])
                c.rz(i, theta=layer_weights[i, 2])
            if n_qubits > 1:
                for i in range(n_qubits):
                    c.cnot(i, (i + 1) % n_qubits)

        if ansatz == CIRCUIT_ANSATZ_TYPES[0]:
            encode()
            for layer in range(n_layers):
                entangling_layer(weights[layer])
        elif ansatz == CIRCUIT_ANSATZ_TYPES[1]:
            for layer in range(n_layers):
                encode()
                entangling_layer(weights[layer])
        else:
            raise ValueError(
                f"Unknown circuit ansatz '{ansatz}'. Options: "
                f"{CIRCUIT_ANSATZ_TYPES}"
            )

    def expval_circuit(self, n_qubits, n_layers, ansatz, readout="z"):
        tc = self._tc

        def fn(inputs, weights):
            c = tc.Circuit(n_qubits)
            self._apply(c, inputs, weights, n_qubits, ansatz)
            vals = [tc.backend.real(c.expectation_ps(z=[i])) for i in range(n_qubits)]
            if readout == "zz":
                vals += [
                    tc.backend.real(c.expectation_ps(z=[i, j]))
                    for i in range(n_qubits)
                    for j in range(i + 1, n_qubits)
                ]
            return jnp.stack(vals, axis=-1)

        return fn

    def state_circuit(self, n_qubits, n_layers, ansatz):
        tc = self._tc

        def fn(inputs, weights):
            c = tc.Circuit(n_qubits)
            self._apply(c, inputs, weights, n_qubits, ansatz)
            return c.state()

        return fn


BACKEND_REGISTRY = dict(
    zip(BACKEND_TYPES, (PennyLaneBackend, TensorCircuitBackend), strict=True)
)


def make_backend(
    backend: str = "pennylane",
    device: str = "default.qubit",
    diff_method: str = "backprop",
    shots: int | None = None,
) -> CircuitBackend:
    if backend not in BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown backend '{backend}'. Available: {list(BACKEND_REGISTRY)}"
        )
    if backend == BACKEND_TYPES[0]:
        return PennyLaneBackend(device=device, diff_method=diff_method, shots=shots)
    return BACKEND_REGISTRY[backend](device=device)


@lru_cache(maxsize=None)
def get_expval_circuit(
    backend: str,
    device: str,
    diff_method: str,
    shots: int | None,
    n_qubits: int,
    n_layers: int,
    ansatz: str,
    readout: str = "z",
) -> Callable:
    """Cached circuit factory (all args hashable -> safe under retracing)."""
    if readout not in READOUT_TYPES:
        raise ValueError(f"Unknown readout '{readout}'. Options: {READOUT_TYPES}")
    be = make_backend(backend, device, diff_method, shots)
    return be.expval_circuit(n_qubits, n_layers, ansatz, readout=readout)


def readout_dim(n_qubits: int, readout: str) -> int:
    """Number of features produced by a readout scheme."""
    if readout == READOUT_TYPES[0]:
        return n_qubits
    if readout == READOUT_TYPES[1]:
        return n_qubits + n_qubits * (n_qubits - 1) // 2
    raise ValueError(f"Unknown readout '{readout}'. Options: {READOUT_TYPES}")


@lru_cache(maxsize=None)
def get_state_circuit(
    backend: str,
    device: str,
    n_qubits: int,
    n_layers: int,
    ansatz: str,
) -> Callable:
    be = make_backend(backend, device)
    return be.state_circuit(n_qubits, n_layers, ansatz)
