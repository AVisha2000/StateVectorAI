"""Quantum simulation backends behind a common protocol.

A backend turns an (ansatz, n_qubits, n_layers) spec into plain JAX-callable
functions, so every layer and metric is backend-agnostic. Dense exact and MPS
execution are separate adapters: capability metadata makes the approximation
and the absence of full-state access explicit.

Returned callables:
- ``expval_circuit``: (inputs(n,), weights) -> (n,) local Pauli-Z expvals
- ``state_circuit``:  (inputs(n,), weights) -> (2**n,) complex statevector
  (used by diagnostics; exact backends only)
"""
from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from typing import Callable, Protocol

import jax.numpy as jnp

from ..registry import BACKEND_TYPES, CIRCUIT_ANSATZ_TYPES, READOUT_TYPES
from .capabilities import BackendCapabilities, resolve_backend_capabilities


class CircuitBackend(Protocol):
    name: str
    capabilities: BackendCapabilities

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
        self.capabilities = resolve_backend_capabilities(
            self.name, device, diff_method, shots
        )

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

            @qml.qnode(dev, interface="jax", diff_method=self.diff_method)
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
        if not self.capabilities.state_access.supported:
            raise ValueError(
                "State access is unavailable for this PennyLane execution mode: "
                f"{self.capabilities.state_access.limitation}."
            )
        return self._qnode(n_qubits, ansatz, "state")


class TensorCircuitBackend:
    """EXPERIMENTAL: TensorCircuit-NG dense statevector adapter (JAX backend).

    Mirrors the PennyLane gate sequence (Rot = RZ·RY·RZ + ring of CNOTs,
    matching ``qml.StronglyEntanglingLayers`` with range 1). This adapter uses
    ``tc.Circuit`` and does not expose MPS or another approximate execution mode.
    Requires ``pip install tensorcircuit-ng``.
    """

    name = "tensorcircuit"

    def __init__(
        self,
        device: str = "statevector",
        diff_method: str = "backprop",
        shots: int | None = None,
    ):
        self.capabilities = resolve_backend_capabilities(
            self.name, device, diff_method, shots
        )
        try:
            import tensorcircuit as tc
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "TensorCircuitBackend requires `pip install tensorcircuit-ng`."
            ) from exc
        tc.set_backend("jax")
        self._tc = tc
        self.device = device
        self.diff_method = diff_method
        self.shots = shots

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


class TensorCircuitMPSBackend(TensorCircuitBackend):
    """Approximate TensorCircuit-NG matrix-product-state adapter.

    The gate sequence is inherited verbatim from ``TensorCircuitBackend`` so
    dense-vs-MPS comparisons change only the state representation and static
    fixed-bond SVD truncation rules. TensorCircuit-NG 1.7's data-dependent
    threshold mode is rejected because it is not JAX-transform safe. Dense
    state materialization is intentionally absent.
    """

    name = "tensorcircuit_mps"

    def __init__(
        self,
        device: str = "mps",
        diff_method: str = "backprop",
        shots: int | None = None,
        mps_max_bond_dimension: int | None = None,
        mps_max_truncation_error: float | None = None,
        mps_relative_truncation: bool = False,
    ):
        self.capabilities = resolve_backend_capabilities(
            self.name,
            device,
            diff_method,
            shots,
            mps_max_bond_dimension,
            mps_max_truncation_error,
            mps_relative_truncation,
        )
        try:
            tc = import_module("tensorcircuit")
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "TensorCircuitMPSBackend requires the optional qllm[mps] "
                "dependency; install it with `pip install 'qllm[mps]'`."
            ) from exc
        tc.set_backend("jax")
        self._tc = tc
        self.device = device
        self.diff_method = diff_method
        self.shots = shots
        self.mps_max_bond_dimension = mps_max_bond_dimension
        self.mps_max_truncation_error = mps_max_truncation_error
        self.mps_relative_truncation = mps_relative_truncation
        self.split_rules = {
            "max_singular_values": mps_max_bond_dimension,
            "relative": False,
        }

    def expval_circuit(self, n_qubits, n_layers, ansatz, readout="z"):
        del n_layers  # encoded in the weights array shape
        if readout not in READOUT_TYPES:
            raise ValueError(
                f"Unknown readout '{readout}'. Options: {READOUT_TYPES}"
            )
        tc = self._tc
        split_rules = self.split_rules

        def fn(inputs, weights):
            c = tc.MPSCircuit(n_qubits, split=split_rules)
            self._apply(c, inputs, weights, n_qubits, ansatz)
            values = [
                tc.backend.real(c.expectation_ps(z=[i], normalize=True))
                for i in range(n_qubits)
            ]
            if readout == "zz":
                values += [
                    tc.backend.real(
                        c.expectation_ps(z=[i, j], normalize=True)
                    )
                    for i in range(n_qubits)
                    for j in range(i + 1, n_qubits)
                ]
            return jnp.stack(values, axis=-1)

        return fn

    def state_circuit(self, n_qubits, n_layers, ansatz):
        del n_qubits, n_layers, ansatz
        raise ValueError(
            "State access is unavailable for tensorcircuit_mps execution: "
            f"{self.capabilities.state_access.limitation}."
        )


BACKEND_REGISTRY = {
    "pennylane": PennyLaneBackend,
    "tensorcircuit": TensorCircuitBackend,
    "tensorcircuit_mps": TensorCircuitMPSBackend,
}
assert tuple(BACKEND_REGISTRY) == BACKEND_TYPES


def make_backend(
    backend: str = "pennylane",
    device: str = "default.qubit",
    diff_method: str = "backprop",
    shots: int | None = None,
    mps_max_bond_dimension: int | None = None,
    mps_max_truncation_error: float | None = None,
    mps_relative_truncation: bool = False,
) -> CircuitBackend:
    if backend not in BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown backend '{backend}'. Available: {list(BACKEND_REGISTRY)}"
        )
    # Validate the selected execution mode before loading an optional backend.
    resolve_backend_capabilities(
        backend,
        device,
        diff_method,
        shots,
        mps_max_bond_dimension,
        mps_max_truncation_error,
        mps_relative_truncation,
    )
    if backend == BACKEND_TYPES[0]:
        return PennyLaneBackend(device=device, diff_method=diff_method, shots=shots)
    options = {
        "device": device,
        "diff_method": diff_method,
        "shots": shots,
    }
    if backend == "tensorcircuit_mps":
        options.update(
            mps_max_bond_dimension=mps_max_bond_dimension,
            mps_max_truncation_error=mps_max_truncation_error,
            mps_relative_truncation=mps_relative_truncation,
        )
    return BACKEND_REGISTRY[backend](**options)


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
    mps_max_bond_dimension: int | None = None,
    mps_max_truncation_error: float | None = None,
    mps_relative_truncation: bool = False,
) -> Callable:
    """Cached circuit factory (all args hashable -> safe under retracing)."""
    if readout not in READOUT_TYPES:
        raise ValueError(f"Unknown readout '{readout}'. Options: {READOUT_TYPES}")
    be = make_backend(
        backend,
        device,
        diff_method,
        shots,
        mps_max_bond_dimension,
        mps_max_truncation_error,
        mps_relative_truncation,
    )
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
    diff_method: str = "backprop",
    shots: int | None = None,
    mps_max_bond_dimension: int | None = None,
    mps_max_truncation_error: float | None = None,
    mps_relative_truncation: bool = False,
) -> Callable:
    capabilities = resolve_backend_capabilities(
        backend,
        device,
        diff_method,
        shots,
        mps_max_bond_dimension,
        mps_max_truncation_error,
        mps_relative_truncation,
    )
    if not capabilities.state_access.supported:
        raise ValueError(
            f"State access is unavailable for {backend} execution mode: "
            f"{capabilities.state_access.limitation}."
        )
    be = make_backend(
        backend,
        device,
        diff_method,
        shots,
        mps_max_bond_dimension,
        mps_max_truncation_error,
        mps_relative_truncation,
    )
    return be.state_circuit(n_qubits, n_layers, ansatz)
