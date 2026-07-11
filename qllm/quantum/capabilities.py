"""Dependency-free capability metadata for QLLM quantum backends.

Capabilities describe the operations exposed by QLLM's adapter, not every
operation that an underlying library might support.  This keeps resource and
evidence payloads conservative and prevents an unexposed library feature from
being mistaken for a platform guarantee.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


CAPABILITY_NAMES = (
    "state_access",
    "expectations",
    "probabilities",
    "sampling",
    "gradients",
    "noise",
    "reset",
    "dynamic_circuits",
)

# QLLM deliberately exposes only methods covered by its CPU parity tests. Other
# PennyLane methods can be added after device/interface-specific validation.
PENNYLANE_DIFF_METHODS = ("backprop", "parameter-shift")


@dataclass(frozen=True)
class Capability:
    """Support status and semantics for one adapter operation."""

    status: Literal["supported", "unsupported", "conditional", "unverified"]
    semantics: str
    limitation: str | None = None

    @property
    def supported(self) -> bool:
        """Whether the adapter guarantees this operation in the selected mode."""
        return self.status == "supported"


@dataclass(frozen=True)
class BackendCapabilities:
    """Immutable execution and operation contract for one backend mode."""

    backend: str
    execution_regime: str
    exactness: Literal["exact", "sampled", "unverified"]
    representation: str
    approximation: dict[str, Any] | None
    state_access: Capability
    expectations: Capability
    probabilities: Capability
    sampling: Capability
    gradients: Capability
    noise: Capability
    reset: Capability
    dynamic_circuits: Capability

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable, schema-versioned capability payload."""
        values = asdict(self)
        operations = {}
        for name in CAPABILITY_NAMES:
            capability = values.pop(name)
            capability["supported"] = capability["status"] == "supported"
            operations[name] = capability
        return {"schema_version": 1, **values, "capabilities": operations}


def _unsupported(reason: str) -> Capability:
    return Capability("unsupported", "unsupported", reason)


def _validate_shots(shots: int | None) -> None:
    if shots is not None and (
        isinstance(shots, bool) or not isinstance(shots, int) or shots <= 0
    ):
        raise ValueError(f"shots must be a positive integer or None; got {shots!r}.")


def resolve_backend_capabilities(
    backend: str,
    device: str = "default.qubit",
    diff_method: str = "backprop",
    shots: int | None = None,
) -> BackendCapabilities:
    """Resolve and validate an actual QLLM backend execution mode.

    This function intentionally imports no JAX or quantum framework, so callers
    can reject unsupported combinations before optional dependencies load.
    ``jax_native_statevector`` is metadata-only: it describes recurrent and
    amplitude-space components implemented directly in JAX and is not accepted
    by :func:`qllm.quantum.backends.make_backend`.
    """
    _validate_shots(shots)
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be a non-empty string.")
    if not isinstance(diff_method, str) or not diff_method.strip():
        raise ValueError("diff_method must be a non-empty string.")

    unavailable_probabilities = _unsupported(
        "probability readout is not exposed by the QLLM backend adapter"
    )
    unavailable_sampling = _unsupported(
        "sample readout is not exposed by the QLLM backend adapter"
    )
    unavailable_noise = _unsupported(
        "noise controls are not exposed by the QLLM backend adapter"
    )
    unavailable_reset = _unsupported(
        "mid-circuit reset is not exposed by the QLLM backend adapter"
    )
    unavailable_dynamic = _unsupported(
        "dynamic circuits are not exposed by the QLLM backend adapter"
    )

    if backend == "pennylane":
        if diff_method not in PENNYLANE_DIFF_METHODS:
            supported = ", ".join(PENNYLANE_DIFF_METHODS)
            raise ValueError(
                "Unsupported PennyLane diff_method "
                f"{diff_method!r}; QLLM currently validates: {supported}."
            )
        if shots is not None and diff_method == "backprop":
            raise ValueError(
                "PennyLane finite-shot execution does not support "
                "diff_method='backprop'; select a finite-shot-compatible "
                "differentiation method such as 'parameter-shift'."
            )
        default_qubit = device == "default.qubit"
        analytic = shots is None
        state_supported = analytic and default_qubit
        if analytic and default_qubit:
            regime = "exact_analytic"
            exactness = "exact"
            representation = "dense_statevector"
            approximation = None
        elif analytic:
            regime = "backend_defined_analytic"
            exactness = "unverified"
            representation = "backend_defined"
            approximation = None
        elif default_qubit:
            regime = "finite_shot_estimate"
            exactness = "sampled"
            representation = "dense_statevector"
            approximation = {
                "method": "finite_shot_sampling",
                "shots": shots,
                "error_metric": "sampling_error",
                "convergence": "not_measured",
            }
        else:
            regime = "finite_shot_estimate"
            exactness = "sampled"
            representation = "backend_defined"
            approximation = {
                "method": "finite_shot_sampling",
                "shots": shots,
                "error_metric": "sampling_error",
                "convergence": "not_measured",
            }
        return BackendCapabilities(
            backend=backend,
            execution_regime=regime,
            exactness=exactness,
            representation=representation,
            approximation=approximation,
            state_access=(
                Capability("supported", "exact_dense_statevector")
                if state_supported
                else _unsupported(
                    "state access requires analytic default.qubit execution"
                )
            ),
            expectations=Capability(
                "supported" if default_qubit else "conditional",
                "exact" if analytic and default_qubit else (
                    "finite_shot_estimate" if not analytic else "backend_defined"
                ),
                None if default_qubit else (
                    "expectation support depends on the selected PennyLane device"
                ),
            ),
            probabilities=unavailable_probabilities,
            sampling=unavailable_sampling,
            gradients=Capability(
                "supported" if analytic and default_qubit else "conditional",
                f"pennylane:{diff_method}",
                None if analytic and default_qubit else (
                    "support depends on the selected PennyLane device and method"
                ),
            ),
            noise=unavailable_noise,
            reset=unavailable_reset,
            dynamic_circuits=unavailable_dynamic,
        )

    if backend == "tensorcircuit":
        if shots is not None:
            raise ValueError(
                "TensorCircuitBackend currently exposes analytic dense-statevector "
                "execution only; shots must be None."
            )
        if diff_method != "backprop":
            raise ValueError(
                "TensorCircuitBackend currently supports only "
                "diff_method='backprop'."
            )
        return BackendCapabilities(
            backend=backend,
            execution_regime="exact_analytic",
            exactness="exact",
            representation="dense_statevector",
            approximation=None,
            state_access=Capability("supported", "exact_dense_statevector"),
            expectations=Capability("supported", "exact"),
            probabilities=unavailable_probabilities,
            sampling=unavailable_sampling,
            gradients=Capability(
                "unverified",
                "jax_autodiff",
                "TensorCircuit is optional and gradient parity is not verified "
                "in this environment",
            ),
            noise=unavailable_noise,
            reset=unavailable_reset,
            dynamic_circuits=unavailable_dynamic,
        )

    if backend == "jax_native_statevector":
        if shots is not None:
            raise ValueError("jax_native_statevector does not support finite shots.")
        if diff_method != "backprop":
            raise ValueError(
                "jax_native_statevector supports only diff_method='backprop'."
            )
        return BackendCapabilities(
            backend=backend,
            execution_regime="exact_analytic",
            exactness="exact",
            representation="dense_statevector",
            approximation=None,
            state_access=Capability("supported", "exact_dense_statevector"),
            expectations=Capability("supported", "exact_state_derived"),
            probabilities=unavailable_probabilities,
            sampling=unavailable_sampling,
            gradients=Capability("supported", "jax_autodiff"),
            noise=unavailable_noise,
            reset=unavailable_reset,
            dynamic_circuits=unavailable_dynamic,
        )

    raise ValueError(
        "Unknown backend capability profile "
        f"{backend!r}. Available: pennylane, tensorcircuit, jax_native_statevector"
    )


def backend_capabilities_payload(
    backend: str,
    device: str = "default.qubit",
    diff_method: str = "backprop",
    shots: int | None = None,
) -> dict[str, Any]:
    """Resolve a backend mode and return its JSON-safe public payload."""
    return resolve_backend_capabilities(backend, device, diff_method, shots).to_payload()
