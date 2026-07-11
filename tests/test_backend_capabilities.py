"""Backend capability metadata and exact-overlap contracts."""
from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qllm.quantum.backends import PennyLaneBackend, get_expval_circuit, get_state_circuit
from qllm.quantum.capabilities import (
    CAPABILITY_NAMES,
    backend_capabilities_payload,
    resolve_backend_capabilities,
)
from qllm.quantum.circuits import weight_shape


@pytest.mark.parametrize(
    ("backend", "device"),
    [
        ("pennylane", "default.qubit"),
        ("tensorcircuit", "statevector"),
        ("jax_native_statevector", "jax:cpu"),
    ],
)
def test_capability_payload_is_complete_and_json_serializable(backend, device):
    payload = backend_capabilities_payload(backend, device)

    assert payload["schema_version"] == 1
    assert tuple(payload["capabilities"]) == CAPABILITY_NAMES
    assert payload["execution_regime"] == "exact_analytic"
    assert payload["exactness"] == "exact"
    assert payload["representation"] == "dense_statevector"
    assert payload["approximation"] is None
    assert json.loads(json.dumps(payload)) == payload


def test_finite_shot_metadata_is_distinct_and_has_no_state_access():
    capabilities = resolve_backend_capabilities(
        "pennylane", "default.qubit", "parameter-shift", 128
    )

    assert capabilities.execution_regime == "finite_shot_estimate"
    assert capabilities.representation == "dense_statevector"
    assert capabilities.exactness == "sampled"
    assert capabilities.approximation == {
        "method": "finite_shot_sampling",
        "shots": 128,
        "error_metric": "sampling_error",
        "convergence": "not_measured",
    }
    assert capabilities.expectations.semantics == "finite_shot_estimate"
    assert not capabilities.state_access.supported


def test_unverified_modes_are_not_reported_as_supported_or_approximate():
    unknown_device = resolve_backend_capabilities(
        "pennylane", "plugin.device", "parameter-shift", None
    )
    tensorcircuit = resolve_backend_capabilities("tensorcircuit", "statevector")

    assert unknown_device.exactness == "unverified"
    assert unknown_device.approximation is None
    assert unknown_device.gradients.status == "conditional"
    assert tensorcircuit.gradients.status == "unverified"
    assert not tensorcircuit.gradients.supported


@pytest.mark.parametrize(
    "args, match",
    [
        (("pennylane", "default.qubit", "backprop", 100), "finite-shot"),
        (("pennylane", "default.qubit", "definitely-invalid", None), "Unsupported"),
        (("tensorcircuit", "statevector", "backprop", 100), "shots must be None"),
        (("tensorcircuit", "statevector", "parameter-shift", None), "only diff_method"),
        (("jax_native_statevector", "jax:cpu", "backprop", 10), "finite shots"),
    ],
)
def test_invalid_execution_modes_fail_without_backend_import(args, match):
    with pytest.raises(ValueError, match=match):
        resolve_backend_capabilities(*args)


def test_tensorcircuit_mode_is_rejected_before_optional_import():
    from qllm.quantum.backends import make_backend

    with pytest.raises(ValueError, match="only diff_method"):
        make_backend("tensorcircuit", diff_method="parameter-shift")


def test_invalid_pennylane_method_cannot_enter_resource_plan(
    tiny_quantum_cfg,
):
    import dataclasses

    import jax

    from qllm.resources import static_resource_plan

    invalid = dataclasses.replace(
        tiny_quantum_cfg,
        model=dataclasses.replace(
            tiny_quantum_cfg.model,
            quantum=dataclasses.replace(
                tiny_quantum_cfg.model.quantum,
                diff_method="definitely-invalid",
            ),
        ),
    )
    with pytest.raises(ValueError, match="Unsupported PennyLane diff_method"):
        static_resource_plan(
            invalid,
            n_params=1,
            requested_device="cpu",
            resolved_device=jax.devices("cpu")[0],
        )


def test_finite_shot_state_factory_is_rejected_explicitly():
    backend = PennyLaneBackend(diff_method="parameter-shift", shots=32)
    with pytest.raises(ValueError, match="State access is unavailable"):
        backend.state_circuit(2, 1, "hardware_efficient")

    with pytest.raises(ValueError, match="State access is unavailable"):
        get_state_circuit(
            "pennylane",
            "default.qubit",
            2,
            1,
            "hardware_efficient",
            "parameter-shift",
            32,
        )


@pytest.mark.parametrize("readout", ["z", "zz"])
def test_pennylane_exact_expvals_match_state_derived_values(readout):
    n_qubits, n_layers = 3, 2
    ansatz = "reuploading"
    key_inputs, key_weights = jax.random.split(jax.random.PRNGKey(19))
    inputs = jax.random.uniform(key_inputs, (n_qubits,), minval=-1.0, maxval=1.0)
    weights = jax.random.uniform(
        key_weights, weight_shape(n_layers, n_qubits), minval=-0.8, maxval=0.8
    )
    state = np.asarray(
        get_state_circuit(
            "pennylane", "default.qubit", n_qubits, n_layers, ansatz
        )(inputs, weights)
    )
    probabilities = np.abs(state) ** 2
    basis = np.arange(2**n_qubits)
    z_values = [
        1.0 - 2.0 * ((basis >> (n_qubits - 1 - wire)) & 1)
        for wire in range(n_qubits)
    ]
    expected = [float(probabilities @ z) for z in z_values]
    if readout == "zz":
        expected.extend(
            float(probabilities @ (z_values[i] * z_values[j]))
            for i in range(n_qubits)
            for j in range(i + 1, n_qubits)
        )

    circuit = get_expval_circuit(
        "pennylane",
        "default.qubit",
        "backprop",
        None,
        n_qubits,
        n_layers,
        ansatz,
        readout,
    )
    np.testing.assert_allclose(
        np.asarray(circuit(inputs, weights)), np.asarray(expected), rtol=1e-5, atol=1e-6
    )
