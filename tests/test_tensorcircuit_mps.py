"""Installed TensorCircuit-NG MPS parity, autodiff, and Flax contracts."""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest


if os.environ.get("QLLM_REQUIRE_TENSORCIRCUIT_MPS") == "1":
    import tensorcircuit as tc
else:
    tc = pytest.importorskip(
        "tensorcircuit",
        reason="install the qllm[mps] extra to run TensorCircuit MPS parity tests",
    )

from qllm.config import QuantumConfig
from qllm.quantum.backends import get_expval_circuit
from qllm.quantum.circuits import weight_shape
from qllm.quantum.layers import QuantumCore
from qllm.quantum.metrics import quantum_diagnostics


N_QUBITS = 4
N_LAYERS = 2
ANSATZ = "reuploading"
# The deterministic fixture's observed worst errors are 5.97e-7 for values
# and 3.42e-6 for gradients on CPU. These limits retain at least a 32x margin
# without weakening the test to generic low-precision agreement.
VALUE_ATOL = 2.0e-5
GRADIENT_ATOL = 1.2e-4


@pytest.fixture(scope="module")
def entangling_point():
    inputs = jnp.asarray([0.71, -0.42, 1.13, -0.87], dtype=jnp.float32)
    weights = jnp.asarray(
        np.random.default_rng(123).uniform(
            -1.2,
            1.2,
            size=weight_shape(N_LAYERS, N_QUBITS),
        ),
        dtype=jnp.float32,
    )
    return inputs, weights


def _circuit(backend: str, readout: str, *, bond_dimension: int = 4):
    if backend == "pennylane":
        return get_expval_circuit(
            backend,
            "default.qubit",
            "backprop",
            None,
            N_QUBITS,
            N_LAYERS,
            ANSATZ,
            readout,
        )
    if backend == "tensorcircuit":
        return get_expval_circuit(
            backend,
            "statevector",
            "backprop",
            None,
            N_QUBITS,
            N_LAYERS,
            ANSATZ,
            readout,
        )
    return get_expval_circuit(
        "tensorcircuit_mps",
        "mps",
        "backprop",
        None,
        N_QUBITS,
        N_LAYERS,
        ANSATZ,
        readout,
        bond_dimension,
        None,
        False,
    )


def test_required_tensorcircuit_ng_version_is_installed():
    assert tc.__version__ == "1.7.0"


@pytest.mark.parametrize("readout", ["z", "zz"])
def test_high_bond_mps_values_match_dense_backends(readout, entangling_point):
    inputs, weights = entangling_point
    pennylane = np.asarray(_circuit("pennylane", readout)(inputs, weights))
    dense_tc = np.asarray(_circuit("tensorcircuit", readout)(inputs, weights))
    mps = np.asarray(_circuit("tensorcircuit_mps", readout)(inputs, weights))

    np.testing.assert_allclose(
        dense_tc,
        pennylane,
        rtol=VALUE_ATOL,
        atol=VALUE_ATOL,
    )
    np.testing.assert_allclose(
        mps,
        pennylane,
        rtol=VALUE_ATOL,
        atol=VALUE_ATOL,
    )


def test_chi_one_changes_only_the_deterministic_entangling_fixture(
    entangling_point,
):
    """This is a fixture regression, not a monotonic-convergence claim."""
    inputs, weights = entangling_point
    high_bond = np.asarray(
        _circuit("tensorcircuit_mps", "zz", bond_dimension=4)(inputs, weights)
    )
    chi_one = np.asarray(
        _circuit("tensorcircuit_mps", "zz", bond_dimension=1)(inputs, weights)
    )

    assert float(np.max(np.abs(chi_one - high_bond))) > 0.1
    chi_one_gradient = jax.grad(
        lambda value: jnp.sum(
            _circuit("tensorcircuit_mps", "zz", bond_dimension=1)(
                inputs,
                value,
            )
            ** 2
        )
    )(weights)
    assert bool(jnp.isfinite(chi_one_gradient).all())
    assert float(jnp.linalg.norm(chi_one_gradient)) > 0.0


def test_high_bond_mps_gradient_matches_dense_backends(entangling_point):
    inputs, weights = entangling_point

    def gradient(circuit):
        return np.asarray(
            jax.grad(lambda value: jnp.sum(circuit(inputs, value) ** 2))(weights)
        )

    pennylane = gradient(_circuit("pennylane", "zz"))
    dense_tc = gradient(_circuit("tensorcircuit", "zz"))
    mps = gradient(_circuit("tensorcircuit_mps", "zz"))

    assert np.isfinite(mps).all()
    assert float(np.linalg.norm(mps)) > 0.0
    np.testing.assert_allclose(
        dense_tc,
        pennylane,
        rtol=GRADIENT_ATOL,
        atol=GRADIENT_ATOL,
    )
    np.testing.assert_allclose(
        mps,
        pennylane,
        rtol=GRADIENT_ATOL,
        atol=GRADIENT_ATOL,
    )


def test_mps_circuit_supports_direct_jit_and_nested_vmap(entangling_point):
    inputs, weights = entangling_point
    circuit = _circuit("tensorcircuit_mps", "z")
    eager = circuit(inputs, weights)
    jitted = jax.jit(circuit)(inputs, weights)
    np.testing.assert_allclose(
        jitted,
        eager,
        rtol=VALUE_ATOL,
        atol=VALUE_ATOL,
    )

    input_batches = jnp.stack(
        [
            jnp.stack([inputs, inputs * 0.7]),
            jnp.stack([inputs * -0.4, inputs * 0.2]),
        ]
    )
    weight_batches = jnp.stack([weights, weights * 0.8])
    nested = jax.jit(
        jax.vmap(
            jax.vmap(circuit, in_axes=(0, None)),
            in_axes=(0, 0),
        )
    )(input_batches, weight_batches)

    assert nested.shape == (2, 2, N_QUBITS)
    assert bool(jnp.isfinite(nested).all())


def test_full_quantum_core_has_finite_nonzero_mps_gradient():
    config = QuantumConfig(
        n_qubits=N_QUBITS,
        n_circuit_layers=N_LAYERS,
        backend="tensorcircuit_mps",
        device="mps",
        diff_method="backprop",
        mps_max_bond_dimension=4,
    )
    core = QuantumCore.from_config(config, out_features=3)
    inputs = jnp.asarray(
        np.random.default_rng(44).normal(size=(2, 3, 5)),
        dtype=jnp.float32,
    )
    params = core.init(jax.random.PRNGKey(8), inputs)["params"]

    eager_output = core.apply({"params": params}, inputs)
    jitted_output = jax.jit(core.apply)({"params": params}, inputs)
    np.testing.assert_allclose(
        jitted_output,
        eager_output,
        rtol=VALUE_ATOL,
        atol=VALUE_ATOL,
    )

    def loss(values):
        return jnp.sum(core.apply({"params": values}, inputs) ** 2)

    gradients = jax.jit(jax.grad(loss))(params)["circuit_weights"]
    assert gradients.shape == (1, *weight_shape(N_LAYERS, N_QUBITS))
    assert bool(jnp.isfinite(gradients).all())
    assert float(jnp.linalg.norm(gradients)) > 0.0


def test_mps_diagnostic_gradient_is_measured_without_dense_state_metrics():
    diagnostics = quantum_diagnostics(
        N_QUBITS,
        N_LAYERS,
        ansatz=ANSATZ,
        backend="tensorcircuit_mps",
        device="mps",
        n_grad_samples=4,
        n_pairs=2,
        n_mw_samples=2,
        seed=11,
        mps_max_bond_dimension=4,
    )

    for key in ("grad_var_first_param", "grad_var_mean", "grad_var_max"):
        assert np.isfinite(diagnostics[key])
        assert diagnostics[key] > 0.0
    assert diagnostics["meyer_wallach_q"] is None
    assert diagnostics["expressibility_kl"] is None
    assert diagnostics["availability"]["gradient_variance"]["status"] == "measured"
    assert diagnostics["availability"]["meyer_wallach_q"]["status"] == "unsupported"
    assert diagnostics["availability"]["expressibility_kl"]["status"] == "unsupported"
