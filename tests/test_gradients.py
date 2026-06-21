"""Gradient correctness: parameter-shift rule vs backprop vs finite differences.

This is the planning doc's required unit-test triad. Agreement of all three
on small circuits validates that autodiff through the quantum layer computes
the true analytic gradient.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml
import pytest

from qllm.quantum.circuits import data_reuploading, weight_shape

N_QUBITS, N_LAYERS = 3, 2


def _qnode(diff_method: str):
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def circuit(inputs, weights):
        data_reuploading(inputs, weights, N_QUBITS)
        return qml.expval(qml.PauliZ(0))

    return circuit


@pytest.fixture(scope="module")
def point():
    key = jax.random.PRNGKey(7)
    k1, k2 = jax.random.split(key)
    inputs = jax.random.uniform(k1, (N_QUBITS,), minval=-1.0, maxval=1.0)
    weights = jax.random.uniform(
        k2, weight_shape(N_LAYERS, N_QUBITS), maxval=2 * jnp.pi
    )
    return inputs, weights


def test_parameter_shift_matches_backprop(point):
    inputs, weights = point
    g_ps = jax.grad(lambda w: _qnode("parameter-shift")(inputs, w))(weights)
    g_bp = jax.grad(lambda w: _qnode("backprop")(inputs, w))(weights)
    np.testing.assert_allclose(g_ps, g_bp, atol=1e-5, rtol=1e-4)


def test_backprop_matches_finite_differences(point):
    inputs, weights = point
    circuit = _qnode("backprop")
    g_bp = np.asarray(jax.grad(lambda w: circuit(inputs, w))(weights))

    eps = 1e-3
    flat = np.asarray(weights).ravel()
    # spot-check a few parameters with central differences
    for idx in [0, len(flat) // 2, len(flat) - 1]:
        bump = np.zeros_like(flat)
        bump[idx] = eps
        wp = jnp.asarray((flat + bump).reshape(weights.shape))
        wm = jnp.asarray((flat - bump).reshape(weights.shape))
        fd = (float(circuit(inputs, wp)) - float(circuit(inputs, wm))) / (2 * eps)
        assert abs(fd - g_bp.ravel()[idx]) < 5e-3


def test_expval_value_agrees_across_diff_methods(point):
    inputs, weights = point
    v_ps = float(_qnode("parameter-shift")(inputs, weights))
    v_bp = float(_qnode("backprop")(inputs, weights))
    assert abs(v_ps - v_bp) < 1e-6
