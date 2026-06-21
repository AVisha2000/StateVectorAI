"""Quantum layers and backends: shapes, gradient flow, jit, output ranges."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qllm.config import ModelConfig, QuantumConfig
from qllm.models.model import build_model
from qllm.quantum.backends import get_expval_circuit
from qllm.quantum.circuits import weight_shape
from qllm.quantum.layers import QuantumCore

QCFG = QuantumConfig(n_qubits=3, n_circuit_layers=2)


def _core(out_features=6):
    core = QuantumCore.from_config(QCFG, out_features=out_features)
    x = jnp.asarray(np.random.default_rng(0).normal(size=(2, 5, 8)))
    params = core.init(jax.random.PRNGKey(0), x)["params"]
    return core, params, x


def test_quantum_core_shape_and_finiteness():
    core, params, x = _core(out_features=6)
    out = core.apply({"params": params}, x)
    assert out.shape == (2, 5, 6)
    assert jnp.isfinite(out).all()


def test_gradients_reach_circuit_weights():
    core, params, x = _core()

    def loss(p):
        return (core.apply({"params": p}, x) ** 2).sum()

    grads = jax.grad(loss)(params)
    g_circ = grads["circuit_weights"]
    # leading axis = n_circuits (parallel quantum heads), default 1
    assert g_circ.shape == (1, *weight_shape(QCFG.n_circuit_layers, QCFG.n_qubits))
    assert float(jnp.abs(g_circ).sum()) > 0.0


def test_quantum_core_jit_matches_eager():
    core, params, x = _core()
    eager = core.apply({"params": params}, x)
    jitted = jax.jit(lambda p, t: core.apply({"params": p}, t))(params, x)
    np.testing.assert_allclose(eager, jitted, rtol=1e-5, atol=1e-6)


def test_expvals_bounded():
    circuit = get_expval_circuit(
        "pennylane", "default.qubit", "backprop", None, 3, 2, "reuploading"
    )
    key = jax.random.PRNGKey(0)
    inputs = jax.random.uniform(key, (3,), minval=-1.5, maxval=1.5)
    weights = jax.random.uniform(key, weight_shape(2, 3), maxval=2 * np.pi)
    out = circuit(inputs, weights)
    assert out.shape == (3,)
    assert (jnp.abs(out) <= 1.0 + 1e-5).all()


def test_circuit_factory_is_cached():
    a = get_expval_circuit(
        "pennylane", "default.qubit", "backprop", None, 2, 1, "reuploading"
    )
    b = get_expval_circuit(
        "pennylane", "default.qubit", "backprop", None, 2, 1, "reuploading"
    )
    assert a is b


def test_hybrid_model_builds_both_variants():
    for attn_type, ffn_type in [("classical", "quantum"), ("quantum_proj", "classical")]:
        cfg = ModelConfig(
            d_model=16,
            n_heads=2,
            n_blocks=1,
            d_ff=32,
            max_seq_len=16,
            attn_type=attn_type,
            ffn_type=ffn_type,
            quantum=QuantumConfig(n_qubits=2, n_circuit_layers=1),
        )
        model, _ = build_model(cfg, vocab_size=7)
        tokens = jnp.array(np.random.default_rng(1).integers(0, 7, (2, 6)))
        params = model.init(jax.random.PRNGKey(0), tokens)["params"]
        logits = model.apply({"params": params}, tokens)
        assert logits.shape == (2, 6, 7)
        assert jnp.isfinite(logits).all()


def test_tensorcircuit_parity_single_layer():
    """TC backend mirrors PennyLane gates; parity holds for 1 layer (range-1)."""
    pytest.importorskip("tensorcircuit")
    from qllm.quantum.backends import TensorCircuitBackend

    n, layers, ansatz = 3, 1, "hardware_efficient"
    pl = get_expval_circuit(
        "pennylane", "default.qubit", "backprop", None, n, layers, ansatz
    )
    tc = TensorCircuitBackend().expval_circuit(n, layers, ansatz)
    key = jax.random.PRNGKey(2)
    inputs = jax.random.uniform(key, (n,), minval=-1.0, maxval=1.0)
    weights = jax.random.uniform(key, weight_shape(layers, n), maxval=2 * np.pi)
    np.testing.assert_allclose(pl(inputs, weights), tc(inputs, weights), atol=1e-5)
