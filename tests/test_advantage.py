"""v0.3 feature tests: advantage detection, readouts, parallel circuits."""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from qllm.config import ModelConfig, QuantumConfig
from qllm.models.model import build_model
from qllm.quantum.advantage import (
    advantage_experiment,
    best_classical_r2,
    classical_kernel_family,
    geometric_difference,
    normalize_trace,
    quantum_fidelity_kernel,
)
from qllm.quantum.backends import readout_dim
from qllm.quantum.layers import QuantumCore
from qllm.train.loop import fit, make_grad_norm_step

# ---------------------------------------------------------------------------
# Advantage module
# ---------------------------------------------------------------------------


def test_quantum_kernel_is_valid():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1.5, 1.5, size=(20, 3))
    K = quantum_fidelity_kernel(X, n_layers=1, seed=0)
    assert K.shape == (20, 20)
    np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-5)  # |<psi|psi>|^2
    np.testing.assert_allclose(K, K.T, atol=1e-7)
    vals = np.linalg.eigvalsh(normalize_trace(K))
    assert vals.min() > -1e-8  # PSD


def test_geometric_difference_self_is_small():
    """g(K || K) ~= 1: a kernel cannot beat itself."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 3))
    K = normalize_trace(classical_kernel_family(X)["rbf_1.0"])
    g, _ = geometric_difference(K, K)
    assert g < 1.5


def test_advantage_controls_separate():
    """The detector's contract: quantum wins on quantum-engineered labels,
    classical wins/ties on classically-engineered labels."""
    rep = advantage_experiment(n_qubits=4, n_samples=120, n_layers=2, seed=0)
    assert rep.g_min > 1.0
    # positive control
    assert rep.r2_engineered["quantum"] > best_classical_r2(rep.r2_engineered)
    # negative control (allow ties within tolerance)
    assert (
        best_classical_r2(rep.r2_classical_natural)
        >= rep.r2_classical_natural["quantum"] - 0.05
    )


# ---------------------------------------------------------------------------
# Architecture upgrades
# ---------------------------------------------------------------------------


def test_readout_dim():
    assert readout_dim(4, "z") == 4
    assert readout_dim(4, "zz") == 4 + 6


def _core_out(qcfg: QuantumConfig, out_features=5):
    core = QuantumCore.from_config(qcfg, out_features=out_features)
    x = jnp.asarray(np.random.default_rng(0).normal(size=(2, 3, 8)))
    params = core.init(jax.random.PRNGKey(0), x)["params"]
    out = core.apply({"params": params}, x)
    return params, out


def test_zz_readout_shapes_and_bounds():
    qcfg = QuantumConfig(n_qubits=3, n_circuit_layers=1, readout="zz")
    params, out = _core_out(qcfg)
    assert out.shape == (2, 3, 5)
    assert jnp.isfinite(out).all()


def test_parallel_circuits_shapes_and_grads():
    qcfg = QuantumConfig(n_qubits=2, n_circuit_layers=1, n_circuits=3)
    core = QuantumCore.from_config(qcfg, out_features=4)
    x = jnp.asarray(np.random.default_rng(1).normal(size=(2, 8)))
    params = core.init(jax.random.PRNGKey(0), x)["params"]
    assert params["circuit_weights"].shape == (3, 1, 2, 3)

    grads = jax.grad(lambda p: (core.apply({"params": p}, x) ** 2).sum())(params)
    per_circuit = jnp.abs(grads["circuit_weights"]).sum(axis=(1, 2, 3))
    assert (per_circuit > 0).all(), "every parallel circuit must receive gradient"


def test_linear_dressing_forward():
    qcfg = QuantumConfig(n_qubits=2, n_circuit_layers=1, dressing="linear")
    _, out = _core_out(qcfg)
    assert jnp.isfinite(out).all()


def test_small_angle_init_scale():
    qcfg = QuantumConfig(n_qubits=2, n_circuit_layers=1, init_scale=0.1)
    params, _ = _core_out(qcfg)
    w = params["circuit_weights"]
    assert float(jnp.abs(w).max()) <= 0.1 + 1e-6


def test_grad_norm_step_reports_groups(tiny_quantum_cfg, tmp_path):
    res = fit(tiny_quantum_cfg, verbose=False, out_dir=tmp_path)
    last = res["summary"]["history"][-1]
    assert "grad_norm_circuit" in last and "grad_norm_classical" in last
    assert np.isfinite(last["grad_norm_circuit"])
    assert last["grad_norm_classical"] > 0


def test_v03_options_through_full_model():
    cfg = ModelConfig(
        d_model=16,
        n_heads=2,
        n_blocks=1,
        d_ff=32,
        max_seq_len=16,
        ffn_type="quantum",
        quantum=QuantumConfig(
            n_qubits=2,
            n_circuit_layers=1,
            readout="zz",
            dressing="linear",
            init_scale=0.3,
            n_circuits=2,
        ),
    )
    model, _ = build_model(cfg, vocab_size=7)
    tokens = jnp.array(np.random.default_rng(2).integers(0, 7, (2, 6)))
    params = model.init(jax.random.PRNGKey(0), tokens)["params"]
    logits = model.apply({"params": params}, tokens)
    assert logits.shape == (2, 6, 7)
    assert jnp.isfinite(logits).all()


def test_grad_norm_step_unit():
    cfg = ModelConfig(
        d_model=16,
        n_heads=2,
        n_blocks=1,
        d_ff=32,
        max_seq_len=16,
        ffn_type="quantum",
        quantum=QuantumConfig(n_qubits=2, n_circuit_layers=1),
    )
    model, _ = build_model(cfg, vocab_size=7)
    tokens = jnp.array(np.random.default_rng(3).integers(0, 7, (2, 7)))
    params = model.init(jax.random.PRNGKey(0), tokens)["params"]

    import optax
    from flax.training.train_state import TrainState

    state = TrainState.create(
        apply_fn=model.apply, params=params, tx=optax.adamw(1e-3)
    )
    g_circ, g_other = make_grad_norm_step()(state, tokens)
    assert float(g_circ) >= 0 and float(g_other) > 0
