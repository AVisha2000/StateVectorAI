"""Recurrent models: quantum HMM cell (QRNNLM) and GRU baseline."""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qllm.classical.recurrent import GRULM
from qllm.config import DataConfig, ModelConfig, QuantumConfig
from qllm.models.model import build_model
from qllm.quantum.recurrent import QRNNLM
from qllm.train.loop import fit

TOKENS = jnp.array(np.random.default_rng(0).integers(0, 2, (3, 12)))


def _qrnn(n_qubits=4, n_layers=2, vocab=2):
    model = QRNNLM(vocab_size=vocab, n_qubits=n_qubits, n_layers=n_layers)
    params = model.init(jax.random.PRNGKey(0), TOKENS)["params"]
    return model, params


def test_qrnn_shapes_and_finiteness():
    model, params = _qrnn()
    logits = model.apply({"params": params}, TOKENS)
    assert logits.shape == (3, 12, 2)
    assert jnp.isfinite(logits).all()


def test_qrnn_emission_probabilities_normalized():
    """At init (scale=1, bias=0): logits = log p, so exp must sum to ~1."""
    model, params = _qrnn()
    logits = model.apply({"params": params}, TOKENS)
    sums = jnp.exp(logits).sum(axis=-1)
    np.testing.assert_allclose(np.asarray(sums), 1.0, atol=1e-3)


def test_qrnn_causality():
    model, params = _qrnn()
    a = model.apply({"params": params}, TOKENS)
    perturbed = TOKENS.at[:, -1].set(1 - TOKENS[:, -1])
    b = model.apply({"params": params}, perturbed)
    np.testing.assert_allclose(a[:, :-1], b[:, :-1], rtol=1e-4, atol=1e-5)


def test_qrnn_gradients_flow():
    model, params = _qrnn()
    targets = jnp.roll(TOKENS, -1, axis=1)

    def loss(p):
        import optax

        logits = model.apply({"params": p}, TOKENS)
        return optax.softmax_cross_entropy_with_integer_labels(
            logits, targets
        ).mean()

    grads = jax.grad(loss)(params)
    for key in ("circuit_weights", "inject_angles"):
        assert float(jnp.abs(grads[key]).sum()) > 0.0, f"no gradient to {key}"


def test_qrnn_rejects_bad_vocab():
    model = QRNNLM(vocab_size=3, n_qubits=4)
    with pytest.raises(AssertionError):
        model.init(jax.random.PRNGKey(0), TOKENS)


def test_gru_shapes():
    model = GRULM(vocab_size=2, hidden=8)
    params = model.init(jax.random.PRNGKey(0), TOKENS)["params"]
    logits = model.apply({"params": params}, TOKENS)
    assert logits.shape == (3, 12, 2)
    assert jnp.isfinite(logits).all()


def test_build_model_arch_dispatch():
    qrnn, _ = build_model(
        ModelConfig(arch="qrnn", quantum=QuantumConfig(n_qubits=4)), vocab_size=2
    )
    assert isinstance(qrnn, QRNNLM)
    gru, _ = build_model(ModelConfig(arch="gru", rnn_hidden=4), vocab_size=2)
    assert isinstance(gru, GRULM)
    with pytest.raises(ValueError):
        build_model(ModelConfig(arch="nope"), vocab_size=2)


def test_fit_qrnn_on_quantum_data(tiny_classical_cfg, tmp_path):
    cfg = dataclasses.replace(
        tiny_classical_cfg,
        model=ModelConfig(
            arch="qrnn", quantum=QuantumConfig(n_qubits=4, n_circuit_layers=2)
        ),
        data=DataConfig(
            kind="monitored_ising",
            gen_qubits=4,
            gen_measured=1,
            gen_sequences=2,
            gen_len=256,
        ),
    )
    res = fit(cfg, verbose=False, out_dir=tmp_path)
    assert np.isfinite(res["summary"]["val_loss"])
    # grad-norm telemetry must engage (qrnn counts as quantum)
    assert "grad_norm_circuit" in res["summary"]["history"][-1]
