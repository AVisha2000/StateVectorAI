"""Ablation infrastructure tests: circuit freezing + parameter matching."""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from flax import traverse_util

from qllm.config import ModelConfig, QuantumConfig, TrainConfig
from qllm.models.model import (
    build_model,
    count_model_params,
    matched_classical_d_ff,
)
from qllm.train.loop import fit, make_optimizer

QMODEL = ModelConfig(
    d_model=16,
    n_heads=2,
    n_blocks=1,
    d_ff=32,
    max_seq_len=16,
    ffn_type="quantum",
    quantum=QuantumConfig(n_qubits=2, n_circuit_layers=1),
)


def _params():
    model, _ = build_model(QMODEL, vocab_size=7)
    tokens = jnp.zeros((1, 6), dtype=jnp.int32)
    return model.init(jax.random.PRNGKey(0), tokens)["params"]


def _circuit_update_norms(freeze: bool):
    params = _params()
    tx = make_optimizer(TrainConfig(), params, freeze_circuit=freeze)
    opt_state = tx.init(params)
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    updates, _ = tx.update(grads, opt_state, params)
    circuit, other = [], []
    for key, value in traverse_util.flatten_dict(updates).items():
        target = circuit if "circuit_weights" in key else other
        target.append(float(jnp.abs(value).max()))
    return circuit, other


def test_optimizer_freezes_circuit_weights():
    circuit, other = _circuit_update_norms(freeze=True)
    assert circuit and other  # both groups present in the tree
    assert all(c == 0.0 for c in circuit), "frozen circuit received updates"
    assert any(o > 0.0 for o in other), "trainable params received no updates"


def test_optimizer_updates_circuit_weights_when_trainable():
    circuit, _ = _circuit_update_norms(freeze=False)
    assert all(c > 0.0 for c in circuit)


def test_matched_d_ff_within_tolerance():
    target = count_model_params(QMODEL, vocab_size=7)
    d_ff = matched_classical_d_ff(QMODEL, vocab_size=7)
    twin = dataclasses.replace(QMODEL, ffn_type="classical", d_ff=d_ff)
    got = count_model_params(twin, vocab_size=7)
    assert abs(got - target) / target < 0.05


def test_frozen_flag_plumbed_through_fit(tiny_quantum_cfg, tmp_path):
    """Same seed => identical init; trained circuit must move, frozen not.

    Combined with test_optimizer_freezes_circuit_weights this verifies the
    frozen run stays exactly at its random initialization.
    """
    frozen_cfg = dataclasses.replace(
        tiny_quantum_cfg,
        model=dataclasses.replace(
            tiny_quantum_cfg.model,
            quantum=dataclasses.replace(
                tiny_quantum_cfg.model.quantum, trainable=False
            ),
        ),
    )
    trained = fit(tiny_quantum_cfg, verbose=False, out_dir=tmp_path / "t")
    frozen = fit(frozen_cfg, verbose=False, out_dir=tmp_path / "f")

    def circuit_weights(res):
        return res["state"].params["block_0"]["ffn"]["QuantumCore_0"][
            "circuit_weights"
        ]

    cw_trained = np.asarray(circuit_weights(trained))
    cw_frozen = np.asarray(circuit_weights(frozen))
    assert not np.allclose(cw_trained, cw_frozen)
    assert np.isfinite(frozen["summary"]["val_loss"])
