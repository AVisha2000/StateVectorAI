"""v0.6: quantum embedding, quantum-QKV attention, results database."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from qllm.config import ModelConfig, QuantumConfig
from qllm.models.model import build_model
from qllm.quantum.layers import QuantumEmbedding, QuantumQKVAttention
from qllm.resultsdb import ResultsDB

QCFG = QuantumConfig(n_qubits=2, n_circuit_layers=1, readout="zz")
TOKENS = jnp.array(np.random.default_rng(0).integers(0, 7, (2, 6)))


def test_quantum_embedding_shapes_and_grads():
    emb = QuantumEmbedding(vocab_size=7, d_model=16, quantum=QCFG)
    params = emb.init(jax.random.PRNGKey(0), TOKENS)["params"]
    out = emb.apply({"params": params}, TOKENS)
    assert out.shape == (2, 6, 16)
    assert jnp.isfinite(out).all()

    grads = jax.grad(lambda p: (emb.apply({"params": p}, TOKENS) ** 2).sum())(
        params
    )
    for key in ("token_angles", "circuit_weights"):
        assert float(jnp.abs(grads[key]).sum()) > 0.0, f"no grad to {key}"


def test_quantum_qkv_attention_shapes_and_grads():
    attn = QuantumQKVAttention(d_model=16, n_heads=2, quantum=QCFG)
    x = jnp.asarray(np.random.default_rng(1).normal(size=(2, 6, 16)))
    params = attn.init(jax.random.PRNGKey(0), x)["params"]
    out = attn.apply({"params": params}, x)
    assert out.shape == (2, 6, 16)
    assert jnp.isfinite(out).all()

    grads = jax.grad(lambda p: (attn.apply({"params": p}, x) ** 2).sum())(params)
    g_circ = grads["quantum_qkv"]["circuit_weights"]
    assert float(jnp.abs(g_circ).sum()) > 0.0


def test_full_quantum_model_causality_and_training_step():
    cfg = ModelConfig(
        d_model=16,
        n_heads=2,
        n_blocks=1,
        d_ff=32,
        max_seq_len=16,
        embed_type="quantum",
        attn_type="quantum_qkv",
        ffn_type="quantum",
        quantum=QCFG,
    )
    model, _ = build_model(cfg, vocab_size=7)
    params = model.init(jax.random.PRNGKey(0), TOKENS)["params"]
    logits_a = model.apply({"params": params}, TOKENS)
    perturbed = TOKENS.at[:, -1].set((TOKENS[:, -1] + 1) % 7)
    logits_b = model.apply({"params": params}, perturbed)
    np.testing.assert_allclose(
        logits_a[:, :-1], logits_b[:, :-1], rtol=1e-4, atol=1e-4
    )

    targets = jnp.roll(TOKENS, -1, axis=1)

    def loss(p):
        lg = model.apply({"params": p}, TOKENS)
        return optax.softmax_cross_entropy_with_integer_labels(lg, targets).mean()

    grads = jax.grad(loss)(params)
    total = sum(
        float(jnp.abs(g).sum()) for g in jax.tree_util.tree_leaves(grads)
    )
    assert np.isfinite(total) and total > 0


def test_resultsdb_roundtrip(tmp_path):
    db = ResultsDB(tmp_path / "r.db")
    assert not db.exists("s", "v", "d", 0, 100)
    first = db.record(
        suite="s", variant="v", dataset="d", seed=0, steps=100,
        n_params=42, val_loss=1.0, val_ppl=2.7, val_bpc=1.44,
        wall_seconds=3.2, config={"model.arch": "transformer"},
    )
    assert db.exists("s", "v", "d", 0, 100)
    rows = db.fetch("s")
    assert len(rows) == 1 and rows[0]["val_ppl"] == 2.7

    # The legacy composite projection remains first-write stable, while each
    # UUID-backed attempt is retained canonically without rewriting evidence.
    second = db.record(
        suite="s", variant="v", dataset="d", seed=0, steps=100,
        n_params=42, val_loss=0.9, val_ppl=2.5, val_bpc=1.3,
        wall_seconds=3.0,
    )
    rows = db.fetch("s")
    assert len(rows) == 1 and rows[0]["val_ppl"] == 2.7
    assert first["run_uuid"] != second["run_uuid"]
    second_row = db.get_run(
        "s", "v", "d", 0, 100, run_uuid=second["run_uuid"]
    )
    assert second_row["val_ppl"] == 2.5

    assert db.delete("s") == 1
    assert db.fetch("s") == []
