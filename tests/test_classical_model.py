"""Classical baseline tests: shapes, causality, gradient flow, jit parity."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from qllm.config import ModelConfig
from qllm.models.model import build_model

VOCAB = 11


def _make(d_model=16, n_heads=2, n_blocks=2):
    cfg = ModelConfig(
        d_model=d_model, n_heads=n_heads, n_blocks=n_blocks, d_ff=32, max_seq_len=16
    )
    model, final_cfg = build_model(cfg, vocab_size=VOCAB)
    tokens = jnp.array(np.random.default_rng(0).integers(0, VOCAB, (2, 8)))
    params = model.init(jax.random.PRNGKey(0), tokens)["params"]
    return model, final_cfg, params, tokens


def test_forward_shape():
    model, _, params, tokens = _make()
    logits = model.apply({"params": params}, tokens)
    assert logits.shape == (2, 8, VOCAB)
    assert jnp.isfinite(logits).all()


def test_causality():
    """Changing a later token must not change logits at earlier positions."""
    model, _, params, tokens = _make()
    logits_a = model.apply({"params": params}, tokens)
    perturbed = tokens.at[:, -1].set((tokens[:, -1] + 1) % VOCAB)
    logits_b = model.apply({"params": params}, perturbed)
    np.testing.assert_allclose(
        logits_a[:, :-1], logits_b[:, :-1], rtol=1e-5, atol=1e-5
    )
    assert not np.allclose(logits_a[:, -1], logits_b[:, -1])


def test_gradients_flow_everywhere():
    model, _, params, tokens = _make()
    targets = jnp.roll(tokens, -1, axis=1)

    def loss_fn(p):
        logits = model.apply({"params": p}, tokens)
        return optax.softmax_cross_entropy_with_integer_labels(
            logits, targets
        ).mean()

    grads = jax.grad(loss_fn)(params)
    norms = [float(jnp.abs(g).sum()) for g in jax.tree_util.tree_leaves(grads)]
    assert all(np.isfinite(norms))
    assert all(n > 0 for n in norms), "some parameter received zero gradient"


def test_jit_matches_eager():
    model, _, params, tokens = _make()
    eager = model.apply({"params": params}, tokens)
    jitted = jax.jit(lambda p, t: model.apply({"params": p}, t))(params, tokens)
    np.testing.assert_allclose(eager, jitted, rtol=1e-5, atol=1e-6)
