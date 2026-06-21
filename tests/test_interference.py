"""Interference output head: expressivity separation from positive mixtures."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from qllm.config import ModelConfig
from qllm.models.model import build_model
from qllm.quantum.interference_head import (
    InterferenceHead, LinearHead, MixtureHead)


def test_heads_shapes_and_finiteness():
    x = jnp.asarray(np.random.default_rng(0).normal(size=(2, 5, 16)))
    for Head, kw in [(LinearHead, {}), (MixtureHead, dict(n_hypotheses=4)),
                     (InterferenceHead, dict(n_hypotheses=2))]:
        m = Head(vocab_size=8, **kw)
        p = m.init(jax.random.PRNGKey(0), x)["params"]
        out = m.apply({"params": p}, x)
        assert out.shape == (2, 5, 8)
        assert jnp.isfinite(out).all()


def test_interference_and_mixture_param_matched():
    """interference-K and mixture-2K must have equal parameter counts (the
    fair control: complex doubles per-branch params)."""
    x = jnp.asarray(np.random.default_rng(0).normal(size=(1, 1, 16)))
    pi = InterferenceHead(vocab_size=16, n_hypotheses=2).init(
        jax.random.PRNGKey(0), x)["params"]
    pm = MixtureHead(vocab_size=16, n_hypotheses=4).init(
        jax.random.PRNGKey(0), x)["params"]
    ni = sum(int(np.prod(v.shape)) for v in jax.tree_util.tree_leaves(pi))
    nm = sum(int(np.prod(v.shape)) for v in jax.tree_util.tree_leaves(pm))
    assert ni == nm


def test_mixture_is_normalized():
    """Positive mixture outputs are exact log-probs (logsumexp == 0)."""
    x = jnp.asarray(np.random.default_rng(1).normal(size=(3, 4, 16)))
    m = MixtureHead(vocab_size=16, n_hypotheses=4)
    p = m.init(jax.random.PRNGKey(0), x)["params"]
    out = m.apply({"params": p}, x)
    np.testing.assert_allclose(
        np.asarray(jax.nn.logsumexp(out, axis=-1)), 0.0, atol=1e-4)


def test_interference_beats_mixture_on_cancellation():
    """The core claim: on an XOR-cancellation target from linear features,
    a coherent head reaches the entropy floor while a param-matched positive
    mixture cannot."""
    rng = np.random.default_rng(0)
    V, q, N = 16, 8, 1024
    a = rng.integers(0, 2, N)
    # allowed: base [0,q); conflict tail [q,V) allowed iff a==1 ... but to need
    # cancellation, forbid the tail when a==0 AND require it present when a==1
    # via a SECOND feature b with XOR:
    b = rng.integers(0, 2, N)
    P = np.zeros((N, V))
    for i in range(N):
        s = list(range(q))
        if a[i] ^ b[i]:
            s += list(range(q, V))
        P[i, s] = 1.0 / len(s)
    X = np.stack([a, b, np.ones(N)], 1).astype(np.float32)
    Xj, Pj = jnp.asarray(X), jnp.asarray(P)
    floor = float(-(Pj * jnp.log(Pj + 1e-12)).sum(-1).mean())

    def fit(Head, **kw):
        m = Head(vocab_size=V, **kw)
        p = m.init(jax.random.PRNGKey(0), Xj[:2])["params"]
        opt = optax.adam(5e-3); st = opt.init(p)

        @jax.jit
        def step(p, st):
            def loss(p):
                lg = m.apply({"params": p}, Xj)
                logp = lg - jax.nn.logsumexp(lg, -1, keepdims=True)
                return -(Pj * logp).sum(-1).mean()
            l, g = jax.value_and_grad(loss)(p)
            u, st = opt.update(g, st, p)
            return optax.apply_updates(p, u), st, l
        l = None
        for _ in range(1500):
            p, st, l = step(p, st)
        return float(l)

    ce_int = fit(InterferenceHead, n_hypotheses=2)
    ce_mix = fit(MixtureHead, n_hypotheses=4)
    # interference essentially reaches the floor; mixture retains a gap. The
    # gap's size depends on target structure (the benchmark's richer
    # cancellation widens it); here we assert the robust qualitative signal.
    assert ce_int - floor < 0.01, f"interference failed: {ce_int - floor:.4f}"
    assert ce_mix - ce_int > 0.001, (
        f"no separation: mix={ce_mix:.4f} int={ce_int:.4f}")
    assert ce_mix >= ce_int, "mixture should never beat interference here"


def test_head_dispatch_in_model():
    t = jnp.array(np.random.default_rng(0).integers(0, 16, (2, 8)))
    for ht, H in [("interference", 2), ("mixture", 4)]:
        cfg = ModelConfig(d_model=32, n_heads=2, n_blocks=1, d_ff=64,
                          max_seq_len=64, head_type=ht, head_hypotheses=H)
        model, _ = build_model(cfg, vocab_size=16)
        p = model.init(jax.random.PRNGKey(0), t)["params"]
        assert model.apply({"params": p}, t).shape == (2, 8, 16)
