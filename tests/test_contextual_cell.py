"""Contextual quantum cell: parity-tracking expressivity + integration."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from qllm.config import ModelConfig, QuantumConfig
from qllm.models.model import build_model
from qllm.quantum.contextual_cell import ContextualQRNN


def test_shapes_causality_grads():
    m = ContextualQRNN(vocab_size=14, n_phase=4, n_layers=2)
    t = jnp.array(np.random.default_rng(0).integers(0, 14, (3, 16)))
    p = m.init(jax.random.PRNGKey(0), t)["params"]
    out = m.apply({"params": p}, t)
    assert out.shape == (3, 16, 14) and jnp.isfinite(out).all()
    t2 = t.at[:, -1].set((t[:, -1] + 1) % 14)
    assert np.allclose(m.apply({"params": p}, t)[:, :-1],
                       m.apply({"params": p}, t2)[:, :-1], atol=1e-5)
    g = jax.grad(lambda pp: (m.apply({"params": pp}, t) ** 2).sum())(p)
    for k in ("token_phase", "token_rot", "zz_phase"):
        assert float(jnp.abs(g[k]).sum()) > 0


def test_tracks_running_parity_exactly():
    """The core capability: phase-accumulate a single running parity and read
    it by interference. This is what the contextuality escape relies on."""
    rng = np.random.default_rng(0)
    seqs = rng.integers(0, 2, (256, 24))
    parity = np.cumsum(seqs, 1) % 2
    X, Y = jnp.asarray(seqs), jnp.asarray(parity)
    m = ContextualQRNN(vocab_size=2, n_phase=3, n_layers=1)
    p = m.init(jax.random.PRNGKey(0), X[:2])["params"]
    opt = optax.adam(1e-2); st = opt.init(p)

    @jax.jit
    def step(p, st):
        def loss(p):
            lg = m.apply({"params": p}, X)
            return optax.softmax_cross_entropy_with_integer_labels(lg, Y).mean()
        l, g = jax.value_and_grad(loss)(p)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l
    for _ in range(1500):
        p, st, _ = step(p, st)
    acc = float((m.apply({"params": p}, X).argmax(-1) == Y).mean())
    assert acc > 0.95, f"cell cannot track running parity: acc={acc:.3f}"


def test_arch_dispatch():
    cfg = ModelConfig(arch="contextual_qrnn",
                      quantum=QuantumConfig(n_qubits=4, n_circuit_layers=2))
    model, _ = build_model(cfg, vocab_size=14)
    assert isinstance(model, ContextualQRNN)


def test_routed_cell_shapes_and_routing_grads():
    from qllm.quantum.contextual_cell import RoutedContextualQRNN
    m = RoutedContextualQRNN(vocab_size=14, n_phase=5, n_layers=1)
    t = jnp.array(np.random.default_rng(0).integers(0, 14, (3, 20)))
    p = m.init(jax.random.PRNGKey(0), t)["params"]
    out = m.apply({"params": p}, t)
    assert out.shape == (3, 20, 14) and jnp.isfinite(out).all()
    g = jax.grad(lambda pp: (m.apply({"params": pp}, t) ** 2).sum())(p)
    assert float(jnp.abs(g["cue_to_qubit"]).sum()) > 0


def test_routed_cell_beats_chance_on_two_stream_parity():
    """Routing works in isolation: 2 interleaved parity streams with explicit
    cues are tracked above chance (the unrouted cell cannot, by design)."""
    from qllm.quantum.contextual_cell import RoutedContextualQRNN
    rng = np.random.default_rng(0)
    N, L = 192, 40
    X = np.zeros((N, L), np.int32); Y = np.zeros((N, L), np.int32)
    for s in range(N):
        pa = pb = active = 0
        for i in range(L):
            if i % 2 == 0:
                active = int(rng.integers(0, 2)); X[s, i] = active; Y[s, i] = active
            else:
                b = int(rng.integers(0, 2))
                if active == 0:
                    pa ^= b
                else:
                    pb ^= b
                X[s, i] = 2 + b; Y[s, i] = pa if active == 0 else pb
    Xj, Yj = jnp.asarray(X), jnp.asarray(Y)
    m = RoutedContextualQRNN(vocab_size=4, n_phase=3, n_layers=1)
    p = m.init(jax.random.PRNGKey(0), Xj[:2])["params"]
    opt = optax.adam(1e-2); st = opt.init(p)
    maskL = jnp.asarray(np.arange(L) % 2 == 1)

    @jax.jit
    def step(p, st):
        def loss(p):
            ce = optax.softmax_cross_entropy_with_integer_labels(
                m.apply({"params": p}, Xj), Yj)
            return (ce * maskL[None, :]).sum() / maskL.sum()
        l, g = jax.value_and_grad(loss)(p)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l
    for _ in range(2000):
        p, st, _ = step(p, st)
    pred = np.asarray(m.apply({"params": p}, Xj).argmax(-1))
    vm = np.broadcast_to(np.asarray(maskL)[None, :], pred.shape)
    acc = (pred[vm] == Y[vm]).mean()
    assert acc > 0.65, f"routing failed in isolation: acc={acc:.3f}"
