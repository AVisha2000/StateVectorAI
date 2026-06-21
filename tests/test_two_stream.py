"""Two-stream LM: encoder variants, conditioning modes, param matching."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from qllm.config import ModelConfig, QuantumConfig
from qllm.models.model import build_model, uses_quantum
from qllm.models.two_stream import TwoStreamLM

CFG = ModelConfig(d_model=32, n_heads=2, n_blocks=2, d_ff=64, max_seq_len=64,
                  quantum=QuantumConfig(n_qubits=4, n_circuit_layers=2))
T = jnp.array(np.random.default_rng(0).integers(0, 16, (2, 12)))


def _params(model):
    p = model.init(jax.random.PRNGKey(0), T)["params"]
    return p, sum(int(np.prod(v.shape)) for v in jax.tree_util.tree_leaves(p))


def test_all_nine_configs_build_and_run():
    cfg = ModelConfig(**{**CFG.__dict__, "vocab_size": 16})
    for kind in ("quantum", "classical", "none"):
        for cond in ("film", "token", "bias"):
            m = TwoStreamLM(cfg=cfg, encoder_kind=kind, condition=cond, d_sent=8)
            p, _ = _params(m)
            out = m.apply({"params": p}, T)
            assert out.shape == (2, 12, 16) and jnp.isfinite(out).all()


def test_quantum_classical_param_matched_within_2pct():
    cfg = ModelConfig(**{**CFG.__dict__, "vocab_size": 16})
    _, nq = _params(TwoStreamLM(cfg=cfg, encoder_kind="quantum",
                                condition="bias", d_sent=8))
    _, nc = _params(TwoStreamLM(cfg=cfg, encoder_kind="classical",
                                condition="bias", d_sent=8, sent_hidden=8))
    assert abs(nq - nc) / nq < 0.02, f"params not matched: q={nq} c={nc}"


def test_causality_token_mode():
    """Virtual-token mode must still be causal in the real tokens."""
    cfg = ModelConfig(**{**CFG.__dict__, "vocab_size": 16})
    m = TwoStreamLM(cfg=cfg, encoder_kind="none", condition="token", d_sent=8)
    p, _ = _params(m)
    a = m.apply({"params": p}, T)
    t2 = T.at[:, -1].set((T[:, -1] + 1) % 16)
    b = m.apply({"params": p}, t2)
    # encoder=none so no leakage path; prefixes must match
    np.testing.assert_allclose(a[:, :-1], b[:, :-1], rtol=1e-4, atol=1e-4)


def test_arch_dispatch_and_uses_quantum():
    q = ModelConfig(arch="two_stream", encoder_kind="quantum",
                    condition="film", quantum=CFG.quantum)
    model, _ = build_model(q, vocab_size=16)
    assert isinstance(model, TwoStreamLM)
    assert uses_quantum(q)
    c = ModelConfig(arch="two_stream", encoder_kind="classical")
    assert not uses_quantum(c)
