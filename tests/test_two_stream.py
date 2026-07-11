"""Two-stream LM: encoder variants, conditioning modes, param matching."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qllm.config import ModelConfig, QuantumConfig, two_stream_position_count
from qllm.data.text import CharTokenizer
from qllm.models.model import build_model, count_model_params, uses_quantum
from qllm.models.two_stream import (
    QuantumSentenceEncoder,
    TwoStreamLM,
    causal_prefix_mean,
    interleave_summary_tokens,
    real_token_slots,
)
from qllm.train.loop import generate

CFG = ModelConfig(d_model=32, n_heads=2, n_blocks=2, d_ff=64, max_seq_len=64,
                  quantum=QuantumConfig(n_qubits=4, n_circuit_layers=2))
T = jnp.array(np.random.default_rng(0).integers(0, 16, (2, 12)))
LEAK_TOKENS = jnp.asarray([[0, 1, 2, 3, 4, 5]], dtype=jnp.int32)
LEAK_CFG = ModelConfig(
    vocab_size=8,
    d_model=8,
    n_heads=2,
    n_blocks=1,
    d_ff=16,
    max_seq_len=12,
    quantum=QuantumConfig(
        n_qubits=2, n_circuit_layers=1, init_scale=0.3
    ),
)


def _params(model):
    p = model.init(jax.random.PRNGKey(0), T)["params"]
    return p, sum(int(np.prod(v.shape)) for v in jax.tree_util.tree_leaves(p))


def test_causal_prefix_mean_includes_current_token_only():
    embeddings = jnp.asarray([[[2.0, 4.0], [4.0, 8.0], [9.0, 3.0]]])
    expected = np.asarray([[[2.0, 4.0], [3.0, 6.0], [5.0, 5.0]]])
    np.testing.assert_allclose(causal_prefix_mean(embeddings), expected)


def test_token_interleave_order_and_real_slot_selection():
    summaries = jnp.asarray([[[10.0], [20.0], [30.0]]])
    real = jnp.asarray([[[1.0], [2.0], [3.0]]])
    interleaved = interleave_summary_tokens(summaries, real)
    np.testing.assert_array_equal(
        interleaved[..., 0], np.asarray([[10.0, 1.0, 20.0, 2.0, 30.0, 3.0]])
    )
    np.testing.assert_array_equal(real_token_slots(interleaved), real)


def test_quantum_encoder_vectorizes_batch_and_time_axes():
    prefixes = jnp.arange(2 * 3 * 8, dtype=jnp.float32).reshape(2, 3, 8) / 10
    encoder = QuantumSentenceEncoder(
        d_sent=4,
        quantum=QuantumConfig(n_qubits=2, n_circuit_layers=1),
    )
    params = encoder.init(jax.random.PRNGKey(1), prefixes)
    summaries = encoder.apply(params, prefixes)
    assert summaries.shape == (2, 3, 4)


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


@pytest.mark.parametrize(
    ("kind", "condition"),
    [
        ("quantum", "film"),
        ("quantum", "token"),
        ("quantum", "bias"),
        ("classical", "film"),
        ("classical", "token"),
        ("classical", "bias"),
        ("none", "token"),
    ],
)
def test_future_tokens_cannot_change_earlier_logits(kind, condition):
    model = TwoStreamLM(
        cfg=LEAK_CFG,
        encoder_kind=kind,
        condition=condition,
        d_sent=4,
        sent_hidden=4,
    )
    variables = model.init(jax.random.PRNGKey(2), LEAK_TOKENS)
    before = model.apply(variables, LEAK_TOKENS)
    cut = 3
    perturbed = LEAK_TOKENS.at[:, cut:].set(
        (LEAK_TOKENS[:, cut:] + 1) % LEAK_CFG.vocab_size
    )
    after = model.apply(variables, perturbed)
    np.testing.assert_allclose(
        before[:, :cut], after[:, :cut], rtol=1e-5, atol=1e-5
    )
    assert not np.allclose(before[:, cut:], after[:, cut:])


def test_direct_model_guard_and_exact_token_capacity_boundary():
    too_small = ModelConfig(**{
        **LEAK_CFG.__dict__, "max_seq_len": 2 * LEAK_TOKENS.shape[1] - 1,
    })
    model = TwoStreamLM(
        cfg=too_small,
        encoder_kind="classical",
        condition="token",
        d_sent=4,
        sent_hidden=4,
    )
    with pytest.raises(ValueError, match="requires 12 internal positions"):
        model.init(jax.random.PRNGKey(3), LEAK_TOKENS)

    exact = ModelConfig(**{
        **LEAK_CFG.__dict__, "max_seq_len": 2 * LEAK_TOKENS.shape[1],
    })
    exact_model = TwoStreamLM(
        cfg=exact,
        encoder_kind="classical",
        condition="token",
        d_sent=4,
        sent_hidden=4,
    )
    variables = exact_model.init(jax.random.PRNGKey(3), LEAK_TOKENS)
    logits = exact_model.apply(variables, LEAK_TOKENS)
    assert logits.shape == (*LEAK_TOKENS.shape, LEAK_CFG.vocab_size)
    assert variables["params"]["pos_embed"].shape == (1, 12, LEAK_CFG.d_model)


def test_parameter_count_clamps_requested_length_to_real_token_capacity():
    cfg = ModelConfig(
        **{
            **LEAK_CFG.__dict__,
            "arch": "two_stream",
            "encoder_kind": "classical",
            "condition": "token",
            "max_seq_len": 10,
        }
    )
    counted = count_model_params(cfg, vocab_size=cfg.vocab_size, seq_len=8)

    model, _ = build_model(cfg, vocab_size=cfg.vocab_size)
    max_real_tokens = cfg.max_seq_len // 2
    variables = model.init(
        jax.random.PRNGKey(0),
        jnp.zeros((1, max_real_tokens), dtype=jnp.int32),
    )
    direct = sum(
        int(np.prod(value.shape))
        for value in jax.tree_util.tree_leaves(variables["params"])
    )
    assert counted == direct


def test_generation_uses_real_token_capacity_and_preserves_output_length():
    tokenizer = CharTokenizer("abcdefgh")
    cfg = ModelConfig(
        **{
            **LEAK_CFG.__dict__,
            "arch": "two_stream",
            "encoder_kind": "classical",
            "condition": "token",
            "max_seq_len": 6,
        }
    )
    model, _ = build_model(cfg, vocab_size=tokenizer.vocab_size)
    max_real_tokens = cfg.max_seq_len // 2
    variables = model.init(
        jax.random.PRNGKey(0),
        jnp.zeros((1, max_real_tokens), dtype=jnp.int32),
    )

    prompt = "abcdefgh"
    max_new_tokens = 4
    generated = generate(
        model,
        variables["params"],
        tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        seed=0,
    )
    assert len(generated) == len(prompt) + max_new_tokens


@pytest.mark.parametrize("condition", ["film", "bias"])
def test_direct_model_guard_for_non_token_conditioning(condition):
    cfg = ModelConfig(**{
        **LEAK_CFG.__dict__, "max_seq_len": LEAK_TOKENS.shape[1] - 1,
    })
    model = TwoStreamLM(
        cfg=cfg,
        encoder_kind="classical",
        condition=condition,
        d_sent=4,
        sent_hidden=4,
    )
    with pytest.raises(ValueError, match="requires 6 internal positions"):
        model.init(jax.random.PRNGKey(4), LEAK_TOKENS)


def test_none_encoder_does_not_interleave_in_token_mode():
    assert two_stream_position_count(6, "none", "token") == 6
    cfg = ModelConfig(**{**LEAK_CFG.__dict__, "max_seq_len": 6})
    model = TwoStreamLM(
        cfg=cfg, encoder_kind="none", condition="token", d_sent=4
    )
    variables = model.init(jax.random.PRNGKey(5), LEAK_TOKENS)
    logits = model.apply(variables, LEAK_TOKENS)
    assert logits.shape == (*LEAK_TOKENS.shape, LEAK_CFG.vocab_size)
    assert "sent_to_tok" not in variables["params"]


def test_arch_dispatch_and_uses_quantum():
    q = ModelConfig(arch="two_stream", encoder_kind="quantum",
                    condition="film", quantum=CFG.quantum)
    model, _ = build_model(q, vocab_size=16)
    assert isinstance(model, TwoStreamLM)
    assert uses_quantum(q)
    c = ModelConfig(arch="two_stream", encoder_kind="classical")
    assert not uses_quantum(c)
