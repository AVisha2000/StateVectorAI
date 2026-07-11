"""Two-stream LM: a sentence encoder guides a classical word transformer.

Hypothesis under test (NOT assumed): can a compact SENTENCE-level summary,
injected into a per-token word model, supply a useful inductive bias — and
does a QUANTUM encoder do so better than a parameter-matched CLASSICAL one?
Every quantum-readout featurizer in this project (q-embed v0.6, transplant
v0.7, interference v0.12) underperformed a classical layer of matched size
on classical text, so the strict control is the whole point: quantum vs
classical sentence encoder at IDENTICAL output dim and (closely) param count.

Streams:
  - SentenceEncoder: cumulative mean over each real-token prefix -> one small
    vector s_t (dim d_sent) per position. Quantum and classical encoders receive
    the identical ``(batch, time, d_model)`` causal-prefix tensor.
    Quantum variant: prefix means -> QuantumCore (VQC, measured) -> s_t.
    Classical control: prefix means -> MLP -> s_t, sized to match params.
  - Word transformer: standard causal decoder, each block CONDITIONED on s
    by one of three modes:
      film    : per-block gain/bias from s_t modulates position t
      token   : [summary_t, real_token_t] pairs are interleaved causally
      bias    : projected s_t is added to real-token embedding t

Prefix t includes input tokens <=t, matching next-token language-model
semantics: logit t may use the current input token but never a future token.
"""
from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn

from ..config import ModelConfig, two_stream_position_count
from ..quantum.layers import QuantumCore


def causal_prefix_mean(embeddings: jnp.ndarray) -> jnp.ndarray:
    """Cumulative mean for every real-token prefix, including the current token."""
    if embeddings.ndim != 3:
        raise ValueError(
            "causal_prefix_mean expects (batch, time, features) embeddings."
        )
    counts = jnp.arange(
        1, embeddings.shape[1] + 1, dtype=embeddings.dtype
    )[None, :, None]
    return jnp.cumsum(embeddings, axis=1) / counts


def interleave_summary_tokens(
    summaries: jnp.ndarray, real_tokens: jnp.ndarray
) -> jnp.ndarray:
    """Order each pair as ``[summary_t, real_token_t]`` without a time loop."""
    if summaries.shape != real_tokens.shape or summaries.ndim != 3:
        raise ValueError(
            "summary and real-token tensors must share (batch, time, features)."
        )
    batch, time, features = real_tokens.shape
    return jnp.stack((summaries, real_tokens), axis=2).reshape(
        batch, 2 * time, features
    )


def real_token_slots(sequence: jnp.ndarray) -> jnp.ndarray:
    """Select real-token positions from an interleaved summary/token sequence."""
    return sequence[:, 1::2, ...]


class ClassicalSentenceEncoder(nn.Module):
    """Causal prefix features -> MLP -> per-position summary (the control)."""

    d_sent: int
    hidden: int

    @nn.compact
    def __call__(self, prefixes: jnp.ndarray) -> jnp.ndarray:
        h = nn.tanh(nn.Dense(self.hidden, name="fc1")(prefixes))
        return nn.Dense(self.d_sent, name="fc2")(h)


class QuantumSentenceEncoder(nn.Module):
    """Causal prefix features -> vectorized VQC -> per-position summary."""

    d_sent: int
    quantum: object  # QuantumConfig

    @nn.compact
    def __call__(self, prefixes: jnp.ndarray) -> jnp.ndarray:
        return QuantumCore.from_config(
            self.quantum, out_features=self.d_sent, name="qcore"
        )(prefixes)


class TwoStreamLM(nn.Module):
    """Classical word transformer conditioned on causal prefix summaries."""

    cfg: ModelConfig
    encoder_kind: str = "quantum"   # quantum | classical | none
    condition: str = "film"         # film | token | bias
    d_sent: int = 8
    sent_hidden: int = 16           # classical encoder hidden (param match)

    @nn.compact
    def __call__(self, tokens: jnp.ndarray) -> jnp.ndarray:
        cfg = self.cfg
        if tokens.ndim != 2:
            raise ValueError("TwoStreamLM expects tokens with shape (batch, time).")
        B, T = tokens.shape
        if T < 1:
            raise ValueError("TwoStreamLM requires at least one input token.")
        internal_positions = two_stream_position_count(
            T, self.encoder_kind, self.condition
        )
        if internal_positions > cfg.max_seq_len:
            detail = (
                " Active token conditioning interleaves one summary before "
                "every real token."
                if self.encoder_kind != "none" and self.condition == "token"
                else ""
            )
            raise ValueError(
                f"TwoStreamLM requires {internal_positions} internal positions "
                f"for {T} input tokens, but model.max_seq_len is "
                f"{cfg.max_seq_len}.{detail}"
            )
        emb = nn.Embed(cfg.vocab_size, cfg.d_model, name="token_embed")(tokens)
        prefixes = causal_prefix_mean(emb)

        # ---- sentence stream ----
        s = None
        if self.encoder_kind == "quantum":
            if cfg.quantum is None:
                raise ValueError(
                    "TwoStreamLM encoder_kind='quantum' requires model.quantum."
                )
            s = QuantumSentenceEncoder(
                d_sent=self.d_sent, quantum=cfg.quantum, name="sent_q"
            )(prefixes)
        elif self.encoder_kind == "classical":
            s = ClassicalSentenceEncoder(
                d_sent=self.d_sent, hidden=self.sent_hidden, name="sent_c"
            )(prefixes)

        # positional + per-position summary/bias injection
        pos = self.param(
            "pos_embed",
            nn.initializers.normal(0.02),
            (1, cfg.max_seq_len, cfg.d_model),
        )
        x = emb
        if s is not None and self.condition == "bias":
            x = x + nn.Dense(cfg.d_model, name="sent_to_bias")(s)
        if s is not None and self.condition == "token":
            summary_tokens = nn.Dense(cfg.d_model, name="sent_to_tok")(s)
            x = interleave_summary_tokens(summary_tokens, x)
        x = x + pos[:, :internal_positions, :]

        # FiLM params from s_t, per position and block
        film = None
        if s is not None and self.condition == "film":
            film = nn.Dense(2 * cfg.d_model * cfg.n_blocks, name="film")(s)
            film = film.reshape(B, T, cfg.n_blocks, 2, cfg.d_model)

        # ---- word transformer ----
        seq = x.shape[1]
        causal = jnp.tril(jnp.ones((seq, seq), dtype=bool))
        for b in range(cfg.n_blocks):
            h = nn.LayerNorm(name=f"ln1_{b}")(x)
            h = nn.SelfAttention(
                num_heads=cfg.n_heads, qkv_features=cfg.d_model,
                use_bias=False, name=f"attn_{b}")(h, mask=causal)
            x = x + h
            h = nn.LayerNorm(name=f"ln2_{b}")(x)
            if film is not None:
                gain = 1.0 + film[:, :, b, 0, :]
                bias = film[:, :, b, 1, :]
                h = h * gain + bias
            h = nn.Dense(cfg.d_ff, name=f"ff1_{b}")(h)
            h = nn.gelu(h)
            h = nn.Dense(cfg.d_model, name=f"ff2_{b}")(h)
            x = x + h

        x = nn.LayerNorm(name="ln_f")(x)
        logits = nn.Dense(cfg.vocab_size, name="lm_head")(x)
        if s is not None and self.condition == "token":
            logits = real_token_slots(logits)
        return logits
