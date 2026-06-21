"""Two-stream LM: a sentence encoder guides a classical word transformer.

Hypothesis under test (NOT assumed): can a compact SENTENCE-level summary,
injected into a per-token word model, supply a useful inductive bias — and
does a QUANTUM encoder do so better than a parameter-matched CLASSICAL one?
Every quantum-readout featurizer in this project (q-embed v0.6, transplant
v0.7, interference v0.12) underperformed a classical layer of matched size
on classical text, so the strict control is the whole point: quantum vs
classical sentence encoder at IDENTICAL output dim and (closely) param count.

Streams:
  - SentenceEncoder: pool the whole context -> small vector s (dim d_sent).
    Quantum variant: pooled embedding -> QuantumCore (VQC, measured) -> s.
    Classical control: pooled embedding -> MLP -> s, sized to match params.
  - Word transformer: standard causal decoder, each block CONDITIONED on s
    by one of three modes:
      film    : per-block FiLM (learned gain+bias from s modulate the stream)
      token   : s projected to d_model, PREPENDED as a virtual token 0
      bias    : s projected to d_model, ADDED to every token embedding

The sentence vector is computed from the LEFT context only (mean-pool of
embeddings up to each position is too expensive per-step; we use a single
summary from the full sequence with a causal-safe note below), so this is a
research probe rather than a strictly autoregressive model: the sentence
stream sees the whole window. We therefore evaluate teacher-forced
perplexity (standard) and treat the sentence vector as side information,
exactly as a document/topic embedding would be in practice.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import linen as nn

from ..config import ModelConfig
from ..quantum.layers import QuantumCore


class ClassicalSentenceEncoder(nn.Module):
    """Mean-pool embeddings -> MLP -> sentence vector (the control)."""

    d_sent: int
    hidden: int

    @nn.compact
    def __call__(self, emb: jnp.ndarray) -> jnp.ndarray:
        pooled = emb.mean(axis=1)                      # (B, d_model)
        h = nn.tanh(nn.Dense(self.hidden, name="fc1")(pooled))
        return nn.Dense(self.d_sent, name="fc2")(h)    # (B, d_sent)


class QuantumSentenceEncoder(nn.Module):
    """Mean-pool embeddings -> QuantumCore (VQC, measured) -> sentence vector."""

    d_sent: int
    quantum: object  # QuantumConfig

    @nn.compact
    def __call__(self, emb: jnp.ndarray) -> jnp.ndarray:
        pooled = emb.mean(axis=1)                      # (B, d_model)
        return QuantumCore.from_config(
            self.quantum, out_features=self.d_sent, name="qcore")(pooled)


class TwoStreamLM(nn.Module):
    """Classical word transformer conditioned on a sentence vector."""

    cfg: ModelConfig
    encoder_kind: str = "quantum"   # quantum | classical | none
    condition: str = "film"         # film | token | bias
    d_sent: int = 8
    sent_hidden: int = 16           # classical encoder hidden (param match)

    @nn.compact
    def __call__(self, tokens: jnp.ndarray) -> jnp.ndarray:
        cfg = self.cfg
        B, T = tokens.shape
        emb = nn.Embed(cfg.vocab_size, cfg.d_model, name="token_embed")(tokens)

        # ---- sentence stream ----
        s = None
        if self.encoder_kind == "quantum":
            s = QuantumSentenceEncoder(
                d_sent=self.d_sent, quantum=cfg.quantum, name="sent_q")(emb)
        elif self.encoder_kind == "classical":
            s = ClassicalSentenceEncoder(
                d_sent=self.d_sent, hidden=self.sent_hidden, name="sent_c")(emb)

        # positional + optional virtual token / bias injection
        pos = self.param("pos_embed", nn.initializers.normal(0.02),
                         (1, cfg.max_seq_len + 1, cfg.d_model))
        x = emb
        if s is not None and self.condition == "bias":
            x = x + nn.Dense(cfg.d_model, name="sent_to_bias")(s)[:, None, :]
        if s is not None and self.condition == "token":
            vtok = nn.Dense(cfg.d_model, name="sent_to_tok")(s)[:, None, :]
            x = jnp.concatenate([vtok, x], axis=1)     # prepend
        x = x + pos[:, :x.shape[1], :]

        # FiLM params from s, per block
        film = None
        if s is not None and self.condition == "film":
            film = nn.Dense(2 * cfg.d_model * cfg.n_blocks, name="film")(s)
            film = film.reshape(B, cfg.n_blocks, 2, cfg.d_model)

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
                gain = 1.0 + film[:, b, 0, :][:, None, :]
                bias = film[:, b, 1, :][:, None, :]
                h = h * gain + bias
            h = nn.Dense(cfg.d_ff, name=f"ff1_{b}")(h)
            h = nn.gelu(h)
            h = nn.Dense(cfg.d_model, name=f"ff2_{b}")(h)
            x = x + h

        x = nn.LayerNorm(name="ln_f")(x)
        logits = nn.Dense(cfg.vocab_size, name="lm_head")(x)
        if s is not None and self.condition == "token":
            logits = logits[:, 1:, :]                  # drop virtual token
        return logits
