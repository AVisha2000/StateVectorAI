"""Classical transformer building blocks (pure JAX/Flax).

``AttentionCore`` is deliberately factored to stop *before* the output
projection: the projection is the swap point shared with the quantum
variant (``qllm.quantum.layers.QuantumProjAttention``), so classical and
hybrid attention differ in exactly one module.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import linen as nn


class AttentionCore(nn.Module):
    """Causal multi-head self-attention WITHOUT the output projection.

    Returns the merged per-head context, shape (batch, seq, d_model).
    """

    d_model: int
    n_heads: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch, seq_len, d_model = x.shape
        assert d_model == self.d_model, (d_model, self.d_model)
        assert d_model % self.n_heads == 0, "d_model must divide n_heads"
        head_dim = d_model // self.n_heads

        qkv = nn.Dense(3 * d_model, name="qkv_proj")(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        def to_heads(t):
            return t.reshape(batch, seq_len, self.n_heads, head_dim).transpose(
                0, 2, 1, 3
            )

        q, k, v = to_heads(q), to_heads(k), to_heads(v)

        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(head_dim)
        causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        scores = jnp.where(causal[None, None], scores, jnp.finfo(scores.dtype).min)
        weights = jax.nn.softmax(scores, axis=-1)

        ctx = jnp.einsum("bhqk,bhkd->bhqd", weights, v)
        return ctx.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)


class CausalSelfAttention(nn.Module):
    """Classical attention = AttentionCore + Dense output projection."""

    d_model: int
    n_heads: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        ctx = AttentionCore(
            d_model=self.d_model, n_heads=self.n_heads, name="attention_core"
        )(x)
        return nn.Dense(self.d_model, name="out_proj")(ctx)


class FeedForward(nn.Module):
    """Standard transformer FFN: Dense -> GELU -> Dense."""

    d_model: int
    d_ff: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = nn.Dense(self.d_ff, name="up_proj")(x)
        h = nn.gelu(h)
        return nn.Dense(self.d_model, name="down_proj")(h)


class LowRankFFN(nn.Module):
    """Rank-r LINEAR bottleneck FFN: Dense(d->r) -> Dense(r->d), no bias,
    no nonlinearity. The exact classical twin of QuantumLinearFFN minus
    the unitary constraint — the control that decides whether transplant
    wins come from circuit structure or just low-rank + warm-start.
    """

    d_model: int
    rank: int = 16

    @nn.compact
    def __call__(self, x):
        h = nn.Dense(self.rank, use_bias=False, name="pre_proj")(x)
        return nn.Dense(self.d_model, use_bias=False, name="post_proj")(h)
