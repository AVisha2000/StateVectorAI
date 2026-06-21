"""Classical recurrent baseline: GRU language model.

The class-matched classical comparison for QRNNLM: a recurrent filter
that must approximate the quantum belief-state dynamics in its hidden
vector. The interesting measured quantity is the parameters-to-floor
curve (what hidden size does a classical recurrence need to extract the
quantum-memory bits that the quantum cell gets natively?).
"""
from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn


class GRULM(nn.Module):
    vocab_size: int
    hidden: int = 16

    @nn.compact
    def __call__(self, tokens: jnp.ndarray) -> jnp.ndarray:
        x = nn.Embed(self.vocab_size, self.hidden, name="embed")(tokens)
        y = nn.RNN(nn.GRUCell(features=self.hidden), name="gru")(x)
        return nn.Dense(self.vocab_size, name="head")(y)
