"""Interference head: a quantum OUTPUT layer (novel; not the hidden-state cell).

Every QLM result I know places the quantum part in the recurrent memory
(CRNNs, our QRNN) or a feature map; the output projection context -> next-
token distribution is always a classical softmax. This makes the OUTPUT
quantum and exploits the one thing a single classical softmax provably
cannot do: amplitude interference.

A classical mixture head forms p(t) = sum_h w_h p_h(t) with w_h, p_h >= 0 —
it can only ADD evidence. A coherent head forms

    p(t) = | sum_h c_h(ctx) * a_h(ctx)[t] |^2

with complex c_h, a_h, squaring AFTER the sum, so hypotheses that each
allow token t can DESTRUCTIVELY INTERFERE and cancel it. "Allowed under
reading A, allowed under reading B, forbidden when both" is a single-layer
constraint here and provably not a single positive-mixture layer. That is
the falsifiable claim. Exact statevector; classically simulable at this size.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn


class InterferenceHead(nn.Module):
    """logits = log | sum_h c_h * a_h |^2, with complex c_h (branch) and
    complex a_h(token) (per-branch amplitudes), all produced from context."""

    vocab_size: int
    n_hypotheses: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        H, V = self.n_hypotheses, self.vocab_size
        c = nn.Dense(2 * H, name="branch")(x)
        c = c[..., :H] + 1j * c[..., H:]                    # (..., H)
        a = nn.Dense(2 * H * V, name="amp")(x)
        a = a.reshape(*x.shape[:-1], H, V, 2)
        a = a[..., 0] + 1j * a[..., 1]                      # (..., H, V)
        psi = jnp.einsum("...h,...hv->...v", c, a)          # (..., V) complex
        return jnp.log(jnp.abs(psi) ** 2 + 1e-9)


class MixtureHead(nn.Module):
    """Parameter-matched classical control: positive mixture of H softmaxes.
    p = sum_h w_h * softmax_h(t), w >= 0 — can only add evidence, never cancel."""

    vocab_size: int
    n_hypotheses: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        H, V = self.n_hypotheses, self.vocab_size
        w = jax.nn.softmax(nn.Dense(H, name="branch")(x), axis=-1)   # (..., H)
        logits_h = nn.Dense(H * V, name="amp")(x)
        logits_h = logits_h.reshape(*x.shape[:-1], H, V)
        p_h = jax.nn.softmax(logits_h, axis=-1)             # (..., H, V)
        p = jnp.einsum("...h,...hv->...v", w, p_h)
        return jnp.log(p + 1e-9)


class LinearHead(nn.Module):
    """Plain classical softmax head (single hypothesis) — capacity floor."""

    vocab_size: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.Dense(self.vocab_size, name="proj")(x)
