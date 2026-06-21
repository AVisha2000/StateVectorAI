"""Contextual quantum cell: O(n)-qubit memory for a task with an Omega(n^2)
classical memory wall (the contextuality separation, arXiv:2209.14353).

The classical wall (measured in v0.10): predicting a parity-FORCED token
requires recalling every earlier revealed value in its live context, and
interleaved contexts force a growing classical latent space. The quantum
escape is NOT to memorize values but to accumulate their PARITY in phases:

  - a register of `n_phase` qubits, each initialized in |+> (Hadamard
    basis), acts as a coherent parity accumulator;
  - revealing observable o with value b applies a controlled-Z-like PHASE
    flip pattern keyed by o, so the register's phase encodes the running
    parity of the values seen, per (learned) context routing;
  - to PREDICT a forced bit, the cell rotates the relevant phase qubit back
    to the computational basis (an interference / inverse-Hadamard read):
    constructive vs destructive interference yields the parity bit with
    O(1) qubits per live context, no value memorization.

Because parity is linear, the entire history's effect collapses into the
register phases — the quantum model needs O(n_phase) qubits where the
classical filter needs space exponential in the number of distinguishable
contexts. This cell is fully simulable here (small n_phase); the point is
the MEMORY-SCALING inductive bias, validated against the classical ladder.

Token convention matches qllm/data/contextual.py: cue tokens in
[0, n_observables) reveal an observable id; value tokens in
[n_observables, n_observables+2) reveal a bit. The cell predicts the next
token's distribution; its edge appears on value tokens at context ends.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from ..config import ModelConfig
from .recurrent import _apply_1q


def _hadamard() -> jnp.ndarray:
    h = np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2)
    return jnp.asarray(h)


def _phase(angle) -> jnp.ndarray:
    """diag(1, e^{i angle}) as a 2x2."""
    return jnp.stack([
        jnp.stack([jnp.ones_like(angle), jnp.zeros_like(angle)], -1),
        jnp.stack([jnp.zeros_like(angle), jnp.exp(1j * angle)], -1),
    ], -2).astype(jnp.complex64)


class ContextualQRNN(nn.Module):
    """Phase-accumulator recurrent cell for the contextual-parity task.

    State: n_phase qubits (dim 2**n_phase). Per token, a learned routing
    decides which phase qubit(s) the token touches and with what angle;
    value tokens imprint phases, cue tokens select context. Readout maps the
    full register (real + imag parts of amplitudes) to next-token logits via
    a small Dense, so the model can learn to read parity by interference.
    """

    vocab_size: int
    n_phase: int = 4
    n_layers: int = 2
    init_scale: float = math.pi

    @nn.compact
    def __call__(self, tokens: jnp.ndarray) -> jnp.ndarray:
        V, nq = self.vocab_size, self.n_phase
        dim = 2**nq
        batch = tokens.shape[0]

        # per-token learned phase angle applied to each phase qubit
        token_phase = self.param(
            "token_phase", nn.initializers.normal(stddev=self.init_scale),
            (V, nq))
        # per-token learned single-qubit pre-rotation (basis steering)
        token_rot = self.param(
            "token_rot", nn.initializers.normal(stddev=0.3), (V, nq))
        # trainable entangling phases between adjacent phase qubits (ZZ ring)
        zz_phase = self.param(
            "zz_phase", nn.initializers.uniform(scale=self.init_scale),
            (self.n_layers,))
        # readout from register (Re,Im of all amplitudes) -> logits
        readout = nn.Dense(V, name="readout")

        basis = np.arange(dim)
        bits = (basis[:, None] >> (nq - 1 - np.arange(nq))[None, :]) & 1
        zsign = 1 - 2 * bits
        zz_vec = jnp.asarray(
            (zsign * np.roll(zsign, -1, axis=1)).sum(axis=1).astype(np.float32))

        H = _hadamard()

        def entangle(psi):
            for layer in range(self.n_layers):
                psi = psi * jnp.exp(-1j * zz_phase[layer] * zz_vec)[None, :]
            return psi

        def step(psi, x_t):
            ph = token_phase[x_t]   # (B, nq)
            ro = token_rot[x_t]     # (B, nq)
            # imprint a Z-phase conditioned on the token. On a |+> qubit a
            # pi phase flips |+> <-> |->, so a controlled accumulation of pi
            # phases encodes PARITY in the X-basis sign — the linear-in-history
            # quantity contextuality needs. token_rot adds learned steering.
            for q in range(nq):
                psi = _apply_1q(psi, _ry_batch(ro[:, q]), q, nq)
                psi = _apply_1q(psi, _zrot(ph[:, q]), q, nq)
            psi = entangle(psi)
            return psi, psi  # carry, and emit the register state

        # start each phase qubit in |+> : uniform real superposition
        psi0 = jnp.full((batch, dim), 1.0 / math.sqrt(dim), dtype=jnp.complex64)
        xs = jnp.swapaxes(tokens, 0, 1)
        _, states = jax.lax.scan(step, psi0, xs)  # (T, B, dim)
        states = jnp.swapaxes(states, 0, 1)       # (B, T, dim)
        # rotate every phase qubit back to the computational basis (Hadamard)
        # so accumulated X-basis sign (parity) becomes a measurable 0/1
        # amplitude — interference read — then map to logits.
        def to_z(psi_bt):
            flat = psi_bt.reshape(-1, dim)
            for q in range(nq):
                flat = _apply_1q(flat, H, q, nq)
            return flat.reshape(*psi_bt.shape)
        read_states = to_z(states)
        probs = jnp.abs(read_states) ** 2          # (B, T, dim) measurement probs
        feats = jnp.concatenate(
            [probs, jnp.real(states), jnp.imag(states)], axis=-1)
        return readout(feats)


def _ry_batch(angle):
    c, s = jnp.cos(angle / 2), jnp.sin(angle / 2)
    return jnp.stack([
        jnp.stack([c, -s], -1),
        jnp.stack([s, c], -1),
    ], -2).astype(jnp.complex64)


def _zrot(angle):
    """RZ(angle) = diag(e^{-i a/2}, e^{+i a/2}); a=pi flips |+><->|->."""
    em = jnp.exp(-0.5j * angle)
    ep = jnp.exp(0.5j * angle)
    z = jnp.zeros_like(em)
    return jnp.stack([
        jnp.stack([em, z], -1),
        jnp.stack([z, ep], -1),
    ], -2).astype(jnp.complex64)


def contextual_qrnn_from_config(cfg: ModelConfig) -> ContextualQRNN:
    return ContextualQRNN(
        vocab_size=cfg.vocab_size,
        n_phase=cfg.quantum.n_qubits,
        n_layers=cfg.quantum.n_circuit_layers,
        init_scale=cfg.quantum.init_scale,
    )


class RoutedContextualQRNN(nn.Module):
    """Cue-conditioned phase-routing cell for the INTERLEAVED contextual task.

    Fixes the v0.13 scrambling failure: the plain ContextualQRNN imprints the
    same phase op regardless of which observable a value belongs to, so
    interleaved contexts collapse into one register. Here the cell routes:

      - A learned soft map ``cue_to_qubit`` (n_cue x n_phase, softmaxed)
        sends each cue token to a distribution over phase qubits.
      - The cell carries a running ``selector`` (B, n_phase): when a CUE
        token arrives it is REPLACED by that cue's qubit distribution; when a
        VALUE token arrives the selector is held. This makes "the qubit the
        current context owns" available at the value step.
      - A value token imprints its bit as an RZ phase WEIGHTED per qubit by
        the selector, so parity accumulates on the active context's qubit
        only. Readout is Hadamard-back + Born, as in the parity primitive.

    Token convention (qllm/data/contextual.py): ids [0, n_cue) are cues
    (observable ids), ids [n_cue, n_cue+2) are value bits, n_cue = vocab-2.
    """

    vocab_size: int
    n_phase: int = 5
    n_layers: int = 1
    init_scale: float = math.pi

    @nn.compact
    def __call__(self, tokens: jnp.ndarray) -> jnp.ndarray:
        V, nq = self.vocab_size, self.n_phase
        dim = 2**nq
        n_cue = V - 2
        batch = tokens.shape[0]

        cue_to_qubit = self.param(
            "cue_to_qubit", nn.initializers.normal(stddev=1.0), (n_cue, nq))
        # per-value-bit base phase (bit 0 vs bit 1 get different RZ angles)
        value_phase = self.param(
            "value_phase", nn.initializers.normal(stddev=self.init_scale), (2, nq))
        zz_phase = self.param(
            "zz_phase", nn.initializers.uniform(scale=self.init_scale),
            (self.n_layers,))
        readout = nn.Dense(V, name="readout")

        basis = np.arange(dim)
        bits = (basis[:, None] >> (nq - 1 - np.arange(nq))[None, :]) & 1
        zsign = 1 - 2 * bits
        zz_vec = jnp.asarray(
            (zsign * np.roll(zsign, -1, axis=1)).sum(axis=1).astype(np.float32))
        H = _hadamard()

        is_value = (tokens >= n_cue)            # (B, T) bool
        # cue index (clamped) and value bit per position
        cue_idx = jnp.clip(tokens, 0, n_cue - 1)
        val_bit = jnp.clip(tokens - n_cue, 0, 1)

        def step(carry, inp):
            psi, selector = carry
            x_cue, x_val, isval = inp  # (B,), (B,), (B,)
            # selector update: cue tokens set it to their qubit distribution
            cue_sel = jax.nn.softmax(cue_to_qubit[x_cue], axis=-1)  # (B, nq)
            isval_f = isval.astype(psi.real.dtype)[:, None]
            selector = isval_f * selector + (1.0 - isval_f) * cue_sel
            # value tokens imprint phase on the selected qubit(s)
            base = value_phase[x_val]            # (B, nq)
            angles = base * selector * isval_f   # zero on cue steps
            for q in range(nq):
                psi = _apply_1q(psi, _zrot(angles[:, q]), q, nq)
            for layer in range(self.n_layers):
                psi = psi * jnp.exp(-1j * zz_phase[layer] * zz_vec)[None, :]
            return (psi, selector), psi

        psi0 = jnp.full((batch, dim), 1.0 / math.sqrt(dim), dtype=jnp.complex64)
        sel0 = jnp.full((batch, nq), 1.0 / nq, dtype=jnp.float32)
        xs = (jnp.swapaxes(cue_idx, 0, 1), jnp.swapaxes(val_bit, 0, 1),
              jnp.swapaxes(is_value, 0, 1))
        _, states = jax.lax.scan(step, (psi0, sel0), xs)
        states = jnp.swapaxes(states, 0, 1)

        def to_z(psi_bt):
            flat = psi_bt.reshape(-1, dim)
            for q in range(nq):
                flat = _apply_1q(flat, H, q, nq)
            return flat.reshape(*psi_bt.shape)
        probs = jnp.abs(to_z(states)) ** 2
        feats = jnp.concatenate(
            [probs, jnp.real(states), jnp.imag(states)], axis=-1)
        return readout(feats)


def routed_contextual_from_config(cfg: ModelConfig) -> RoutedContextualQRNN:
    return RoutedContextualQRNN(
        vocab_size=cfg.vocab_size,
        n_phase=cfg.quantum.n_qubits,
        n_layers=cfg.quantum.n_circuit_layers,
        init_scale=cfg.quantum.init_scale,
    )
