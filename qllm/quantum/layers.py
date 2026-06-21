"""Quantum layers as drop-in Flax modules.

``QuantumCore`` is the reusable hybrid primitive ("dressed quantum circuit"):

    Dense(d -> C*n_qubits) -> dressing -> C parallel VQCs -> Dense(-> out)

v0.3 upgrades, motivated by the v0.2 ablation (trained ~= frozen ~= matched
classical twin):

- ``readout='zz'``: weight-2 correlators give O(n^2) features per circuit
  while every observable stays low-weight (the trainable, local-cost regime).
- ``dressing='linear'``: removes the tanh so the circuit is the ONLY
  nonlinearity in the block — makes trained-vs-frozen a sharp test of
  whether the circuit itself learns. (Angle encodings are periodic, so
  unbounded angles are well-defined.)
- ``n_circuits``: parallel quantum heads — scale capacity by adding blocks,
  not qubits, per the barren-plateau scaling result.
- ``init_scale``: small-angle (near-identity) initialization keeps early
  gradients alive (Grant et al. 2019 style plateau mitigation).
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import linen as nn

from ..config import QuantumConfig
from .backends import get_expval_circuit, readout_dim
from .circuits import weight_shape


class QuantumCore(nn.Module):
    """Classical pre-projection -> parallel VQCs -> post-projection."""

    n_qubits: int
    n_circuit_layers: int
    out_features: int
    ansatz: str = "reuploading"
    backend: str = "pennylane"
    device: str = "default.qubit"
    diff_method: str = "backprop"
    shots: int | None = None
    readout: str = "z"
    dressing: str = "tanh"
    init_scale: float = 2.0 * math.pi
    n_circuits: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        c, n = self.n_circuits, self.n_qubits
        circuit_weights = self.param(
            "circuit_weights",
            nn.initializers.uniform(scale=self.init_scale),
            (c, *weight_shape(self.n_circuit_layers, n)),
        )

        z = nn.Dense(c * n, name="pre_proj")(x)
        if self.dressing == "tanh":
            z = (math.pi / 2.0) * jnp.tanh(z)
        elif self.dressing != "linear":
            raise ValueError(f"Unknown dressing '{self.dressing}'")

        circuit = get_expval_circuit(
            self.backend,
            self.device,
            self.diff_method,
            self.shots,
            n,
            self.n_circuit_layers,
            self.ansatz,
            self.readout,
        )

        flat = z.reshape(-1, c, n)
        per_sample = jax.vmap(circuit, in_axes=(0, 0))      # over circuits
        expvals = jax.vmap(per_sample, in_axes=(0, None))(  # over batch
            flat, circuit_weights
        )
        m = readout_dim(n, self.readout)
        expvals = expvals.reshape(*x.shape[:-1], c * m)

        return nn.Dense(self.out_features, name="post_proj")(expvals)

    @classmethod
    def from_config(cls, qcfg: QuantumConfig, out_features: int, **kwargs):
        return cls(
            n_qubits=qcfg.n_qubits,
            n_circuit_layers=qcfg.n_circuit_layers,
            out_features=out_features,
            ansatz=qcfg.ansatz,
            backend=qcfg.backend,
            device=qcfg.device,
            diff_method=qcfg.diff_method,
            shots=qcfg.shots,
            readout=qcfg.readout,
            dressing=qcfg.dressing,
            init_scale=qcfg.init_scale,
            n_circuits=qcfg.n_circuits,
            **kwargs,
        )


class VQCFeedForward(nn.Module):
    """Quantum replacement for the transformer FFN block."""

    d_model: int
    quantum: QuantumConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return QuantumCore.from_config(self.quantum, out_features=self.d_model)(x)


class QuantumProjAttention(nn.Module):
    """Causal self-attention with the output projection replaced by a VQC.

    Reuses the classical ``AttentionCore`` (scores/softmax/context stay
    classical) and swaps only the final projection for ``QuantumCore`` —
    the minimal, cleanly-ablatable quantum insertion point in attention.
    """

    d_model: int
    n_heads: int
    quantum: QuantumConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        from ..classical.layers import AttentionCore

        ctx = AttentionCore(
            d_model=self.d_model, n_heads=self.n_heads, name="attention_core"
        )(x)
        return QuantumCore.from_config(
            self.quantum, out_features=self.d_model, name="quantum_out_proj"
        )(ctx)


class QuantumEmbedding(nn.Module):
    """Words-as-quantum-states token embedding (DisCoCat/lambeq-inspired).

    Coecke et al.'s categorical QNLP represents each word as a quantum
    state; full DisCoCat composes them along grammar structure (lambeq),
    which targets classification rather than autoregressive LM. The
    transferable principle implemented here: token t -> learnable
    preparation angles -> shared variational circuit -> low-weight
    measurement features -> Dense to d_model. The token's representation
    IS a quantum state, read out through observables.
    """

    vocab_size: int
    d_model: int
    quantum: QuantumConfig

    @nn.compact
    def __call__(self, tokens: jnp.ndarray) -> jnp.ndarray:
        q = self.quantum
        n = q.n_qubits
        from .circuits import weight_shape as _ws

        token_angles = self.param(
            "token_angles", nn.initializers.normal(stddev=0.5),
            (self.vocab_size, n),
        )
        circuit_weights = self.param(
            "circuit_weights",
            nn.initializers.uniform(scale=q.init_scale),
            _ws(q.n_circuit_layers, n),
        )
        circuit = get_expval_circuit(
            q.backend, q.device, q.diff_method, q.shots,
            n, q.n_circuit_layers, q.ansatz, q.readout,
        )
        angles = token_angles[tokens]                # (B, T, n)
        flat = angles.reshape(-1, n)
        feats = jax.vmap(circuit, in_axes=(0, None))(flat, circuit_weights)
        feats = feats.reshape(*tokens.shape, -1)
        return nn.Dense(self.d_model, name="proj")(feats)


class QuantumQKVAttention(nn.Module):
    """Causal attention whose Q, K, V projections come from one VQC.

    The quantum feature map generates all three attention streams
    (QuantumCore -> 3*d_model, split); scores/softmax stay classical.
    Deeper quantum involvement than the output-projection swap, at the
    same per-token circuit cost.
    """

    d_model: int
    n_heads: int
    quantum: QuantumConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch, seq_len, d_model = x.shape
        head_dim = d_model // self.n_heads

        qkv = QuantumCore.from_config(
            self.quantum, out_features=3 * d_model, name="quantum_qkv"
        )(x)
        qh, kh, vh = jnp.split(qkv, 3, axis=-1)

        def to_heads(t):
            return t.reshape(batch, seq_len, self.n_heads, head_dim).transpose(
                0, 2, 1, 3
            )

        qh, kh, vh = to_heads(qh), to_heads(kh), to_heads(vh)
        scores = jnp.einsum("bhqd,bhkd->bhqk", qh, kh) / math.sqrt(head_dim)
        causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        scores = jnp.where(causal[None, None], scores, jnp.finfo(scores.dtype).min)
        weights = jax.nn.softmax(scores, axis=-1)
        ctx = jnp.einsum("bhqk,bhkd->bhqd", weights, vh)
        ctx = ctx.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)
        return nn.Dense(self.d_model, name="out_proj")(ctx)


class QuantumLinearFFN(nn.Module):
    """Amplitude-space unitary FFN: Dense(d->2^n) -> U(theta)|amplitudes> -> Dense.

    Unlike QuantumCore (angle encode -> expvals, a nonlinear feature map),
    this layer applies the circuit unitary DIRECTLY to the activation
    vector as amplitudes — the computation model in which a classical
    weight matrix's rotation part can be transplanted exactly (see
    qllm.quantum.transplant). Bias-free Dense maps so the transplanted
    linear algebra is preserved verbatim.
    """

    d_model: int
    quantum: QuantumConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        from .transplant import apply_circuit

        q = self.quantum
        n, dim = q.n_qubits, 2**q.n_qubits
        circuit_weights = self.param(
            "circuit_weights",
            nn.initializers.uniform(scale=q.init_scale),
            (q.n_circuit_layers, n, 3),
        )
        zz_phase = self.param(
            "zz_phase",
            nn.initializers.uniform(scale=q.init_scale),
            (q.n_circuit_layers,),
        )
        global_phase = self.param(
            "global_phase", nn.initializers.zeros, ()
        )
        z = nn.Dense(dim, use_bias=False, name="pre_proj")(x)
        flat = z.reshape(-1, dim).astype(jnp.complex64)
        out = jnp.real(
            apply_circuit(flat, circuit_weights, zz_phase, n, global_phase)
        )
        out = out.reshape(*x.shape[:-1], dim)
        return nn.Dense(self.d_model, use_bias=False, name="post_proj")(out)
