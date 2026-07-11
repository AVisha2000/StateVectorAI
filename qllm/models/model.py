"""Model assembly: the plugin architecture's composition root.

``build_attention`` / ``build_ffn`` resolve config strings to Flax modules.
Quantum classes are imported lazily so classical-only experiments never
import PennyLane. Adding a new component = add a branch here + a config
string; nothing else in the pipeline changes.
"""
from __future__ import annotations

import dataclasses

import jax.numpy as jnp
from flax import linen as nn

from ..classical.layers import CausalSelfAttention, FeedForward
from ..config import ModelConfig, model_block_config, two_stream_position_count
from ..registry import (
    ARCH_TYPES,
    ATTN_TYPES,
    CONDITION_TYPES,
    EMBED_TYPES,
    ENCODER_TYPES,
    FFN_TYPES,
    HEAD_TYPES,
    QUANTUM_ARCH_TYPES,
    QUANTUM_ATTN_TYPES,
    QUANTUM_FFN_TYPES,
)


def uses_quantum(cfg: ModelConfig) -> bool:
    if cfg.arch in QUANTUM_ARCH_TYPES:
        return True
    if cfg.arch == "two_stream":
        return cfg.encoder_kind == "quantum"
    if cfg.arch == "gru":
        return False
    if cfg.embed_type == "quantum":
        return True
    if cfg.blocks is not None:
        return any(
            b.ffn_type in QUANTUM_FFN_TYPES
            or b.attn_type in QUANTUM_ATTN_TYPES
            for b in cfg.blocks
        )
    return (
        cfg.ffn_type in QUANTUM_FFN_TYPES
        or cfg.attn_type in QUANTUM_ATTN_TYPES
    )


def build_attention(cfg: ModelConfig, name: str) -> nn.Module:
    if cfg.attn_type == "classical":
        return CausalSelfAttention(
            d_model=cfg.d_model, n_heads=cfg.n_heads, name=name
        )
    if cfg.attn_type == "quantum_proj":
        from ..quantum.layers import QuantumProjAttention

        return QuantumProjAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            quantum=cfg.quantum,
            name=name,
        )
    if cfg.attn_type == "quantum_qkv":
        from ..quantum.layers import QuantumQKVAttention

        return QuantumQKVAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            quantum=cfg.quantum,
            name=name,
        )
    raise ValueError(f"Unknown attn_type '{cfg.attn_type}'. Options: {ATTN_TYPES}")


def build_ffn(cfg: ModelConfig, name: str) -> nn.Module:
    if cfg.ffn_type == "classical":
        return FeedForward(d_model=cfg.d_model, d_ff=cfg.d_ff, name=name)
    if cfg.ffn_type == "quantum":
        from ..quantum.layers import VQCFeedForward

        return VQCFeedForward(d_model=cfg.d_model, quantum=cfg.quantum, name=name)
    if cfg.ffn_type == "quantum_linear":
        from ..quantum.layers import QuantumLinearFFN

        return QuantumLinearFFN(d_model=cfg.d_model, quantum=cfg.quantum, name=name)
    if cfg.ffn_type == "lowrank":
        from ..classical.layers import LowRankFFN

        return LowRankFFN(d_model=cfg.d_model, rank=cfg.ffn_rank, name=name)
    raise ValueError(f"Unknown ffn_type '{cfg.ffn_type}'. Options: {FFN_TYPES}")


class TransformerBlock(nn.Module):
    """Pre-LayerNorm residual block; sub-layers resolved from config."""

    cfg: ModelConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        attn = build_attention(self.cfg, name="attn")
        ffn = build_ffn(self.cfg, name="ffn")
        x = x + attn(nn.LayerNorm(name="ln_attn")(x))
        x = x + ffn(nn.LayerNorm(name="ln_ffn")(x))
        return x


class QLLM(nn.Module):
    """Decoder-only language model with swappable classical/quantum blocks."""

    cfg: ModelConfig

    @nn.compact
    def __call__(self, tokens: jnp.ndarray) -> jnp.ndarray:
        cfg = self.cfg
        _, seq_len = tokens.shape
        assert seq_len <= cfg.max_seq_len, (seq_len, cfg.max_seq_len)

        if cfg.embed_type == "quantum":
            from ..quantum.layers import QuantumEmbedding

            x = QuantumEmbedding(
                vocab_size=cfg.vocab_size,
                d_model=cfg.d_model,
                quantum=cfg.quantum,
                name="token_embed_q",
            )(tokens)
        else:
            x = nn.Embed(cfg.vocab_size, cfg.d_model, name="token_embed")(tokens)
        pos = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.02),
            (cfg.max_seq_len, cfg.d_model),
        )
        x = x + pos[:seq_len]

        for i in range(cfg.n_blocks):
            x = TransformerBlock(model_block_config(cfg, i), name=f"block_{i}")(x)

        x = nn.LayerNorm(name="ln_final")(x)
        if cfg.head_type == "interference":
            from ..quantum.interference_head import InterferenceHead

            return InterferenceHead(vocab_size=cfg.vocab_size,
                                    n_hypotheses=cfg.head_hypotheses,
                                    name="lm_head_interference")(x)
        if cfg.head_type == "mixture":
            from ..quantum.interference_head import MixtureHead

            return MixtureHead(vocab_size=cfg.vocab_size,
                               n_hypotheses=cfg.head_hypotheses,
                               name="lm_head_mixture")(x)
        if cfg.head_type == "linear":
            return nn.Dense(cfg.vocab_size, name="lm_head")(x)
        raise ValueError(f"Unknown head_type '{cfg.head_type}'. Options: {HEAD_TYPES}")


def build_model(cfg: ModelConfig, vocab_size: int):
    """Finalize the model config with the runtime vocab size and build.

    Dispatches on ``cfg.arch``: the decoder-only transformer (default),
    the recurrent quantum LM (learnable quantum HMM), or the GRU
    baseline. All expose tokens -> logits, so the pipeline is shared.
    """
    if cfg.arch not in ARCH_TYPES:
        raise ValueError(f"Unknown arch '{cfg.arch}'. Options: {ARCH_TYPES}")
    if cfg.embed_type not in EMBED_TYPES:
        raise ValueError(
            f"Unknown embed_type '{cfg.embed_type}'. Options: {EMBED_TYPES}"
        )
    if cfg.head_type not in HEAD_TYPES:
        raise ValueError(f"Unknown head_type '{cfg.head_type}'. Options: {HEAD_TYPES}")
    if cfg.encoder_kind not in ENCODER_TYPES:
        raise ValueError(
            f"Unknown encoder_kind '{cfg.encoder_kind}'. Options: {ENCODER_TYPES}"
        )
    if cfg.condition not in CONDITION_TYPES:
        raise ValueError(
            f"Unknown condition '{cfg.condition}'. Options: {CONDITION_TYPES}"
        )
    final_cfg = dataclasses.replace(cfg, vocab_size=vocab_size)
    if cfg.arch == "transformer":
        return QLLM(final_cfg), final_cfg
    if cfg.arch == "two_stream":
        from .two_stream import TwoStreamLM

        return TwoStreamLM(
            cfg=final_cfg, encoder_kind=cfg.encoder_kind,
            condition=cfg.condition, d_sent=cfg.d_sent,
            sent_hidden=cfg.sent_hidden), final_cfg
    if cfg.arch == "qrnn":
        from ..quantum.recurrent import qrnn_from_config

        return qrnn_from_config(final_cfg), final_cfg
    if cfg.arch == "contextual_qrnn":
        from ..quantum.contextual_cell import contextual_qrnn_from_config

        return contextual_qrnn_from_config(final_cfg), final_cfg
    if cfg.arch == "routed_contextual":
        from ..quantum.contextual_cell import routed_contextual_from_config

        return routed_contextual_from_config(final_cfg), final_cfg
    if cfg.arch == "gru":
        from ..classical.recurrent import GRULM

        return GRULM(vocab_size=vocab_size, hidden=cfg.rnn_hidden), final_cfg
    raise AssertionError(f"Unhandled registered architecture: {cfg.arch}")


def count_model_params(cfg: ModelConfig, vocab_size: int, seq_len: int = 8) -> int:
    """Total trainable parameter count for a config (via a dummy init)."""
    import jax
    import numpy as np

    model, _ = build_model(cfg, vocab_size)
    real_capacity = cfg.max_seq_len
    if cfg.arch == "two_stream":
        positions_per_token = two_stream_position_count(
            1, cfg.encoder_kind, cfg.condition
        )
        real_capacity = cfg.max_seq_len // positions_per_token
        if real_capacity < 1:
            raise ValueError(
                "two-stream conditioning has no usable real-token capacity"
            )
    tokens = jnp.zeros((1, min(seq_len, real_capacity)), dtype=jnp.int32)
    params = model.init(jax.random.PRNGKey(0), tokens)["params"]
    return sum(int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(params))


def matched_classical_d_ff(cfg: ModelConfig, vocab_size: int) -> int:
    """``d_ff`` for a classical twin parameter-matched to a (quantum) config.

    Total parameter count is exactly linear in ``d_ff``, so two probe inits
    solve for it. This is the Quixer-style parameter-matched baseline: any
    claim of quantum contribution must beat this twin, not just an
    arbitrary classical model.
    """
    target = count_model_params(cfg, vocab_size)
    base = dataclasses.replace(cfg, ffn_type="classical")
    p1 = count_model_params(dataclasses.replace(base, d_ff=1), vocab_size)
    p2 = count_model_params(dataclasses.replace(base, d_ff=2), vocab_size)
    slope = p2 - p1
    return max(1, round(1 + (target - p1) / slope))
