"""Resource heuristics for queue-time quantum run guardrails."""
from __future__ import annotations

from ..config import ExperimentConfig
from ..registry import (
    QUANTUM_ARCH_TYPES,
    QUANTUM_ATTN_TYPES,
    QUANTUM_FFN_TYPES,
)


def quantum_resource_estimate(cfg: ExperimentConfig) -> dict:
    model = cfg.model
    q = model.quantum
    block_attn = [model.attn_type] * max(int(model.n_blocks or 1), 1)
    block_ffn = [model.ffn_type] * max(int(model.n_blocks or 1), 1)
    if model.blocks is not None:
        block_attn = [block.attn_type for block in model.blocks]
        block_ffn = [block.ffn_type for block in model.blocks]
    uses_quantum_attn = any(t in QUANTUM_ATTN_TYPES for t in block_attn)
    uses_quantum_ffn = any(t in QUANTUM_FFN_TYPES for t in block_ffn)
    uses_quantum_recurrent = model.arch in QUANTUM_ARCH_TYPES
    uses_two_stream_quantum = model.arch == "two_stream" and model.encoder_kind == "quantum"
    token_calls = cfg.train.batch_size * cfg.train.seq_len
    attn_blocks = sum(1 for t in block_attn if t in QUANTUM_ATTN_TYPES)
    ffn_blocks = sum(1 for t in block_ffn if t in QUANTUM_FFN_TYPES)
    component_multiplier = 0
    if uses_quantum_attn:
        component_multiplier += 2 * attn_blocks
    if uses_quantum_ffn:
        component_multiplier += ffn_blocks
    if uses_quantum_recurrent:
        component_multiplier += 2
    if uses_two_stream_quantum:
        component_multiplier += 1
    n_qubits = int(q.n_qubits) if q is not None else 0
    n_layers = int(q.n_circuit_layers) if q is not None else 0
    state_dim = 2 ** n_qubits if q is not None else 0
    score = (
        state_dim
        * max(n_layers, 1)
        * max(token_calls, 1)
        * max(component_multiplier, 1)
    )
    if not any((uses_quantum_attn, uses_quantum_ffn, uses_quantum_recurrent, uses_two_stream_quantum)):
        band = "classical"
    elif score >= 16_000_000:
        band = "extreme"
    elif score >= 6_000_000:
        band = "high"
    elif score >= 1_500_000:
        band = "medium"
    else:
        band = "low"
    advice = []
    if band in {"high", "extreme"}:
        advice.append("Reduce batch size before increasing qubits or depth.")
        advice.append("Use seq_len 32 or lower for quantum attention sweeps.")
    if uses_quantum_attn and n_qubits >= 8:
        advice.append("Quantum attention is per-token; prefer q<=8 while scanning depth.")
    if n_layers >= 10:
        advice.append("Depth >=10 can trigger large JIT intermediates; scale depth after width is stable.")
    return {
        "band": band,
        "score": score,
        "state_dim": state_dim,
        "token_calls": token_calls,
        "component_multiplier": component_multiplier,
        "uses_quantum_attention": uses_quantum_attn,
        "advice": advice,
    }
