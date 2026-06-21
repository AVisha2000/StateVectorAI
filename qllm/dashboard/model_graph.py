"""Read-only architecture graph payloads for QLLM Lab."""
from __future__ import annotations

import dataclasses
from typing import Any

from ..config import ExperimentConfig, ModelConfig, QuantumConfig


def _get(cfg: ModelConfig | dict, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(f"model.{key}", cfg.get(key, default))
    return getattr(cfg, key, default)


def _block_get(block: Any, key: str, default: Any = None) -> Any:
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


def _quantum_dict(cfg: ModelConfig | dict) -> dict:
    if isinstance(cfg, dict):
        return {
            "n_qubits": int(cfg.get("model.quantum.n_qubits", 0) or 0),
            "n_circuit_layers": int(cfg.get("model.quantum.n_circuit_layers", 0) or 0),
            "ansatz": cfg.get("model.quantum.ansatz", "reuploading"),
            "backend": cfg.get("model.quantum.backend", "pennylane"),
            "device": cfg.get("model.quantum.device", "default.qubit"),
            "shots": cfg.get("model.quantum.shots"),
            "readout": cfg.get("model.quantum.readout", "z"),
            "trainable": str(cfg.get("model.quantum.trainable", "True")) != "False",
        }
    qcfg: QuantumConfig = cfg.quantum
    return dataclasses.asdict(qcfg)


def _node(node_id: str, label: str, kind: str, **meta) -> dict:
    return {"id": node_id, "label": label, "kind": kind, "meta": meta}


def _kind_for_component(value: str) -> str:
    if value in {"quantum", "quantum_linear", "quantum_proj", "quantum_qkv"}:
        return "quantum"
    return "classical"


def model_graph_from_config(cfg: ExperimentConfig | ModelConfig | dict) -> dict:
    """Build a compact, UI-safe architecture graph from model config."""
    model = cfg.model if isinstance(cfg, ExperimentConfig) else cfg
    arch = _get(model, "arch", "transformer")
    quantum = _quantum_dict(model)
    nodes: list[dict] = []
    edges: list[list[str]] = []

    def add(node: dict, previous: str | None = None) -> str:
        nodes.append(node)
        if previous:
            edges.append([previous, node["id"]])
        return node["id"]

    if arch == "gru":
        prev = add(_node("tokens", "Tokens", "input"))
        prev = add(_node("embed", "Classical Embedding", "classical",
                         d_model=_get(model, "rnn_hidden", 16)), prev)
        prev = add(_node("gru", "GRU Recurrent Core", "classical",
                         hidden=_get(model, "rnn_hidden", 16)), prev)
        add(_node("head", "LM Head", "output",
                  head_type=_get(model, "head_type", "linear")), prev)
    elif arch == "qrnn":
        prev = add(_node("tokens", "Tokens", "input"))
        prev = add(_node("embed", "Classical Embedding", "classical"), prev)
        prev = add(_node("q_memory", "Quantum Memory Cell", "quantum",
                         n_qubits=quantum["n_qubits"],
                         circuit_depth=quantum["n_circuit_layers"],
                         trainable=quantum["trainable"]), prev)
        add(_node("head", "LM Head", "output",
                  head_type=_get(model, "head_type", "linear")), prev)
    else:
        prev = add(_node("tokens", "Tokens", "input"))
        prev = add(_node("embed", "Classical Embedding", "classical",
                         d_model=_get(model, "d_model", 64),
                         max_seq_len=_get(model, "max_seq_len", 128)), prev)
        encoder_kind = _get(model, "encoder_kind", "none")
        if encoder_kind != "none":
            enc_kind = "quantum" if encoder_kind == "quantum" else "classical"
            prev = add(_node(
                "sentence_encoder",
                f"{encoder_kind.title()} Sentence Encoder",
                enc_kind,
                condition=_get(model, "condition", "film"),
                d_sent=_get(model, "d_sent", 8),
                n_qubits=quantum["n_qubits"] if enc_kind == "quantum" else None,
            ), prev)
        n_blocks = int(_get(model, "n_blocks", 2) or 2)
        blocks = _get(model, "blocks")
        if isinstance(model, dict) and blocks is None:
            block_keys = sorted(
                {
                    int(key.split(".")[2])
                    for key in model
                    if key.startswith("model.blocks.") and key.endswith(".attn_type")
                }
            )
            blocks = [
                {
                    "attn_type": model.get(f"model.blocks.{i}.attn_type"),
                    "ffn_type": model.get(f"model.blocks.{i}.ffn_type"),
                }
                for i in block_keys
            ] or None
        for i in range(n_blocks):
            block = blocks[i] if blocks is not None and i < len(blocks) else None
            attn_type = _block_get(block, "attn_type", _get(model, "attn_type", "classical"))
            ffn_type = _block_get(block, "ffn_type", _get(model, "ffn_type", "classical"))
            attn_kind = _kind_for_component(attn_type)
            prev = add(_node(
                f"block_{i}_attn",
                f"Block {i + 1} {'Quantum' if attn_kind == 'quantum' else 'Classical'} Attention",
                attn_kind,
                attn_type=attn_type,
                heads=_get(model, "n_heads", 4),
                d_model=_get(model, "d_model", 64),
                n_qubits=quantum["n_qubits"] if attn_kind == "quantum" else None,
            ), prev)
            ffn_kind = _kind_for_component(ffn_type)
            prev = add(_node(
                f"block_{i}_ffn",
                f"Block {i + 1} {'Quantum' if ffn_kind == 'quantum' else 'Classical'} FFN",
                ffn_kind,
                ffn_type=ffn_type,
                d_ff=_get(model, "d_ff", 256),
                n_qubits=quantum["n_qubits"] if ffn_kind == "quantum" else None,
                circuit_depth=quantum["n_circuit_layers"] if ffn_kind == "quantum" else None,
            ), prev)
        add(_node("head", "LM Head", "output",
                  head_type=_get(model, "head_type", "linear")), prev)

    has_quantum = any(n["kind"] == "quantum" for n in nodes)
    return {
        "nodes": nodes,
        "edges": edges,
        "quantum": quantum if has_quantum else None,
        "summary": {
            "arch": arch,
            "uses_quantum": has_quantum,
            "model_family": model_family(model),
            "quantum_components": [n["id"] for n in nodes if n["kind"] == "quantum"],
        },
    }


def uses_quantum_config(cfg: ExperimentConfig | ModelConfig | dict) -> bool:
    return bool(model_graph_from_config(cfg)["summary"]["uses_quantum"])


def model_family(cfg: ExperimentConfig | ModelConfig | dict) -> str:
    model = cfg.model if isinstance(cfg, ExperimentConfig) else cfg
    arch = _get(model, "arch", "transformer")
    if arch in {"gru", "qrnn"}:
        return arch
    encoder_kind = _get(model, "encoder_kind", "none")
    if encoder_kind != "none":
        return "two-stream"
    blocks = _get(model, "blocks")
    if blocks is not None:
        attn_types = {_block_get(block, "attn_type", "classical") for block in blocks}
        ffn_types = {_block_get(block, "ffn_type", "classical") for block in blocks}
        if any(t in {"quantum_proj", "quantum_qkv"} for t in attn_types):
            return "hybrid-attention"
        if any(t in {"quantum", "quantum_linear"} for t in ffn_types):
            return "hybrid-ffn"
    if isinstance(model, dict) and any(
        key.startswith("model.blocks.") and str(value).startswith("quantum")
        for key, value in model.items()
    ):
        return "hybrid-transformer"
    if _get(model, "attn_type", "classical") == "quantum_proj":
        return "quantum-attention"
    if _get(model, "ffn_type", "classical") == "quantum":
        return "quantum-ffn"
    return "transformer"
