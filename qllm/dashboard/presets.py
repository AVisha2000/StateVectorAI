"""Curated, UI-safe experiment presets for QLLM Lab."""
from __future__ import annotations

import dataclasses

from ..config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    QuantumConfig,
    TrackingConfig,
    TrainConfig,
    to_flat_dict,
)


_BASE_TRAIN = TrainConfig(
    seed=0, steps=300, batch_size=16, seq_len=64, lr=0.001,
    weight_decay=0.01, grad_clip=1.0, eval_every=50, eval_batches=8,
)
_BASE_DATA = DataConfig(kind="text", corpus_path="data/input.txt", val_fraction=0.1)
_BASE_TRACKING = TrackingConfig(
    enabled=False, log_quantum_diagnostics=True, log_grad_norms=True
)

_QUANTUM_CONTROL_SPECS: dict[str, dict] = {
    "quantum-ffn-4q": {
        "summary": "Tune FFN circuit width and depth without changing the surrounding transformer.",
        "warning": "More qubits raise simulation cost sharply. Increase depth after qubit count is stable.",
        "fields": {
            "n_qubits": {"label": "Qubit count", "min": 2, "max": 10, "gpu_max": 16},
            "n_circuit_layers": {"label": "Circuit depth", "min": 1, "max": 8, "gpu_max": 16},
        },
    },
    "quantum-attn-4q": {
        "summary": "Tune the quantum attention projection while keeping the classical FFN fixed.",
        "warning": (
            "Attention swaps are per-token and memory-heavy. For 8+ qubits or depth 8+, "
            "use batch size 1-4 and seq_len 16-32 first."
        ),
        "fields": {
            "n_qubits": {"label": "Qubit count", "min": 2, "max": 8, "gpu_max": 12},
            "n_circuit_layers": {"label": "Circuit depth", "min": 1, "max": 8, "gpu_max": 12},
        },
    },
    "qrnn-small": {
        "summary": "Tune the quantum memory cell directly for recurrent sequence probes.",
        "warning": "QRNN cost grows quickly with qubits and may plateau if depth rises before optimization is stable.",
        "fields": {
            "n_qubits": {"label": "Memory qubits", "min": 2, "max": 12, "gpu_max": 18},
            "n_circuit_layers": {"label": "Memory depth", "min": 1, "max": 10, "gpu_max": 18},
        },
    },
    "two-stream-quantum-bias": {
        "summary": "Tune the sentence-level quantum encoder while leaving the token stack unchanged.",
        "warning": "Treat larger encoders as research probes first; compare against the classical twin at the same seed.",
        "fields": {
            "n_qubits": {"label": "Encoder qubits", "min": 2, "max": 10, "gpu_max": 16},
            "n_circuit_layers": {"label": "Encoder depth", "min": 1, "max": 8, "gpu_max": 16},
        },
    },
}


def _cfg(model: ModelConfig, run_name: str) -> ExperimentConfig:
    return ExperimentConfig(
        model=model,
        train=_BASE_TRAIN,
        data=_BASE_DATA,
        tracking=dataclasses.replace(_BASE_TRACKING, run_name=run_name),
    )


def _quantum_controls(preset_id: str, cfg: ExperimentConfig) -> dict:
    spec = _QUANTUM_CONTROL_SPECS.get(preset_id)
    if not spec:
        return {"enabled": False}
    qcfg = cfg.model.quantum
    fields = []
    for key, meta in spec["fields"].items():
        fields.append(
            {
                "key": key,
                "label": meta["label"],
                "min": meta["min"],
                "max": meta["max"],
                "gpu_max": meta.get("gpu_max", meta["max"]),
                "step": 1,
                "default": getattr(qcfg, key),
            }
        )
    return {
        "enabled": True,
        "summary": spec["summary"],
        "warning": spec["warning"],
        "comparison_note": (
            "Quantum overrides apply only to the selected preset. "
            "Linked classical comparison jobs stay fixed."
        ),
        "fields": fields,
    }


def _normalize_quantum_overrides(
    preset_id: str,
    overrides: dict | None,
    allow_gpu_scale: bool = False,
) -> dict[str, int]:
    if not overrides:
        return {}
    spec = _QUANTUM_CONTROL_SPECS.get(preset_id)
    if not spec:
        raise ValueError(f"Preset '{preset_id}' does not expose quantum tuning controls.")
    out: dict[str, int] = {}
    for key, meta in spec["fields"].items():
        raw = overrides.get(key)
        if raw in (None, ""):
            continue
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{meta['label']} must be an integer.") from exc
        max_value = meta.get("gpu_max", meta["max"]) if allow_gpu_scale else meta["max"]
        if value < meta["min"] or value > max_value:
            mode = "GPU range" if allow_gpu_scale else "safe CPU range"
            raise ValueError(
                f"{meta['label']} must stay between {meta['min']} and {max_value} ({mode})."
            )
        out[key] = value
    extra = set(overrides) - set(spec["fields"])
    if extra:
        raise ValueError(f"Unknown quantum override(s): {', '.join(sorted(extra))}.")
    return out


def apply_quantum_overrides(
    preset_id: str,
    cfg: ExperimentConfig,
    overrides: dict | None,
    allow_gpu_scale: bool = False,
) -> ExperimentConfig:
    normalized = _normalize_quantum_overrides(preset_id, overrides, allow_gpu_scale)
    if not normalized:
        return cfg
    qcfg = dataclasses.replace(cfg.model.quantum, **normalized)
    return dataclasses.replace(
        cfg,
        model=dataclasses.replace(cfg.model, quantum=qcfg),
    )


def _preset(
    *,
    label: str,
    kind: str,
    cost: str,
    summary: str,
    description: str,
    architecture: str,
    quantum_role: str,
    recommended_use: str,
    risks: str,
    config: ExperimentConfig,
    classical_twin_id: str | None = None,
) -> dict:
    return {
        "label": label,
        "kind": kind,
        "cost": cost,
        "summary": summary,
        "description": description,
        "architecture": architecture,
        "quantum_role": quantum_role,
        "recommended_use": recommended_use,
        "risks": risks,
        "classical_twin_id": classical_twin_id,
        "comparison_policy": "optional_default_on" if classical_twin_id else "none",
        "config": config,
    }


PRESETS: dict[str, dict] = {
    "classical-small": _preset(
        label="Classical Small",
        kind="classical",
        cost="Fast CPU sanity run",
        summary="Decoder-only transformer baseline with classical attention and FFN.",
        description=(
            "A compact decoder-only transformer. This is the main control model: "
            "token embeddings, causal self-attention, feed-forward blocks, and a "
            "linear output head are all classical neural-network layers."
        ),
        architecture="Transformer, 2 blocks, 4 heads, d_model=64, d_ff=256.",
        quantum_role="None. This is the baseline used for comparison.",
        recommended_use="Run first on every dataset to establish a classical floor.",
        risks="Small model; poor performance can mean undertraining or insufficient capacity.",
        config=_cfg(
            ModelConfig(d_model=64, n_heads=4, n_blocks=2, d_ff=256,
                        max_seq_len=128, attn_type="classical",
                        ffn_type="classical"),
            "classical-small",
        ),
    ),
    "quantum-ffn-4q": _preset(
        label="Quantum FFN 4q",
        kind="quantum",
        cost="Slower; useful quantum-swap comparison",
        summary="Classical transformer with the FFN replaced by a 4-qubit VQC.",
        description=(
            "A transformer where the feed-forward sub-layer is replaced by a "
            "4-qubit variational quantum circuit. Attention and embeddings stay "
            "classical, so differences isolate the FFN swap."
        ),
        architecture="Transformer, 2 blocks, classical attention, quantum FFN.",
        quantum_role="4-qubit reuploading VQC inside the FFN path.",
        recommended_use="Compare against classical-small on the same dataset/seed/steps.",
        risks="Quantum simulation is slower; gains must beat the classical twin and wall-time cost.",
        classical_twin_id="classical-small",
        config=_cfg(
            ModelConfig(d_model=64, n_heads=4, n_blocks=2, d_ff=256,
                        max_seq_len=128, ffn_type="quantum",
                        quantum=QuantumConfig(n_qubits=4, n_circuit_layers=2)),
            "quantum-ffn-4q",
        ),
    ),
    "quantum-attn-4q": _preset(
        label="Quantum Attention 4q",
        kind="quantum",
        cost="Slower; tests attention projection swap",
        summary="Transformer with quantum attention output projection.",
        description=(
            "A transformer where the attention output projection is replaced by "
            "a 4-qubit quantum projection while the FFN remains classical."
        ),
        architecture="Transformer, 2 blocks, quantum attention projection, classical FFN.",
        quantum_role="4-qubit VQC in the attention projection path.",
        recommended_use="Probe whether quantum projection helps before touching the FFN.",
        risks="May match classical quality only through classical surrounding layers.",
        classical_twin_id="classical-small",
        config=_cfg(
            ModelConfig(d_model=64, n_heads=4, n_blocks=2, d_ff=256,
                        max_seq_len=128, attn_type="quantum_proj",
                        ffn_type="classical",
                        quantum=QuantumConfig(n_qubits=4, n_circuit_layers=2)),
            "quantum-attn-4q",
        ),
    ),
    "gru-small": _preset(
        label="GRU Small",
        kind="classical",
        cost="Fast recurrent baseline",
        summary="Small classical GRU language model for recurrent comparison.",
        description=(
            "A compact classical recurrent language model. Use it as the control "
            "for quantum recurrent presets."
        ),
        architecture="GRU language model, hidden size 64.",
        quantum_role="None. Recurrent classical baseline.",
        recommended_use="Compare against QRNN on sequence-memory datasets.",
        risks="Different inductive bias from transformer baselines; compare within recurrent family.",
        config=_cfg(
            ModelConfig(arch="gru", rnn_hidden=64, max_seq_len=128),
            "gru-small",
        ),
    ),
    "qrnn-small": _preset(
        label="QRNN Small",
        kind="quantum",
        cost="Quantum recurrent run; use short smoke settings first",
        summary="Quantum recurrent language model using a learnable quantum memory.",
        description=(
            "A quantum recurrent model where a parameterized quantum memory cell "
            "processes token sequences. It is intended for memory/separation probes."
        ),
        architecture="Quantum recurrent language model with a 4-qubit memory cell.",
        quantum_role="Quantum recurrence supplies the model memory.",
        recommended_use="Use on generated sequence tasks and compare to GRU.",
        risks="Optimization can plateau; short text runs may be uninformative.",
        classical_twin_id="gru-small",
        config=_cfg(
            ModelConfig(arch="qrnn", max_seq_len=128,
                        quantum=QuantumConfig(n_qubits=4, n_circuit_layers=2)),
            "qrnn-small",
        ),
    ),
    "two-stream-classical-bias": _preset(
        label="Two-Stream Classical Bias",
        kind="classical",
        cost="Medium; two-stream classical control",
        summary="Two-stream LM with a classical sentence encoder bias.",
        description=(
            "A two-stream language model where a classical sentence encoder biases "
            "the token model. It controls for the two-stream architecture itself."
        ),
        architecture="Two-stream transformer with classical sentence encoder and bias conditioning.",
        quantum_role="None. Classical control for two-stream quantum bias.",
        recommended_use="Baseline for the two-stream quantum-bias preset.",
        risks="Larger control than classical-small; compare within two-stream family.",
        config=_cfg(
            ModelConfig(arch="two_stream", encoder_kind="classical",
                        condition="bias", d_sent=8, sent_hidden=8,
                        d_model=64, n_heads=4, n_blocks=2, d_ff=256,
                        max_seq_len=128),
            "two-stream-classical-bias",
        ),
    ),
    "two-stream-quantum-bias": _preset(
        label="Two-Stream Quantum Bias",
        kind="hybrid",
        cost="Medium; probes sentence-level quantum conditioning",
        summary="Two-stream LM where a quantum sentence encoder biases token modeling.",
        description=(
            "A two-stream language model where sentence-level state comes from a "
            "quantum encoder and is injected as a bias into token prediction."
        ),
        architecture="Two-stream transformer with quantum sentence encoder and bias conditioning.",
        quantum_role="4-qubit quantum sentence encoder conditions the token stream.",
        recommended_use="Compare against two-stream-classical-bias across multiple seeds.",
        risks="Signal has been statistically fragile; overlapping error bars are expected.",
        classical_twin_id="two-stream-classical-bias",
        config=_cfg(
            ModelConfig(arch="two_stream", encoder_kind="quantum",
                        condition="bias", d_sent=8, sent_hidden=8,
                        d_model=64, n_heads=4, n_blocks=2, d_ff=256,
                        max_seq_len=128,
                        quantum=QuantumConfig(n_qubits=4, n_circuit_layers=2)),
            "two-stream-quantum-bias",
        ),
    ),
}


def preset_meta(preset_id: str) -> dict:
    if preset_id not in PRESETS:
        raise KeyError(f"Unknown preset '{preset_id}'")
    payload = PRESETS[preset_id]
    cfg = payload["config"]
    return {
        "id": preset_id,
        "label": payload["label"],
        "kind": payload["kind"],
        "cost": payload["cost"],
        "summary": payload["summary"],
        "description": payload["description"],
        "architecture": payload["architecture"],
        "quantum_role": payload["quantum_role"],
        "recommended_use": payload["recommended_use"],
        "risks": payload["risks"],
        "classical_twin_id": payload["classical_twin_id"],
        "classical_analogue": _classical_analogue_meta(preset_id, payload),
        "comparison_policy": payload["comparison_policy"],
        "quantum_controls": _quantum_controls(preset_id, cfg),
        "defaults": {
            "steps": cfg.train.steps,
            "seed": cfg.train.seed,
            "eval_every": cfg.train.eval_every,
            "run_name": cfg.tracking.run_name,
        },
        "config": to_flat_dict(cfg),
    }


def list_presets() -> list[dict]:
    return [preset_meta(pid) for pid in PRESETS]


def build_preset(preset_id: str) -> ExperimentConfig:
    if preset_id not in PRESETS:
        raise KeyError(f"Unknown preset '{preset_id}'")
    return PRESETS[preset_id]["config"]


def _classical_analogue_meta(preset_id: str, payload: dict) -> dict | None:
    twin_id = payload["classical_twin_id"]
    if twin_id:
        twin = PRESETS[twin_id]
        return {
            "kind": "classical_analogue",
            "analogue_type": "component_swap",
            "resolver": "curated_twin",
            "label": twin["label"],
            "source_preset_id": preset_id,
            "analogue_preset_id": twin_id,
            "reason": (
                f"Uses curated classical twin '{twin_id}' from preset metadata. "
                "Queueing preserves dataset, seed, steps, eval cadence, preprocessing, batch size, and sequence length."
            ),
            "fairness_requirements": [
                "same_dataset",
                "same_seed",
                "same_steps",
                "same_eval_every",
                "same_train_split",
                "same_preprocessing",
                "same_batch_size",
                "same_sequence_length",
            ],
            "known_limitations": [
                "Curated twin is the repository-defined fair comparison, not necessarily exact parameter matching before training."
            ],
        }
    if payload["kind"] in {"quantum", "hybrid"}:
        return {
            "kind": "classical_analogue",
            "analogue_type": "component_swap",
            "resolver": "automatic_component_swap",
            "label": "Automatic classical analogue",
            "source_preset_id": preset_id,
            "analogue_preset_id": None,
            "reason": "Quantum components are replaced by classical components while preserving the surrounding model and training protocol.",
            "fairness_requirements": [
                "same_dataset",
                "same_seed",
                "same_steps",
                "same_eval_every",
                "same_train_split",
                "same_preprocessing",
                "same_batch_size",
                "same_sequence_length",
            ],
            "known_limitations": [
                "Parameter matching is verified after training metrics are recorded."
            ],
        }
    return None


def classical_twin_id(preset_id: str) -> str | None:
    return preset_meta(preset_id)["classical_twin_id"]
