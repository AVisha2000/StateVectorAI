"""Configuration system for QLLM experiments.

Frozen dataclasses (hashable -> safe as static fields on Flax modules) loaded
from YAML. The quantum component is selected purely via config flags
(`attn_type`, `ffn_type`), which is the core of the plugin architecture:
the same training/eval pipeline runs classical and hybrid models.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class QuantumConfig:
    """Configuration of a quantum sub-layer (circuit + backend)."""

    n_qubits: int = 4
    n_circuit_layers: int = 2
    ansatz: str = "reuploading"  # see qllm.quantum.circuits.ANSATZ_REGISTRY
    backend: str = "pennylane"   # see qllm.quantum.backends.BACKEND_REGISTRY
    device: str = "default.qubit"
    diff_method: str = "backprop"  # backprop = full JAX autodiff (simulators)
    shots: int | None = None       # None = analytic expectation values
    trainable: bool = True         # False = freeze circuit at random init
                                   # (random-feature control arm of ablations)
    readout: str = "z"             # z = <Z_i> (n feats) | zz = + <Z_i Z_j> pairs
                                   # (n + n(n-1)/2 feats; richer, still low-weight)
    dressing: str = "tanh"         # tanh = bounded angles | linear = NO classical
                                   # nonlinearity: the circuit must supply it all
    init_scale: float = 6.283185307179586  # uniform[0, init_scale) circuit init;
                                   # ~0.1-0.5 = near-identity (plateau mitigation)
    n_circuits: int = 1            # parallel quantum heads per layer ("scale by
                                   # blocks, not qubits")


@dataclass(frozen=True)
class BlockConfig:
    """Optional per-transformer-block component overrides."""

    attn_type: str = "classical"
    ffn_type: str = "classical"
    quantum: QuantumConfig | None = None


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 0  # filled in from tokenizer at runtime
    arch: str = "transformer"  # transformer | qrnn | gru
    rnn_hidden: int = 16       # hidden size for arch=gru
    embed_type: str = "classical"  # classical | quantum (words-as-quantum-states)
    head_type: str = "linear"  # linear | interference | mixture (output head)
    head_hypotheses: int = 4
    # two-stream sentence-encoder fields
    encoder_kind: str = "none"   # none | quantum | classical
    condition: str = "film"      # film | token | bias
    d_sent: int = 8
    sent_hidden: int = 8
    ffn_rank: int = 16             # bottleneck rank for ffn_type=lowrank
    d_model: int = 64
    n_heads: int = 4
    n_blocks: int = 2
    d_ff: int = 256
    max_seq_len: int = 128
    attn_type: str = "classical"   # classical | quantum_proj
    ffn_type: str = "classical"    # classical | quantum
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    blocks: tuple[BlockConfig, ...] | None = None


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    steps: int = 200
    batch_size: int = 16
    seq_len: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_every: int = 50
    eval_batches: int = 8


@dataclass(frozen=True)
class DataConfig:
    corpus_path: str = "data/input.txt"
    val_fraction: float = 0.1
    kind: str = "text"            # text | monitored_ising | markov_control
    # quantum-sequence generator parameters (monitored kicked-Ising):
    gen_qubits: int = 6           # system size (memory = gen_qubits - gen_measured)
    gen_measured: int = 2         # measured qubits/step -> vocab = 2**gen_measured
    gen_sequences: int = 64
    gen_len: int = 2048           # tokens per trajectory
    gen_theta_zz: float = 0.7853981633974483   # pi/4
    gen_theta_x: float = 0.7853981633974483    # pi/4 = maximally chaotic
    gen_steps_per_token: int = 1  # Floquet periods between emissions
    gen_seed: int = 0
    markov_order: int = 3
    # contextual-parity task (provable contextuality separation)
    ctx_observables: int = 12
    ctx_context_size: int = 4
    ctx_n_live: int = 3
    seq_cancel_density: float = 0.25         # for kind=markov_control (the matched twin)


@dataclass(frozen=True)
class TrackingConfig:
    enabled: bool = True
    experiment: str = "qllm"
    run_name: str | None = None
    tracking_uri: str = "sqlite:///mlflow.db"
    log_quantum_diagnostics: bool = True
    log_grad_norms: bool = True  # per-eval circuit-vs-classical grad split
    # own-dashboard per-step logging (replaces MLflow); set these to enable
    dashboard_db: str | None = None     # path to qllm_results.db, or None
    dashboard_suite: str | None = None
    dashboard_variant: str | None = None
    dashboard_dataset: str | None = None
    dashboard_seed: int | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)


_SECTION_TYPES = {
    "model": ModelConfig,
    "train": TrainConfig,
    "data": DataConfig,
    "tracking": TrackingConfig,
}


def _build(cls: type, payload: dict[str, Any]):
    """Recursively construct a (possibly nested) frozen dataclass from a dict."""
    kwargs: dict[str, Any] = {}
    fields = {f.name: f for f in dataclasses.fields(cls)}
    for key, value in payload.items():
        if key not in fields:
            raise KeyError(f"Unknown config key '{key}' for {cls.__name__}")
        ftype = fields[key].type
        if key == "quantum" and isinstance(value, dict):
            kwargs[key] = _build(QuantumConfig, value)
        elif key == "blocks" and value is not None:
            kwargs[key] = tuple(_build(BlockConfig, item) for item in value)
        else:
            kwargs[key] = value
        del ftype
    return cls(**kwargs)


def from_dict(payload: dict[str, Any]) -> ExperimentConfig:
    sections = {}
    for name, cls in _SECTION_TYPES.items():
        sections[name] = _build(cls, payload.get(name, {}) or {})
    return ExperimentConfig(**sections)


def model_block_config(model: ModelConfig, index: int) -> ModelConfig:
    """Return a ModelConfig view for one transformer block."""
    if model.blocks is None:
        return model
    block = model.blocks[index]
    return dataclasses.replace(
        model,
        attn_type=block.attn_type,
        ffn_type=block.ffn_type,
        quantum=block.quantum or model.quantum,
        blocks=None,
    )


def validate_config(cfg: ExperimentConfig) -> list[str]:
    """Return validation errors for editable configs."""
    errors: list[str] = []
    model = cfg.model
    if model.blocks is not None and model.n_blocks != len(model.blocks):
        errors.append("model.n_blocks must match len(model.blocks).")
    allowed_attn = {"classical", "quantum_proj", "quantum_qkv"}
    allowed_ffn = {"classical", "quantum", "quantum_linear", "lowrank"}
    blocks = model.blocks or tuple(
        BlockConfig(model.attn_type, model.ffn_type, model.quantum)
        for _ in range(model.n_blocks)
    )
    for i, block in enumerate(blocks):
        if block.attn_type not in allowed_attn:
            errors.append(f"Block {i + 1}: unknown attention type '{block.attn_type}'.")
        if block.ffn_type not in allowed_ffn:
            errors.append(f"Block {i + 1}: unknown FFN type '{block.ffn_type}'.")
        if (
            block.attn_type in {"quantum_proj", "quantum_qkv"}
            or block.ffn_type in {"quantum", "quantum_linear"}
        ) and (block.quantum is None and model.quantum is None):
            errors.append(f"Block {i + 1}: quantum component requires quantum config.")
    return errors


def load_yaml(path: str | Path) -> ExperimentConfig:
    with open(path) as fh:
        payload = yaml.safe_load(fh) or {}
    return from_dict(payload)


def to_flat_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    """Flatten config to dotted keys for experiment-tracker param logging."""
    flat: dict[str, Any] = {}

    def walk(obj: Any, prefix: str) -> None:
        for f in dataclasses.fields(obj):
            value = getattr(obj, f.name)
            key = f"{prefix}{f.name}"
            if dataclasses.is_dataclass(value):
                walk(value, key + ".")
            elif isinstance(value, tuple) and value and dataclasses.is_dataclass(value[0]):
                for i, item in enumerate(value):
                    walk(item, f"{key}.{i}.")
            else:
                flat[key] = value

    walk(cfg, "")
    return flat


def replace_model(cfg: ExperimentConfig, **kwargs) -> ExperimentConfig:
    """Convenience: return a copy with model fields replaced."""
    return dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, **kwargs))
