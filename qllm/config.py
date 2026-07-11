"""Configuration system for QLLM experiments.

Frozen dataclasses (hashable -> safe as static fields on Flax modules) loaded
from YAML. The quantum component is selected purely via config flags
(`attn_type`, `ffn_type`), which is the core of the plugin architecture:
the same training/eval pipeline runs classical and hybrid models.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .registry import (
    ANSATZ_TYPES,
    ARCH_TYPES,
    ATTN_TYPES,
    BACKEND_TYPES,
    CONDITION_TYPES,
    DATASET_KINDS,
    DRESSING_TYPES,
    EMBED_TYPES,
    ENCODER_TYPES,
    FFN_TYPES,
    HEAD_TYPES,
    QUANTUM_ARCH_TYPES,
    QUANTUM_ATTN_TYPES,
    QUANTUM_FFN_TYPES,
    QRNN_ONLY_ANSATZ_TYPES,
    READOUT_TYPES,
    choices_text,
)


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
    quantum: QuantumConfig | None = field(default_factory=QuantumConfig)
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
    if not isinstance(payload, dict):
        raise TypeError(f"{cls.__name__} config must be a mapping.")
    kwargs: dict[str, Any] = {}
    fields = {f.name: f for f in dataclasses.fields(cls)}
    for key, value in payload.items():
        if key not in fields:
            raise KeyError(f"Unknown config key '{key}' for {cls.__name__}")
        ftype = fields[key].type
        if key == "quantum":
            if value is None:
                kwargs[key] = None
            elif isinstance(value, dict):
                kwargs[key] = _build(QuantumConfig, value)
            else:
                raise TypeError(f"{cls.__name__}.{key} config must be a mapping or null.")
        elif key == "blocks" and value is not None:
            if not isinstance(value, (list, tuple)):
                raise TypeError("ModelConfig.blocks must be a list or tuple.")
            kwargs[key] = tuple(_build(BlockConfig, item) for item in value)
        else:
            kwargs[key] = value
        del ftype
    return cls(**kwargs)


def from_dict(payload: dict[str, Any]) -> ExperimentConfig:
    if not isinstance(payload, dict):
        raise TypeError("Experiment config must be a mapping.")
    unknown = sorted(set(payload) - set(_SECTION_TYPES))
    if unknown:
        names = ", ".join(unknown)
        raise KeyError(f"Unknown config section(s): {names}")
    sections = {}
    for name, cls in _SECTION_TYPES.items():
        section = payload.get(name, {})
        if section is None:
            section = {}
        sections[name] = _build(cls, section)
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


def two_stream_position_count(
    seq_len: int, encoder_kind: str, condition: str
) -> int:
    """Internal positions required for a two-stream input token sequence."""
    if encoder_kind != "none" and condition == "token":
        return 2 * seq_len
    return seq_len


def validate_config(cfg: ExperimentConfig) -> list[str]:
    """Return actionable errors for every shared config entry point.

    The function is deliberately dependency-free and never initializes a
    model, dataset, circuit, or backend.  CLI, dashboard, and training paths
    can therefore reject the same invalid configuration before side effects.
    """
    errors: list[str] = []
    model = cfg.model
    train = cfg.train
    data = cfg.data

    def choice(path: str, value: object, options: tuple[str, ...]) -> bool:
        if not isinstance(value, str) or value not in options:
            errors.append(
                f"{path} must be one of: {choices_text(options)}; got {value!r}."
            )
            return False
        return True

    def positive_int(path: str, value: object) -> bool:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            errors.append(f"{path} must be a positive integer; got {value!r}.")
            return False
        return True

    def nonnegative_int(path: str, value: object) -> bool:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            errors.append(f"{path} must be a non-negative integer; got {value!r}.")
            return False
        return True

    def finite_number(
        path: str, value: object, *, positive: bool = False,
        nonnegative: bool = False,
    ) -> bool:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"{path} must be a finite number; got {value!r}.")
            return False
        numeric = float(value)
        if not math.isfinite(numeric):
            errors.append(f"{path} must be finite; got {value!r}.")
            return False
        if positive and numeric <= 0:
            errors.append(f"{path} must be greater than 0; got {value!r}.")
            return False
        if nonnegative and numeric < 0:
            errors.append(f"{path} must be at least 0; got {value!r}.")
            return False
        return True

    def quantum(path: str, qcfg: QuantumConfig | None) -> None:
        if qcfg is None:
            errors.append(f"{path} must be provided.")
            return
        if not isinstance(qcfg, QuantumConfig):
            errors.append(f"{path} must be a QuantumConfig mapping.")
            return
        positive_int(f"{path}.n_qubits", qcfg.n_qubits)
        positive_int(f"{path}.n_circuit_layers", qcfg.n_circuit_layers)
        choice(f"{path}.ansatz", qcfg.ansatz, ANSATZ_TYPES)
        choice(f"{path}.backend", qcfg.backend, BACKEND_TYPES)
        choice(f"{path}.readout", qcfg.readout, READOUT_TYPES)
        choice(f"{path}.dressing", qcfg.dressing, DRESSING_TYPES)
        if qcfg.ansatz in QRNN_ONLY_ANSATZ_TYPES and model.arch != "qrnn":
            errors.append(
                f"{path}.ansatz='ising' is supported only when model.arch='qrnn'."
            )
        if not isinstance(qcfg.device, str) or not qcfg.device.strip():
            errors.append(f"{path}.device must be a non-empty string.")
        if not isinstance(qcfg.diff_method, str) or not qcfg.diff_method.strip():
            errors.append(f"{path}.diff_method must be a non-empty string.")
        if qcfg.shots is not None:
            positive_int(f"{path}.shots", qcfg.shots)
        if not isinstance(qcfg.trainable, bool):
            errors.append(f"{path}.trainable must be true or false.")
        finite_number(f"{path}.init_scale", qcfg.init_scale, positive=True)
        positive_int(f"{path}.n_circuits", qcfg.n_circuits)

    # Registry-backed model choices.
    arch_ok = choice("model.arch", model.arch, ARCH_TYPES)
    choice("model.attn_type", model.attn_type, ATTN_TYPES)
    choice("model.ffn_type", model.ffn_type, FFN_TYPES)
    choice("model.embed_type", model.embed_type, EMBED_TYPES)
    choice("model.head_type", model.head_type, HEAD_TYPES)
    choice("model.encoder_kind", model.encoder_kind, ENCODER_TYPES)
    choice("model.condition", model.condition, CONDITION_TYPES)

    nonnegative_int("model.vocab_size", model.vocab_size)
    positive_int("model.rnn_hidden", model.rnn_hidden)
    positive_int("model.head_hypotheses", model.head_hypotheses)
    positive_int("model.d_sent", model.d_sent)
    positive_int("model.sent_hidden", model.sent_hidden)
    positive_int("model.ffn_rank", model.ffn_rank)
    d_model_ok = positive_int("model.d_model", model.d_model)
    n_heads_ok = positive_int("model.n_heads", model.n_heads)
    n_blocks_ok = positive_int("model.n_blocks", model.n_blocks)
    positive_int("model.d_ff", model.d_ff)
    max_seq_ok = positive_int("model.max_seq_len", model.max_seq_len)
    attention_arch = arch_ok and model.arch in ("transformer", "two_stream")
    if (
        attention_arch
        and d_model_ok
        and n_heads_ok
        and model.d_model % model.n_heads != 0
    ):
        errors.append("model.d_model must be divisible by model.n_heads.")
    if model.arch != "two_stream" and model.encoder_kind != "none":
        errors.append(
            "model.encoder_kind must be 'none' unless model.arch='two_stream'."
        )
    if model.arch != "transformer" and model.head_type != "linear":
        errors.append(
            "model.head_type interference/mixture is supported only for "
            "model.arch='transformer'."
        )
    if model.arch != "transformer" and model.embed_type != "classical":
        errors.append(
            "model.embed_type='quantum' is supported only for "
            "model.arch='transformer'."
        )
    if model.arch != "transformer" and (
        model.attn_type != "classical" or model.ffn_type != "classical"
    ):
        errors.append(
            "model.attn_type and model.ffn_type must be 'classical' unless "
            "model.arch='transformer'; recurrent and two-stream models use "
            "their architecture-specific components."
        )
    if model.blocks is not None and model.arch != "transformer":
        errors.append("model.blocks is supported only for model.arch='transformer'.")
    if model.blocks is not None and (
        not n_blocks_ok or model.n_blocks != len(model.blocks)
    ):
        errors.append("model.n_blocks must match len(model.blocks).")

    blocks = model.blocks or (
        (BlockConfig(model.attn_type, model.ffn_type, None),)
        if not n_blocks_ok
        else tuple(
            BlockConfig(model.attn_type, model.ffn_type, None)
            for _ in range(model.n_blocks)
        )
    )
    global_quantum_required = (
        model.arch in QUANTUM_ARCH_TYPES
        or model.embed_type == "quantum"
        or (model.arch == "two_stream" and model.encoder_kind == "quantum")
        or any(
            block.attn_type in QUANTUM_ATTN_TYPES
            or block.ffn_type in QUANTUM_FFN_TYPES
            for block in blocks
            if isinstance(block, BlockConfig)
        )
    )
    if model.quantum is not None:
        quantum("model.quantum", model.quantum)
    elif global_quantum_required:
        errors.append("model.quantum must be provided for the selected quantum model.")

    for index, block in enumerate(blocks):
        path = f"model.blocks.{index}"
        if not isinstance(block, BlockConfig):
            errors.append(f"{path} must be a BlockConfig mapping.")
            continue
        choice(f"{path}.attn_type", block.attn_type, ATTN_TYPES)
        choice(f"{path}.ffn_type", block.ffn_type, FFN_TYPES)
        active_qcfg = block.quantum or model.quantum
        if block.quantum is not None:
            quantum(f"{path}.quantum", block.quantum)
        if (
            block.attn_type in QUANTUM_ATTN_TYPES
            or block.ffn_type in QUANTUM_FFN_TYPES
        ) and active_qcfg is None:
            errors.append(f"{path} quantum component requires quantum config.")

    # Training constraints used by both text and trajectory samplers.
    if isinstance(train.seed, bool) or not isinstance(train.seed, int):
        errors.append(f"train.seed must be an integer; got {train.seed!r}.")
    positive_int("train.steps", train.steps)
    positive_int("train.batch_size", train.batch_size)
    seq_len_ok = positive_int("train.seq_len", train.seq_len)
    finite_number("train.lr", train.lr, positive=True)
    finite_number("train.weight_decay", train.weight_decay, nonnegative=True)
    finite_number("train.grad_clip", train.grad_clip, positive=True)
    positive_int("train.eval_every", train.eval_every)
    positive_int("train.eval_batches", train.eval_batches)
    if attention_arch and seq_len_ok and max_seq_ok:
        required_positions = (
            two_stream_position_count(
                train.seq_len, model.encoder_kind, model.condition
            )
            if model.arch == "two_stream"
            else train.seq_len
        )
        if required_positions > model.max_seq_len:
            if (
                model.arch == "two_stream"
                and model.encoder_kind != "none"
                and model.condition == "token"
            ):
                errors.append(
                    "model.max_seq_len is the internal positional capacity and "
                    "must be >= 2 * train.seq_len for two-stream token "
                    f"conditioning with an active encoder; required "
                    f"{required_positions}, got {model.max_seq_len}."
                )
            else:
                errors.append("train.seq_len must be <= model.max_seq_len.")

    # Dataset and generator constraints.  Keeping every generator knob valid,
    # even when inactive, prevents malformed configs from becoming valid merely
    # by switching ``data.kind`` and makes later edits reproducible.
    data_kind_ok = choice("data.kind", data.kind, DATASET_KINDS)
    if not isinstance(data.corpus_path, str) or not data.corpus_path.strip():
        errors.append("data.corpus_path must be a non-empty string.")
    val_ok = finite_number("data.val_fraction", data.val_fraction)
    if val_ok and not 0.0 < float(data.val_fraction) < 1.0:
        errors.append("data.val_fraction must be strictly between 0 and 1.")
    gen_qubits_ok = positive_int("data.gen_qubits", data.gen_qubits)
    measured_ok = positive_int("data.gen_measured", data.gen_measured)
    sequences_ok = positive_int("data.gen_sequences", data.gen_sequences)
    gen_len_ok = positive_int("data.gen_len", data.gen_len)
    finite_number("data.gen_theta_zz", data.gen_theta_zz)
    finite_number("data.gen_theta_x", data.gen_theta_x)
    positive_int("data.gen_steps_per_token", data.gen_steps_per_token)
    if isinstance(data.gen_seed, bool) or not isinstance(data.gen_seed, int):
        errors.append(f"data.gen_seed must be an integer; got {data.gen_seed!r}.")
    markov_ok = positive_int("data.markov_order", data.markov_order)
    observables_ok = positive_int("data.ctx_observables", data.ctx_observables)
    context_ok = positive_int("data.ctx_context_size", data.ctx_context_size)
    positive_int("data.ctx_n_live", data.ctx_n_live)
    density_ok = finite_number("data.seq_cancel_density", data.seq_cancel_density)
    if density_ok and not 0.0 <= float(data.seq_cancel_density) < 1.0:
        errors.append("data.seq_cancel_density must be in [0, 1).")
    if gen_qubits_ok and measured_ok and data.gen_measured >= data.gen_qubits:
        errors.append("data.gen_measured must be smaller than data.gen_qubits.")
    if observables_ok and context_ok and data.ctx_context_size > data.ctx_observables:
        errors.append("data.ctx_context_size must be <= data.ctx_observables.")

    synthetic = data_kind_ok and data.kind != "text"
    if synthetic:
        if sequences_ok and data.gen_sequences < 2:
            errors.append(
                "data.gen_sequences must be at least 2 so train/validation "
                "splits use distinct trajectories."
            )
        if gen_len_ok and seq_len_ok and data.gen_len <= train.seq_len:
            errors.append(
                "data.gen_len must be greater than train.seq_len for "
                "within-trajectory sampling."
            )
    if data.kind == "markov_control" and gen_len_ok and markov_ok:
        if data.gen_len <= data.markov_order + 1:
            errors.append(
                "data.gen_len must be greater than data.markov_order + 1."
            )
    if data.kind in ("interference", "seq_cancellation"):
        if observables_ok and data.ctx_observables < 4:
            errors.append(
                "data.ctx_observables must be at least 4 for cancellation datasets."
            )
    if data.kind == "seq_cancellation" and gen_len_ok and context_ok:
        if data.gen_len <= data.ctx_context_size:
            errors.append(
                "data.gen_len must be greater than data.ctx_context_size."
            )

    # QRNN emissions require a power-of-two vocabulary with fewer measured
    # emission qubits than recurrent state qubits.  Synthetic Ising vocab is
    # known before loading; an explicitly finalized vocab is checked too.
    if model.arch == "qrnn" and model.quantum is not None:
        vocab = model.vocab_size if isinstance(model.vocab_size, int) else 0
        if data.kind in ("monitored_ising", "markov_control") and measured_ok:
            vocab = 2 ** data.gen_measured
        if vocab > 0:
            measured = vocab.bit_length() - 1
            if 2 ** measured != vocab:
                errors.append("QRNN vocabulary size must be a power of two.")
            elif isinstance(model.quantum.n_qubits, int) and (
                measured >= model.quantum.n_qubits
            ):
                errors.append(
                    "QRNN emission qubits must be fewer than "
                    "model.quantum.n_qubits."
                )
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
