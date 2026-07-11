"""Canonical, dependency-free registries for configurable QLLM choices.

This module intentionally contains strings only.  Configuration validation can
therefore use the same supported-choice definitions as runtime dispatch without
importing JAX, Flax, PennyLane, or an optional backend.
"""
from __future__ import annotations

from types import MappingProxyType


ARCH_TYPES = (
    "transformer",
    "qrnn",
    "gru",
    "contextual_qrnn",
    "routed_contextual",
    "two_stream",
)
ATTN_TYPES = ("classical", "quantum_proj", "quantum_qkv")
EMBED_TYPES = ("classical", "quantum")
FFN_TYPES = ("classical", "quantum", "quantum_linear", "lowrank")
HEAD_TYPES = ("linear", "interference", "mixture")

DATASET_KINDS = (
    "text",
    "monitored_ising",
    "markov_control",
    "contextual",
    "interference",
    "seq_cancellation",
)

# ``ising`` is a recurrent evolution mode, not a PennyLane/TensorCircuit
# feed-forward circuit.  It is accepted only for ``arch=qrnn`` by the shared
# semantic validator.
CIRCUIT_ANSATZ_TYPES = ("hardware_efficient", "reuploading")
QRNN_ONLY_ANSATZ_TYPES = ("ising",)
ANSATZ_TYPES = CIRCUIT_ANSATZ_TYPES + QRNN_ONLY_ANSATZ_TYPES

BACKEND_TYPES = ("pennylane", "tensorcircuit", "tensorcircuit_mps")
READOUT_TYPES = ("z", "zz")
DRESSING_TYPES = ("tanh", "linear")
ENCODER_TYPES = ("none", "quantum", "classical")
CONDITION_TYPES = ("film", "token", "bias")

QUANTUM_ARCH_TYPES = ("qrnn", "contextual_qrnn", "routed_contextual")
QUANTUM_ATTN_TYPES = tuple(
    value for value in ATTN_TYPES if value.startswith("quantum")
)
QUANTUM_FFN_TYPES = tuple(
    value for value in FFN_TYPES if value.startswith("quantum")
)
QUANTUM_COMPONENT_TYPES = (
    "quantum",
    "quantum_linear",
    *QUANTUM_ATTN_TYPES,
)

# Grouped views are useful to dashboard/config consumers that need to present
# component or conditioning choices without rebuilding parallel lists.
COMPONENT_REGISTRY = MappingProxyType(
    {
        "attention": ATTN_TYPES,
        "embedding": EMBED_TYPES,
        "feed_forward": FFN_TYPES,
        "head": HEAD_TYPES,
    }
)
CONDITIONING_REGISTRY = MappingProxyType(
    {"encoder": ENCODER_TYPES, "condition": CONDITION_TYPES}
)
SUPPORTED_CHOICES = MappingProxyType(
    {
        "architecture": ARCH_TYPES,
        "quantum_architecture": QUANTUM_ARCH_TYPES,
        "dataset": DATASET_KINDS,
        "ansatz": ANSATZ_TYPES,
        "circuit_ansatz": CIRCUIT_ANSATZ_TYPES,
        "backend": BACKEND_TYPES,
        "readout": READOUT_TYPES,
        "dressing": DRESSING_TYPES,
        **COMPONENT_REGISTRY,
        **CONDITIONING_REGISTRY,
    }
)


def choices_text(values: tuple[str, ...]) -> str:
    """Return stable, user-facing choice text for validation errors."""
    return ", ".join(values)


def supported_choices_payload() -> dict[str, list[str]]:
    """JSON-safe copy used by dashboard clients and external tooling."""
    return {name: list(values) for name, values in SUPPORTED_CHOICES.items()}
