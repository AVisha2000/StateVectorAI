"""Canonical, dependency-free registries for configurable QLLM choices.

This module intentionally contains only dependency-free scalar metadata.
Configuration validation and dashboard inference can therefore share canonical
contracts without importing JAX, Flax, PennyLane, or an optional backend.
"""
from __future__ import annotations

from collections.abc import Mapping
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
TASK_TYPES = (
    "sequence_modeling",
    "ground_state",
    "combinatorial_optimization",
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
        "task_type": TASK_TYPES,
        "ansatz": ANSATZ_TYPES,
        "circuit_ansatz": CIRCUIT_ANSATZ_TYPES,
        "backend": BACKEND_TYPES,
        "readout": READOUT_TYPES,
        "dressing": DRESSING_TYPES,
        **COMPONENT_REGISTRY,
        **CONDITIONING_REGISTRY,
    }
)

# A metric becomes eligible for comparative inference only through one of
# these contracts. The extraction key lives beside admission so callers cannot
# relabel a value from an unrelated result column.
MetricSpec = Mapping[str, str | bool]
METRIC_TYPES: Mapping[str, MetricSpec] = MappingProxyType(
    {
        "strict_autoregressive_next_token": MappingProxyType(
            {
                "lower_is_better": True,
                "units": "ppl",
                "pairable": True,
                "extraction_key": "val_ppl",
                "comparator_class": "matched_control",
            }
        ),
        "validation_perplexity": MappingProxyType(
            {
                "lower_is_better": True,
                "units": "ppl",
                "pairable": True,
                "extraction_key": "val_ppl",
                "comparator_class": "matched_control",
            }
        ),
        "ground_state_energy_error": MappingProxyType(
            {
                "lower_is_better": True,
                "units": "problem_energy_units",
                "pairable": False,
                "extraction_key": "energy_error",
                "comparator_class": "exact_reference_diagnostic",
            }
        ),
    }
)

# Comparison eligibility is server-owned. Result payloads may report their
# identity, but they cannot self-register by setting ``registration_status``.
SolverRunnerKey = tuple[str, str, str, str]
SolverRunnerSpec = Mapping[str, object]
SOLVER_RUNNERS: Mapping[SolverRunnerKey, SolverRunnerSpec] = MappingProxyType(
    {
        (
            "qllm.train.vqe.run_vqe",
            "analytic_statevector_v1",
            "qllm_vqe",
            "analytic_backprop_v1",
        ): MappingProxyType(
            {
                "task_type": "ground_state",
                "computation_kind": "quantum",
                "registration_status": "diagnostic_only",
                "comparison_eligible": False,
                "reason": (
                    "The current VQE runner uses analytic shots=None and is "
                    "not eligible for solver competition."
                ),
            }
        )
    }
)


def choices_text(values: tuple[str, ...]) -> str:
    """Return stable, user-facing choice text for validation errors."""
    return ", ".join(values)


def metric_type_spec(
    metric_type: str | None, *, require_pairable: bool = False
) -> MetricSpec | None:
    """Resolve one metric contract, failing closed for unknown metrics."""
    if not isinstance(metric_type, str):
        return None
    spec = METRIC_TYPES.get(metric_type)
    if spec is None or (require_pairable and not bool(spec.get("pairable"))):
        return None
    return spec


def solver_runner_registration(
    *,
    runner_id: object,
    runner_version: object,
    solver_id: object,
    solver_version: object,
) -> SolverRunnerSpec | None:
    """Resolve canonical solver-runner identity without trusting run metadata."""
    values = (runner_id, runner_version, solver_id, solver_version)
    if not all(isinstance(value, str) and value.strip() for value in values):
        return None
    return SOLVER_RUNNERS.get(values)


def supported_choices_payload() -> dict[str, object]:
    """JSON-safe copy used by dashboard clients and external tooling."""
    from .problems import ground_state_instances_payload

    payload: dict[str, object] = {
        name: list(values) for name, values in SUPPORTED_CHOICES.items()
    }
    payload["metric_types"] = {
        name: dict(spec) for name, spec in METRIC_TYPES.items()
    }
    payload["ground_state_instances"] = ground_state_instances_payload()
    return payload
