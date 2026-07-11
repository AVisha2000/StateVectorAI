"""Honest, provenance-labelled resource accounting for local QLLM runs."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax

from .config import ExperimentConfig, ModelConfig, QuantumConfig
from .registry import QUANTUM_ARCH_TYPES, QUANTUM_ATTN_TYPES, QUANTUM_FFN_TYPES


def resolve_execution_device(target: str = "auto"):
    """Resolve an execution target without changing process-global JAX config."""
    requested = str(target or "auto").strip().lower()
    if requested not in {"auto", "cpu", "gpu"}:
        raise ValueError("device_target must be one of: auto, cpu, gpu.")
    try:
        devices = list(jax.devices() if requested == "auto" else jax.devices(requested))
    except RuntimeError as exc:
        raise RuntimeError(f"Requested JAX {requested} device is unavailable.") from exc
    if not devices:
        if requested == "auto":
            raise RuntimeError("JAX reported no execution devices.")
        raise RuntimeError(f"Requested JAX {requested} device is unavailable.")
    return devices[0]


def device_identity(device: Any) -> dict[str, Any]:
    """Return stable, non-secret identity fields for one resolved JAX device."""
    return {
        "platform": str(getattr(device, "platform", "unknown")),
        "device_kind": str(getattr(device, "device_kind", "unknown")),
        "id": int(getattr(device, "id", 0)),
    }


def _quantum_component_multiplier(model: ModelConfig) -> tuple[int, dict[str, int]]:
    breakdown = {
        name: logical_instances
        for name, _qcfg, logical_instances, _implementation in _active_quantum_configs(model)
    }
    return sum(breakdown.values()), breakdown


def _active_quantum_configs(
    model: ModelConfig,
) -> list[tuple[str, QuantumConfig, int, str]]:
    """Return active components and logical circuit instances per token.

    ``QuantumCore`` evaluates ``n_circuits`` parallel circuits. The quantum
    embedding, recurrent state updates, and amplitude-space ``quantum_linear``
    path each evaluate one logical state/circuit instance per token.
    """
    active: list[tuple[str, QuantumConfig, int, str]] = []
    global_q = model.quantum
    if global_q is not None:
        if model.embed_type == "quantum":
            active.append(("embedding", global_q, 1, "configured_circuit_backend"))
        if model.arch in QUANTUM_ARCH_TYPES:
            active.append(("recurrent", global_q, 1, "jax_native_statevector"))
        if model.arch == "two_stream" and model.encoder_kind == "quantum":
            active.append(
                (
                    "two_stream_encoder",
                    global_q,
                    global_q.n_circuits,
                    "configured_circuit_backend",
                )
            )
    if model.blocks is not None:
        for index, block in enumerate(model.blocks):
            qcfg = block.quantum or global_q
            if qcfg is None:
                continue
            if block.attn_type in QUANTUM_ATTN_TYPES:
                active.append(
                    (
                        f"block_{index}.attention",
                        qcfg,
                        qcfg.n_circuits,
                        "configured_circuit_backend",
                    )
                )
            if block.ffn_type in QUANTUM_FFN_TYPES:
                active.append(
                    (
                        f"block_{index}.feed_forward",
                        qcfg,
                        qcfg.n_circuits if block.ffn_type == "quantum" else 1,
                        (
                            "configured_circuit_backend"
                            if block.ffn_type == "quantum"
                            else "jax_native_statevector"
                        ),
                    )
                )
    elif global_q is not None:
        for index in range(max(int(model.n_blocks or 1), 1)):
            if model.attn_type in QUANTUM_ATTN_TYPES:
                active.append(
                    (
                        f"block_{index}.attention",
                        global_q,
                        global_q.n_circuits,
                        "configured_circuit_backend",
                    )
                )
            if model.ffn_type in QUANTUM_FFN_TYPES:
                active.append(
                    (
                        f"block_{index}.feed_forward",
                        global_q,
                        global_q.n_circuits if model.ffn_type == "quantum" else 1,
                        (
                            "configured_circuit_backend"
                            if model.ffn_type == "quantum"
                            else "jax_native_statevector"
                        ),
                    )
                )
    return active


def static_resource_plan(
    cfg: ExperimentConfig,
    *,
    n_params: int,
    requested_device: str,
    resolved_device: Any,
) -> dict[str, Any]:
    """Build immutable configured/exact resource facts known before training."""
    model = cfg.model
    multiplier, component_breakdown = _quantum_component_multiplier(model)
    active_configs = _active_quantum_configs(model)
    is_quantum = bool(active_configs)
    state_dimensions = {
        name: 2 ** int(qcfg.n_qubits)
        for name, qcfg, _logical_instances, _implementation in active_configs
    }
    state_dimension = max(state_dimensions.values(), default=0)
    per_step = int(cfg.train.batch_size) * int(cfg.train.seq_len) * multiplier
    configured_backends = sorted({qcfg.backend for _, qcfg, _, _ in active_configs})
    configured_devices = sorted({qcfg.device for _, qcfg, _, _ in active_configs})
    actual_backends = sorted(
        {
            qcfg.backend
            if implementation == "configured_circuit_backend"
            else implementation
            for _, qcfg, _, implementation in active_configs
        }
    )
    execution_identity = device_identity(resolved_device)
    actual_devices = sorted(
        {
            qcfg.device
            if implementation == "configured_circuit_backend"
            else f"jax:{execution_identity['platform']}:{execution_identity['id']}"
            for _, qcfg, _, implementation in active_configs
        }
    )
    return {
        "schema_version": 1,
        "n_params": int(n_params),
        "n_params_status": "exact",
        "parameters": {"value": int(n_params), "status": "exact", "unit": "count"},
        "state_dim": int(state_dimension),
        "state_dim_status": "exact",
        "state_dimension": {
            "value": int(state_dimension),
            "status": "exact",
            "basis": "maximum 2**n_qubits across active state-vector components; 0 for classical",
            "component_dimensions": state_dimensions,
        },
        "logical_circuit_forward_instances_per_train_step": {
            "value": int(per_step),
            "status": (
                "derived_logical_forward_instances" if is_quantum else "exact"
            ),
            "methodology": {
                "token_positions": int(cfg.train.batch_size) * int(cfg.train.seq_len),
                "component_multiplier": int(multiplier),
                "component_breakdown": component_breakdown,
            },
        },
        "backend_execution_calls": {
            "value": None,
            "status": "unsupported" if is_quantum else "not_applicable",
            "reason": (
                "backend execution calls are not instrumented; logical forward instances "
                "must not be interpreted as device or QPU executions"
                if is_quantum
                else "classical model has no quantum backend executions"
            ),
        },
        "quantum_backend": {
            "configured": configured_backends,
            "configured_devices": configured_devices,
            "actual": actual_backends,
            "actual_devices": actual_devices,
            "components": {
                name: {
                    "implementation": implementation,
                    "backend": (
                        qcfg.backend
                        if implementation == "configured_circuit_backend"
                        else "jax_native_statevector"
                    ),
                    "device": (
                        qcfg.device
                        if implementation == "configured_circuit_backend"
                        else execution_identity
                    ),
                    "configured_backend": qcfg.backend,
                    "configured_device": qcfg.device,
                    "configured_backend_status": (
                        "active"
                        if implementation == "configured_circuit_backend"
                        else "not_applicable_to_native_jax_component"
                    ),
                    "shots": (
                        qcfg.shots
                        if implementation == "configured_circuit_backend"
                        else None
                    ),
                    "configured_shots": qcfg.shots,
                    "shots_status": (
                        "active"
                        if implementation == "configured_circuit_backend"
                        else "not_applicable_to_native_jax_component"
                    ),
                    "logical_instances_per_token": int(logical_instances),
                }
                for name, qcfg, logical_instances, implementation in active_configs
            },
        },
        "backend": (
            actual_backends[0]
            if len(actual_backends) == 1
            else actual_backends or "none"
        ),
        "backend_status": "actual_implementation",
        "device": (
            actual_devices[0]
            if len(actual_devices) == 1
            else actual_devices or "none"
        ),
        "execution_device": {
            "requested": str(requested_device),
            "resolved": execution_identity,
        },
        "precision": {
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "default_float_bits": 64 if bool(jax.config.jax_enable_x64) else 32,
            "parameter_dtypes": [],
            "status": "configured_before_parameter_dtype_inventory",
        },
        "timing": {
            "status": "pending_runtime_measurement",
            "clock": "time.perf_counter",
        },
        "memory": {
            "status": "pending_runtime_measurement",
            "scope": "resolved_jax_device",
        },
    }


def parameter_precision(params: Any) -> dict[str, Any]:
    dtypes = sorted(
        {
            str(getattr(leaf, "dtype", "unknown"))
            for leaf in jax.tree_util.tree_leaves(params)
        }
    )
    return {
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "default_float_bits": 64 if bool(jax.config.jax_enable_x64) else 32,
        "parameter_dtypes": dtypes,
        "status": "measured_from_parameter_tree",
    }


def device_memory_evidence(device: Any) -> dict[str, Any]:
    """Read allocator/device memory counters, explicitly preserving unsupported states."""
    try:
        raw = device.memory_stats()
    except (AttributeError, RuntimeError, NotImplementedError) as exc:
        return {
            "status": "unsupported",
            "scope": "resolved_jax_device",
            "peak_bytes": None,
            "available_bytes": None,
            "capacity_bytes": None,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    if not isinstance(raw, Mapping):
        return {
            "status": "unsupported",
            "scope": "resolved_jax_device",
            "peak_bytes": None,
            "available_bytes": None,
            "capacity_bytes": None,
            "reason": "JAX device memory_stats returned no mapping",
        }
    peak = raw.get("peak_bytes_in_use")
    if peak is None:
        peak = raw.get("peak_bytes")
    capacity = raw.get("bytes_limit")
    if capacity is None:
        capacity = raw.get("total_bytes")
    in_use = raw.get("bytes_in_use")
    available = (
        max(int(capacity) - int(in_use), 0)
        if capacity is not None and in_use is not None
        else None
    )
    return {
        "status": (
            "measured" if peak is not None or capacity is not None else "unsupported"
        ),
        "scope": "resolved_jax_device_allocator",
        "peak_bytes": int(peak) if peak is not None else None,
        "available_bytes": available,
        "capacity_bytes": int(capacity) if capacity is not None else None,
        "raw_keys": sorted(str(key) for key in raw),
        "reason": (
            None
            if peak is not None or capacity is not None
            else "recognized memory counters unavailable"
        ),
    }


def runtime_resource_ledger(
    static_plan: Mapping[str, Any],
    *,
    params: Any,
    completed_steps_this_attempt: int,
    evaluation_forward_instances: int,
    first_step_seconds: float | None,
    steady_step_seconds: list[float],
    loop_wall_seconds: float,
    attempt_wall_seconds: float,
    fit_wall_seconds: float,
    device: Any,
) -> dict[str, Any]:
    """Complete a static plan with measured timing and allocator evidence."""
    resource = dict(static_plan)
    per_step = int(
        (
            static_plan.get("logical_circuit_forward_instances_per_train_step")
            or {}
        ).get("value")
        or 0
    )
    train_instances = per_step * int(completed_steps_this_attempt)
    total_instances = train_instances + int(evaluation_forward_instances)
    status = "derived_logical_forward_instances" if per_step else "exact"
    resource["logical_circuit_forward_instances"] = {
        "value": total_instances,
        "status": status,
        "breakdown": {
            "train": train_instances,
            "evaluation": int(evaluation_forward_instances),
            "scope": "training and validation forwards in this execution attempt",
            "excluded": (
                "model initialization, optimizer/gradient backend executions, "
                "and optional circuit diagnostics"
            ),
        },
    }
    resource["circuit_calls"] = total_instances
    resource["circuit_calls_kind"] = status
    resource["circuit_calls_status"] = status
    resource["circuit_calls_breakdown"] = {
        "train": train_instances,
        "evaluation": int(evaluation_forward_instances),
        "scope": "training and validation forwards in this execution attempt",
        "excluded": (
            "model initialization, optimizer/gradient backend executions, "
            "and optional circuit diagnostics"
        ),
        "methodology": static_plan.get(
            "logical_circuit_forward_instances_per_train_step"
        ),
    }
    steady_total = float(sum(steady_step_seconds))
    resource["timing"] = {
        "clock": "time.perf_counter",
        "completion_barrier": "jax.block_until_ready(loss)",
        "compile_plus_first_executed_train_step_seconds": first_step_seconds,
        "compile_plus_first_executed_train_step_status": (
            "measured" if first_step_seconds is not None else "not_executed"
        ),
        "steady_state_train_step_seconds_mean": (
            steady_total / len(steady_step_seconds) if steady_step_seconds else None
        ),
        "steady_state_train_step_seconds_total": (
            steady_total if steady_step_seconds else None
        ),
        "steady_state_train_steps": len(steady_step_seconds),
        "loop_wall_seconds": float(loop_wall_seconds),
        "loop_wall_scope": "training loop including validation and checkpoints",
        "attempt_wall_seconds": float(attempt_wall_seconds),
        "attempt_wall_scope": (
            "active attempt from pre-loop checkpoint setup through completed loop"
        ),
        "fit_wall_seconds": float(fit_wall_seconds),
        "fit_wall_scope": (
            "fit entry through completed training/evaluation before final result publication"
        ),
        "status": "measured",
    }
    resource["precision"] = parameter_precision(params)
    memory = device_memory_evidence(device)
    resource["memory"] = memory
    resource["peak_memory_bytes"] = memory["peak_bytes"]
    resource["peak_memory_status"] = memory["status"]
    resource["available_memory_bytes"] = memory["available_bytes"]
    resource["available_memory_status"] = (
        "measured" if memory["available_bytes"] is not None else "unsupported"
    )
    return resource
