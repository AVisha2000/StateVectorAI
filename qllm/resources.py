"""Honest, provenance-labelled resource accounting for local QLLM runs."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax

from .config import ExperimentConfig, ModelConfig, QuantumConfig
from .quantum.capabilities import backend_capabilities_payload
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
        for name, _qcfg, logical_instances, _implementation in active_quantum_configs(model)
    }
    return sum(breakdown.values()), breakdown


def active_quantum_configs(
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


def _component_capabilities(
    qcfg: QuantumConfig, implementation: str
) -> dict[str, Any]:
    """Resolve the actual adapter semantics stored with a run resource plan."""
    if implementation == "configured_circuit_backend":
        return backend_capabilities_payload(
            qcfg.backend,
            qcfg.device,
            qcfg.diff_method,
            qcfg.shots,
            mps_max_bond_dimension=qcfg.mps_max_bond_dimension,
            mps_max_truncation_error=qcfg.mps_max_truncation_error,
            mps_relative_truncation=qcfg.mps_relative_truncation,
        )
    return backend_capabilities_payload(
        "jax_native_statevector", "jax_runtime", "backprop", None
    )


def quantum_component_resource_evidence(
    qcfg: QuantumConfig,
    implementation: str,
    *,
    logical_instances: int,
) -> dict[str, Any]:
    """Describe logical dimension, configured storage, and unknown runtime cost.

    The returned element counts describe one stored simulated state after a
    circuit operation.  They are never allocator, differentiation-tape, or
    device peak-memory measurements.
    """
    capabilities = _component_capabilities(qcfg, implementation)
    logical_dimension = 2 ** int(qcfg.n_qubits)
    representation = str(capabilities.get("representation") or "unverified")
    is_configured_mps = (
        implementation == "configured_circuit_backend"
        and qcfg.backend == "tensorcircuit_mps"
    )
    if is_configured_mps:
        max_bond_dimension = int(qcfg.mps_max_bond_dimension)
        stored_elements = (
            2 * int(qcfg.n_qubits) * max_bond_dimension * max_bond_dimension
        )
        storage_status = "configured_conservative_upper_bound"
        storage_basis = (
            "2 * n_qubits * mps_max_bond_dimension**2; conservative configured "
            "upper bound on stored post-truncation MPS state-tensor elements"
        )
    elif representation == "dense_statevector":
        max_bond_dimension = None
        stored_elements = logical_dimension
        storage_status = "representation_derived_not_measured"
        storage_basis = (
            "one complex statevector element per logical basis state; this is a "
            "representation-derived state size, not measured allocated memory"
        )
    else:
        max_bond_dimension = None
        stored_elements = None
        storage_status = "unavailable_for_backend_defined_representation"
        storage_basis = (
            "the backend capability profile does not expose a supported storage "
            "representation from which an element count can be derived"
        )

    excluded = [
        "automatic-differentiation intermediates",
        "gate-application intermediates",
        "nonlocal-gate and SWAP intermediates",
        "allocator overhead and peak memory",
        "total process or device memory",
        "runtime",
    ]
    approximation = capabilities.get("approximation")
    approximation_evidence: dict[str, Any] = {
        "capability_profile": approximation,
        "status": "configured_not_observed" if approximation else "not_applicable",
    }
    if is_configured_mps:
        approximation_evidence.update(
            {
                "truncation_mode": "fixed_bond_dimension_only",
                "threshold_support": "unsupported_for_jit_vmap_training",
                "configured_svd_split_threshold": None,
                "configured_svd_split_threshold_status": (
                    "unsupported_for_jit_vmap_training"
                ),
                "relative_truncation": False,
                "relative_truncation_status": (
                    "unsupported_for_jit_vmap_training"
                ),
                "realized_truncation_error": None,
                "realized_truncation_error_status": "unmeasured",
                "discarded_weight": None,
                "discarded_weight_status": "unmeasured",
                "convergence": None,
                "convergence_status": "unmeasured",
            }
        )

    return {
        "capabilities": capabilities,
        "representation": representation,
        "logical_hilbert_dimension": logical_dimension,
        "logical_hilbert_dimension_status": "exact_mathematical_dimension",
        "logical_dimension_is_dense_allocation_evidence": False,
        "storage": {
            "stored_state_tensor_elements_per_instance": stored_elements,
            "stored_state_tensor_elements_per_instance_status": storage_status,
            "stored_state_tensor_elements_per_instance_basis": storage_basis,
            "stored_state_tensor_elements_across_logical_instances_per_token": (
                stored_elements * int(logical_instances)
                if stored_elements is not None
                else None
            ),
            "logical_instances_per_token": int(logical_instances),
            "configured_max_bond_dimension": max_bond_dimension,
            "observed_bond_dimension": None,
            "observed_bond_dimension_status": (
                "unmeasured" if is_configured_mps else "not_applicable"
            ),
            "peak_memory_bytes": None,
            "peak_memory_status": "unmeasured",
            "total_memory_bytes": None,
            "total_memory_status": "unmeasured",
            "excludes": excluded,
        },
        "approximation_evidence": approximation_evidence,
    }


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
    active_configs = active_quantum_configs(model)
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
    component_evidence = {
        name: quantum_component_resource_evidence(
            qcfg, implementation, logical_instances=logical_instances
        )
        for name, qcfg, logical_instances, implementation in active_configs
    }
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
            "meaning": "maximum logical Hilbert-space dimension",
            "basis": (
                "maximum logical 2**n_qubits across active quantum components; "
                "0 for classical. This mathematical dimension is not evidence "
                "that dense state storage was allocated"
            ),
            "allocation_status": "not_an_allocation_measurement",
            "component_dimensions": state_dimensions,
            "component_representations": {
                name: evidence["representation"]
                for name, evidence in component_evidence.items()
            },
            "component_storage": {
                name: evidence["storage"]
                for name, evidence in component_evidence.items()
            },
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
                    "capabilities": component_evidence[name]["capabilities"],
                    "representation": component_evidence[name]["representation"],
                    "logical_hilbert_dimension": component_evidence[name][
                        "logical_hilbert_dimension"
                    ],
                    "logical_dimension_is_dense_allocation_evidence": False,
                    "storage": component_evidence[name]["storage"],
                    "approximation_evidence": component_evidence[name][
                        "approximation_evidence"
                    ],
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
