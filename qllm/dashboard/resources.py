"""Resource heuristics for queue-time quantum run guardrails."""
from __future__ import annotations

from ..config import ExperimentConfig
from ..resources import (
    active_quantum_configs,
    quantum_component_resource_evidence,
)


def quantum_resource_estimate(cfg: ExperimentConfig) -> dict:
    """Return a conservative configured-work review, not a memory prediction.

    The legacy ``score`` and ``band`` fields remain for queue compatibility,
    but are labelled as a coarse proxy.  In particular, the MPS state-tensor
    bound cannot predict differentiation or nonlocal-gate intermediates.
    """
    ground_state = cfg.problem.task_type == "ground_state"
    if ground_state and cfg.model.quantum is not None:
        active = [
            (
                "vqe_ansatz",
                cfg.model.quantum,
                1,
                "configured_circuit_backend",
            )
        ]
    else:
        active = active_quantum_configs(cfg.model)
    token_calls = (
        0 if ground_state else int(cfg.train.batch_size) * int(cfg.train.seq_len)
    )
    configured_work_units = int(cfg.train.steps) if ground_state else token_calls
    components: dict[str, dict] = {}
    work_proxy = 0

    for name, qcfg, logical_instances, implementation in active:
        evidence = quantum_component_resource_evidence(
            qcfg, implementation, logical_instances=logical_instances
        )
        storage = evidence["storage"]
        stored_per_instance = storage[
            "stored_state_tensor_elements_per_instance"
        ]
        if stored_per_instance is None:
            # Keep the queue guard conservative for backend-defined
            # representations without mislabelling this fallback as storage.
            proxy_per_instance = evidence["logical_hilbert_dimension"]
            proxy_status = "logical_dimension_fallback_not_storage_evidence"
        else:
            proxy_per_instance = int(stored_per_instance)
            proxy_status = storage[
                "stored_state_tensor_elements_per_instance_status"
            ]
        layers = max(int(qcfg.n_circuit_layers), 1)
        component_proxy = (
            int(proxy_per_instance) * int(logical_instances) * layers
        )
        work_proxy += component_proxy
        components[name] = {
            "implementation": implementation,
            "configured_backend": qcfg.backend,
            "actual_backend": (
                qcfg.backend
                if implementation == "configured_circuit_backend"
                else "jax_native_statevector"
            ),
            "n_qubits": int(qcfg.n_qubits),
            "n_circuit_layers": int(qcfg.n_circuit_layers),
            "logical_instances_per_token": int(logical_instances),
            "logical_hilbert_dimension": evidence[
                "logical_hilbert_dimension"
            ],
            "logical_dimension_is_dense_allocation_evidence": False,
            "representation": evidence["representation"],
            "storage": storage,
            "approximation_evidence": evidence["approximation_evidence"],
            "configured_work_proxy_per_token": component_proxy,
            "configured_work_proxy_status": proxy_status,
        }

    score = int(configured_work_units * work_proxy) if active else 0
    if not active:
        band = "classical"
    elif score >= 16_000_000:
        band = "extreme"
    elif score >= 6_000_000:
        band = "high"
    elif score >= 1_500_000:
        band = "medium"
    else:
        band = "low"

    uses_quantum_attn = any(name.endswith(".attention") for name in components)
    max_qubits = max((row["n_qubits"] for row in components.values()), default=0)
    max_layers = max(
        (row["n_circuit_layers"] for row in components.values()), default=0
    )
    component_multiplier = sum(
        row["logical_instances_per_token"] for row in components.values()
    )
    state_dim = max(
        (row["logical_hilbert_dimension"] for row in components.values()),
        default=0,
    )
    known_storage_per_token = sum(
        row["storage"][
            "stored_state_tensor_elements_across_logical_instances_per_token"
        ]
        or 0
        for row in components.values()
    )
    unknown_storage_components = [
        name
        for name, row in components.items()
        if row["storage"]["stored_state_tensor_elements_per_instance"] is None
    ]
    mps_components = [
        name
        for name, row in components.items()
        if row["representation"] == "matrix_product_state"
    ]

    peak_memory_caveat = (
        "Peak memory is unmeasured and cannot be inferred from this configured "
        "state-storage proxy; automatic-differentiation, gate-application, "
        "nonlocal-gate/SWAP, allocator, and runtime intermediates are excluded."
    )
    advice: list[str] = []
    if active:
        advice.append(peak_memory_caveat)
    if mps_components:
        advice.append(
            "MPS element counts are fixed-bond configured post-truncation "
            "state-tensor bounds. Threshold and relative truncation are unsupported "
            "for JIT/vmap training; observed bond dimensions, discarded weight, "
            "convergence, and realized truncation error are not measured."
        )
    if band in {"high", "extreme"}:
        advice.append("Reduce batch size before increasing qubits or depth.")
        advice.append("Use seq_len 32 or lower for quantum attention sweeps.")
    if uses_quantum_attn and max_qubits >= 8:
        advice.append("Quantum attention is per-token; prefer q<=8 while scanning depth.")
    if max_layers >= 10:
        advice.append(
            "Depth >=10 can trigger large JIT intermediates; scale depth after width is stable."
        )

    return {
        "band": band,
        "band_status": "coarse_threshold_on_configured_work_proxy",
        "score": score,
        "score_status": (
            "coarse_configured_state_work_proxy_not_memory_or_runtime_prediction"
        ),
        "score_methodology": {
            "formula": (
                "train.steps * sum(component state-element proxy * logical "
                "instances * circuit layers)"
                if ground_state
                else "train.batch_size * train.seq_len * sum(component state-element "
                "proxy * logical instances per token * circuit layers)"
            ),
            "mps_proxy": (
                "2 * n_qubits * configured max bond dimension**2"
            ),
            "dense_proxy": "logical 2**n_qubits statevector elements",
            "precision": "coarse_configured_guardrail_only",
        },
        "state_dim": int(state_dim),
        "state_dim_status": "exact_logical_dimension_not_storage_allocation",
        "token_calls": int(token_calls),
        "objective_steps": int(cfg.train.steps) if ground_state else None,
        "work_unit": "optimizer_step" if ground_state else "token_instance",
        "component_multiplier": int(component_multiplier),
        "uses_quantum_attention": uses_quantum_attn,
        "components": components,
        "representations": sorted(
            {row["representation"] for row in components.values()}
        ),
        "approximation_evidence": {
            name: row["approximation_evidence"]
            for name, row in components.items()
        },
        "state_storage": {
            "known_configured_or_representation_derived_elements_per_token": (
                int(known_storage_per_token)
            ),
            "unknown_components": unknown_storage_components,
            "status": (
                "partial" if unknown_storage_components else (
                    "configured_or_representation_derived" if active else "not_applicable"
                )
            ),
            "logical_dimension_is_dense_allocation_evidence": False,
        },
        "peak_memory": {
            "bytes": None,
            "status": "unmeasured" if active else "not_applicable",
            "caveat": peak_memory_caveat if active else None,
        },
        "advice": advice,
    }
