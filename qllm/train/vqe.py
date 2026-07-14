"""Durable analytic CPU VQE runner for registered toy ground-state tasks."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.training.train_state import TrainState

from ..config import ExperimentConfig, to_flat_dict, validate_config
from ..problems import GroundStateInstance, get_ground_state_instance
from ..quantum.backends import get_state_circuit
from ..quantum.capabilities import BackendCapabilities, resolve_backend_capabilities
from ..quantum.circuits import weight_shape
from ..registry import metric_type_spec
from ..research_protocol import normalize_seed_axes
from ..resources import device_identity, resolve_execution_device
from .artifacts import (
    RunOptions,
    atomic_write_bytes,
    atomic_write_json,
    build_run_manifest,
    read_checkpoint,
    resolve_run_options,
    restore_checkpoint,
    validate_checkpoint_manifest,
    validate_manifest,
    write_checkpoint,
    write_immutable_manifest,
)


GROUND_STATE_PRIMARY_METRIC_TYPE = "ground_state_energy_error"
_LOWER_BOUND_TOLERANCE = 1e-5


def _primary_metric_name(metric_type: str) -> str:
    spec = metric_type_spec(metric_type)
    if (
        metric_type != GROUND_STATE_PRIMARY_METRIC_TYPE
        or spec is None
        or str(spec.get("extraction_key")) != "energy_error"
    ):
        raise ValueError(
            "The VQE runner requires metric_type "
            f"{GROUND_STATE_PRIMARY_METRIC_TYPE!r} with extraction key "
            "'energy_error'."
        )
    return "energy_error"


def _require_valid_problem(
    cfg: ExperimentConfig, primary_metric_type: str
) -> tuple[GroundStateInstance, BackendCapabilities]:
    _primary_metric_name(primary_metric_type)
    if cfg.problem.task_type != "ground_state":
        raise ValueError(
            "qllm.train.vqe.run_vqe requires "
            "problem.task_type='ground_state'."
        )
    if not cfg.problem.instance_id:
        raise ValueError("A registered ground-state problem.instance_id is required.")
    try:
        instance = get_ground_state_instance(cfg.problem.instance_id)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc

    errors = validate_config(cfg)
    if errors:
        raise ValueError("Invalid experiment config:\n- " + "\n- ".join(errors))
    qcfg = cfg.model.quantum
    if qcfg is None or qcfg.trainable is not True:
        raise ValueError(
            "Ground-state VQE requires a trainable model.quantum configuration."
        )
    if qcfg.shots is not None:
        raise ValueError("Ground-state VQE supports analytic shots=None only.")
    if qcfg.diff_method != "backprop":
        raise ValueError(
            "The state-derived VQE objective supports diff_method='backprop' only."
        )
    if qcfg.n_qubits != instance.n_qubits:
        raise ValueError(
            "model.quantum.n_qubits must match the registered instance."
        )
    capabilities = resolve_backend_capabilities(
        qcfg.backend,
        qcfg.device,
        qcfg.diff_method,
        qcfg.shots,
        qcfg.mps_max_bond_dimension,
        qcfg.mps_max_truncation_error,
        qcfg.mps_relative_truncation,
    )
    if not capabilities.state_access.supported or capabilities.exactness != "exact":
        raise ValueError(
            "Ground-state VQE requires exact backend state access for the "
            "initial diagnostic slice."
        )
    if not capabilities.gradients.supported:
        raise ValueError(
            "Ground-state VQE requires a backend mode with verified analytic "
            "gradients."
        )
    return instance, capabilities


def _pauli_matrix(pauli: str) -> np.ndarray:
    matrices = {
        "I": np.eye(2, dtype=np.complex128),
        "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
    }
    try:
        return matrices[pauli]
    except KeyError as exc:
        raise ValueError(f"Unsupported Pauli operator {pauli!r}.") from exc


def hamiltonian_matrix(instance: GroundStateInstance) -> np.ndarray:
    """Build the dense reference matrix from immutable registered Pauli terms."""
    dimension = 2 ** instance.n_qubits
    result = np.zeros((dimension, dimension), dtype=np.complex128)
    for term in instance.terms:
        operators = ["I"] * instance.n_qubits
        for pauli, qubit in zip(term.pauli, term.qubits, strict=True):
            if qubit < 0 or qubit >= instance.n_qubits:
                raise ValueError(
                    f"Pauli term references out-of-range qubit {qubit}."
                )
            if operators[qubit] != "I":
                raise ValueError(f"Pauli term repeats qubit {qubit}.")
            operators[qubit] = pauli
        matrix = _pauli_matrix(operators[0])
        for operator in operators[1:]:
            matrix = np.kron(matrix, _pauli_matrix(operator))
        result += float(term.coefficient) * matrix
    if not np.allclose(result, result.conj().T):
        raise ValueError("Registered ground-state Hamiltonian must be Hermitian.")
    return result


def _exact_ground_energy(
    instance: GroundStateInstance, matrix: np.ndarray
) -> float:
    energy = float(np.linalg.eigvalsh(matrix).min())
    exact = next(
        (
            reference
            for reference in instance.classical_references
            if reference.reference_id == "exact_diagonalization"
            and reference.role == "oracle"
        ),
        None,
    )
    if exact is None or not exact.certified:
        raise ValueError(
            "Registered VQE instances require a certified exact-diagonalization "
            "reference."
        )
    if not np.isclose(energy, exact.energy, atol=1e-12, rtol=1e-12):
        raise ValueError(
            "Registered exact-reference energy disagrees with diagonalization."
        )
    return energy


def _product_state_reference_energy(
    instance: GroundStateInstance, reference
) -> float:
    certificate = reference.to_payload()["certificate"]
    if not isinstance(certificate, dict):
        raise ValueError("The product-state certificate must be a mapping.")
    vectors = certificate.get("bloch_vectors")
    if not isinstance(vectors, list) or len(vectors) != instance.n_qubits:
        raise ValueError(
            "The product-state reference requires one certified Bloch vector "
            "per qubit."
        )
    normalized: list[dict[str, float]] = []
    for vector in vectors:
        if not isinstance(vector, dict):
            raise ValueError("Product-state Bloch vectors must be mappings.")
        try:
            values = {
                axis: float(vector[axis])
                for axis in ("x", "y", "z")
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Product-state Bloch vectors require finite x, y, and z values."
            ) from exc
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError("Product-state Bloch vectors must be finite.")
        if sum(value * value for value in values.values()) > 1.0 + 1e-12:
            raise ValueError("Product-state Bloch vectors must lie in the unit ball.")
        normalized.append(values)

    energy = 0.0
    for term in instance.terms:
        expectation = 1.0
        for operator, qubit in zip(term.pauli, term.qubits, strict=True):
            if operator != "I":
                expectation *= normalized[qubit][operator.lower()]
        energy += float(term.coefficient) * expectation
    return energy


def _validate_reference_ladder(
    instance: GroundStateInstance, matrix: np.ndarray
) -> float:
    exact_energy = _exact_ground_energy(instance, matrix)
    product = next(
        (
            reference
            for reference in instance.classical_references
            if reference.reference_id == "best_product_state"
        ),
        None,
    )
    if product is None:
        raise ValueError(
            "Registered VQE instances require a product-state descriptive challenger."
        )
    if product.role != "descriptive_challenger" or not product.certified:
        raise ValueError(
            "The product-state reference must be a certified descriptive challenger."
        )
    certified_energy = _product_state_reference_energy(instance, product)
    if not np.isclose(certified_energy, product.energy, atol=1e-12, rtol=1e-12):
        raise ValueError(
            "Registered product-state reference energy disagrees with its certificate."
        )
    return exact_energy


def _checked_energy_error(energy: float, exact_ground_energy: float) -> float:
    if not np.isfinite(energy):
        raise FloatingPointError("VQE energy is non-finite.")
    delta = float(energy) - float(exact_ground_energy)
    if delta < -_LOWER_BOUND_TOLERANCE:
        raise ValueError(
            "Variational energy violated the certified exact lower bound."
        )
    return abs(delta)


def _resource_plan(
    instance: GroundStateInstance,
    capabilities: BackendCapabilities,
    qcfg,
    *,
    requested_device: str,
    resolved_device,
    configured_steps: int,
    eval_every: int,
    primary_metric_type: str,
    primary_metric_name: str,
) -> dict[str, Any]:
    configured_evaluations = len(
        {
            *range(eval_every, configured_steps + 1, eval_every),
            configured_steps,
        }
    )
    return {
        "schema_version": 1,
        "execution_kind": "analytic_statevector_simulator_diagnostic",
        "evidence_scope": "simulator_diagnostic_only",
        "qpu_evidence": False,
        "primary_metric": {
            "metric_type": primary_metric_type,
            "extraction_key": primary_metric_name,
        },
        "measurement_budget": {
            "shots": None,
            "status": "not_applicable_analytic_statevector",
        },
        "configured_optimizer_steps": int(configured_steps),
        "configured_standalone_energy_evaluations": int(
            configured_evaluations + 1
        ),
        "configured_logical_objective_invocations": int(
            configured_steps + configured_evaluations + 1
        ),
        "analytic_gradient_mode": {
            "method": qcfg.diff_method,
            "status": "configured_simulator_autodiff",
            "hardware_circuit_equivalent": None,
        },
        "measured_hardware_circuit_executions": {
            "value": None,
            "status": "not_applicable_no_qpu_execution",
        },
        "state_dimension": {
            "value": 2 ** instance.n_qubits,
            "status": "exact_mathematical_dimension_not_measured_memory",
        },
        "quantum_backend": capabilities.to_payload(),
        "execution_device": {
            "requested": requested_device,
            "resolved": device_identity(resolved_device),
        },
        "classical_reference_ladder": [
            reference.to_payload()
            for reference in instance.classical_references
        ],
    }


def _checkpoint_source(
    raw_options: RunOptions,
) -> tuple[RunOptions, Path | None, dict | None, str | None]:
    if not raw_options.resume_from:
        return raw_options, None, None, None
    source_path = Path(raw_options.resume_from).resolve()
    source_payload = read_checkpoint(source_path)
    source_manifest = source_payload["manifest"]
    source_run_uuid = str(source_manifest["run_uuid"])
    requested_run_uuid = raw_options.run_uuid
    mode = (
        "recovery"
        if requested_run_uuid is None or requested_run_uuid == source_run_uuid
        else "fork"
    )
    source_artifact_dir = source_path.parent.parent
    if mode == "recovery":
        if (
            raw_options.experiment_uuid is not None
            and raw_options.experiment_uuid != source_manifest["experiment_uuid"]
        ):
            raise ValueError(
                "Recovery must retain the checkpoint experiment_uuid."
            )
        if (
            raw_options.artifact_dir is not None
            and Path(raw_options.artifact_dir).resolve()
            != source_artifact_dir.resolve()
        ):
            raise ValueError(
                "Recovery must retain the checkpoint artifact directory."
            )
        raw_options = dataclasses.replace(
            raw_options,
            experiment_uuid=str(source_manifest["experiment_uuid"]),
            run_uuid=source_run_uuid,
            artifact_dir=str(source_artifact_dir),
        )
    else:
        if raw_options.artifact_dir is None:
            raise ValueError(
                "Forking a VQE checkpoint requires an explicit artifact_dir."
            )
        if (
            raw_options.parent_run_uuid is not None
            and raw_options.parent_run_uuid != source_run_uuid
        ):
            raise ValueError(
                "Fork parent_run_uuid must match the checkpoint source run_uuid."
            )
        raw_options = dataclasses.replace(
            raw_options,
            parent_run_uuid=source_run_uuid,
        )
    return raw_options, source_path, source_payload, mode


def _prior_runtime_accounting(source_payload: dict | None) -> tuple[float, int]:
    if source_payload is None:
        return 0.0, 0
    runtime = source_payload.get("rng_state") or {}
    if not isinstance(runtime, dict):
        raise ValueError("VQE checkpoint runtime accounting must be a mapping.")
    raw_seconds = runtime.get("cumulative_wall_seconds", 0.0)
    raw_attempts = runtime.get("attempt_count", 1)
    if isinstance(raw_seconds, bool) or not isinstance(raw_seconds, (int, float)):
        raise ValueError("VQE checkpoint cumulative wall time must be numeric.")
    seconds = float(raw_seconds)
    if not np.isfinite(seconds) or seconds < 0.0:
        raise ValueError(
            "VQE checkpoint cumulative wall time must be finite and non-negative."
        )
    if (
        isinstance(raw_attempts, bool)
        or not isinstance(raw_attempts, int)
        or raw_attempts < 0
    ):
        raise ValueError(
            "VQE checkpoint attempt count must be a non-negative integer."
        )
    return seconds, raw_attempts


def _manifest_only_retry(artifact_dir: Path) -> dict | None:
    if not artifact_dir.exists():
        return None
    if not artifact_dir.is_dir():
        raise ValueError(f"Artifact path is not a directory: {artifact_dir}")
    entries = {entry.name for entry in artifact_dir.iterdir()}
    checkpoint_dir = artifact_dir / "checkpoints"
    checkpoint_entries = (
        list(checkpoint_dir.iterdir()) if checkpoint_dir.is_dir() else []
    )
    temp_only = all(
        entry.is_file()
        and entry.name.endswith(".tmp")
        and entry.name.startswith((".latest.msgpack.", ".best.msgpack."))
        for entry in checkpoint_entries
    )
    if (
        "manifest.json" in entries
        and entries <= {"manifest.json", "checkpoints"}
        and (not checkpoint_entries or temp_only)
    ):
        try:
            return validate_manifest(
                json.loads((artifact_dir / "manifest.json").read_text("utf-8"))
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"Existing incomplete VQE run manifest is invalid: {exc}"
            ) from exc
    if entries:
        raise ValueError(
            "Refusing to overwrite a non-empty artifact directory for a new "
            f"VQE run: {artifact_dir}."
        )
    return None


def run_vqe(
    cfg: ExperimentConfig,
    verbose: bool = True,
    out_dir: str | Path = "results",
    should_cancel=None,
    run_options: RunOptions | None = None,
    progress_callback=None,
    publish_guard=None,
    primary_metric_type: str = GROUND_STATE_PRIMARY_METRIC_TYPE,
) -> dict:
    """Minimize a registered Hamiltonian on CPU without touching LM training."""
    instance, capabilities = _require_valid_problem(cfg, primary_metric_type)
    supplied_options = run_options or RunOptions()
    normalized_options = supplied_options.normalized()
    requested_device = normalized_options.device_target
    if requested_device != "cpu":
        raise ValueError(
            "The initial VQE slice requires explicit device_target='cpu' for "
            "analytic simulation."
        )
    resolved_device = resolve_execution_device("cpu")
    with jax.default_device(resolved_device):
        return _run_on_device(
            cfg,
            instance,
            capabilities,
            verbose=verbose,
            out_dir=out_dir,
            should_cancel=should_cancel,
            raw_options=normalized_options,
            requested_device=requested_device,
            progress_callback=progress_callback,
            publish_guard=publish_guard,
            primary_metric_type=primary_metric_type,
            resolved_device=resolved_device,
        )


def _run_on_device(
    cfg: ExperimentConfig,
    instance: GroundStateInstance,
    capabilities: BackendCapabilities,
    *,
    verbose: bool,
    out_dir: str | Path,
    should_cancel,
    raw_options: RunOptions,
    requested_device: str,
    progress_callback,
    publish_guard,
    primary_metric_type: str,
    resolved_device,
) -> dict:
    qcfg = cfg.model.quantum
    assert qcfg is not None
    primary_metric_name = _primary_metric_name(primary_metric_type)
    attempt_started = time.perf_counter()
    run_name = cfg.tracking.run_name or instance.instance_id

    raw_options, source_path, source_payload, resume_mode = _checkpoint_source(
        raw_options
    )
    source_manifest = source_payload["manifest"] if source_payload else None
    if resume_mode == "recovery":
        source_run_name = str(source_manifest.get("run_name") or "")
        if source_run_name and run_name != source_run_name:
            raise ValueError(
                "Recovery must retain the checkpoint run_name identity."
            )
        source_requested = (
            (source_manifest.get("resource_plan") or {})
            .get("execution_device", {})
            .get("requested")
        )
        if isinstance(source_requested, str):
            requested_device = source_requested

    options = resolve_run_options(raw_options)
    artifact_dir = (
        Path(raw_options.artifact_dir)
        if raw_options.artifact_dir is not None
        else Path(out_dir) / "runs" / str(options.run_uuid)
    )
    retry_manifest = None
    if source_payload is None:
        retry_manifest = _manifest_only_retry(artifact_dir)
    elif resume_mode == "fork":
        _manifest_only_retry(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = artifact_dir / "checkpoints"
    latest_checkpoint = checkpoint_dir / "latest.msgpack"
    best_checkpoint = checkpoint_dir / "best.msgpack"

    init_key = jax.random.PRNGKey(cfg.train.seed)
    initial_weights = jax.random.uniform(
        init_key,
        weight_shape(qcfg.n_circuit_layers, qcfg.n_qubits),
        minval=0.0,
        maxval=float(qcfg.init_scale),
        dtype=jnp.float32,
    )
    initial_params = {"circuit_weights": initial_weights}
    matrix = hamiltonian_matrix(instance)
    exact_ground_energy = _validate_reference_ladder(instance, matrix)
    circuit = get_state_circuit(
        qcfg.backend,
        qcfg.device,
        qcfg.n_qubits,
        qcfg.n_circuit_layers,
        qcfg.ansatz,
        qcfg.diff_method,
        qcfg.shots,
        qcfg.mps_max_bond_dimension,
        qcfg.mps_max_truncation_error,
        qcfg.mps_relative_truncation,
    )
    hamiltonian = jnp.asarray(matrix, dtype=jnp.complex64)
    inputs = jnp.zeros((qcfg.n_qubits,), dtype=jnp.float32)

    def energy_for(params):
        statevector = circuit(inputs, params["circuit_weights"])
        return jnp.real(jnp.vdot(statevector, hamiltonian @ statevector))

    tx = optax.chain(
        optax.clip_by_global_norm(cfg.train.grad_clip),
        optax.adamw(cfg.train.lr, weight_decay=cfg.train.weight_decay),
    )
    state = TrainState.create(apply_fn=energy_for, params=initial_params, tx=tx)

    if source_payload is not None and source_path is not None:
        state, source_payload = restore_checkpoint(source_path, state)
        source_manifest = source_payload["manifest"]
    prior_wall_seconds, prior_attempt_count = _prior_runtime_accounting(
        source_payload
    )
    attempt_count = prior_attempt_count + 1

    if source_manifest is not None:
        seed_axes = dict(source_manifest.get("seed_axes") or {})
        initialization = dict(source_manifest.get("initialization") or {})
    else:
        seed_axes = normalize_seed_axes(
            cfg.train.seed,
            data_kind=None,
            circuit_applicable=True,
            minibatch_applicable=False,
            explicit=raw_options.seed_axes,
            reject_unsupported=True,
        )
        initialization = {
            "mode": "seeded_uniform_circuit_weights",
            "parameters_sha256": hashlib.sha256(
                serialization.to_bytes(initial_params)
            ).hexdigest(),
            "source": "train.seed",
            "seed": int(cfg.train.seed),
            "init_scale": float(qcfg.init_scale),
            "train_seed_controls_parameters": True,
        }

    if source_payload is not None and source_path is not None:
        if resume_mode == "fork":
            resume_lineage = {
                "mode": "fork",
                "source_checkpoint": str(source_path),
                "source_checkpoint_sha256": hashlib.sha256(
                    source_path.read_bytes()
                ).hexdigest(),
                "source_run_uuid": source_manifest["run_uuid"],
                "source_manifest_hash": source_manifest["manifest_hash"],
                "source_completed_step": int(source_payload["completed_step"]),
                "parent_run_uuid": options.parent_run_uuid,
                "resume_event_uuid": str(uuid.uuid4()),
            }
        else:
            resume_lineage = dict(source_manifest.get("resume_lineage") or {})
    else:
        resume_lineage = {}

    resource_plan = _resource_plan(
        instance,
        capabilities,
        qcfg,
        requested_device=requested_device,
        resolved_device=resolved_device,
        configured_steps=cfg.train.steps,
        eval_every=cfg.train.eval_every,
        primary_metric_type=primary_metric_type,
        primary_metric_name=primary_metric_name,
    )
    current_manifest = build_run_manifest(
        cfg,
        instance,
        options,
        run_name=run_name,
        seed_axes=seed_axes,
        initialization=initialization,
        resume_lineage=resume_lineage,
        resource_plan=resource_plan,
    )
    if source_manifest is not None:
        validate_checkpoint_manifest(source_manifest, current_manifest)
        if resume_mode == "recovery":
            source_device = (
                (source_manifest.get("resource_plan") or {})
                .get("execution_device", {})
                .get("resolved")
            )
            if source_device and source_device != device_identity(resolved_device):
                raise ValueError(
                    "Exact VQE recovery must retain the execution device."
                )
            manifest = source_manifest
        else:
            manifest = current_manifest
    elif retry_manifest is not None:
        if retry_manifest.get("manifest_hash") != current_manifest.get(
            "manifest_hash"
        ):
            raise ValueError(
                "Existing incomplete VQE manifest differs from the retried "
                "configuration or environment."
            )
        manifest = retry_manifest
    else:
        manifest = current_manifest

    def require_publish_ownership() -> None:
        if publish_guard is not None and not bool(publish_guard()):
            raise RuntimeError("Run publication ownership was lost.")

    require_publish_ownership()
    write_immutable_manifest(artifact_dir / "manifest.json", manifest)

    completed_step = int(source_payload["completed_step"]) if source_payload else 0
    history = list(source_payload["history"]) if source_payload else []
    best_metric = source_payload.get("best_metric") if source_payload else None
    best_step = source_payload.get("best_step") if source_payload else None
    if int(np.asarray(state.step)) != completed_step:
        raise ValueError(
            "Checkpoint completed_step does not match the VQE TrainState step."
        )
    if completed_step < 0 or completed_step > cfg.train.steps:
        raise ValueError(
            "Checkpoint completed_step is outside the configured VQE run."
        )

    dashboard_config = to_flat_dict(cfg) | {
        "research.task_type": "ground_state",
        "research.metric_type": primary_metric_type,
        "research.reference_ladder": [
            reference.to_payload()
            for reference in instance.classical_references
        ],
        "research.simulator_diagnostic": True,
        "research.comparative_inference_enabled": False,
    }
    dash = None
    dash_key = None
    if cfg.tracking.dashboard_db:
        from ..resultsdb import ResultsDB

        dash = ResultsDB(cfg.tracking.dashboard_db)
        dash_key = (
            f"{cfg.tracking.dashboard_suite}/"
            f"{cfg.tracking.dashboard_variant}/{instance.instance_id}/"
            f"{cfg.train.seed}/{cfg.train.steps}"
        )
        require_publish_ownership()
        dash.start_run(
            run_key=dash_key,
            run_name=run_name,
            suite=cfg.tracking.dashboard_suite or "adhoc",
            variant=cfg.tracking.dashboard_variant or run_name,
            dataset=cfg.tracking.dashboard_dataset or instance.instance_id,
            seed=(
                cfg.tracking.dashboard_seed
                if cfg.tracking.dashboard_seed is not None
                else cfg.train.seed
            ),
            total_steps=cfg.train.steps,
            config=dashboard_config,
            run_uuid=options.run_uuid,
            experiment_uuid=options.experiment_uuid,
            manifest=manifest,
            primary_metric_type=primary_metric_type,
        )

    def save_checkpoint(
        path: Path, *, cumulative_wall_seconds: float | None = None
    ) -> None:
        require_publish_ownership()
        if cumulative_wall_seconds is None:
            cumulative_wall_seconds = (
                prior_wall_seconds + time.perf_counter() - attempt_started
            )
        write_checkpoint(
            path,
            state,
            completed_step=completed_step,
            rng_state={
                "algorithm": "deterministic_after_initialization",
                "attempt_count": attempt_count,
                "cumulative_wall_seconds": cumulative_wall_seconds,
            },
            history=history,
            best_metric=(float(best_metric) if best_metric is not None else None),
            best_step=(int(best_step) if best_step is not None else None),
            manifest=manifest,
            resume_lineage=resume_lineage,
        )

    def notify_progress() -> None:
        if progress_callback is None:
            return
        progress_callback(
            {
                "completed_step": completed_step,
                "checkpoint_path": str(latest_checkpoint.resolve()),
                "best_checkpoint_path": (
                    str(best_checkpoint.resolve())
                    if best_checkpoint.exists()
                    else None
                ),
                "artifact_dir": str(artifact_dir.resolve()),
                "manifest": manifest,
            }
        )

    save_checkpoint(latest_checkpoint)
    notify_progress()

    value_and_grad = jax.jit(jax.value_and_grad(energy_for))
    energy_only = jax.jit(energy_for)
    cancelled = False
    cadence = raw_options.checkpoint_every or cfg.train.eval_every
    attempt_gradient_steps = 0
    attempt_evaluation_calls = 0
    first_step_seconds: float | None = None
    steady_step_seconds: list[float] = []

    for step in range(completed_step + 1, cfg.train.steps + 1):
        if should_cancel is not None and should_cancel():
            cancelled = True
            break
        step_started = time.perf_counter()
        energy_before, grads = value_and_grad(state.params)
        jax.block_until_ready(energy_before)
        leaves = jax.tree_util.tree_leaves(grads)
        if not bool(jnp.isfinite(energy_before)) or not all(
            bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves
        ):
            raise FloatingPointError("VQE energy or gradient is non-finite.")
        _checked_energy_error(float(energy_before), exact_ground_energy)
        state = state.apply_gradients(grads=grads)
        completed_step = step
        attempt_gradient_steps += 1
        step_seconds = time.perf_counter() - step_started
        if first_step_seconds is None:
            first_step_seconds = step_seconds
        else:
            steady_step_seconds.append(step_seconds)

        evaluated = step % cfg.train.eval_every == 0 or step == cfg.train.steps
        if evaluated:
            energy_value = float(energy_only(state.params))
            attempt_evaluation_calls += 1
            energy_error = _checked_energy_error(
                energy_value, exact_ground_energy
            )
            row = {
                "step": step,
                "energy": energy_value,
                "exact_ground_energy": exact_ground_energy,
                "energy_error": energy_error,
            }
            history.append(row)
            if best_metric is None or energy_error < float(best_metric):
                best_metric = energy_error
                best_step = step
                save_checkpoint(best_checkpoint)
            if dash is not None:
                require_publish_ownership()
                dash.log_step(
                    dash_key,
                    step,
                    {
                        "energy": energy_value,
                        "exact_ground_energy": exact_ground_energy,
                        "energy_error": energy_error,
                    },
                    run_uuid=options.run_uuid,
                )
            if verbose:
                print(
                    f"[{run_name}] step {step:5d} energy "
                    f"{energy_value:.6f} error {energy_error:.3e}"
                )

        if step % cadence == 0 or evaluated or step == cfg.train.steps:
            save_checkpoint(latest_checkpoint)
        notify_progress()

    if cancelled or not latest_checkpoint.exists():
        save_checkpoint(latest_checkpoint)

    final_energy = float(energy_only(state.params))
    attempt_evaluation_calls += 1
    final = {
        "energy": final_energy,
        "exact_ground_energy": exact_ground_energy,
        "energy_error": _checked_energy_error(
            final_energy, exact_ground_energy
        ),
    }
    n_params = sum(
        int(np.prod(leaf.shape)) for leaf in jax.tree_util.tree_leaves(state.params)
    )
    attempt_wall_seconds = time.perf_counter() - attempt_started
    wall_seconds = prior_wall_seconds + attempt_wall_seconds
    save_checkpoint(latest_checkpoint, cumulative_wall_seconds=wall_seconds)
    resources = {
        **resource_plan,
        "completed_optimizer_gradient_steps": {
            "value": int(completed_step),
            "status": "derived_from_durable_train_state",
        },
        "attempt_optimizer_gradient_steps": {
            "value": int(attempt_gradient_steps),
            "status": "measured_control_flow_count",
        },
        "attempt_energy_evaluations": {
            "value": int(attempt_evaluation_calls),
            "status": "measured_host_invocation_count",
        },
        "attempt_logical_analytic_objective_invocations": {
            "value": int(attempt_gradient_steps + attempt_evaluation_calls),
            "status": "measured_host_invocation_count",
        },
        "recorded_evaluation_points": {
            "value": len(history),
            "status": "derived_from_durable_history",
        },
        "analytic_state_preparations": {
            "value": int(attempt_gradient_steps + attempt_evaluation_calls),
            "status": (
                "logical_simulator_objective_invocations; not physical or "
                "hardware circuit executions"
            ),
        },
        "hardware_equivalent_circuit_executions": {
            "value": None,
            "status": "not_defined_for_analytic_backpropagation",
        },
        "timing": {
            "first_optimizer_step_seconds": first_step_seconds,
            "steady_optimizer_step_seconds": steady_step_seconds,
            "prior_attempt_count": prior_attempt_count,
            "attempt_count": attempt_count,
            "prior_wall_seconds": prior_wall_seconds,
            "attempt_wall_seconds": attempt_wall_seconds,
            "cumulative_wall_seconds": wall_seconds,
        },
    }

    if dash is not None and not cancelled:
        require_publish_ownership()
        dash.record(
            suite=cfg.tracking.dashboard_suite or "adhoc",
            variant=cfg.tracking.dashboard_variant or run_name,
            dataset=cfg.tracking.dashboard_dataset or instance.instance_id,
            seed=(
                cfg.tracking.dashboard_seed
                if cfg.tracking.dashboard_seed is not None
                else cfg.train.seed
            ),
            steps=cfg.train.steps,
            n_params=n_params,
            val_loss=None,
            val_ppl=None,
            val_bpc=None,
            wall_seconds=wall_seconds,
            config=dashboard_config,
            run_uuid=options.run_uuid,
            experiment_uuid=options.experiment_uuid,
            manifest_hash=manifest["manifest_hash"],
            manifest=manifest,
            finalize_manifest=False,
            resources=resources,
            primary_metric_type=primary_metric_type,
            metric_values=final,
        )
    if dash is not None and not raw_options.defer_dashboard_terminal:
        dash.finish_run(
            dash_key,
            status="cancelled" if cancelled else "done",
            run_uuid=options.run_uuid,
        )

    if not cancelled:
        require_publish_ownership()
        atomic_write_bytes(
            artifact_dir / "params.msgpack",
            serialization.to_bytes(state.params),
        )
    summary = {
        "schema_version": 1,
        "run_name": run_name,
        "task_type": "ground_state",
        "instance_id": instance.instance_id,
        "instance_hash": instance.content_hash,
        "solver_kind": "vqe",
        "evidence_kind": "analytic_simulator_diagnostic",
        "claim_eligible": False,
        "comparative_inference_enabled": False,
        "paired_stats": None,
        "energy_units": instance.energy_units,
        "artifact_dir": str(artifact_dir.resolve()),
        "experiment_uuid": options.experiment_uuid,
        "run_uuid": options.run_uuid,
        "manifest_hash": manifest["manifest_hash"],
        "n_params": n_params,
        "wall_seconds": round(wall_seconds, 6),
        "attempt_wall_seconds": round(attempt_wall_seconds, 6),
        "steps": cfg.train.steps,
        "completed_step": completed_step,
        "cancelled": cancelled,
        "resumed": source_payload is not None,
        "resumed_from": str(source_path) if source_path is not None else None,
        "resume_lineage": resume_lineage,
        "checkpoint_path": str(latest_checkpoint.resolve()),
        "best_checkpoint_path": (
            str(best_checkpoint.resolve()) if best_checkpoint.exists() else None
        ),
        "best_step": best_step,
        "primary_metric_type": primary_metric_type,
        "primary_metric_name": primary_metric_name,
        "primary_metric_value": final[primary_metric_name],
        "instance": dict(instance.metadata),
        "reference_ladder": [
            reference.to_payload()
            for reference in instance.classical_references
        ],
        "execution": (
            "analytic CPU simulator diagnostic; no QPU evidence; shots=None"
        ),
        "resources": resources,
        "history": history,
        "final_energy": final["energy"],
        **final,
    }
    require_publish_ownership()
    atomic_write_json(artifact_dir / "summary.json", summary)
    return {
        "state": state,
        "summary": summary,
        "manifest": manifest,
        "run_options": options,
        "instance": instance,
    }
