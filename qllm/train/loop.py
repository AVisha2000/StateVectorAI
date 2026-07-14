"""Training loop, evaluation, and generation.

One pipeline for every model variant: the loop never knows whether the
blocks inside the model are classical or quantum — that is the payoff of
the plugin architecture.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
import uuid
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization, traverse_util
from flax.training.train_state import TrainState

from ..config import (
    ExperimentConfig,
    TrainConfig,
    to_flat_dict,
    two_stream_position_count,
    validate_config,
)
from ..data.datasets import load_dataset_bundle
from ..data.text import CharTokenizer, sample_batch, train_val_split
from ..models.model import build_model, uses_quantum
from ..registry import metric_type_spec
from ..research_protocol import normalize_seed_axes
from ..resources import (
    active_quantum_configs,
    resolve_execution_device,
    runtime_resource_ledger,
    static_resource_plan,
)
from ..tracking import ExperimentTracker, log_quantum_diagnostics
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


DEFAULT_SEQUENCE_PRIMARY_METRIC_TYPE = "strict_autoregressive_next_token"


def _sequence_primary_metric_name(metric_type: str) -> str:
    """Resolve the one metric this sequence-only runner can publish."""
    spec = metric_type_spec(metric_type)
    if spec is None:
        raise ValueError(f"Unsupported primary metric_type {metric_type!r}.")
    extraction_key = str(spec["extraction_key"])
    if extraction_key != "val_ppl":
        raise ValueError(
            f"metric_type {metric_type!r} extracts {extraction_key!r}; "
            "the sequence runner only produces 'val_ppl'. Use the "
            "metric-specific sibling runner instead."
        )
    return extraction_key


def _require_sequence_task(cfg: ExperimentConfig) -> None:
    if cfg.problem.task_type != "sequence_modeling":
        raise ValueError(
            f"task_type {cfg.problem.task_type!r} requires a task-specific "
            "sibling runner; qllm.train.loop.fit is sequence-modeling only."
        )


def _format_quantum_diagnostics_for_display(value):
    """Format measured scalars without assuming unavailable evidence is numeric."""
    if isinstance(value, dict):
        return {
            key: _format_quantum_diagnostics_for_display(item)
            for key, item in value.items()
        }
    if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        return f"{float(value):.3e}"
    return value


def _quantum_tracking_context(model_cfg):
    """Describe the quantum components that the finalized model will execute.

    Per-block overrides are authoritative.  The model-level quantum config is
    only a default and must never be reported as the executed backend when an
    active component overrides it or uses a native JAX implementation.
    """
    active = active_quantum_configs(model_cfg)
    components = {}
    for name, qcfg, logical_instances, implementation in active:
        configured_backend = qcfg.backend
        configured_device = qcfg.device
        if implementation == "configured_circuit_backend":
            actual_backend = configured_backend
            actual_device = configured_device
        else:
            actual_backend = "jax_native_statevector"
            actual_device = "jax_runtime"
        components[name] = {
            "implementation": implementation,
            "backend": actual_backend,
            "device": actual_device,
            "configured_backend": configured_backend,
            "configured_device": configured_device,
            "n_qubits": int(qcfg.n_qubits),
            "ansatz": qcfg.ansatz,
            "logical_instances_per_token": int(logical_instances),
        }

    components_json = json.dumps(
        components, sort_keys=True, separators=(",", ":")
    )
    base_tags = {
        "quantum_component_count": len(components),
        "quantum_components_schema_version": 1,
        "quantum_components_sha256": hashlib.sha256(
            components_json.encode("utf-8")
        ).hexdigest(),
    }
    if not active:
        return {
            "tags": base_tags
            | {
                "qubits": 0,
                "ansatz": "none",
                "backend": "none",
                "device": "none",
                "quantum_configuration_status": "classical",
            },
            "components": components,
            "diagnostic_config": None,
            "diagnostic_unavailability": None,
        }

    first_config = active[0][1]
    first_implementation = active[0][3]
    homogeneous = all(
        qcfg == first_config and implementation == first_implementation
        for _name, qcfg, _logical_instances, implementation in active[1:]
    )
    if homogeneous:
        first_component = next(iter(components.values()))
        tags = base_tags | {
            "qubits": int(first_config.n_qubits),
            "ansatz": first_config.ansatz,
            "backend": first_component["backend"],
            "device": first_component["device"],
            "quantum_configuration_status": "single_configuration",
        }
        if first_implementation == "configured_circuit_backend":
            return {
                "tags": tags,
                "components": components,
                "diagnostic_config": first_config,
                "diagnostic_unavailability": None,
            }
        diagnostic_unavailability = {
            "status": "not_measured",
            "reason": "native_jax_component_diagnostics_unsupported",
            "semantics": "single_configuration",
            "components": components,
        }
        return {
            "tags": tags,
            "components": components,
            "diagnostic_config": None,
            "diagnostic_unavailability": diagnostic_unavailability,
        }

    diagnostic_unavailability = {
        "status": "not_measured",
        "reason": "mixed_quantum_configurations",
        "semantics": "mixed_configuration",
        "components": components,
    }
    return {
        "tags": base_tags
        | {
            "qubits": "mixed",
            "ansatz": "mixed",
            "backend": "mixed",
            "device": "mixed",
            "quantum_configuration_status": "mixed_configuration",
        },
        "components": components,
        "diagnostic_config": None,
        "diagnostic_unavailability": diagnostic_unavailability,
    }


def _collect_quantum_diagnostics(tracker, context):
    """Measure one honest configuration or return explicit unavailability."""
    qcfg = context["diagnostic_config"]
    if qcfg is not None:
        return log_quantum_diagnostics(tracker, qcfg)
    return context["diagnostic_unavailability"]


def make_train_step():
    @jax.jit
    def train_step(state: TrainState, batch: jnp.ndarray):
        inputs, targets = batch[:, :-1], batch[:, 1:]

        def loss_fn(params):
            logits = state.apply_fn({"params": params}, inputs)
            return optax.softmax_cross_entropy_with_integer_labels(
                logits, targets
            ).mean()

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss

    return train_step


def make_eval_step():
    @jax.jit
    def eval_step(state: TrainState, batch: jnp.ndarray):
        inputs, targets = batch[:, :-1], batch[:, 1:]
        logits = state.apply_fn({"params": state.params}, inputs)
        return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

    return eval_step


def make_grad_norm_step():
    """Per-group gradient norms: is the circuit actually receiving signal?

    The v0.2 trained~=frozen result demanded this diagnostic: it splits
    the gradient L2 norm into circuit_weights vs everything else, logged
    at every eval point so under-training of the quantum block is visible
    DURING runs, not post-hoc.
    """

    @jax.jit
    def grad_norm_step(state: TrainState, batch: jnp.ndarray):
        inputs, targets = batch[:, :-1], batch[:, 1:]

        def loss_fn(params):
            logits = state.apply_fn({"params": params}, inputs)
            return optax.softmax_cross_entropy_with_integer_labels(
                logits, targets
            ).mean()

        grads = jax.grad(loss_fn)(state.params)
        flat = traverse_util.flatten_dict(grads)
        circuit_sq = sum(
            jnp.sum(v**2) for k, v in flat.items() if "circuit_weights" in k
        )
        other_sq = sum(
            jnp.sum(v**2) for k, v in flat.items() if "circuit_weights" not in k
        )
        return jnp.sqrt(circuit_sq), jnp.sqrt(other_sq)

    return grad_norm_step


def evaluate(
    eval_step,
    state: TrainState,
    val_ids: np.ndarray,
    cfg: TrainConfig,
    seed_offset: int = 9999,
) -> dict[str, float]:
    """Average loss over a fixed (seeded) set of validation batches."""
    rng = np.random.default_rng(cfg.seed + seed_offset)
    losses = []
    for _ in range(cfg.eval_batches):
        batch = jnp.asarray(sample_batch(rng, val_ids, cfg.batch_size, cfg.seq_len))
        losses.append(eval_step(state, batch))
    loss = float(jnp.mean(jnp.stack(losses)))
    return {
        "val_loss": loss,
        "val_ppl": float(np.exp(loss)),
        "val_bpc": loss / float(np.log(2)),
    }


def count_params(params) -> int:
    return sum(int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(params))


def make_optimizer(train_cfg: TrainConfig, params, freeze_circuit: bool = False):
    """AdamW + grad clipping; optionally freeze quantum circuit weights.

    Freezing uses ``optax.multi_transform`` keyed on parameter paths: any
    leaf whose path contains ``circuit_weights`` receives zero updates, so
    a frozen circuit stays EXACTLY at its random initialization — the
    random-feature control arm of the 2x2 ablation.
    """
    base = optax.chain(
        optax.clip_by_global_norm(train_cfg.grad_clip),
        optax.adamw(train_cfg.lr, weight_decay=train_cfg.weight_decay),
    )
    if not freeze_circuit:
        return base
    flat = traverse_util.flatten_dict(params)
    labels = traverse_util.unflatten_dict(
        {k: ("frozen" if "circuit_weights" in k else "trainable") for k in flat}
    )
    return optax.multi_transform(
        {"trainable": base, "frozen": optax.set_to_zero()}, labels
    )


def _retry_manifest_identity(manifest: dict) -> dict:
    """Stable retry contract, excluding dataset access-path provenance only."""
    body = {
        key: value for key, value in manifest.items() if key != "manifest_hash"
    }
    data = dict(body.get("data") or {})
    data.pop("provenance", None)
    metadata = dict(data.get("metadata") or {})
    metadata.pop("provenance", None)
    data["metadata"] = metadata
    body["data"] = data
    return body


def fit(
    cfg: ExperimentConfig,
    verbose: bool = True,
    out_dir: str | Path = "results",
    init_params=None,
    should_cancel=None,
    run_options: RunOptions | None = None,
    progress_callback=None,
    publish_guard=None,
    primary_metric_type: str = DEFAULT_SEQUENCE_PRIMARY_METRIC_TYPE,
) -> dict:
    """Train on the requested JAX device without mutating global JAX config."""
    _require_sequence_task(cfg)
    _sequence_primary_metric_name(primary_metric_type)
    normalized_options = (run_options or RunOptions()).normalized()
    resolved_device = resolve_execution_device(normalized_options.device_target)
    with jax.default_device(resolved_device):
        return _fit_on_device(
            cfg,
            verbose=verbose,
            out_dir=out_dir,
            init_params=init_params,
            should_cancel=should_cancel,
            run_options=normalized_options,
            progress_callback=progress_callback,
            publish_guard=publish_guard,
            primary_metric_type=primary_metric_type,
            resolved_device=resolved_device,
        )


def _fit_on_device(
    cfg: ExperimentConfig,
    verbose: bool = True,
    out_dir: str | Path = "results",
    init_params=None,
    should_cancel=None,
    run_options: RunOptions | None = None,
    progress_callback=None,
    publish_guard=None,
    primary_metric_type: str = DEFAULT_SEQUENCE_PRIMARY_METRIC_TYPE,
    resolved_device=None,
) -> dict:
    """Train a model end to end from an ExperimentConfig.

    Returns dict with the final TrainState, model, tokenizer, and a JSON-able
    summary (written under a UUID-scoped artifact directory by default).
    """
    _require_sequence_task(cfg)
    primary_metric_name = _sequence_primary_metric_name(primary_metric_type)
    fit_started = time.perf_counter()
    if init_params is not None and run_options is not None and run_options.resume_from:
        raise ValueError("init_params and resume_from are mutually exclusive.")

    validation_errors = validate_config(cfg)
    if validation_errors:
        details = "\n- ".join(validation_errors)
        raise ValueError(f"Invalid experiment config:\n- {details}")

    dataset = load_dataset_bundle(cfg.data)
    tokenizer = dataset.tokenizer
    train_ids, val_ids = train_val_split(dataset.ids, cfg.data.val_fraction)

    runtime_cfg = dataclasses.replace(
        cfg,
        model=dataclasses.replace(cfg.model, vocab_size=tokenizer.vocab_size),
    )
    runtime_errors = validate_config(runtime_cfg)
    if runtime_errors:
        details = "\n- ".join(runtime_errors)
        raise ValueError(f"Invalid runtime experiment config:\n- {details}")

    model, model_cfg = build_model(
        runtime_cfg.model, vocab_size=tokenizer.vocab_size
    )

    raw_options = (run_options or RunOptions()).normalized()

    def require_publish_ownership() -> None:
        if publish_guard is not None and not bool(publish_guard()):
            raise RuntimeError("Run publication ownership was lost.")
    rng_np = np.random.default_rng(cfg.train.seed)
    init_key = jax.random.PRNGKey(cfg.train.seed)
    sample = jnp.asarray(
        sample_batch(rng_np, train_ids, cfg.train.batch_size, cfg.train.seq_len)
    )
    params = (
        init_params
        if init_params is not None
        else model.init(init_key, sample[:, :-1])["params"]
    )
    n_params = count_params(params)
    if resolved_device is None:  # private-call defensive fallback
        resolved_device = resolve_execution_device(raw_options.device_target)
    resource_plan = static_resource_plan(
        runtime_cfg,
        n_params=n_params,
        requested_device=raw_options.device_target,
        resolved_device=resolved_device,
    )

    freeze_circuit = uses_quantum(model_cfg) and not model_cfg.quantum.trainable
    tx = make_optimizer(cfg.train, params, freeze_circuit=freeze_circuit)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    run_name = cfg.tracking.run_name or (
        f"{model_cfg.attn_type}-attn_{model_cfg.ffn_type}-ffn"
    )
    source_payload = None
    source_manifest = None
    source_artifact_dir = None
    if raw_options.resume_from:
        source_path = Path(raw_options.resume_from).resolve()
        state, source_payload = restore_checkpoint(source_path, state)
        source_manifest = source_payload["manifest"]
        source_artifact_dir = (
            source_path.parent.parent
            if source_path.parent.name == "checkpoints"
            else source_path.parent
        )
        if raw_options.experiment_uuid is None:
            raw_options = dataclasses.replace(
                raw_options,
                experiment_uuid=source_manifest.get("experiment_uuid"),
            )
        if raw_options.run_uuid is None:
            raw_options = dataclasses.replace(
                raw_options,
                run_uuid=source_manifest.get("run_uuid"),
            )
        elif raw_options.run_uuid != source_manifest.get("run_uuid") and (
            raw_options.artifact_dir is None
        ):
            raise ValueError(
                "Forking a resumed checkpoint requires an explicit new "
                "run_uuid and artifact_dir."
            )
        recovering_source = raw_options.run_uuid == source_manifest.get("run_uuid")
        if recovering_source:
            source_run_name = str(source_manifest.get("run_name") or "")
            if source_run_name and run_name != source_run_name:
                raise ValueError(
                    "Recovery must retain the checkpoint run_name identity."
                )
            if raw_options.artifact_dir is not None and (
                Path(raw_options.artifact_dir).resolve()
                != source_artifact_dir.resolve()
            ):
                raise ValueError(
                    "Recovery must retain the checkpoint artifact directory; "
                    "use a new run_uuid for an explicit fork."
                )
            latest_path = source_artifact_dir / "checkpoints" / "latest.msgpack"
            if latest_path.resolve() != source_path and latest_path.is_file():
                try:
                    latest_payload = read_checkpoint(latest_path)
                except ValueError:
                    latest_payload = None
                if latest_payload is not None:
                    latest_manifest = latest_payload["manifest"]
                    if (
                        latest_manifest.get("run_uuid")
                        != source_manifest.get("run_uuid")
                        or latest_manifest.get("experiment_uuid")
                        != source_manifest.get("experiment_uuid")
                    ):
                        raise ValueError(
                            "The artifact directory latest checkpoint belongs "
                            "to a different immutable run."
                        )
                    if int(latest_payload["completed_step"]) > int(
                        source_payload["completed_step"]
                    ):
                        raise ValueError(
                            "Refusing to roll back the same run from an older "
                            "checkpoint while a newer latest checkpoint exists; "
                            "use a new run_uuid and artifact_dir to fork it."
                        )
        if raw_options.artifact_dir is None:
            raw_options = dataclasses.replace(
                raw_options, artifact_dir=str(source_artifact_dir)
            )
    options = resolve_run_options(raw_options)
    artifact_dir = (
        Path(raw_options.artifact_dir)
        if raw_options.artifact_dir is not None
        else Path(out_dir) / "runs" / str(options.run_uuid)
    )
    if artifact_dir.exists() and not artifact_dir.is_dir():
        raise ValueError(f"Artifact path is not a directory: {artifact_dir}")
    preexisting_artifacts = artifact_dir.exists() and any(artifact_dir.iterdir())
    recovering_source = bool(
        source_manifest is not None
        and raw_options.run_uuid == source_manifest.get("run_uuid")
    )
    incomplete_identity_retry = False
    retry_manifest = None
    if (
        not recovering_source
        and options.run_uuid is not None
        and preexisting_artifacts
    ):
        entries = {entry.name for entry in artifact_dir.iterdir()}
        checkpoint_entries = (
            list((artifact_dir / "checkpoints").iterdir())
            if (artifact_dir / "checkpoints").is_dir()
            else []
        )
        partial_checkpoint_temps_only = all(
            entry.is_file()
            and entry.name.endswith(".tmp")
            and entry.name.startswith((".latest.msgpack.", ".best.msgpack."))
            for entry in checkpoint_entries
        )
        manifest_only = (
            "manifest.json" in entries
            and entries <= {"manifest.json", "checkpoints"}
            and (not checkpoint_entries or partial_checkpoint_temps_only)
        )
        if manifest_only:
            try:
                retry_manifest = validate_manifest(
                    json.loads((artifact_dir / "manifest.json").read_text("utf-8"))
                )
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"Existing incomplete run manifest is invalid: {exc}"
                ) from exc
            incomplete_identity_retry = bool(
                retry_manifest.get("run_uuid") == options.run_uuid
                and retry_manifest.get("experiment_uuid")
                == options.experiment_uuid
                and retry_manifest.get("run_name") == run_name
            )
    if not recovering_source and preexisting_artifacts and not incomplete_identity_retry:
        raise ValueError(
            "Refusing to overwrite a non-empty artifact directory for a new run: "
            f"{artifact_dir}. Use a unique run name/artifact_dir, or resume its "
            "checkpoint explicitly."
        )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = artifact_dir / "checkpoints"
    latest_checkpoint = checkpoint_dir / "latest.msgpack"
    best_checkpoint = checkpoint_dir / "best.msgpack"

    seed_axes = normalize_seed_axes(
        cfg.train.seed,
        generator_seed=cfg.data.gen_seed,
        data_kind=cfg.data.kind,
        circuit_applicable=uses_quantum(model_cfg),
        explicit=raw_options.seed_axes,
        reject_unsupported=True,
    )
    if source_manifest is not None:
        seed_axes = dict(source_manifest.get("seed_axes") or seed_axes)
        initialization = dict(source_manifest.get("initialization") or {})
    else:
        parameter_hash = hashlib.sha256(
            serialization.to_bytes(params)
        ).hexdigest()
        if init_params is not None:
            initialization = {
                "mode": "warm_start",
                "parameters_sha256": parameter_hash,
                "source": str(
                    raw_options.caller_metadata.get(
                        "warm_start_source", "external_init_params"
                    )
                ),
                "train_seed_controls_parameters": False,
            }
            sources = dict(seed_axes.get("sources") or {})
            sources["initialization"] = "warm_start_params"
            if uses_quantum(model_cfg):
                sources["circuit"] = "warm_start_params"
            seed_axes = {
                **seed_axes,
                "initialization": None,
                "circuit": None if uses_quantum(model_cfg) else seed_axes.get("circuit"),
                "sources": sources,
                "coupled_axes": [
                    axis
                    for axis in seed_axes.get("coupled_axes", [])
                    if axis not in {"initialization", "circuit"}
                ],
                "warm_start_parameters_sha256": parameter_hash,
            }
        else:
            initialization = {
                "mode": "seeded_random",
                "parameters_sha256": parameter_hash,
                "source": "train.seed",
                "seed": int(cfg.train.seed),
                "train_seed_controls_parameters": True,
            }
    retry_lineage = dict((retry_manifest or {}).get("resume_lineage") or {})
    resume_lineage: dict = {}
    if source_payload is not None:
        resume_event_uuid = retry_lineage.get("resume_event_uuid")
        if retry_manifest is not None and not resume_event_uuid:
            raise ValueError(
                "Existing incomplete fork manifest lacks resume event identity."
            )
        resume_lineage = {
            "source_checkpoint": str(Path(raw_options.resume_from).resolve()),
            "source_checkpoint_sha256": hashlib.sha256(
                Path(raw_options.resume_from).read_bytes()
            ).hexdigest(),
            "source_run_uuid": source_manifest.get("run_uuid"),
            "source_manifest_hash": source_manifest.get("manifest_hash"),
            "source_completed_step": int(source_payload["completed_step"]),
            "mode": (
                "recovery"
                if options.run_uuid == source_manifest.get("run_uuid")
                else "fork"
            ),
            "resume_event_uuid": str(resume_event_uuid or uuid.uuid4()),
        }
        if options.parent_run_uuid is not None:
            resume_lineage["parent_run_uuid"] = options.parent_run_uuid
        elif resume_lineage["mode"] == "fork":
            resume_lineage["parent_run_uuid"] = source_manifest.get("run_uuid")
    elif retry_manifest is not None:
        resume_lineage = retry_lineage
    current_manifest = build_run_manifest(
        runtime_cfg,
        dataset,
        options,
        run_name=run_name,
        seed_axes=seed_axes,
        resume_lineage=resume_lineage,
        initialization=initialization,
        resource_plan=resource_plan,
    )
    if source_manifest is not None:
        validate_checkpoint_manifest(source_manifest, current_manifest)
        if resume_lineage.get("mode") == "recovery":
            source_execution = (source_manifest.get("resource_plan") or {}).get(
                "execution_device"
            )
            current_execution = (current_manifest.get("resource_plan") or {}).get(
                "execution_device"
            )
            if (
                source_execution
                and current_execution
                and source_execution != current_execution
            ):
                raise ValueError(
                    "Exact recovery must retain resource_plan.execution_device; "
                    "use a new run_uuid and artifact_dir for a cross-device fork."
                )
    if retry_manifest is not None:
        if _retry_manifest_identity(retry_manifest) != _retry_manifest_identity(
            current_manifest
        ):
            raise ValueError(
                "Existing incomplete run manifest differs from the retried "
                "configuration, source checkpoint, or environment."
            )
        manifest = retry_manifest
    elif source_manifest is not None:
        if resume_lineage["mode"] == "recovery":
            if options.experiment_uuid != source_manifest.get("experiment_uuid"):
                raise ValueError(
                    "Recovery must retain the checkpoint experiment_uuid."
                )
            manifest = source_manifest
        else:
            manifest = current_manifest
    else:
        manifest = current_manifest
    require_publish_ownership()
    write_immutable_manifest(artifact_dir / "manifest.json", manifest)

    history: list[dict] = []
    completed_step = 0
    best_metric: float | None = None
    best_step: int | None = None
    if source_payload is not None:
        completed_step = int(source_payload["completed_step"])
        state_step = int(np.asarray(state.step))
        if state_step != completed_step:
            raise ValueError(
                "Checkpoint completed_step does not match TrainState.step: "
                f"{completed_step} != {state_step}."
            )
        if completed_step < 0 or completed_step > cfg.train.steps:
            raise ValueError(
                "Checkpoint completed_step is outside the configured run: "
                f"{completed_step} not in [0, {cfg.train.steps}]."
            )
        rng_np.bit_generator.state = source_payload["rng_state"]
        history = list(source_payload["history"])
        raw_best = source_payload.get("best_metric")
        best_metric = float(raw_best) if raw_best is not None else None
        raw_best_step = source_payload.get("best_step")
        best_step = int(raw_best_step) if raw_best_step is not None else None

    # own-dashboard per-step logger (replaces MLflow when configured)
    dash = None
    dash_key = None
    dash_config = to_flat_dict(cfg)
    if model_cfg.arch == "two_stream":
        from ..research_protocol import TWO_STREAM_CAUSAL_PROTOCOL

        dash_config["research.two_stream_protocol"] = TWO_STREAM_CAUSAL_PROTOCOL
    if cfg.tracking.dashboard_db:
        from ..resultsdb import ResultsDB

        dash = ResultsDB(cfg.tracking.dashboard_db)
        dash_key = (
            f"{cfg.tracking.dashboard_suite}/{cfg.tracking.dashboard_variant}/"
            f"{cfg.tracking.dashboard_dataset}/{cfg.tracking.dashboard_seed}/"
            f"{cfg.train.steps}"
        )
        require_publish_ownership()
        dash.start_run(
            run_key=dash_key, run_name=run_name,
            suite=cfg.tracking.dashboard_suite or "adhoc",
            variant=cfg.tracking.dashboard_variant or run_name,
            dataset=cfg.tracking.dashboard_dataset or cfg.data.kind,
            seed=cfg.tracking.dashboard_seed
            if cfg.tracking.dashboard_seed is not None else cfg.train.seed,
            total_steps=cfg.train.steps, config=dash_config,
            run_uuid=options.run_uuid,
            experiment_uuid=options.experiment_uuid,
            manifest=manifest,
            primary_metric_type=primary_metric_type)

    tracker = ExperimentTracker(cfg.tracking)
    tracker.log_params(
        to_flat_dict(cfg)
        | {
            "n_params": n_params,
            "vocab_size": tokenizer.vocab_size,
            "experiment_uuid": options.experiment_uuid,
            "run_uuid": options.run_uuid,
            "manifest_hash": manifest["manifest_hash"],
        }
    )
    quantum_tracking = _quantum_tracking_context(model_cfg)
    tracker.set_tags(quantum_tracking["tags"])

    if verbose:
        print(
            f"[{run_name}] vocab={tokenizer.vocab_size} params={n_params:,} "
            f"attn={model_cfg.attn_type} ffn={model_cfg.ffn_type}"
            + (" [circuit FROZEN]" if freeze_circuit else "")
        )

    diagnostics = None
    if uses_quantum(model_cfg) and cfg.tracking.log_quantum_diagnostics:
        diagnostics = _collect_quantum_diagnostics(tracker, quantum_tracking)
        if verbose:
            pretty = _format_quantum_diagnostics_for_display(diagnostics)
            print(f"[{run_name}] quantum diagnostics: {pretty}")

    train_step = make_train_step()
    eval_step = make_eval_step()
    grad_norm_step = (
        make_grad_norm_step()
        if uses_quantum(model_cfg) and cfg.tracking.log_grad_norms
        else None
    )

    attempt_started = time.perf_counter()
    cancelled = False
    cadence = raw_options.checkpoint_every or cfg.train.eval_every
    first_step_seconds: float | None = None
    steady_step_seconds: list[float] = []
    evaluation_forward_instances = 0

    def save_checkpoint(path: Path) -> None:
        require_publish_ownership()
        write_checkpoint(
            path,
            state,
            completed_step=completed_step,
            rng_state=rng_np.bit_generator.state,
            history=history,
            best_metric=best_metric,
            best_step=best_step,
            manifest=manifest,
            resume_lineage=resume_lineage,
        )

    def notify_progress() -> None:
        if progress_callback is None:
            return
        progress_callback({
            "completed_step": completed_step,
            "checkpoint_path": str(latest_checkpoint.resolve()),
            "best_checkpoint_path": (
                str(best_checkpoint.resolve()) if best_checkpoint.exists() else None
            ),
            "artifact_dir": str(artifact_dir.resolve()),
            "manifest": manifest,
        })

    # Establish a resumable step-zero/restored checkpoint before the first JIT
    # or training batch. This closes the restart gap between manifest creation
    # and the first periodic checkpoint, and lets the queue persist the exact
    # artifact location immediately.
    save_checkpoint(latest_checkpoint)
    notify_progress()

    loop_started = time.perf_counter()
    attempt_start_step = completed_step
    for step in range(completed_step + 1, cfg.train.steps + 1):
        if should_cancel is not None and should_cancel():
            cancelled = True
            break
        batch = jnp.asarray(
            sample_batch(rng_np, train_ids, cfg.train.batch_size, cfg.train.seq_len)
        )
        step_started = time.perf_counter()
        state, loss = train_step(state, batch)
        jax.block_until_ready(loss)
        step_seconds = time.perf_counter() - step_started
        if first_step_seconds is None:
            first_step_seconds = step_seconds
        else:
            steady_step_seconds.append(step_seconds)
        completed_step = step

        if completed_step == attempt_start_step + 1 and verbose:
            print(f"[{run_name}] first executed step (incl. jit) {step_seconds:.1f}s")
        if step % 10 == 0 or step == 1:
            tracker.log_metrics({"train_loss": float(loss)}, step=step)
            if dash is not None:
                require_publish_ownership()
                dash.log_step(dash_key, step, {"train_loss": float(loss)},
                              train_loss=float(loss), run_uuid=options.run_uuid)
            if verbose and step % 50 == 0:
                print(f"[{run_name}] step {step:5d}  train_loss {float(loss):.4f}")

        evaluated = step % cfg.train.eval_every == 0 or step == cfg.train.steps
        if evaluated:
            ev = evaluate(eval_step, state, val_ids, cfg.train)
            per_train_step = int(
                resource_plan["logical_circuit_forward_instances_per_train_step"]["value"]
            )
            evaluation_forward_instances += per_train_step * int(cfg.train.eval_batches)
            if grad_norm_step is not None:
                g_circ, g_other = grad_norm_step(state, batch)
                jax.block_until_ready(g_other)
                evaluation_forward_instances += per_train_step
                ev["grad_norm_circuit"] = float(g_circ)
                ev["grad_norm_classical"] = float(g_other)
                ev["grad_norm_ratio"] = float(g_circ) / (float(g_other) + 1e-12)
            tracker.log_metrics(ev, step=step)
            if dash is not None:
                require_publish_ownership()
                dash.log_step(dash_key, step,
                              {k: float(v) for k, v in ev.items()},
                              val_ppl=float(ev.get("val_ppl", 0)) or None,
                              run_uuid=options.run_uuid)
            history.append({"step": step, **ev})
            if best_metric is None or float(ev["val_loss"]) < best_metric:
                best_metric = float(ev["val_loss"])
                best_step = step
                save_checkpoint(best_checkpoint)
            if verbose:
                extra = (
                    f"  g_circ/g_cls {ev['grad_norm_ratio']:.3e}"
                    if "grad_norm_ratio" in ev
                    else ""
                )
                print(
                    f"[{run_name}] step {step:5d}  val_loss {ev['val_loss']:.4f}  "
                    f"val_ppl {ev['val_ppl']:.2f}  val_bpc {ev['val_bpc']:.3f}"
                    + extra
                )

        if step % cadence == 0 or evaluated or step == cfg.train.steps:
            save_checkpoint(latest_checkpoint)
        notify_progress()

    # Cancellation can arrive between periodic checkpoints.  Always leave a
    # complete latest checkpoint for deterministic recovery, including step 0.
    if cancelled or not latest_checkpoint.exists():
        save_checkpoint(latest_checkpoint)

    loop_wall = time.perf_counter() - loop_started
    wall = time.perf_counter() - attempt_started
    fit_wall = time.perf_counter() - fit_started
    resources = runtime_resource_ledger(
        resource_plan,
        params=state.params,
        completed_steps_this_attempt=completed_step - attempt_start_step,
        evaluation_forward_instances=evaluation_forward_instances,
        first_step_seconds=first_step_seconds,
        steady_step_seconds=steady_step_seconds,
        loop_wall_seconds=loop_wall,
        attempt_wall_seconds=wall,
        fit_wall_seconds=fit_wall,
        device=resolved_device,
    )
    final = history[-1] if history else {}
    if dash is not None:
        if final and not cancelled:
            require_publish_ownership()
            dash.record(
                suite=cfg.tracking.dashboard_suite or "adhoc",
                variant=cfg.tracking.dashboard_variant or run_name,
                dataset=cfg.tracking.dashboard_dataset or cfg.data.kind,
                seed=cfg.tracking.dashboard_seed
                if cfg.tracking.dashboard_seed is not None else cfg.train.seed,
                steps=cfg.train.steps,
                n_params=n_params,
                val_loss=float(final.get("val_loss", 0.0)),
                val_ppl=float(final.get("val_ppl", 0.0)),
                val_bpc=float(final.get("val_bpc", 0.0)),
                wall_seconds=wall,
                config=dash_config,
                run_uuid=options.run_uuid,
                experiment_uuid=options.experiment_uuid,
                manifest_hash=manifest["manifest_hash"],
                manifest=manifest,
                finalize_manifest=False,
                resources=resources,
                primary_metric_type=primary_metric_type,
                metric_values=final,
            )
        if not options.defer_dashboard_terminal:
            dash.finish_run(
                dash_key,
                status="cancelled" if cancelled else "done",
                run_uuid=options.run_uuid,
            )

    out = artifact_dir
    if not cancelled:
        require_publish_ownership()
        atomic_write_bytes(
            out / "params.msgpack", serialization.to_bytes(state.params)
        )
    summary = {
        "run_name": run_name,
        "artifact_dir": str(out.resolve()),
        "experiment_uuid": options.experiment_uuid,
        "run_uuid": options.run_uuid,
        "manifest_hash": manifest["manifest_hash"],
        "n_params": n_params,
        "vocab_size": tokenizer.vocab_size,
        "wall_seconds": round(wall, 2),
        "resources": resources,
        "steps": cfg.train.steps,
        "completed_step": completed_step,
        "cancelled": cancelled,
        "resumed": source_payload is not None,
        "resumed_from": raw_options.resume_from,
        "resume_lineage": resume_lineage,
        "checkpoint_path": str(latest_checkpoint.resolve()),
        "best_checkpoint_path": (
            str(best_checkpoint.resolve()) if best_checkpoint.exists() else None
        ),
        "best_step": best_step,
        "primary_metric_type": primary_metric_type,
        "primary_metric_name": primary_metric_name,
        "primary_metric_value": final.get(primary_metric_name),
        "quantum_diagnostics": diagnostics,
        "history": history,
        **final,
    }
    require_publish_ownership()
    atomic_write_json(out / "summary.json", summary)

    tracker.log_metrics({"wall_seconds": wall})
    tracker.log_artifact(out / "summary.json")
    tracker.end()

    return {
        "state": state,
        "model": model,
        "model_cfg": model_cfg,
        "dataset": dataset,
        "tokenizer": tokenizer,
        "summary": summary,
        "manifest": manifest,
        "run_options": options,
    }


def generation_capability(
    model_cfg,
    tokenizer,
    *,
    prompt: str = "\n",
    max_new_tokens: int = 200,
    temperature: float = 0.8,
) -> dict:
    """Return an explicit architecture/tokenizer/context generation contract."""
    arch = str(getattr(model_cfg, "arch", "unknown"))
    capacity = int(getattr(model_cfg, "max_seq_len", 0) or 0)
    if arch == "two_stream":
        positions_per_token = two_stream_position_count(
            1, model_cfg.encoder_kind, model_cfg.condition
        )
        capacity //= positions_per_token
    reason = None
    try:
        token_count = int(max_new_tokens)
    except (TypeError, ValueError, OverflowError):
        token_count = None
    try:
        numeric_temperature = float(temperature)
    except (TypeError, ValueError, OverflowError):
        numeric_temperature = None
    if not isinstance(tokenizer, CharTokenizer):
        reason = "character generation requires a CharTokenizer"
    elif not hasattr(tokenizer, "stoi") or not hasattr(tokenizer, "decode"):
        reason = "tokenizer does not expose stoi/decode"
    elif capacity < 1:
        reason = "model has no usable real-token context capacity"
    elif int(getattr(model_cfg, "vocab_size", 0) or 0) != tokenizer.vocab_size:
        reason = "model/tokenizer vocabulary sizes do not match"
    elif token_count is None or token_count < 1:
        reason = "max_new_tokens must be a positive integer"
    elif (
        numeric_temperature is None
        or not math.isfinite(numeric_temperature)
        or numeric_temperature <= 0
    ):
        reason = "temperature must be finite and positive"
    elif not isinstance(prompt, str):
        reason = "prompt must be a string"
    return {
        "supported": reason is None,
        "status": "supported" if reason is None else "unsupported",
        "reason": reason,
        "architecture": arch,
        "context_capacity": capacity,
        "settings": {
            "max_new_tokens": token_count,
            "temperature": numeric_temperature,
            "vocab_size": getattr(tokenizer, "vocab_size", None),
        },
    }


def generate_outcome(
    model,
    params,
    tokenizer,
    *,
    model_cfg,
    prompt: str = "\n",
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    seed: int = 0,
) -> dict:
    """Architecture-neutral generation with explicit unsupported outcomes."""
    capability = generation_capability(
        model_cfg,
        tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    try:
        numeric_seed = int(seed)
    except (TypeError, ValueError, OverflowError):
        numeric_seed = None
        capability = {
            **capability,
            "supported": False,
            "status": "unsupported",
            "reason": "seed must be an integer",
        }
    base = {
        "ok": False,
        "supported": capability["supported"],
        "status": capability["status"],
        "kind": "prompt_generation",
        "reason": capability["reason"],
        "generated_text": None,
        "architecture": capability["architecture"],
        "context_capacity": capability["context_capacity"],
        "settings": {**capability["settings"], "seed": numeric_seed},
        "capability": capability,
    }
    if not capability["supported"]:
        return base

    context_len = capability["context_capacity"]
    ids = [tokenizer.stoi[c] for c in prompt if c in tokenizer.stoi] or [0]
    max_new_tokens = int(capability["settings"]["max_new_tokens"])
    temperature = float(capability["settings"]["temperature"])
    key = jax.random.PRNGKey(numeric_seed)

    @jax.jit
    def step_fn(window, t_index, sample_key):
        logits = model.apply({"params": params}, window[None])[0]
        sample_key, sub = jax.random.split(sample_key)
        next_id = jax.random.categorical(
            sub, logits[t_index] / float(temperature)
        )
        return next_id, sample_key

    for _ in range(max_new_tokens):
        window = ids[-context_len:]
        padded = np.zeros(context_len, dtype=np.int32)
        padded[: len(window)] = window
        next_id, key = step_fn(
            jnp.asarray(padded), jnp.asarray(len(window) - 1), key
        )
        ids.append(int(next_id))

    return {
        **base,
        "ok": True,
        "supported": True,
        "status": "supported",
        "reason": None,
        "generated_text": tokenizer.decode(ids),
    }


def generate(
    model,
    params,
    tokenizer: CharTokenizer,
    prompt: str = "\n",
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    seed: int = 0,
    model_cfg=None,
) -> str:
    """String compatibility wrapper over :func:`generate_outcome`."""
    finalized = model_cfg or getattr(model, "cfg", None)
    if finalized is None:
        raise ValueError("generate requires the finalized ModelConfig.")
    outcome = generate_outcome(
        model,
        params,
        tokenizer,
        model_cfg=finalized,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        seed=seed,
    )
    if not outcome["ok"]:
        raise ValueError(outcome["reason"] or "generation is unsupported")
    return str(outcome["generated_text"])
