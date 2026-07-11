"""Immutable run identity and portable, atomic training checkpoints.

This module deliberately keeps operational run controls out of
``ExperimentConfig``.  Scientific configuration remains hashable and stable,
while UUIDs, artifact locations, checkpoint cadence, and caller metadata can
vary between executions without changing the model/data protocol.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from flax import serialization


CHECKPOINT_VERSION = 2
MANIFEST_VERSION = 2
_NONFINITE_FLOAT = "__qllm_nonfinite_float__"


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def canonical_json(value: Any) -> str:
    """Canonical JSON used by every run identity hash."""
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _operational_jsonable(value: Any) -> Any:
    """JSON-safe runtime state that explicitly preserves non-finite floats."""
    if dataclasses.is_dataclass(value):
        return _operational_jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _operational_jsonable(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_operational_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_operational_jsonable(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return _operational_jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            label = "nan"
        elif value > 0:
            label = "positive_infinity"
        else:
            label = "negative_infinity"
        return {_NONFINITE_FLOAT: label}
    return value


def _restore_operational_json(value: Any) -> Any:
    if isinstance(value, list):
        return [_restore_operational_json(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {_NONFINITE_FLOAT}:
            label = value[_NONFINITE_FLOAT]
            if label == "nan":
                return float("nan")
            if label == "positive_infinity":
                return float("inf")
            if label == "negative_infinity":
                return float("-inf")
            raise ValueError(f"Unknown non-finite float label {label!r}.")
        return {
            key: _restore_operational_json(item) for key, item in value.items()
        }
    return value


def operational_json(value: Any) -> str:
    return json.dumps(
        _operational_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _valid_uuid(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError(f"{label} must be a valid UUID; got {value!r}.") from exc


@dataclass(frozen=True)
class RunOptions:
    """Operational execution controls shared by CLI and dashboard callers."""

    experiment_uuid: str | None = None
    run_uuid: str | None = None
    resume_from: str | Path | None = None
    checkpoint_every: int = 0
    artifact_dir: str | Path | None = None
    caller_metadata: Mapping[str, Any] = field(default_factory=dict)
    parent_run_uuid: str | None = None
    seed_axes: Mapping[str, Any] | None = None
    defer_dashboard_terminal: bool = False

    def normalized(self) -> "RunOptions":
        cadence = int(self.checkpoint_every)
        if cadence < 0:
            raise ValueError("checkpoint_every must be non-negative.")
        return dataclasses.replace(
            self,
            experiment_uuid=_valid_uuid(self.experiment_uuid, "experiment_uuid"),
            run_uuid=_valid_uuid(self.run_uuid, "run_uuid"),
            parent_run_uuid=_valid_uuid(self.parent_run_uuid, "parent_run_uuid"),
            resume_from=(
                str(Path(self.resume_from).resolve())
                if self.resume_from is not None
                else None
            ),
            artifact_dir=(
                str(Path(self.artifact_dir))
                if self.artifact_dir is not None
                else None
            ),
            checkpoint_every=cadence,
            caller_metadata=_jsonable(dict(self.caller_metadata)),
            seed_axes=(
                _jsonable(dict(self.seed_axes))
                if self.seed_axes is not None
                else None
            ),
        )


def resolve_run_options(options: RunOptions | None = None) -> RunOptions:
    normalized = (options or RunOptions()).normalized()
    return dataclasses.replace(
        normalized,
        experiment_uuid=normalized.experiment_uuid or str(uuid.uuid4()),
        run_uuid=normalized.run_uuid or str(uuid.uuid4()),
    )


def code_identity(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Identify HEAD, tracked changes, and untracked runtime source files."""
    root = Path(repo_root or Path(__file__).resolve().parents[2])
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "diff", "--binary", "--no-ext-diff", "HEAD", "--", "."],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=30,
        ).stdout
        untracked_raw = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        source_digest = hashlib.sha256()
        source_count = 0
        source_prefixes = ("qllm/", "scripts/", "benchmarks/", "configs/")
        source_suffixes = (".py", ".yaml", ".yml", ".toml", ".json")
        resolved_root = root.resolve()
        for relative in sorted(line.strip() for line in untracked_raw.splitlines()):
            normalized = relative.replace("\\", "/")
            if not normalized.startswith(source_prefixes):
                continue
            if not normalized.lower().endswith(source_suffixes):
                continue
            candidate = (root / relative).resolve()
            try:
                candidate.relative_to(resolved_root)
            except ValueError:
                continue
            if not candidate.is_file():
                continue
            source_digest.update(normalized.encode("utf-8"))
            source_digest.update(b"\0")
            source_digest.update(candidate.read_bytes())
            source_digest.update(b"\0")
            source_count += 1
        payload = {
            "status": "dirty" if diff or source_count else "clean",
            "commit": commit,
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "untracked_source_sha256": source_digest.hexdigest(),
            "untracked_source_count": source_count,
        }
    except (OSError, subprocess.SubprocessError) as exc:
        payload = {
            "status": "unavailable",
            "commit": None,
            "tracked_diff_sha256": None,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return {**payload, "hash": sha256_json(payload)}


def environment_identity() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("qllm", "jax", "jaxlib", "flax", "optax", "numpy", "pennylane"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    payload = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "byteorder": sys.byteorder,
        "packages": packages,
    }
    return {**payload, "hash": sha256_json(payload)}


def _resume_config_payload(cfg: Any) -> dict[str, Any]:
    """Scientific/training state that must match for an exact continuation.

    Tracking destinations, display names, and the diagnostics-at-start toggle
    are operational. They remain in the immutable full config hash, but do not
    change parameters, optimizer state, sampler state, or per-step metrics.
    Gradient logging is retained because it changes the evaluation history
    carried by the checkpoint.
    """
    if isinstance(cfg, Mapping):
        tracking = cfg.get("tracking") or {}
        tracking_grad_norms = (
            tracking.get("log_grad_norms")
            if isinstance(tracking, Mapping)
            else None
        )
        return {
            "model": _jsonable(cfg.get("model")),
            "train": _jsonable(cfg.get("train")),
            "data": _jsonable(cfg.get("data")),
            "tracking_evidence": {"log_grad_norms": tracking_grad_norms},
        }
    tracking = getattr(cfg, "tracking", None)
    return {
        "model": _jsonable(getattr(cfg, "model", None)),
        "train": _jsonable(getattr(cfg, "train", None)),
        "data": _jsonable(getattr(cfg, "data", None)),
        "tracking_evidence": {
            "log_grad_norms": getattr(tracking, "log_grad_norms", None),
        },
    }


def _seed_axes_identity(seed_axes: Mapping[str, Any]) -> dict[str, Any]:
    """Effective random axes, excluding descriptive normalization metadata."""
    names = (
        "generator",
        "split",
        "initialization",
        "minibatch",
        "circuit",
        "hardware_calibration",
    )
    return {
        "legacy_seed": _jsonable(seed_axes.get("legacy_seed")),
        **{name: _jsonable(seed_axes.get(name)) for name in names},
    }


def build_run_manifest(
    cfg,
    dataset,
    options: RunOptions,
    *,
    run_name: str,
    seed_axes: Mapping[str, Any],
    repo_root: str | Path | None = None,
    resume_lineage: Mapping[str, Any] | None = None,
    initialization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a canonical identity document for one logical run."""
    options = resolve_run_options(options)
    config_payload = _jsonable(cfg)
    code = code_identity(repo_root)
    environment = environment_identity()
    data = {
        "config_hash": dataset.config_hash,
        "content_hash": dataset.content_hash,
        "shape": list(dataset.shape),
        "sampler_policy": dataset.sampler_policy,
        "provenance": _jsonable(dataset.provenance),
        "metadata": _jsonable(dataset.metadata),
    }
    stable_data_identity = {
        "config_hash": dataset.config_hash,
        "content_hash": dataset.content_hash,
        "shape": list(dataset.shape),
        "sampler_policy": dataset.sampler_policy,
    }
    seed_axes_payload = _jsonable(seed_axes)
    initialization_payload = _jsonable(initialization or {})
    identity = {
        "schema_version": MANIFEST_VERSION,
        "experiment_uuid": options.experiment_uuid,
        "run_uuid": options.run_uuid,
        "run_name": run_name,
        "config_hash": sha256_json(config_payload),
        "resume_compatibility_hash": sha256_json(_resume_config_payload(cfg)),
        "code_hash": code["hash"],
        "data_hash": sha256_json(stable_data_identity),
        "environment_hash": environment["hash"],
        "seed_axes_hash": sha256_json(_seed_axes_identity(seed_axes_payload)),
        "initialization_hash": sha256_json(initialization_payload),
    }
    manifest = {
        **identity,
        "config": config_payload,
        "code": code,
        "data": data,
        "environment": environment,
        "seed_axes": seed_axes_payload,
        "initialization": initialization_payload,
        "caller_metadata": _jsonable(options.caller_metadata),
        "resume_lineage": _jsonable(resume_lineage or {}),
    }
    manifest["manifest_hash"] = sha256_json(manifest)
    return manifest


def build_record_manifest(
    *,
    suite: str,
    variant: str,
    dataset: str,
    seed: int,
    steps: int,
    config: Mapping[str, Any] | None = None,
    experiment_uuid: str | None = None,
    run_uuid: str | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build an honest identity for direct/non-``fit`` result records.

    Direct probes often expose only a dataset label rather than token content.
    The manifest records that limitation explicitly instead of fabricating a
    content hash; the resulting data hash identifies the available label-level
    evidence and its unavailable-content status.
    """
    experiment_uuid = _valid_uuid(experiment_uuid, "experiment_uuid") or str(
        uuid.uuid4()
    )
    run_uuid = _valid_uuid(run_uuid, "run_uuid") or str(uuid.uuid4())
    config_payload = _jsonable(dict(config or {}))
    code = code_identity(repo_root)
    environment = environment_identity()
    data = {
        "identity_status": "label_only",
        "dataset": str(dataset),
        "content_hash": None,
        "reason": "direct result recorder did not receive a DatasetBundle",
    }
    seed_axes = {
        "legacy_seed": int(seed),
        "generator": None,
        "split": None,
        "initialization": None,
        "minibatch": None,
        "circuit": None,
        "hardware_calibration": None,
        "assessment_status": "record_only_unresolved",
    }
    initialization = {
        "mode": "direct_measurement",
        "parameters_hash": None,
        "source": "caller_unavailable",
    }
    manifest = {
        "schema_version": MANIFEST_VERSION,
        "experiment_uuid": experiment_uuid,
        "run_uuid": run_uuid,
        "run_name": f"{suite}/{variant}/{dataset}/s{seed}/{steps}",
        "config_hash": sha256_json(config_payload),
        "resume_compatibility_hash": sha256_json(
            _resume_config_payload(config_payload)
        ),
        "code_hash": code["hash"],
        "data_hash": sha256_json(data),
        "environment_hash": environment["hash"],
        "seed_axes_hash": sha256_json(_seed_axes_identity(seed_axes)),
        "initialization_hash": sha256_json(initialization),
        "config": config_payload,
        "code": code,
        "data": data,
        "environment": environment,
        "seed_axes": seed_axes,
        "initialization": initialization,
        "caller_metadata": {"source": "ResultsDB.record"},
        "resume_lineage": {},
    }
    manifest["manifest_hash"] = sha256_json(manifest)
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate immutable identity fields and every recomputable hash."""
    if not isinstance(manifest, Mapping):
        raise ValueError("Run manifest must be a mapping.")
    normalized = _jsonable(dict(manifest))
    try:
        schema_version = int(normalized.get("schema_version", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("Run manifest schema_version is invalid.") from exc
    if schema_version < 1 or schema_version > MANIFEST_VERSION:
        raise ValueError(f"Unsupported run manifest schema_version {schema_version}.")
    if _valid_uuid(normalized.get("experiment_uuid"), "experiment_uuid") is None:
        raise ValueError("Run manifest experiment_uuid is missing.")
    if _valid_uuid(normalized.get("run_uuid"), "run_uuid") is None:
        raise ValueError("Run manifest run_uuid is missing.")

    claimed_manifest_hash = str(normalized.get("manifest_hash") or "")
    body = dict(normalized)
    body.pop("manifest_hash", None)
    if claimed_manifest_hash != sha256_json(body):
        raise ValueError("Run manifest_hash does not match its canonical payload.")

    config = normalized.get("config")
    if normalized.get("config_hash") != sha256_json(config):
        raise ValueError("Run manifest config_hash is inconsistent.")
    if schema_version >= 2 and normalized.get(
        "resume_compatibility_hash"
    ) != sha256_json(_resume_config_payload(config)):
        raise ValueError("Run manifest resume_compatibility_hash is inconsistent.")

    for name in ("code", "environment"):
        payload = normalized.get(name)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Run manifest {name} identity is missing.")
        nested = dict(payload)
        nested_hash = nested.pop("hash", None)
        if nested_hash != sha256_json(nested):
            raise ValueError(f"Run manifest {name} identity hash is inconsistent.")
        if normalized.get(f"{name}_hash") != nested_hash:
            raise ValueError(f"Run manifest {name}_hash is inconsistent.")

    data = normalized.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("Run manifest data identity is missing.")
    if data.get("identity_status") == "label_only":
        stable_data = dict(data)
    else:
        stable_data = {
            "config_hash": data.get("config_hash"),
            "content_hash": data.get("content_hash"),
            "shape": data.get("shape"),
            "sampler_policy": data.get("sampler_policy"),
        }
    if normalized.get("data_hash") != sha256_json(stable_data):
        raise ValueError("Run manifest data_hash is inconsistent.")

    if schema_version >= 2:
        if normalized.get("seed_axes_hash") != sha256_json(
            _seed_axes_identity(normalized.get("seed_axes") or {})
        ):
            raise ValueError("Run manifest seed_axes_hash is inconsistent.")
        if normalized.get("initialization_hash") != sha256_json(
            normalized.get("initialization") or {}
        ):
            raise ValueError("Run manifest initialization_hash is inconsistent.")
    return normalized


def _atomic_temp(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(raw)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    return temp


def atomic_write_bytes(path: str | Path, payload: bytes) -> Path:
    target = Path(path)
    temp = _atomic_temp(target, payload)
    try:
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)
    return target


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    body = (json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    return atomic_write_bytes(path, body)


def write_immutable_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    """Atomically create a manifest, or verify an identical existing one."""
    target = Path(path)
    validated = validate_manifest(manifest)
    payload = (canonical_json(validated) + "\n").encode("utf-8")
    if target.exists():
        try:
            existing = validate_manifest(json.loads(target.read_text("utf-8")))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Existing run manifest is invalid: {target}: {exc}") from exc
        if canonical_json(existing) != canonical_json(validated):
            raise ValueError(f"Immutable run manifest already differs: {target}")
        return target
    temp = _atomic_temp(target, payload)
    try:
        try:
            os.link(temp, target)
        except FileExistsError:
            if target.read_bytes() != payload:
                raise ValueError(f"Immutable run manifest already differs: {target}")
    finally:
        temp.unlink(missing_ok=True)
    return target


def checkpoint_payload(
    state,
    *,
    completed_step: int,
    rng_state: Mapping[str, Any],
    history: list[dict[str, Any]],
    best_metric: float | None,
    best_step: int | None,
    manifest: Mapping[str, Any],
    resume_lineage: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "train_state": serialization.to_state_dict(state),
        "completed_step": int(completed_step),
        "rng_state_json": canonical_json(rng_state),
        "history_json": operational_json(history),
        "best_metric": best_metric,
        "best_step": best_step,
        "manifest_json": canonical_json(manifest),
        "seed_axes_json": canonical_json(manifest.get("seed_axes", {})),
        "resume_lineage_json": canonical_json(resume_lineage),
    }
    payload["payload_sha256"] = hashlib.sha256(
        serialization.msgpack_serialize(payload)
    ).hexdigest()
    return payload


def write_checkpoint(
    path: str | Path,
    state,
    *,
    completed_step: int,
    rng_state: Mapping[str, Any],
    history: list[dict[str, Any]],
    best_metric: float | None,
    best_step: int | None,
    manifest: Mapping[str, Any],
    resume_lineage: Mapping[str, Any],
) -> Path:
    payload = checkpoint_payload(
        state,
        completed_step=completed_step,
        rng_state=rng_state,
        history=history,
        best_metric=best_metric,
        best_step=best_step,
        manifest=manifest,
        resume_lineage=resume_lineage,
    )
    return atomic_write_bytes(path, serialization.msgpack_serialize(payload))


def read_checkpoint(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = serialization.msgpack_restore(source.read_bytes())
        if not isinstance(payload, dict):
            raise TypeError("checkpoint root is not a mapping")
        version = int(payload.get("checkpoint_version", -1))
        if version not in (1, CHECKPOINT_VERSION):
            raise ValueError("unsupported checkpoint version")
        if version >= 2:
            claimed_checksum = str(payload.get("payload_sha256") or "")
            checksum_body = dict(payload)
            checksum_body.pop("payload_sha256", None)
            actual_checksum = hashlib.sha256(
                serialization.msgpack_serialize(checksum_body)
            ).hexdigest()
            if claimed_checksum != actual_checksum:
                raise ValueError("checkpoint payload checksum mismatch")
        for key in (
            "train_state",
            "completed_step",
            "rng_state_json",
            "history_json",
            "manifest_json",
        ):
            if key not in payload:
                raise ValueError(f"missing checkpoint field '{key}'")
        payload["rng_state"] = json.loads(payload["rng_state_json"])
        payload["history"] = _restore_operational_json(
            json.loads(payload["history_json"])
        )
        payload["manifest"] = validate_manifest(
            json.loads(payload["manifest_json"])
        )
        payload["seed_axes"] = json.loads(payload.get("seed_axes_json") or "{}")
        payload["resume_lineage"] = json.loads(
            payload.get("resume_lineage_json") or "{}"
        )
        return payload
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid checkpoint '{source}': {exc}") from exc


def restore_checkpoint(path: str | Path, template_state):
    payload = read_checkpoint(path)
    try:
        state = serialization.from_state_dict(template_state, payload["train_state"])
    except Exception as exc:
        raise ValueError(f"Checkpoint TrainState is incompatible: {exc}") from exc
    return state, payload


def validate_checkpoint_manifest(
    checkpoint_manifest: Mapping[str, Any], current_manifest: Mapping[str, Any]
) -> None:
    mismatches = []
    compatibility_name = (
        "resume_compatibility_hash"
        if checkpoint_manifest.get("resume_compatibility_hash") is not None
        and current_manifest.get("resume_compatibility_hash") is not None
        else "config_hash"
    )
    for name in (
        compatibility_name,
        "code_hash",
        "data_hash",
        "environment_hash",
        "initialization_hash",
    ):
        if checkpoint_manifest.get(name) != current_manifest.get(name):
            mismatches.append(name)
    checkpoint_axes_hash = checkpoint_manifest.get("seed_axes_hash") or sha256_json(
        _seed_axes_identity(checkpoint_manifest.get("seed_axes", {}))
    )
    current_axes_hash = current_manifest.get("seed_axes_hash") or sha256_json(
        _seed_axes_identity(current_manifest.get("seed_axes", {}))
    )
    if checkpoint_axes_hash != current_axes_hash:
        mismatches.append("seed_axes_hash")
    if mismatches:
        raise ValueError(
            "Checkpoint is incompatible with the current run identity: "
            + ", ".join(mismatches)
        )


def checkpoint_is_valid(path: str | Path | None) -> bool:
    if not path:
        return False
    try:
        read_checkpoint(path)
        return True
    except ValueError:
        return False
