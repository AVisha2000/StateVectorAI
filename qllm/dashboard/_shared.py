"""Dependency-leaf helpers shared by dashboard payload builders."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from .security import resolve_within


def decode_config(row: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the decoded flat config from a persisted dashboard row."""
    if not row:
        return {}
    config = row.get("config")
    if isinstance(config, Mapping):
        return dict(config)
    try:
        decoded = json.loads(row.get("config_json") or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return dict(decoded) if isinstance(decoded, Mapping) else {}


def curve(db: Any, run_key: str | None, run_uuid: str | None = None) -> dict[str, list]:
    """Project ordered persisted step rows into the dashboard curve shape."""
    if not run_key:
        return {}
    series: dict[str, list] = defaultdict(list)
    for step in db.fetch_steps(run_key, run_uuid=run_uuid):
        series[step["name"]].append({"step": step["step"], "value": step["value"]})
    return dict(series)


def quantum_scale(job: Mapping[str, Any]) -> dict[str, int] | None:
    """Return valid quantum scale metadata from override or study-cell config."""
    config = decode_config(job)
    override_qubits = config.get("lab.quantum_override.n_qubits")
    override_depth = config.get("lab.quantum_override.n_circuit_layers")
    qubits = (
        override_qubits
        if override_qubits is not None
        else config.get("lab.study_cell.n_qubits")
    )
    depth = (
        override_depth
        if override_depth is not None
        else config.get("lab.study_cell.n_circuit_layers")
    )
    try:
        return {"n_qubits": int(qubits), "n_circuit_layers": int(depth)}
    except (TypeError, ValueError):
        return None


def job_variant(job: Mapping[str, Any]) -> str:
    """Return the canonical lab variant for a job without changing its payload."""
    run_key = job.get("run_key")
    if run_key:
        parts = str(run_key).split("/")
        if len(parts) >= 2:
            return parts[1]
    scale = quantum_scale(job)
    preset_id = str(job.get("preset_id", ""))
    if scale:
        return f"{preset_id}-q{scale['n_qubits']}-d{scale['n_circuit_layers']}"
    return preset_id


def artifact_dir(results_dir: str | Path, job: Mapping[str, Any]) -> Path:
    """Resolve a job artifact directory within the configured results root."""
    root = Path(results_dir).resolve()
    persisted = job.get("artifact_dir")
    if persisted:
        return resolve_within(root, persisted, label="persisted artifact directory")
    checkpoint = job.get("checkpoint_path")
    if checkpoint:
        safe_checkpoint = resolve_within(root, checkpoint, label="persisted checkpoint")
        return resolve_within(
            root, safe_checkpoint.parent.parent, label="checkpoint artifact directory"
        )
    return resolve_within(root, str(job["run_name"]), label="legacy run artifact")
