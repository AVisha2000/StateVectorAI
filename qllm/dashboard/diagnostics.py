"""Read-only quantum diagnostic payloads for completed dashboard artifacts.

This module deliberately does not construct models or circuits.  It exposes
only measurements already persisted in a job's confined ``summary.json``;
the optional scaling result is a pure numeric fit over those saved rows.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel

from ..quantum.metrics import gradient_variance_scaling_fit
from ..resultsdb import ResultsDB
from .security import resolve_within


DIMENSION_NAMES = (
    "gradient_variance",
    "parameter_shift_gradient_snr",
    "expressibility_kl",
    "meyer_wallach_q",
    "scaling_fit",
)
_GRADIENT_KEYS = ("grad_var_first_param", "grad_var_mean", "grad_var_max")


class DiagnosticDimension(BaseModel):
    """One persisted mechanism or trainability observation."""

    status: Literal["measured", "unavailable"]
    value: float | dict[str, float | int | bool] | None
    source: str
    reason: str | None = None
    provenance: dict[str, Any]

    class Config:
        extra = "forbid"


class InterpretationWarning(BaseModel):
    code: str
    severity: Literal["info", "warning"]
    title: str
    message: str
    evidence: dict[str, Any]

    class Config:
        extra = "forbid"


class DiagnosticsJob(BaseModel):
    id: int
    run_name: str
    status: str
    group_id: str | None = None

    class Config:
        extra = "forbid"


class DiagnosticsDimensions(BaseModel):
    gradient_variance: DiagnosticDimension
    parameter_shift_gradient_snr: DiagnosticDimension
    expressibility_kl: DiagnosticDimension
    meyer_wallach_q: DiagnosticDimension
    scaling_fit: DiagnosticDimension

    class Config:
        extra = "forbid"


class DiagnosticsPayload(BaseModel):
    """Response model for a future diagnostics route."""

    job: DiagnosticsJob
    diagnostics: DiagnosticsDimensions
    interpretation_warnings: list[InterpretationWarning]

    class Config:
        extra = "forbid"


def _response_dict(payload: DiagnosticsPayload) -> dict[str, Any]:
    """Support the Pydantic versions used by dashboard dependency profiles."""
    if hasattr(payload, "model_dump"):
        return payload.model_dump(mode="json")
    return payload.dict()


def _artifact_dir(results_dir: str | Path, job: Mapping[str, Any]) -> Path:
    root = Path(results_dir).resolve()
    artifact_dir = job.get("artifact_dir")
    if artifact_dir:
        return resolve_within(root, artifact_dir, label="persisted artifact directory")
    checkpoint = job.get("checkpoint_path")
    if checkpoint:
        safe_checkpoint = resolve_within(root, checkpoint, label="persisted checkpoint")
        return resolve_within(
            root, safe_checkpoint.parent.parent, label="checkpoint artifact directory"
        )
    return resolve_within(root, str(job["run_name"]), label="legacy run artifact")


def _read_summary(
    results_dir: str | Path, job: Mapping[str, Any]
) -> tuple[Mapping[str, Any] | None, str | None]:
    """Load exactly one path-confined JSON summary without raising for artifacts."""
    try:
        artifact_dir = _artifact_dir(results_dir, job)
        summary_path = resolve_within(
            artifact_dir, artifact_dir / "summary.json", label="summary artifact"
        )
    except (KeyError, TypeError, ValueError) as exc:
        return None, str(exc)
    if not summary_path.is_file():
        return None, "summary.json artifact is missing"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, "summary.json is not valid JSON"
    if not isinstance(summary, Mapping):
        return None, "summary.json must contain a JSON object"
    return summary, None


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _availability_reason(diagnostics: Mapping[str, Any], name: str) -> str | None:
    availability = diagnostics.get("availability")
    if not isinstance(availability, Mapping):
        return None
    record = availability.get(name)
    if not isinstance(record, Mapping):
        return None
    reason = record.get("reason")
    return reason if isinstance(reason, str) and reason else None


def _unavailable(
    name: str, reason: str, *, source: str = "summary.json", **provenance: Any
) -> DiagnosticDimension:
    return DiagnosticDimension(
        status="unavailable",
        value=None,
        source=source,
        reason=reason,
        provenance={"dimension": name, "artifact": "summary.json", **provenance},
    )


def _scalar_dimension(
    name: str, diagnostics: Mapping[str, Any]
) -> DiagnosticDimension:
    value = _finite_number(diagnostics.get(name))
    if value is not None:
        return DiagnosticDimension(
            status="measured",
            value=value,
            source="summary.quantum_diagnostics",
            provenance={"dimension": name, "artifact": "summary.json"},
        )
    return _unavailable(
        name,
        _availability_reason(diagnostics, name)
        or "persisted value is missing, malformed, or non-finite",
    )


def _mapping_dimension(
    name: str, value: object, *, source: str, diagnostics: Mapping[str, Any]
) -> DiagnosticDimension:
    if isinstance(value, Mapping) and value:
        parsed: dict[str, float | int | bool] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                return _unavailable(name, "persisted value is malformed")
            number = _finite_number(item)
            if number is None:
                return _unavailable(
                    name, "persisted value is missing, malformed, or non-finite"
                )
            parsed[key] = number
        return DiagnosticDimension(
            status="measured",
            value=parsed,
            source=source,
            provenance={"dimension": name, "artifact": "summary.json"},
        )
    return _unavailable(
        name,
        _availability_reason(diagnostics, name)
        or "persisted value is missing, malformed, or non-finite",
    )


def _gradient_dimension(diagnostics: Mapping[str, Any]) -> DiagnosticDimension:
    values = {key: diagnostics.get(key) for key in _GRADIENT_KEYS}
    return _mapping_dimension(
        "gradient_variance",
        values,
        source="summary.quantum_diagnostics",
        diagnostics=diagnostics,
    )


def _snr_dimension(summary: Mapping[str, Any], diagnostics: Mapping[str, Any]) -> DiagnosticDimension:
    if "parameter_shift_gradient_snr" in diagnostics:
        value, source = diagnostics["parameter_shift_gradient_snr"], "summary.quantum_diagnostics"
    else:
        value, source = summary.get("parameter_shift_gradient_snr"), "summary.parameter_shift_gradient_snr"
    return _mapping_dimension(
        "parameter_shift_gradient_snr", value, source=source, diagnostics=diagnostics
    )


def _config(job: Mapping[str, Any]) -> Mapping[str, Any]:
    config = job.get("config")
    if isinstance(config, Mapping):
        return config
    try:
        decoded = json.loads(job.get("config_json") or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return decoded if isinstance(decoded, Mapping) else {}


def _n_qubits(summary: Mapping[str, Any], job: Mapping[str, Any]) -> int | None:
    direct = summary.get("n_qubits")
    if isinstance(direct, int) and not isinstance(direct, bool) and direct > 0:
        return direct
    resources = summary.get("resources")
    if isinstance(resources, Mapping):
        backend = resources.get("quantum_backend")
        tags = backend.get("tags") if isinstance(backend, Mapping) else None
        value = tags.get("qubits") if isinstance(tags, Mapping) else None
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
        components = backend.get("components") if isinstance(backend, Mapping) else None
        if isinstance(components, Mapping):
            counts = {
                item.get("n_qubits")
                for item in components.values()
                if isinstance(item, Mapping)
                and isinstance(item.get("n_qubits"), int)
                and not isinstance(item.get("n_qubits"), bool)
                and item.get("n_qubits") > 0
            }
            if len(counts) == 1:
                return counts.pop()
    config = _config(job)
    for key in (
        "model.quantum.n_qubits",
        "lab.quantum_override.n_qubits",
        "lab.study_cell.n_qubits",
    ):
        value = config.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    model = config.get("model")
    quantum = model.get("quantum") if isinstance(model, Mapping) else None
    value = quantum.get("n_qubits") if isinstance(quantum, Mapping) else None
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _scaling_dimension(
    db: ResultsDB, job: Mapping[str, Any], results_dir: str | Path
) -> DiagnosticDimension:
    group_id = job.get("group_id")
    if not isinstance(group_id, str) or not group_id:
        return _unavailable("scaling_fit", "job has no persisted group_id")
    values_by_qubits: dict[int, list[float]] = {}
    persisted_rows = 0
    for candidate in db.fetch_lab_jobs(limit=1_000):
        if candidate.get("group_id") != group_id:
            continue
        summary, error = _read_summary(results_dir, candidate)
        if error or summary is None:
            continue
        diagnostics = summary.get("quantum_diagnostics")
        if not isinstance(diagnostics, Mapping):
            continue
        variance = _finite_number(diagnostics.get("grad_var_mean"))
        qubits = _n_qubits(summary, candidate)
        if variance is not None and variance > 0.0 and qubits is not None:
            values_by_qubits.setdefault(qubits, []).append(variance)
            persisted_rows += 1
    rows = [
        {
            "n_qubits": float(qubits),
            "grad_var_mean": sum(values) / len(values),
        }
        for qubits, values in sorted(values_by_qubits.items())
    ]
    if len(rows) < 2:
        return _unavailable(
            "scaling_fit",
            "at least two distinct persisted same-group qubit counts are required",
            group_id=group_id,
            persisted_rows=persisted_rows,
            distinct_qubit_counts=len(rows),
        )
    try:
        fit = gradient_variance_scaling_fit(rows)
    except (TypeError, ValueError, OverflowError):
        return _unavailable("scaling_fit", "persisted scaling rows could not be fit")
    if any(_finite_number(value) is None for key, value in fit.items() if key != "exponential_decay_detected"):
        return _unavailable("scaling_fit", "scaling fit produced a non-finite value")
    return DiagnosticDimension(
        status="measured",
        value=fit,
        source="persisted same-group summary.quantum_diagnostics",
        provenance={
            "dimension": "scaling_fit",
            "group_id": group_id,
            "persisted_rows": persisted_rows,
            "distinct_qubit_counts": len(rows),
            "qubit_counts": [int(row["n_qubits"]) for row in rows],
            "fit": "qllm.quantum.metrics.gradient_variance_scaling_fit",
        },
    )


def diagnostics_payload(
    db: ResultsDB, job_id: int, results_dir: str | Path = "results"
) -> dict[str, Any]:
    """Return saved diagnostics for one job without recomputing quantum data."""
    job = db.get_lab_job(job_id)
    if job is None:
        raise KeyError(f"Unknown job {job_id}")
    summary, summary_error = _read_summary(results_dir, job)
    if summary is None:
        dimensions = {
            name: _unavailable(name, summary_error or "summary is unavailable")
            for name in DIMENSION_NAMES
        }
    else:
        persisted = summary.get("quantum_diagnostics")
        diagnostics = persisted if isinstance(persisted, Mapping) else {}
        absent_reason = (
            "summary.quantum_diagnostics is missing or malformed"
            if not isinstance(persisted, Mapping)
            else None
        )
        dimensions = {
            "gradient_variance": _gradient_dimension(diagnostics),
            "parameter_shift_gradient_snr": _snr_dimension(summary, diagnostics),
            "expressibility_kl": _scalar_dimension("expressibility_kl", diagnostics),
            "meyer_wallach_q": _scalar_dimension("meyer_wallach_q", diagnostics),
            "scaling_fit": _scaling_dimension(db, job, results_dir),
        }
        if absent_reason:
            for name in DIMENSION_NAMES:
                if name != "scaling_fit":
                    dimensions[name] = _unavailable(name, absent_reason)
    payload = DiagnosticsPayload(
        job={"id": job["id"], "run_name": job["run_name"], "status": job["status"], "group_id": job.get("group_id")},
        diagnostics=dimensions,
        interpretation_warnings=[
            InterpretationWarning(
                code="diagnostics_scope",
                severity="warning",
                title="Mechanism and trainability observations only",
                message="These saved diagnostics are mechanism/trainability observations, not evidence of quantum advantage.",
                evidence={"dimensions": list(DIMENSION_NAMES)},
            ),
            InterpretationWarning(
                code="diagnostics_not_aggregated",
                severity="info",
                title="No aggregate diagnostic",
                message="The payload provides separate dimensions and intentionally provides no composite score.",
                evidence={"aggregation": "not provided"},
            ),
        ],
    )
    return _response_dict(payload)
