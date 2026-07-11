"""Experiment tracking: MLflow as system of record + quantum-metrics logger.

Implements the planning doc's recommendation: MLflow (self-hosted, file
store by default) for params/metrics/artifacts, plus a thin custom layer
that logs circuit diagnostics (gradient variance, Meyer-Wallach Q,
expressibility KL) as first-class metrics with consistent `q_` prefixes,
so scaling plots can be assembled later via ``mlflow.search_runs``.

The tracker degrades to a no-op if MLflow is unavailable or disabled, so
the training loop and tests never depend on it.
"""
from __future__ import annotations

import math
from numbers import Real
from pathlib import Path
from typing import Any

from .config import QuantumConfig, TrackingConfig


class ExperimentTracker:
    """Thin MLflow wrapper with graceful no-op fallback."""

    def __init__(self, cfg: TrackingConfig):
        self.cfg = cfg
        self._mlflow = None
        if not cfg.enabled:
            return
        try:
            import mlflow

            mlflow.set_tracking_uri(cfg.tracking_uri)
            mlflow.set_experiment(cfg.experiment)
            mlflow.start_run(run_name=cfg.run_name)
            self._mlflow = mlflow
        except Exception as exc:  # pragma: no cover - env-dependent
            print(f"[tracking] disabled ({type(exc).__name__}: {exc})")

    @property
    def active(self) -> bool:
        return self._mlflow is not None

    def log_params(self, params: dict[str, Any]) -> None:
        if self.active:
            self._mlflow.log_params({k: str(v)[:500] for k, v in params.items()})

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self.active:
            self._mlflow.log_metrics(
                {k: float(v) for k, v in metrics.items()}, step=step
            )

    def set_tags(self, tags: dict[str, Any]) -> None:
        if self.active:
            self._mlflow.set_tags(tags)

    def log_artifact(self, path: str | Path) -> None:
        if self.active:
            self._mlflow.log_artifact(str(path))

    def end(self) -> None:
        if self.active:
            self._mlflow.end_run()

    def __enter__(self) -> "ExperimentTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.end()


def log_quantum_diagnostics(
    tracker: ExperimentTracker,
    qcfg: QuantumConfig,
    n_grad_samples: int = 64,
    n_pairs: int = 200,
    n_mw_samples: int = 32,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute and log circuit diagnostics for the configured quantum layer.

    Logged once per run (they characterize the circuit at init, not the
    trained model): grad variance (barren-plateau probe), Meyer-Wallach Q,
    expressibility KL. Unsupported diagnostics remain explicit ``None`` values
    with an ``availability`` reason. Returns the diagnostics dict either way,
    so callers can inspect it even with tracking disabled.
    """
    from .quantum import metrics as qmetrics

    diag = qmetrics.quantum_diagnostics(
        qcfg.n_qubits,
        qcfg.n_circuit_layers,
        ansatz=qcfg.ansatz,
        backend=qcfg.backend,
        device=qcfg.device,
        n_grad_samples=n_grad_samples,
        n_pairs=n_pairs,
        n_mw_samples=n_mw_samples,
        seed=seed,
        diff_method=qcfg.diff_method,
        shots=qcfg.shots,
        mps_max_bond_dimension=qcfg.mps_max_bond_dimension,
        mps_max_truncation_error=qcfg.mps_max_truncation_error,
        mps_relative_truncation=qcfg.mps_relative_truncation,
    )

    availability = diag.get("availability")
    if not isinstance(availability, dict):
        raise ValueError("quantum diagnostics must include availability metadata")
    mlflow_metrics: dict[str, float] = {}
    for key, value in diag.items():
        if key == "availability":
            continue
        availability_key = (
            "gradient_variance"
            if key in qmetrics.GRADIENT_METRIC_KEYS
            else key
        )
        status = availability.get(availability_key)
        if not isinstance(status, dict):
            raise ValueError(
                f"quantum diagnostic {key!r} lacks availability metadata"
            )
        if value is None:
            if status.get("status") == "measured":
                raise ValueError(
                    f"quantum diagnostic {key!r} is marked measured but has no value"
                )
            continue
        if status.get("status") != "measured":
            raise ValueError(
                f"quantum diagnostic {key!r} has a value but is not marked measured"
            )
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(
                f"measured quantum diagnostic {key!r} must be numeric; "
                f"got {type(value).__name__}"
            )
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(
                f"measured quantum diagnostic {key!r} must be finite; got {value!r}"
            )
        mlflow_metrics[f"q_{key}"] = numeric
    tracker.log_metrics(mlflow_metrics)
    return diag
