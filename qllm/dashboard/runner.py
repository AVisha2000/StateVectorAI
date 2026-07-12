"""Single-worker local experiment queue for QLLM Lab."""
from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import socket
import sqlite3
import threading
import time
import traceback
import uuid
from pathlib import Path

from ..config import (
    DataConfig,
    TrackingConfig,
    from_dict,
    to_flat_dict,
    validate_config,
)
from ..claims import METRIC_TYPES, get_claim, infer_claim_id
from ..research_protocol import TWO_STREAM_CAUSAL_PROTOCOL, normalize_seed_axes
from ..resultsdb import ResultsDB
from ..train.artifacts import RunOptions, read_checkpoint, validate_manifest
from . import with_dashboard
from .analogues import (
    AnalogueSpec,
    analogue_config_fields,
    classical_analogue_for_config,
    classical_analogue_for_preset,
    config_from_flat_payload,
)
from .datasets import get_dataset
from .gpu_reservation import (
    apply_reservation_config,
    reservation_metadata,
    update_reservation_state,
)
from .model_graph import uses_quantum_config
from .model_specs import config_payload, create_spec
from .presets import apply_quantum_overrides, build_preset
from .resources import quantum_resource_estimate
from .security import resolve_data_path, resolve_within


logger = logging.getLogger(__name__)


def _transient_sqlite_error(exc: sqlite3.OperationalError) -> bool:
    message = str(exc).casefold()
    return "locked" in message or "busy" in message


def _lease_heartbeat_loop(
    db: ResultsDB,
    job_id: int,
    worker_id: str,
    lease_seconds: float,
    stop: threading.Event,
    ownership_lost: threading.Event,
) -> None:
    """Renew a claimed job lease until stopped or ownership is no longer safe."""
    interval = max(0.05, min(lease_seconds / 3.0, 30.0))
    while not stop.wait(interval):
        try:
            renewed = db.heartbeat_lab_job(
                job_id,
                worker_id,
                lease_seconds=lease_seconds,
            )
        except sqlite3.OperationalError as exc:
            if _transient_sqlite_error(exc):
                # Lock contention is transient; a later renewal still proves
                # whether this worker owns the lease.
                logger.warning("Transient SQLite failure renewing job %s lease", job_id)
                continue
            logger.exception("Unexpected SQLite failure renewing job %s lease", job_id)
            ownership_lost.set()
            return
        except Exception:
            logger.exception("Unexpected failure renewing job %s lease", job_id)
            ownership_lost.set()
            return
        if not renewed:
            ownership_lost.set()
            return


def _compatible_seed_request(snapshot: dict | None, cfg) -> dict | None:
    """Retain supported shared requests and drop structurally N/A axes."""
    if not isinstance(snapshot, dict):
        return None
    raw = snapshot.get("requested")
    requested = dict(raw if isinstance(raw, dict) else snapshot)
    if not uses_quantum_config(cfg):
        requested.pop("circuit", None)
    return requested or None


class ExperimentQueue:
    def __init__(
        self,
        db_path: str = "results/qllm_results.db",
        start_worker: bool = True,
        *,
        lease_seconds: float = 300.0,
        poll_seconds: float = 0.5,
        worker_id: str | None = None,
        results_dir: str | Path | None = None,
        data_dir: str | Path | None = None,
    ):
        self.db_path = db_path
        self.results_dir = Path(
            results_dir
            if results_dir is not None
            else os.environ.get("QLLM_RESULTS", "results")
        ).resolve()
        self.data_dir = Path(
            data_dir
            if data_dir is not None
            else os.environ.get("QLLM_DATA", "data")
        ).resolve()
        self.lease_seconds = float(lease_seconds)
        self.poll_seconds = float(poll_seconds)
        if (
            not math.isfinite(self.lease_seconds)
            or not math.isfinite(self.poll_seconds)
            or self.lease_seconds <= 0
            or self.poll_seconds <= 0
        ):
            raise ValueError(
                "lease_seconds and poll_seconds must be finite and positive."
            )
        self.worker_id = worker_id or (
            f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"
        )
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._worker = None
        self.db().recover_stale_lab_jobs(
            checkpoint_resolver=self._recoverable_checkpoint
        )
        if start_worker:
            self._worker = threading.Thread(target=self._run, daemon=True)
            self._worker.start()

    def db(self) -> ResultsDB:
        return ResultsDB(self.db_path)

    def _confined_path(self, value: str | Path, *, label: str) -> Path:
        return resolve_within(self.results_dir, value, label=label)

    def _confined_data_path(self, value: str | Path, *, label: str) -> Path:
        return resolve_data_path(self.data_dir, value, label=label)

    def _authorized_artifact_layout(self, value: str | Path) -> Path:
        """Authorize the run root and every dashboard-managed write target."""
        artifact_dir = self._confined_path(value, label="artifact directory")
        for relative, label in (
            ("manifest.json", "run manifest"),
            ("summary.json", "run summary"),
            ("params.msgpack", "run parameters"),
            ("checkpoints", "checkpoint directory"),
            ("checkpoints/latest.msgpack", "latest checkpoint"),
            ("checkpoints/best.msgpack", "best checkpoint"),
        ):
            resolve_within(
                artifact_dir,
                artifact_dir / relative,
                label=label,
            )
        return artifact_dir

    def close(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._worker is not None:
            self._worker.join(timeout=timeout)

    def worker_status(self) -> str:
        """Return the stable display contract for the in-process worker."""
        if self._worker is None:
            return "in-process / disabled"
        if self._worker.is_alive() and not self._stop.is_set():
            return "in-process / active"
        return "in-process / stopped"

    def submit(
        self, preset_id: str, dataset_name: str, run_name: str | None,
        seed: int, steps: int, eval_every: int,
        device_target: str = "auto",
        queue_classical_comparison: bool = False,
        quantum_overrides: dict | None = None,
        group_id: str | None = None,
        batch_size: int | None = None,
        seq_len: int | None = None,
        claim_id: str | None = None,
        seed_axes: dict | None = None,
        metric_type: str | None = None,
        resume_from: str | None = None,
        checkpoint_every: int = 0,
        experiment_uuid: str | None = None,
        run_uuid: str | None = None,
        artifact_dir: str | None = None,
    ) -> dict:
        db = self.db()
        dataset = get_dataset(db, dataset_name)
        if dataset is None:
            raise ValueError(f"Unknown dataset '{dataset_name}'")
        corpus_path = str(
            self._confined_data_path(
                dataset["corpus_path"], label="dataset corpus path"
            )
        )
        device_target = (device_target or "auto").strip().lower()
        if device_target not in ("auto", "cpu", "gpu"):
            raise ValueError("Device target must be one of: auto, cpu, gpu.")
        if device_target == "gpu" and not self.gpu_ready():
            raise ValueError(
                "GPU was requested, but JAX does not currently see a GPU. "
                "Open the GPU page and follow the CUDA/JAX setup guidance."
            )
        allow_gpu_scale = device_target == "gpu"
        explicit_run_name = run_name is not None
        cfg = self._config_for_source(preset_id)
        cfg = apply_quantum_overrides(
            preset_id, cfg, quantum_overrides, allow_gpu_scale=allow_gpu_scale
        )
        run_name = (run_name or cfg.tracking.run_name or preset_id).strip()
        if not run_name:
            raise ValueError("Run name is required.")
        steps = int(steps)
        eval_every = int(eval_every)
        if steps < 1:
            raise ValueError("Steps must be at least 1.")
        if eval_every < 1:
            raise ValueError("Eval interval must be at least 1.")
        if batch_size is not None and int(batch_size) < 1:
            raise ValueError("Batch size must be at least 1.")
        seed = int(seed)
        analogue = self._analogue_for_source(preset_id, cfg)
        wants_comparison = bool(queue_classical_comparison and analogue)
        checkpoint_every = int(checkpoint_every or 0)
        if checkpoint_every < 0:
            raise ValueError("Checkpoint interval must be non-negative.")
        if resume_from and wants_comparison:
            raise ValueError(
                "A checkpoint resumes one logical model run; queue its matched "
                "comparison separately."
            )
        parent_run_uuid = None
        if resume_from:
            resume_from = str(
                self._confined_path(resume_from, label="resume checkpoint")
            )
        if artifact_dir:
            artifact_dir = str(
                self._confined_path(artifact_dir, label="artifact directory")
            )
        if resume_from:
            resume_payload = read_checkpoint(resume_from)
            source_manifest = resume_payload["manifest"]
            source_run_uuid = source_manifest.get("run_uuid")
            source_experiment_uuid = source_manifest.get("experiment_uuid")
            if not source_run_uuid or not source_experiment_uuid:
                raise ValueError("Resume checkpoint is missing run identity.")
            if run_uuid is None:
                run_uuid = source_run_uuid
            if experiment_uuid is None:
                experiment_uuid = source_experiment_uuid
            if run_uuid != source_run_uuid:
                if not artifact_dir:
                    raise ValueError(
                        "Forking a checkpoint requires an explicit new run_uuid "
                        "and artifact_dir."
                    )
                parent_run_uuid = source_run_uuid
            elif artifact_dir is None:
                source_path = Path(resume_from).resolve()
                artifact_dir = str(
                    source_path.parent.parent
                    if source_path.parent.name == "checkpoints"
                    else source_path.parent
                )
            source_name = str(source_manifest.get("run_name") or "")
            if source_name and not explicit_run_name:
                run_name = source_name
            elif source_name and run_name != source_name and run_uuid == source_run_uuid:
                raise ValueError(
                    "Recovery must retain the checkpoint run_name/artifact identity."
                )
        cfg = self._config_for_job(
            cfg, corpus_path, run_name,
            seed, steps, eval_every, batch_size, seq_len,
        )
        self._require_valid_config(cfg, "candidate")
        resolved_claim_id = infer_claim_id(
            explicit=claim_id,
            preset_id=preset_id,
        )
        if claim_id and resolved_claim_id is None:
            raise ValueError(f"Unknown or ambiguous claim_id '{claim_id}'.")
        claim = get_claim(resolved_claim_id) if resolved_claim_id else None
        resolved_metric_type = str(
            metric_type
            or (claim or {}).get("metric_type")
            or "strict_autoregressive_next_token"
        )
        if resolved_metric_type not in METRIC_TYPES:
            raise ValueError(f"Unsupported metric_type '{resolved_metric_type}'.")
        if claim and resolved_metric_type != claim["metric_type"]:
            raise ValueError(
                f"metric_type must match claim '{resolved_claim_id}': "
                f"{claim['metric_type']}"
            )
        axes = normalize_seed_axes(
            seed,
            generator_seed=cfg.data.gen_seed,
            data_kind=cfg.data.kind,
            circuit_applicable=uses_quantum_config(cfg),
            explicit=seed_axes,
            reject_unsupported=True,
        )
        prepared_twin_cfg = None
        if wants_comparison and analogue:
            twin_name = f"{run_name}-classical"
            prepared_twin_cfg = self._config_for_job(
                self._config_from_analogue(analogue),
                corpus_path,
                twin_name,
                seed,
                steps,
                eval_every,
                batch_size,
                seq_len,
            )
            self._require_valid_config(prepared_twin_cfg, "classical analogue")
        estimate = quantum_resource_estimate(cfg)
        if estimate["band"] == "extreme" and estimate["uses_quantum_attention"]:
            raise ValueError(
                "This quantum-attention run is likely to exhaust GPU memory. "
                "Try batch size 1-4, seq_len 16-32, or reduce qubits/depth. "
                f"Current estimate: {estimate['band']}."
            )
        job_config = to_flat_dict(cfg)
        job_config["research.claim_id"] = resolved_claim_id
        job_config["research.metric_type"] = resolved_metric_type
        job_config["research.seed_axes"] = axes
        job_config["lab.config_snapshot_version"] = 1
        job_config["lab.checkpoint_every"] = checkpoint_every
        job_config["lab.submission.comparison_mode"] = (
            "paired" if wants_comparison else "single"
        )
        if resume_from:
            job_config["lab.resume_from"] = str(resume_from)
        if artifact_dir:
            job_config["lab.artifact_dir"] = str(artifact_dir)
        if cfg.model.arch == "two_stream":
            job_config["lab.two_stream_protocol"] = TWO_STREAM_CAUSAL_PROTOCOL
        if quantum_overrides:
            job_config["lab.quantum_override.n_qubits"] = cfg.model.quantum.n_qubits
            job_config["lab.quantum_override.n_circuit_layers"] = (
                cfg.model.quantum.n_circuit_layers
            )
            job_config["lab.study_cell.n_qubits"] = cfg.model.quantum.n_qubits
            job_config["lab.study_cell.n_circuit_layers"] = (
                cfg.model.quantum.n_circuit_layers
            )
        if batch_size is not None:
            job_config["lab.train_override.batch_size"] = cfg.train.batch_size
        if seq_len is not None:
            job_config["lab.train_override.seq_len"] = cfg.train.seq_len
        job_config["lab.resource.band"] = estimate["band"]
        job_config["lab.resource.score"] = estimate["score"]
        job_config = apply_reservation_config(
            job_config,
            reservation_metadata(device_target, estimate),
        )
        job_config.update(
            analogue_config_fields(
                analogue,
                state="queued" if wants_comparison else "missing",
                role="candidate" if wants_comparison else "candidate_missing_baseline",
            )
        )
        group_id = group_id or uuid.uuid4().hex
        experiment_uuid = str(uuid.UUID(str(experiment_uuid or uuid.uuid4())))
        run_uuid = str(uuid.UUID(str(run_uuid or uuid.uuid4())))
        if artifact_dir is None:
            artifact_dir = str((self.results_dir / "runs" / run_uuid).resolve())
            job_config["lab.artifact_dir"] = artifact_dir
        artifact_dir = str(self._authorized_artifact_layout(artifact_dir))
        job_config["lab.artifact_dir"] = artifact_dir
        primary_job = {
            "status": "queued",
            "preset_id": preset_id,
            "dataset_name": dataset_name,
            "run_name": run_name,
            "seed": seed,
            "steps": steps,
            "eval_every": eval_every,
            "config": job_config,
            "group_id": group_id,
            "device_target": device_target,
            "comparison_role": "candidate" if wants_comparison else "primary",
            "experiment_uuid": experiment_uuid,
            "run_uuid": run_uuid,
            "parent_run_uuid": parent_run_uuid,
            "resume_from": str(resume_from) if resume_from else None,
            "artifact_dir": str(artifact_dir) if artifact_dir else None,
        }
        comparison_job = None
        if wants_comparison and analogue:
            analogue_source_id, _, analogue = self._source_for_analogue(
                db, analogue, run_name, preset_id, 0
            )
            twin_name = f"{run_name}-classical"
            if prepared_twin_cfg is None:  # defensive; guarded above
                raise AssertionError("Missing validated classical analogue config.")
            twin_cfg = prepared_twin_cfg
            twin_config = to_flat_dict(twin_cfg)
            twin_config["research.claim_id"] = resolved_claim_id
            twin_config["research.metric_type"] = resolved_metric_type
            twin_config["lab.config_snapshot_version"] = 1
            twin_config["lab.checkpoint_every"] = checkpoint_every
            twin_config["lab.submission.comparison_mode"] = "paired"
            twin_config["research.seed_axes"] = normalize_seed_axes(
                seed,
                generator_seed=twin_cfg.data.gen_seed,
                data_kind=twin_cfg.data.kind,
                circuit_applicable=uses_quantum_config(twin_cfg),
                explicit=_compatible_seed_request(axes, twin_cfg),
                reject_unsupported=True,
            )
            if quantum_overrides:
                twin_config["lab.study_cell.n_qubits"] = cfg.model.quantum.n_qubits
                twin_config["lab.study_cell.n_circuit_layers"] = (
                    cfg.model.quantum.n_circuit_layers
                )
            if twin_cfg.model.arch == "two_stream":
                twin_config["lab.two_stream_protocol"] = TWO_STREAM_CAUSAL_PROTOCOL
            twin_estimate = quantum_resource_estimate(twin_cfg)
            twin_config["lab.resource.band"] = twin_estimate["band"]
            twin_config["lab.resource.score"] = twin_estimate["score"]
            twin_config = apply_reservation_config(
                twin_config,
                reservation_metadata(device_target, twin_estimate),
            )
            twin_config.update(
                analogue_config_fields(
                    analogue,
                    state="baseline",
                    role="baseline",
                )
            )
            twin_run_uuid = str(uuid.uuid4())
            twin_artifact_dir = str(
                self._authorized_artifact_layout(
                    self.results_dir / "runs" / twin_run_uuid
                )
            )
            twin_config["lab.artifact_dir"] = twin_artifact_dir
            twin_job = {
                "status": "queued",
                "preset_id": analogue_source_id,
                "dataset_name": dataset_name,
                "run_name": twin_name,
                "seed": seed,
                "steps": steps,
                "eval_every": eval_every,
                "config": twin_config,
                "group_id": group_id,
                "device_target": device_target,
                "comparison_role": "baseline",
                "experiment_uuid": experiment_uuid,
                "run_uuid": twin_run_uuid,
                "artifact_dir": twin_artifact_dir,
            }
            job_id, twin_job_id = db.create_lab_job_pair(primary_job, twin_job)
            comparison_job = self.get(twin_job_id)
        else:
            existing = db.get_lab_job_by_run_uuid(run_uuid)
            if existing is not None:
                if not db.lab_job_submission_matches(existing, primary_job):
                    raise ValueError(
                        "Existing dashboard job conflicts with submitted run_uuid."
                    )
                if existing["status"] in ("error", "cancelled") and resume_from:
                    db.requeue_lab_job_from_checkpoint(
                        int(existing["id"]),
                        resume_from=str(resume_from),
                        checkpoint_path=str(resume_from),
                        completed_step=int(resume_payload["completed_step"]),
                        config=job_config,
                        artifact_dir=str(artifact_dir) if artifact_dir else None,
                    )
                    self._wake.set()
                return self.get(int(existing["id"])) or {}
            job_id = db.create_lab_job(primary_job)
        self._wake.set()
        out = self.get(job_id) or {}
        if comparison_job is not None:
            out["comparison_job"] = comparison_job
        return out

    def submit_scaling_sweep(
        self, preset_id: str, dataset_name: str, run_name: str | None,
        seed: int, steps: int, eval_every: int, device_target: str,
        qubits: list[int], depths: list[int],
        queue_classical_comparison: bool = False,
        batch_size: int | None = None,
        seq_len: int | None = None,
        claim_id: str | None = None,
        seed_axes: dict | None = None,
        metric_type: str | None = None,
        checkpoint_every: int = 0,
        experiment_uuid: str | None = None,
    ) -> dict:
        if not qubits:
            raise ValueError("At least one qubit count is required.")
        if not depths:
            raise ValueError("At least one circuit depth is required.")
        if len(qubits) * len(depths) > 64:
            raise ValueError("Scaling sweeps are capped at 64 jobs per batch.")
        group_id = uuid.uuid4().hex
        experiment_uuid = experiment_uuid or str(uuid.uuid4())
        label = (run_name or preset_id).strip() or preset_id
        jobs = []
        for qbits in qubits:
            for depth in depths:
                jobs.append(
                    self.submit(
                        preset_id=preset_id,
                        dataset_name=dataset_name,
                        run_name=f"{label}-q{qbits}-d{depth}",
                        seed=seed,
                        steps=steps,
                        eval_every=eval_every,
                        device_target=device_target,
                        queue_classical_comparison=queue_classical_comparison,
                        quantum_overrides={
                            "n_qubits": int(qbits),
                            "n_circuit_layers": int(depth),
                        },
                        group_id=group_id,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        claim_id=claim_id,
                        seed_axes=seed_axes,
                        metric_type=metric_type,
                        checkpoint_every=checkpoint_every,
                        experiment_uuid=experiment_uuid,
                    )
                )
        return {"group_id": group_id, "count": len(jobs), "jobs": jobs}

    def submit_model_spec(
        self, spec_id: int, dataset_name: str, run_name: str | None,
        seed: int, steps: int, eval_every: int,
        device_target: str = "auto",
        queue_classical_comparison: bool = False,
        batch_size: int | None = None,
        seq_len: int | None = None,
        claim_id: str | None = None,
        seed_axes: dict | None = None,
        metric_type: str | None = None,
        resume_from: str | None = None,
        checkpoint_every: int = 0,
        experiment_uuid: str | None = None,
        run_uuid: str | None = None,
        artifact_dir: str | None = None,
    ) -> dict:
        spec = self.db().get_model_spec(spec_id)
        if spec is None:
            raise ValueError(f"Unknown model spec {spec_id}")
        return self.submit(
            preset_id=f"model-spec:{spec_id}",
            dataset_name=dataset_name,
            run_name=(run_name if resume_from else (run_name or spec["name"])),
            seed=seed,
            steps=steps,
            eval_every=eval_every,
            device_target=device_target,
            queue_classical_comparison=queue_classical_comparison,
            batch_size=batch_size,
            seq_len=seq_len,
            claim_id=claim_id,
            seed_axes=seed_axes,
            metric_type=metric_type,
            resume_from=resume_from,
            checkpoint_every=checkpoint_every,
            experiment_uuid=experiment_uuid,
            run_uuid=run_uuid,
            artifact_dir=artifact_dir,
        )

    def classical_analogue_for_job(self, job_id: int) -> dict:
        job = self.get(job_id)
        if job is None:
            raise ValueError(f"Unknown job {job_id}")
        cfg = self._config_from_job_snapshot(job)
        spec = self._analogue_for_source(job["preset_id"], cfg, source_job_id=job["id"])
        if spec is None:
            return {
                "available": False,
                "reason": "No quantum or hybrid component was detected for this job.",
            }
        payload = spec.to_payload(include_config=False)
        payload["available"] = True
        payload["analogue_state"] = (
            "queued" if job.get("compare_to_job_id") else "missing"
        )
        payload["analogue_job_id"] = job.get("compare_to_job_id")
        return payload

    def queue_classical_analogue(self, job_id: int, rerun: bool = False) -> dict:
        db = self.db()
        job = self.get(job_id)
        if job is None:
            raise ValueError(f"Unknown job {job_id}")
        if job.get("comparison_role") == "baseline":
            raise ValueError("This job is already a classical analogue baseline.")
        if job.get("compare_to_job_id") and not rerun:
            out = self.get(job["id"]) or {}
            comparison_job = self.get(int(job["compare_to_job_id"]))
            if comparison_job:
                out["comparison_job"] = comparison_job
            return out

        dataset = get_dataset(db, job["dataset_name"])
        if dataset is None:
            raise ValueError(f"Unknown dataset '{job['dataset_name']}'")
        corpus_path = str(
            self._confined_data_path(
                dataset["corpus_path"], label="dataset corpus path"
            )
        )
        cfg = self._config_from_job_snapshot(job)
        analogue = self._analogue_for_source(
            job["preset_id"], cfg, source_job_id=int(job["id"])
        )
        if analogue is None:
            raise ValueError("No classical analogue is available for this job.")

        group_id = job.get("group_id") or uuid.uuid4().hex
        twin_name = f"{job['run_name']}-classical"
        batch_size, seq_len = self._train_overrides_from_decoded_job(job)
        twin_cfg = self._config_for_job(
            self._config_from_analogue(analogue), corpus_path, twin_name,
            int(job["seed"]), int(job["steps"]), int(job["eval_every"]),
            batch_size, seq_len,
        )
        self._require_valid_config(twin_cfg, "classical analogue")
        analogue_source_id, _, analogue = self._source_for_analogue(
            db, analogue, job["run_name"], job["preset_id"], int(job["id"])
        )
        twin_config = to_flat_dict(twin_cfg)
        twin_config["lab.config_snapshot_version"] = 1
        twin_config["lab.submission.comparison_mode"] = "post_hoc_baseline"
        twin_config.update({
            key: value
            for key, value in (job.get("config") or {}).items()
            if str(key).startswith("research.")
            or key == "lab.checkpoint_every"
        })
        source_overrides = self._quantum_overrides_from_decoded_job(job) or {}
        if {"n_qubits", "n_circuit_layers"} <= set(source_overrides):
            twin_config["lab.study_cell.n_qubits"] = source_overrides["n_qubits"]
            twin_config["lab.study_cell.n_circuit_layers"] = source_overrides[
                "n_circuit_layers"
            ]
        twin_config["research.seed_axes"] = normalize_seed_axes(
            int(job["seed"]),
            generator_seed=twin_cfg.data.gen_seed,
            data_kind=twin_cfg.data.kind,
            circuit_applicable=uses_quantum_config(twin_cfg),
            explicit=_compatible_seed_request(
                (job.get("config") or {}).get("research.seed_axes"), twin_cfg
            ),
            reject_unsupported=True,
        )
        twin_estimate = quantum_resource_estimate(twin_cfg)
        twin_config["lab.resource.band"] = twin_estimate["band"]
        twin_config["lab.resource.score"] = twin_estimate["score"]
        twin_config = apply_reservation_config(
            twin_config,
            reservation_metadata(job.get("device_target") or "auto", twin_estimate),
        )
        twin_config.update(
            analogue_config_fields(
                analogue,
                state="baseline",
                role="baseline",
                analogue_job_id=int(job["id"]),
            )
        )
        candidate_config = dict(job.get("config") or {})
        candidate_config.setdefault("lab.submission.comparison_mode", "single")
        if {"n_qubits", "n_circuit_layers"} <= set(source_overrides):
            candidate_config["lab.study_cell.n_qubits"] = source_overrides[
                "n_qubits"
            ]
            candidate_config["lab.study_cell.n_circuit_layers"] = (
                source_overrides["n_circuit_layers"]
            )
        candidate_config.update(
            analogue_config_fields(
                analogue,
                state="queued",
                role="candidate",
            )
        )
        twin_run_uuid = str(uuid.uuid4())
        twin_artifact_dir = str(
            self._authorized_artifact_layout(
                self.results_dir / "runs" / twin_run_uuid
            )
        )
        twin_config["lab.artifact_dir"] = twin_artifact_dir
        twin_job_id = db.create_linked_lab_job(int(job["id"]), {
            "status": "queued",
            "preset_id": analogue_source_id,
            "dataset_name": job["dataset_name"],
            "run_name": twin_name,
            "seed": int(job["seed"]),
            "steps": int(job["steps"]),
            "eval_every": int(job["eval_every"]),
            "config": twin_config,
            "group_id": group_id,
            "parent_job_id": int(job["id"]),
            "compare_to_job_id": int(job["id"]),
            "device_target": job.get("device_target") or "auto",
            "comparison_role": "baseline",
            "experiment_uuid": job.get("experiment_uuid"),
            "run_uuid": twin_run_uuid,
            "artifact_dir": twin_artifact_dir,
        }, primary_config=candidate_config, group_id=group_id)
        comparison_job = self.get(twin_job_id)
        self._wake.set()
        out = self.get(int(job["id"])) or {}
        if comparison_job is not None:
            out["comparison_job"] = comparison_job
        return out

    def queue_classical_analogues_for_group(self, group_id: str) -> dict:
        jobs = [
            self._decode(row)
            for row in self.db().fetch_lab_jobs(limit=1000)
            if row.get("group_id") == group_id
        ]
        queued = []
        skipped = []
        for job in jobs:
            if job.get("comparison_role") == "baseline" or job.get("compare_to_job_id"):
                skipped.append({"job_id": job["id"], "reason": "already linked"})
                continue
            try:
                queued.append(self.queue_classical_analogue(int(job["id"])))
            except ValueError as exc:
                skipped.append({"job_id": job["id"], "reason": str(exc)})
        return {
            "group_id": group_id,
            "count": len(queued),
            "jobs": queued,
            "skipped": skipped,
        }

    @staticmethod
    def gpu_ready() -> bool:
        try:
            import jax

            return any(d.platform in ("gpu", "cuda", "rocm", "tpu")
                       for d in jax.devices())
        except Exception:
            return False

    def _config_for_job(self, cfg, corpus_path: str, run_name: str,
                        seed: int, steps: int, eval_every: int,
                        batch_size: int | None = None,
                        seq_len: int | None = None):
        train_updates = {"seed": seed, "steps": steps, "eval_every": eval_every}
        if batch_size is not None:
            train_updates["batch_size"] = int(batch_size)
        if seq_len is not None:
            train_updates["seq_len"] = int(seq_len)
        return dataclasses.replace(
            cfg,
            train=dataclasses.replace(cfg.train, **train_updates),
            data=DataConfig(kind="text", corpus_path=corpus_path),
            tracking=dataclasses.replace(
                cfg.tracking,
                enabled=False,
                run_name=run_name,
                log_quantum_diagnostics=False,
            ),
        )

    @staticmethod
    def _require_valid_config(cfg, label: str) -> None:
        errors = validate_config(cfg)
        if errors:
            raise ValueError(
                f"Invalid {label} config:\n- " + "\n- ".join(errors)
            )

    @staticmethod
    def _config_from_analogue(analogue: AnalogueSpec):
        if analogue.analogue_preset_id:
            return build_preset(analogue.analogue_preset_id)
        if analogue.config is None:
            raise ValueError("Automatic analogue resolver did not return a config.")
        return analogue.config

    def list(self) -> list[dict]:
        return [self._decode(j) for j in self.db().fetch_lab_jobs()]

    def get(self, job_id: int) -> dict | None:
        row = self.db().get_lab_job(job_id)
        return self._decode(row) if row else None

    def cancel(self, job_id: int) -> dict:
        db = self.db()
        job = db.request_cancel_lab_job(job_id)
        if job is None:
            raise ValueError(f"Unknown job {job_id}")
        self._wake.set()
        return self.get(job_id) or {}

    def should_cancel(self, job_id: int) -> bool:
        return self.db().lab_job_cancel_requested(job_id)

    def _run(self) -> None:
        while not self._stop.is_set():
            db = self.db()
            db.recover_stale_lab_jobs(
                checkpoint_resolver=self._recoverable_checkpoint
            )
            job = db.claim_next_lab_job(
                self.worker_id, lease_seconds=self.lease_seconds
            )
            if job is None:
                self._wake.wait(self.poll_seconds)
                self._wake.clear()
                continue
            self._run_claimed(job)

    def _recoverable_checkpoint(self, job: dict) -> dict | None:
        """Resolve only artifacts whose immutable identity matches the job."""
        fresh_allowed = bool(
            int(job.get("completed_step") or 0) == 0
            and not job.get("manifest_hash")
            and not job.get("manifest_json")
            and not job.get("checkpoint_path")
            and not job.get("resume_from")
        )
        try:
            artifact_dir = (
                self._authorized_artifact_layout(str(job["artifact_dir"]))
                if job.get("artifact_dir")
                else None
            )
        except ValueError:
            return None
        candidates: list[Path] = []
        if artifact_dir is not None:
            latest = artifact_dir / "checkpoints" / "latest.msgpack"
            candidates.append(latest)
            best = artifact_dir / "checkpoints" / "best.msgpack"
            if best not in candidates:
                candidates.append(best)
        if job.get("checkpoint_path"):
            checkpoint = Path(str(job["checkpoint_path"]))
            if checkpoint not in candidates:
                candidates.append(checkpoint)
        if job.get("best_checkpoint_path"):
            best_checkpoint = Path(str(job["best_checkpoint_path"]))
            if best_checkpoint not in candidates:
                candidates.append(best_checkpoint)
        own_checkpoints: list[tuple[int, Path]] = []
        for candidate in candidates:
            try:
                candidate = self._confined_path(
                    candidate, label="persisted checkpoint"
                )
                payload = read_checkpoint(candidate)
            except ValueError:
                continue
            manifest = payload["manifest"]
            if manifest.get("run_uuid") != job.get("run_uuid"):
                continue
            if manifest.get("experiment_uuid") != job.get("experiment_uuid"):
                continue
            completed_step = int(payload["completed_step"])
            if completed_step < 0 or completed_step > int(job["steps"]):
                continue
            own_checkpoints.append((completed_step, candidate))
        if own_checkpoints:
            completed_step, candidate = max(
                own_checkpoints, key=lambda item: item[0]
            )
            return {
                "path": str(candidate.resolve()),
                "completed_step": completed_step,
                "fresh": False,
            }
        if job.get("resume_from"):
            try:
                bootstrap = self._confined_path(
                    str(job["resume_from"]), label="persisted resume checkpoint"
                )
            except ValueError:
                return None
            try:
                payload = read_checkpoint(bootstrap)
            except ValueError:
                payload = None
            if payload is not None:
                manifest = payload["manifest"]
                source_run_uuid = manifest.get("run_uuid")
                same_run = (
                    source_run_uuid == job.get("run_uuid")
                    and manifest.get("experiment_uuid")
                    == job.get("experiment_uuid")
                )
                fork_source = bool(
                    job.get("parent_run_uuid")
                    and source_run_uuid == job.get("parent_run_uuid")
                )
                completed_step = int(payload["completed_step"])
                if (
                    (same_run or fork_source)
                    and 0 <= completed_step <= int(job["steps"])
                ):
                    return {
                        "path": str(bootstrap.resolve()),
                        "completed_step": completed_step,
                        "fresh": False,
                    }
            # A persisted resume contract must never degrade silently to a
            # fresh random initialization when its bootstrap is unavailable,
            # corrupt, or bound to another run.
            return None
        if artifact_dir is None or not artifact_dir.exists():
            return (
                {"path": None, "completed_step": 0, "fresh": True}
                if fresh_allowed
                else None
            )
        if not artifact_dir.is_dir():
            return None
        entries = {entry.name for entry in artifact_dir.iterdir()}
        try:
            checkpoints_dir = resolve_within(
                artifact_dir,
                artifact_dir / "checkpoints",
                label="checkpoint directory",
            )
        except ValueError:
            return None
        checkpoint_entries = (
            list(checkpoints_dir.iterdir()) if checkpoints_dir.is_dir() else []
        )
        partial_checkpoint_temps_only = all(
            entry.is_file()
            and entry.name.endswith(".tmp")
            and entry.name.startswith((".latest.msgpack.", ".best.msgpack."))
            for entry in checkpoint_entries
        )
        if not entries:
            return (
                {"path": None, "completed_step": 0, "fresh": True}
                if fresh_allowed
                else None
            )
        if (
            entries <= {"manifest.json", "checkpoints"}
            and "manifest.json" in entries
            and (not checkpoint_entries or partial_checkpoint_temps_only)
        ):
            try:
                manifest_path = resolve_within(
                    artifact_dir,
                    artifact_dir / "manifest.json",
                    label="persisted manifest",
                )
                manifest = validate_manifest(
                    json.loads(manifest_path.read_text("utf-8"))
                )
            except (OSError, json.JSONDecodeError, ValueError):
                return None
            if (
                manifest.get("run_uuid") == job.get("run_uuid")
                and manifest.get("experiment_uuid") == job.get("experiment_uuid")
                and int(job.get("completed_step") or 0) == 0
                and not job.get("manifest_hash")
            ):
                return {"path": None, "completed_step": 0, "fresh": True}
        return None

    def _run_one(self, job_id: int) -> None:
        """Claim and run one job synchronously (tests and local tooling)."""
        db = self.db()
        job = db.get_lab_job(job_id)
        if job is None or job["status"] in ("cancelled", "done", "error"):
            return
        if job["status"] == "queued":
            job = db.claim_lab_job(
                job_id, self.worker_id, lease_seconds=self.lease_seconds
            )
        elif job.get("worker_id") != self.worker_id:
            return
        if job is not None:
            self._run_claimed(job)

    def _run_claimed(self, job: dict) -> None:
        db = self.db()
        job_id = int(job["id"])
        heartbeat_stop = threading.Event()
        ownership_lost = threading.Event()

        heartbeat_thread = threading.Thread(
            target=_lease_heartbeat_loop,
            args=(
                db,
                job_id,
                self.worker_id,
                self.lease_seconds,
                heartbeat_stop,
                ownership_lost,
            ),
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            try:
                existing_config = json.loads(job.get("config_json") or "{}")
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Job config_json is malformed; refusing to replace the raw "
                    "evidence."
                ) from exc
            if not isinstance(existing_config, dict):
                raise ValueError(
                    "Job config_json must decode to an object; raw evidence was preserved."
                )
            required_snapshot_keys = {
                "model.arch",
                "train.seed",
                "train.steps",
                "train.eval_every",
                "data.kind",
                "tracking.run_name",
            }
            has_snapshot = bool(
                existing_config.get("lab.config_snapshot_version")
                or required_snapshot_keys <= set(existing_config)
            )
            device_target = job.get("device_target") or "auto"
            if has_snapshot:
                cfg = config_from_flat_payload(existing_config)
                snapshot_mismatches = []
                for label, snapshot_value, job_value in (
                    ("train.seed", cfg.train.seed, int(job["seed"])),
                    ("train.steps", cfg.train.steps, int(job["steps"])),
                    (
                        "train.eval_every",
                        cfg.train.eval_every,
                        int(job["eval_every"]),
                    ),
                    ("tracking.run_name", cfg.tracking.run_name, job["run_name"]),
                ):
                    if snapshot_value != job_value:
                        snapshot_mismatches.append(
                            f"{label}: snapshot={snapshot_value!r}, job={job_value!r}"
                        )
                if snapshot_mismatches:
                    raise ValueError(
                        "Persisted job config conflicts with immutable queue fields: "
                        + "; ".join(snapshot_mismatches)
                    )
            else:
                dataset = get_dataset(db, job["dataset_name"])
                if dataset is None:
                    raise ValueError(f"Unknown dataset '{job['dataset_name']}'")
                cfg = self._config_for_source(job["preset_id"])
                cfg = apply_quantum_overrides(
                    job["preset_id"],
                    cfg,
                    self._quantum_overrides_from_job(job),
                    allow_gpu_scale=device_target == "gpu",
                )
                cfg = self._config_for_job(
                    cfg,
                    dataset["corpus_path"],
                    job["run_name"],
                    int(job["seed"]),
                    int(job["steps"]),
                    int(job["eval_every"]),
                    *self._train_overrides_from_job(job),
                )
            safe_corpus_path = str(
                self._confined_data_path(
                    cfg.data.corpus_path, label="persisted dataset corpus path"
                )
            )
            cfg = dataclasses.replace(
                cfg,
                data=dataclasses.replace(cfg.data, corpus_path=safe_corpus_path),
            )
            self._require_valid_config(cfg, "persisted job")
            if device_target == "gpu" and not self.gpu_ready():
                raise ValueError(
                    "GPU was requested, but JAX does not currently see a GPU."
                )
            variant = self._variant_for_job(job)
            cfg = with_dashboard(
                cfg,
                suite="lab",
                variant=variant,
                dataset=job["dataset_name"],
                seed=int(job["seed"]),
                db=self.db_path,
            )
            run_key = (
                f"lab/{variant}/{job['dataset_name']}/"
                f"{job['seed']}/{job['steps']}"
            )
            runtime_config = to_flat_dict(cfg)
            if has_snapshot:
                runtime_config = {
                    key: value
                    for key, value in runtime_config.items()
                    if key.startswith("tracking.")
                }
            running_config = {**existing_config, **runtime_config}
            if not job.get("run_uuid") or not job.get("experiment_uuid"):
                raise ValueError("Claimed job is missing durable UUID identity.")
            persisted_resume = job.get("resume_from")
            if persisted_resume:
                persisted_resume = str(
                    self._confined_path(
                        persisted_resume, label="persisted resume checkpoint"
                    )
                )
            artifact_dir = str(
                self._authorized_artifact_layout(
                    job.get("artifact_dir")
                    or self.results_dir / "runs" / str(job["run_uuid"])
                )
            )
            running_config["lab.artifact_dir"] = artifact_dir
            running_config = update_reservation_state(
                running_config, "active", job_id
            )
            if not db.prepare_claimed_lab_job(
                job_id,
                self.worker_id,
                run_key=run_key,
                config=running_config,
                artifact_dir=artifact_dir,
            ):
                ownership_lost.set()
                return
            job = {
                **job,
                "artifact_dir": artifact_dir,
                "resume_from": persisted_resume,
            }
            from ..train.loop import fit

            def on_progress(progress: dict) -> None:
                checkpoint_path = progress.get("checkpoint_path")
                if checkpoint_path:
                    checkpoint_path = str(
                        resolve_within(
                            job["artifact_dir"],
                            checkpoint_path,
                            label="reported checkpoint",
                        )
                    )
                best_checkpoint_path = progress.get("best_checkpoint_path")
                if best_checkpoint_path:
                    best_checkpoint_path = str(
                        resolve_within(
                            job["artifact_dir"],
                            best_checkpoint_path,
                            label="reported best checkpoint",
                        )
                    )
                artifact_dir = (
                    str(
                        resolve_within(
                            job["artifact_dir"],
                            Path(checkpoint_path).parent.parent,
                            label="reported artifact directory",
                        )
                    )
                    if checkpoint_path
                    else None
                )
                if not db.heartbeat_lab_job(
                    job_id,
                    self.worker_id,
                    lease_seconds=self.lease_seconds,
                    completed_step=progress.get("completed_step"),
                    checkpoint_path=checkpoint_path,
                    best_checkpoint_path=best_checkpoint_path,
                    artifact_dir=artifact_dir,
                    manifest=progress.get("manifest"),
                ):
                    ownership_lost.set()

            result = fit(
                cfg,
                verbose=False,
                should_cancel=lambda: (
                    ownership_lost.is_set() or self.should_cancel(job_id)
                ),
                run_options=RunOptions(
                    experiment_uuid=job.get("experiment_uuid"),
                    run_uuid=job.get("run_uuid"),
                    resume_from=job.get("resume_from"),
                    checkpoint_every=self._checkpoint_every_from_job(job),
                    artifact_dir=job.get("artifact_dir"),
                    caller_metadata={
                        "source": "dashboard",
                        "job_id": job_id,
                    },
                    parent_run_uuid=job.get("parent_run_uuid"),
                    seed_axes=self._seed_axes_from_job(job),
                    defer_dashboard_terminal=True,
                    device_target=job.get("device_target") or "auto",
                ),
                progress_callback=on_progress,
                publish_guard=lambda: db.heartbeat_lab_job(
                    job_id,
                    self.worker_id,
                    lease_seconds=self.lease_seconds,
                ),
            )
            if ownership_lost.is_set():
                return
            status = "cancelled" if result["summary"].get("cancelled") else "done"
            done_config = update_reservation_state(running_config, "released")
            summary_checkpoint = result["summary"].get("checkpoint_path")
            summary_best_checkpoint = result["summary"].get("best_checkpoint_path")
            summary_artifact_dir = result["summary"].get("artifact_dir")
            if summary_checkpoint:
                summary_checkpoint = str(
                    resolve_within(
                        job["artifact_dir"],
                        summary_checkpoint,
                        label="final checkpoint",
                    )
                )
            if summary_best_checkpoint:
                summary_best_checkpoint = str(
                    resolve_within(
                        job["artifact_dir"],
                        summary_best_checkpoint,
                        label="final best checkpoint",
                    )
                )
            if summary_artifact_dir:
                summary_artifact_dir = str(
                    resolve_within(
                        job["artifact_dir"],
                        summary_artifact_dir,
                        label="final artifact directory",
                    )
                )
            finished = db.finish_claimed_lab_job(
                job_id,
                self.worker_id,
                status=status,
                config=done_config,
                completed_step=result["summary"].get("completed_step"),
                checkpoint_path=summary_checkpoint,
                best_checkpoint_path=summary_best_checkpoint,
                artifact_dir=summary_artifact_dir,
                manifest=result.get("manifest"),
            )
            if not finished:
                ownership_lost.set()
            else:
                db.finish_run(
                    run_key,
                    status=status,
                    run_uuid=job.get("run_uuid"),
                )
        except Exception as exc:  # pragma: no cover - defensive worker boundary
            latest = db.get_lab_job(job_id)
            try:
                decoded_error_config = json.loads(
                    (latest or job).get("config_json") or "{}"
                )
                error_config = (
                    update_reservation_state(decoded_error_config, "released")
                    if isinstance(decoded_error_config, dict)
                    else None
                )
            except (json.JSONDecodeError, TypeError, ValueError):
                error_config = None
            finished = db.finish_claimed_lab_job(
                job_id,
                self.worker_id,
                status="error",
                config=error_config,
                error=f"{exc}\n{traceback.format_exc(limit=6)}",
            )
            if finished and latest and latest.get("run_key"):
                db.finish_run(
                    latest["run_key"],
                    status="error",
                    run_uuid=latest.get("run_uuid"),
                )
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=max(0.1, min(self.lease_seconds, 1.0)))

    def _decode(self, job: dict) -> dict:
        out = dict(job)
        try:
            out["config"] = json.loads(out.get("config_json") or "{}")
        except json.JSONDecodeError:
            out["config"] = {}
        return out

    def _config_for_source(self, preset_id: str):
        if str(preset_id).startswith("model-spec:"):
            spec_id = int(str(preset_id).split(":", 1)[1])
            spec = self.db().get_model_spec(spec_id)
            if spec is None:
                raise ValueError(f"Unknown model spec {spec_id}")
            return from_dict(spec["config"])
        return build_preset(preset_id)

    def _config_from_job_snapshot(self, job: dict):
        config = job.get("config") or {}
        if config:
            return config_from_flat_payload(config)
        cfg = self._config_for_source(job["preset_id"])
        cfg = apply_quantum_overrides(
            job["preset_id"],
            cfg,
            self._quantum_overrides_from_decoded_job(job),
            allow_gpu_scale=(job.get("device_target") or "auto") == "gpu",
        )
        return cfg

    def _analogue_for_source(
        self,
        preset_id: str,
        cfg,
        *,
        source_job_id: int | None = None,
    ) -> AnalogueSpec | None:
        if str(preset_id).startswith("model-spec:"):
            return classical_analogue_for_config(
                cfg,
                source_preset_id=preset_id,
                source_job_id=source_job_id,
            )
        spec = classical_analogue_for_preset(preset_id)
        if spec and source_job_id is not None:
            return dataclasses.replace(spec, source_job_id=source_job_id)
        return spec

    def _source_for_analogue(
        self,
        db: ResultsDB,
        analogue: AnalogueSpec,
        run_name: str,
        source_id: str,
        source_job_id: int,
    ) -> tuple[str, object, AnalogueSpec]:
        if analogue.analogue_preset_id:
            return analogue.analogue_preset_id, build_preset(analogue.analogue_preset_id), analogue
        if analogue.config is None:
            raise ValueError("Automatic analogue resolver did not return a config.")
        spec = create_spec(db, {
            "name": f"{run_name} classical analogue",
            "source": f"classical-analogue:{source_id}:job:{source_job_id}",
            "parent_id": self._parent_model_spec_id(source_id),
            "version": 1,
            "notes": analogue.reason,
            "config": config_payload(analogue.config),
        })
        analogue = analogue.with_model_spec(spec["id"])
        return f"model-spec:{spec['id']}", analogue.config, analogue

    @staticmethod
    def _parent_model_spec_id(source_id: str) -> int | None:
        if not str(source_id).startswith("model-spec:"):
            return None
        try:
            return int(str(source_id).split(":", 1)[1])
        except ValueError:
            return None

    @staticmethod
    def _quantum_overrides_from_job(job: dict) -> dict[str, int] | None:
        try:
            config = json.loads(job.get("config_json") or "{}")
        except json.JSONDecodeError:
            return None
        overrides = {}
        for key, field in (
            ("lab.quantum_override.n_qubits", "n_qubits"),
            ("lab.quantum_override.n_circuit_layers", "n_circuit_layers"),
        ):
            if key in config:
                overrides[field] = int(config[key])
        return overrides or None

    @staticmethod
    def _quantum_overrides_from_decoded_job(job: dict) -> dict[str, int] | None:
        config = job.get("config") or {}
        overrides = {}
        for key, field in (
            ("lab.quantum_override.n_qubits", "n_qubits"),
            ("lab.quantum_override.n_circuit_layers", "n_circuit_layers"),
        ):
            if key in config:
                overrides[field] = int(config[key])
        return overrides or None

    @staticmethod
    def _train_overrides_from_job(job: dict) -> tuple[int | None, int | None]:
        try:
            config = json.loads(job.get("config_json") or "{}")
        except json.JSONDecodeError:
            return None, None
        batch_size = config.get("lab.train_override.batch_size")
        seq_len = config.get("lab.train_override.seq_len")
        return (
            int(batch_size) if batch_size is not None else None,
            int(seq_len) if seq_len is not None else None,
        )

    @staticmethod
    def _checkpoint_every_from_job(job: dict) -> int:
        try:
            config = json.loads(job.get("config_json") or "{}")
        except json.JSONDecodeError:
            return 0
        return int(config.get("lab.checkpoint_every") or 0)

    @staticmethod
    def _seed_axes_from_job(job: dict) -> dict | None:
        try:
            config = json.loads(job.get("config_json") or "{}")
        except json.JSONDecodeError:
            return None
        axes = config.get("research.seed_axes") or config.get("lab.seed_axes")
        return dict(axes) if isinstance(axes, dict) else None

    @staticmethod
    def _train_overrides_from_decoded_job(job: dict) -> tuple[int | None, int | None]:
        config = job.get("config") or {}
        batch_size = config.get("lab.train_override.batch_size")
        seq_len = config.get("lab.train_override.seq_len")
        return (
            int(batch_size) if batch_size is not None else None,
            int(seq_len) if seq_len is not None else None,
        )

    @staticmethod
    def _variant_for_job(job: dict) -> str:
        variant = job["preset_id"]
        if job.get("comparison_role") in {"control", "frozen_control"}:
            variant = f"{variant}-control"
        overrides = ExperimentQueue._quantum_overrides_from_job(job) or {}
        if not {"n_qubits", "n_circuit_layers"} <= set(overrides):
            try:
                config = json.loads(job.get("config_json") or "{}")
            except json.JSONDecodeError:
                config = {}
            overrides = {
                "n_qubits": config.get("lab.study_cell.n_qubits"),
                "n_circuit_layers": config.get(
                    "lab.study_cell.n_circuit_layers"
                ),
            }
        if {"n_qubits", "n_circuit_layers"} <= set(overrides):
            if overrides["n_qubits"] is None or overrides["n_circuit_layers"] is None:
                return variant
            return (
                f"{variant}-q{overrides['n_qubits']}"
                f"-d{overrides['n_circuit_layers']}"
            )
        return variant
