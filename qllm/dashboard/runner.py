"""Single-worker local experiment queue for QLLM Lab."""
from __future__ import annotations

import dataclasses
import json
import queue
import threading
import traceback
import uuid

from ..config import DataConfig, TrackingConfig, from_dict, to_flat_dict
from ..resultsdb import ResultsDB
from . import with_dashboard
from .analogues import (
    AnalogueSpec,
    analogue_config_fields,
    classical_analogue_for_config,
    classical_analogue_for_preset,
    config_from_flat_payload,
)
from .datasets import get_dataset
from .model_specs import config_payload, create_spec
from .presets import apply_quantum_overrides, build_preset
from .resources import quantum_resource_estimate


class ExperimentQueue:
    def __init__(self, db_path: str = "results/qllm_results.db", start_worker: bool = True):
        self.db_path = db_path
        self._q: queue.Queue[int] = queue.Queue()
        self._cancel: set[int] = set()
        self._lock = threading.Lock()
        self._worker = None
        if start_worker:
            self._worker = threading.Thread(target=self._run, daemon=True)
            self._worker.start()

    def db(self) -> ResultsDB:
        return ResultsDB(self.db_path)

    def submit(
        self, preset_id: str, dataset_name: str, run_name: str | None,
        seed: int, steps: int, eval_every: int,
        device_target: str = "auto",
        queue_classical_comparison: bool = False,
        quantum_overrides: dict | None = None,
        group_id: str | None = None,
        batch_size: int | None = None,
        seq_len: int | None = None,
    ) -> dict:
        db = self.db()
        dataset = get_dataset(db, dataset_name)
        if dataset is None:
            raise ValueError(f"Unknown dataset '{dataset_name}'")
        device_target = (device_target or "auto").strip().lower()
        if device_target not in ("auto", "cpu", "gpu"):
            raise ValueError("Device target must be one of: auto, cpu, gpu.")
        if device_target == "gpu" and not self.gpu_ready():
            raise ValueError(
                "GPU was requested, but JAX does not currently see a GPU. "
                "Open the GPU page and follow the CUDA/JAX setup guidance."
            )
        allow_gpu_scale = device_target == "gpu"
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
        if seq_len is not None and int(seq_len) < 8:
            raise ValueError("Sequence length must be at least 8.")
        seed = int(seed)
        analogue = self._analogue_for_source(preset_id, cfg)
        wants_comparison = bool(queue_classical_comparison and analogue)
        cfg = self._config_for_job(
            cfg, dataset["corpus_path"], run_name,
            seed, steps, eval_every, batch_size, seq_len,
        )
        estimate = quantum_resource_estimate(cfg)
        if estimate["band"] == "extreme" and estimate["uses_quantum_attention"]:
            raise ValueError(
                "This quantum-attention run is likely to exhaust GPU memory. "
                "Try batch size 1-4, seq_len 16-32, or reduce qubits/depth. "
                f"Current estimate: {estimate['band']}."
            )
        job_config = to_flat_dict(cfg)
        if quantum_overrides:
            job_config["lab.quantum_override.n_qubits"] = cfg.model.quantum.n_qubits
            job_config["lab.quantum_override.n_circuit_layers"] = (
                cfg.model.quantum.n_circuit_layers
            )
        if batch_size is not None:
            job_config["lab.train_override.batch_size"] = cfg.train.batch_size
        if seq_len is not None:
            job_config["lab.train_override.seq_len"] = cfg.train.seq_len
        job_config["lab.resource.band"] = estimate["band"]
        job_config["lab.resource.score"] = estimate["score"]
        job_config.update(
            analogue_config_fields(
                analogue,
                state="queued" if wants_comparison else "missing",
                role="candidate" if wants_comparison else "candidate_missing_baseline",
            )
        )
        group_id = group_id or uuid.uuid4().hex
        job_id = db.create_lab_job({
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
        })
        comparison_job = None
        if wants_comparison and analogue:
            analogue_source_id, twin_cfg, analogue = self._source_for_analogue(
                db, analogue, run_name, preset_id, job_id
            )
            twin_name = f"{run_name}-classical"
            twin_cfg = self._config_for_job(
                twin_cfg, dataset["corpus_path"], twin_name,
                seed, steps, eval_every, batch_size, seq_len)
            twin_config = to_flat_dict(twin_cfg)
            twin_config.update(
                analogue_config_fields(
                    analogue,
                    state="baseline",
                    role="baseline",
                    analogue_job_id=job_id,
                )
            )
            twin_job_id = db.create_lab_job({
                "status": "queued",
                "preset_id": analogue_source_id,
                "dataset_name": dataset_name,
                "run_name": twin_name,
                "seed": seed,
                "steps": steps,
                "eval_every": eval_every,
                "config": twin_config,
                "group_id": group_id,
                "parent_job_id": job_id,
                "compare_to_job_id": job_id,
                "device_target": device_target,
                "comparison_role": "baseline",
            })
            job_config.update(
                analogue_config_fields(
                    analogue,
                    state="queued",
                    role="candidate",
                    analogue_job_id=twin_job_id,
                )
            )
            db.update_lab_job(job_id, compare_to_job_id=twin_job_id, config=job_config)
            comparison_job = self.get(twin_job_id)
        self._q.put(job_id)
        if comparison_job is not None:
            self._q.put(comparison_job["id"])
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
    ) -> dict:
        if not qubits:
            raise ValueError("At least one qubit count is required.")
        if not depths:
            raise ValueError("At least one circuit depth is required.")
        if len(qubits) * len(depths) > 64:
            raise ValueError("Scaling sweeps are capped at 64 jobs per batch.")
        group_id = uuid.uuid4().hex
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
    ) -> dict:
        spec = self.db().get_model_spec(spec_id)
        if spec is None:
            raise ValueError(f"Unknown model spec {spec_id}")
        return self.submit(
            preset_id=f"model-spec:{spec_id}",
            dataset_name=dataset_name,
            run_name=run_name or spec["name"],
            seed=seed,
            steps=steps,
            eval_every=eval_every,
            device_target=device_target,
            queue_classical_comparison=queue_classical_comparison,
            batch_size=batch_size,
            seq_len=seq_len,
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
        cfg = self._config_from_job_snapshot(job)
        analogue = self._analogue_for_source(
            job["preset_id"], cfg, source_job_id=int(job["id"])
        )
        if analogue is None:
            raise ValueError("No classical analogue is available for this job.")

        group_id = job.get("group_id") or uuid.uuid4().hex
        analogue_source_id, twin_cfg, analogue = self._source_for_analogue(
            db, analogue, job["run_name"], job["preset_id"], int(job["id"])
        )
        twin_name = f"{job['run_name']}-classical"
        batch_size, seq_len = self._train_overrides_from_decoded_job(job)
        twin_cfg = self._config_for_job(
            twin_cfg, dataset["corpus_path"], twin_name,
            int(job["seed"]), int(job["steps"]), int(job["eval_every"]),
            batch_size, seq_len,
        )
        twin_config = to_flat_dict(twin_cfg)
        twin_config.update(
            analogue_config_fields(
                analogue,
                state="baseline",
                role="baseline",
                analogue_job_id=int(job["id"]),
            )
        )
        twin_job_id = db.create_lab_job({
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
        })

        candidate_config = dict(job.get("config") or {})
        candidate_config.update(
            analogue_config_fields(
                analogue,
                state="queued",
                role="candidate",
                analogue_job_id=twin_job_id,
            )
        )
        db.update_lab_job(
            int(job["id"]),
            group_id=group_id,
            compare_to_job_id=twin_job_id,
            comparison_role="candidate",
            config=candidate_config,
        )
        comparison_job = self.get(twin_job_id)
        self._q.put(twin_job_id)
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

    def list(self) -> list[dict]:
        return [self._decode(j) for j in self.db().fetch_lab_jobs()]

    def get(self, job_id: int) -> dict | None:
        row = self.db().get_lab_job(job_id)
        return self._decode(row) if row else None

    def cancel(self, job_id: int) -> dict:
        db = self.db()
        job = db.get_lab_job(job_id)
        if job is None:
            raise ValueError(f"Unknown job {job_id}")
        if job["status"] in ("done", "error", "cancelled"):
            return self._decode(job)
        with self._lock:
            self._cancel.add(job_id)
        if job["status"] == "queued":
            db.update_lab_job(job_id, status="cancelled", error="Cancelled before start.")
        return self.get(job_id) or {}

    def should_cancel(self, job_id: int) -> bool:
        with self._lock:
            return job_id in self._cancel

    def _run(self) -> None:
        while True:
            job_id = self._q.get()
            try:
                self._run_one(job_id)
            finally:
                self._q.task_done()

    def _run_one(self, job_id: int) -> None:
        db = self.db()
        job = db.get_lab_job(job_id)
        if job is None or job["status"] == "cancelled":
            return
        try:
            dataset = get_dataset(db, job["dataset_name"])
            if dataset is None:
                raise ValueError(f"Unknown dataset '{job['dataset_name']}'")
            cfg = self._config_for_source(job["preset_id"])
            device_target = job.get("device_target") or "auto"
            cfg = apply_quantum_overrides(
                job["preset_id"], cfg, self._quantum_overrides_from_job(job),
                allow_gpu_scale=device_target == "gpu",
            )
            if device_target == "gpu" and not self.gpu_ready():
                raise ValueError(
                    "GPU was requested, but JAX does not currently see a GPU."
                )
            cfg = self._config_for_job(
                cfg, dataset["corpus_path"], job["run_name"],
                int(job["seed"]), int(job["steps"]), int(job["eval_every"]),
                *self._train_overrides_from_job(job),
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
            db.update_lab_job(job_id, status="running", run_key=run_key,
                              config=to_flat_dict(cfg), error=None)
            from ..train.loop import fit

            result = fit(
                cfg,
                verbose=False,
                should_cancel=lambda: self.should_cancel(job_id),
            )
            status = "cancelled" if result["summary"].get("cancelled") else "done"
            db.update_lab_job(job_id, status=status)
        except Exception as exc:  # pragma: no cover - defensive worker boundary
            latest = db.get_lab_job(job_id)
            if latest and latest.get("run_key"):
                try:
                    db.finish_run(latest["run_key"], status="error")
                except Exception:
                    pass
            db.update_lab_job(
                job_id,
                status="error",
                error=f"{exc}\n{traceback.format_exc(limit=6)}",
            )

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
        overrides = ExperimentQueue._quantum_overrides_from_job(job) or {}
        if {"n_qubits", "n_circuit_layers"} <= set(overrides):
            return (
                f"{job['preset_id']}-q{overrides['n_qubits']}"
                f"-d{overrides['n_circuit_layers']}"
            )
        return job["preset_id"]
