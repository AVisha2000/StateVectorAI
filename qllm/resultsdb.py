"""Results database: one row per benchmark run, SQLite, stdlib-only.

MLflow (sqlite:///mlflow.db) remains the per-step tracker; this is the
clean cross-suite results table that makes comparison plots and
resumability trivial: a (suite, variant, dataset, seed, steps) key is
unique, so re-running a suite skips finished work and a crashed/timed-out
sweep continues where it stopped — on this machine or on a GPU box later.
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Callable

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    suite TEXT NOT NULL,
    variant TEXT NOT NULL,
    dataset TEXT NOT NULL,
    seed INTEGER NOT NULL,
    steps INTEGER NOT NULL,
    n_params INTEGER NOT NULL,
    val_loss REAL,
    val_ppl REAL,
    val_bpc REAL,
    wall_seconds REAL,
    config_json TEXT,
    UNIQUE(suite, variant, dataset, seed, steps)
);
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    suite TEXT NOT NULL,
    variant TEXT NOT NULL,
    dataset TEXT NOT NULL,
    seed INTEGER NOT NULL,
    name TEXT NOT NULL,
    value REAL,
    UNIQUE(suite, variant, dataset, seed, name)
);
-- per-step training curves (replaces MLflow's role)
CREATE TABLE IF NOT EXISTS steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_key TEXT NOT NULL,
    step INTEGER NOT NULL,
    name TEXT NOT NULL,
    value REAL
);
CREATE INDEX IF NOT EXISTS idx_steps_run ON steps(run_key);
-- Canonical idempotent step store.  The legacy ``steps`` table is never
-- rewritten because historical databases may contain duplicate observations.
CREATE TABLE IF NOT EXISTS run_steps (
    run_uuid TEXT NOT NULL,
    run_key TEXT NOT NULL,
    step INTEGER NOT NULL,
    name TEXT NOT NULL,
    value REAL,
    ts TEXT NOT NULL,
    PRIMARY KEY(run_uuid, step, name)
);
CREATE INDEX IF NOT EXISTS idx_run_steps_key ON run_steps(run_key, step);
CREATE TABLE IF NOT EXISTS run_manifests (
    run_uuid TEXT PRIMARY KEY,
    experiment_uuid TEXT NOT NULL,
    manifest_hash TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    completed_step INTEGER NOT NULL DEFAULT 0,
    checkpoint_path TEXT,
    best_checkpoint_path TEXT,
    ts TEXT NOT NULL,
    updated_ts TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS run_results (
    run_uuid TEXT PRIMARY KEY,
    experiment_uuid TEXT NOT NULL,
    ts TEXT NOT NULL,
    suite TEXT NOT NULL,
    variant TEXT NOT NULL,
    dataset TEXT NOT NULL,
    seed INTEGER NOT NULL,
    steps INTEGER NOT NULL,
    n_params INTEGER NOT NULL,
    val_loss REAL,
    val_ppl REAL,
    val_bpc REAL,
    wall_seconds REAL,
    config_json TEXT,
    manifest_hash TEXT
);
-- live registry: a row per in-flight (or finished) training run, so the
-- dashboard can show progress while sweeps fill the DB.
CREATE TABLE IF NOT EXISTS live_runs (
    run_key TEXT PRIMARY KEY,
    run_name TEXT,
    suite TEXT, variant TEXT, dataset TEXT, seed INTEGER,
    total_steps INTEGER,
    current_step INTEGER,
    status TEXT,            -- running | done | error
    started_ts TEXT,
    updated_ts TEXT,
    last_train_loss REAL,
    last_val_ppl REAL,
    config_json TEXT
);
CREATE TABLE IF NOT EXISTS lab_datasets (
    name TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    source TEXT NOT NULL,
    split TEXT,
    text_column TEXT,
    corpus_path TEXT NOT NULL,
    n_rows INTEGER,
    n_chars INTEGER,
    preview TEXT,
    requested_revision TEXT,
    resolved_fingerprint TEXT,
    revision_applicable INTEGER,
    row_limit INTEGER,
    char_limit INTEGER,
    byte_limit INTEGER,
    rows_examined INTEGER,
    n_bytes INTEGER,
    sha256 TEXT,
    truncated INTEGER DEFAULT 0,
    truncation_reason TEXT,
    warnings_json TEXT,
    ts TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS lab_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    updated_ts TEXT NOT NULL,
    status TEXT NOT NULL,
    preset_id TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    run_name TEXT NOT NULL,
    seed INTEGER NOT NULL,
    steps INTEGER NOT NULL,
    eval_every INTEGER NOT NULL,
    run_key TEXT,
    error TEXT,
    config_json TEXT,
    group_id TEXT,
    parent_job_id INTEGER,
    compare_to_job_id INTEGER,
    device_target TEXT DEFAULT 'auto',
    comparison_role TEXT DEFAULT 'primary',
    experiment_uuid TEXT,
    run_uuid TEXT,
    parent_run_uuid TEXT,
    manifest_hash TEXT,
    manifest_json TEXT,
    worker_id TEXT,
    claimed_ts REAL,
    heartbeat_ts REAL,
    lease_expires_ts REAL,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    recovery_count INTEGER NOT NULL DEFAULT 0,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    cancel_requested_ts REAL,
    resume_from TEXT,
    checkpoint_path TEXT,
    best_checkpoint_path TEXT,
    artifact_dir TEXT,
    completed_step INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS model_specs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    updated_ts TEXT NOT NULL,
    name TEXT NOT NULL,
    source TEXT,
    parent_id INTEGER,
    version INTEGER NOT NULL DEFAULT 1,
    notes TEXT,
    config_json TEXT NOT NULL,
    graph_json TEXT
);
CREATE TABLE IF NOT EXISTS studies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    updated_ts TEXT NOT NULL,
    name TEXT NOT NULL,
    research_question TEXT,
    task TEXT,
    description TEXT,
    dataset_names_json TEXT NOT NULL,
    candidate_preset_id TEXT NOT NULL,
    baseline_policy TEXT NOT NULL DEFAULT 'analogue',
    control_preset_ids_json TEXT,
    seeds_json TEXT NOT NULL,
    sweep_json TEXT,
    status TEXT NOT NULL DEFAULT 'draft',
    group_id TEXT NOT NULL,
    protocol_json TEXT
);
CREATE TABLE IF NOT EXISTS study_jobs (
    study_id INTEGER NOT NULL,
    job_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    sweep_json TEXT,
    PRIMARY KEY(study_id, job_id)
);
"""


class ResultsDB:
    def __init__(self, path: str | Path = "results/qllm_results.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as con:
            con.executescript(_SCHEMA)
            self._ensure_lab_dataset_columns(con)
            self._ensure_lab_job_columns(con)
            self._ensure_run_identity_columns(con)
            con.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_lab_jobs_run_uuid "
                "ON lab_jobs(run_uuid) WHERE run_uuid IS NOT NULL"
            )
            con.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_runs_run_uuid "
                "ON runs(run_uuid) WHERE run_uuid IS NOT NULL"
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_lab_jobs_status_id "
                "ON lab_jobs(status, id)"
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_lab_jobs_status_lease "
                "ON lab_jobs(status, lease_expires_ts)"
            )

    def _ensure_lab_dataset_columns(self, con: sqlite3.Connection) -> None:
        """Add import-provenance fields without rewriting existing rows."""
        cols = {
            r["name"] for r in con.execute("PRAGMA table_info(lab_datasets)").fetchall()
        }
        additions = {
            "requested_revision": "TEXT",
            "resolved_fingerprint": "TEXT",
            "revision_applicable": "INTEGER",
            "row_limit": "INTEGER",
            "char_limit": "INTEGER",
            "byte_limit": "INTEGER",
            "rows_examined": "INTEGER",
            "n_bytes": "INTEGER",
            "sha256": "TEXT",
            "truncated": "INTEGER DEFAULT 0",
            "truncation_reason": "TEXT",
            "warnings_json": "TEXT",
        }
        for name, ddl in additions.items():
            if name not in cols:
                con.execute(f"ALTER TABLE lab_datasets ADD COLUMN {name} {ddl}")

    def _ensure_lab_job_columns(self, con: sqlite3.Connection) -> None:
        cols = {r["name"] for r in con.execute("PRAGMA table_info(lab_jobs)").fetchall()}
        additions = {
            "group_id": "TEXT",
            "parent_job_id": "INTEGER",
            "compare_to_job_id": "INTEGER",
            "device_target": "TEXT DEFAULT 'auto'",
            "comparison_role": "TEXT DEFAULT 'primary'",
            "experiment_uuid": "TEXT",
            "run_uuid": "TEXT",
            "parent_run_uuid": "TEXT",
            "manifest_hash": "TEXT",
            "manifest_json": "TEXT",
            "worker_id": "TEXT",
            "claimed_ts": "REAL",
            "heartbeat_ts": "REAL",
            "lease_expires_ts": "REAL",
            "attempt_count": "INTEGER NOT NULL DEFAULT 0",
            "recovery_count": "INTEGER NOT NULL DEFAULT 0",
            "cancel_requested": "INTEGER NOT NULL DEFAULT 0",
            "cancel_requested_ts": "REAL",
            "resume_from": "TEXT",
            "checkpoint_path": "TEXT",
            "best_checkpoint_path": "TEXT",
            "artifact_dir": "TEXT",
            "completed_step": "INTEGER NOT NULL DEFAULT 0",
        }
        for name, ddl in additions.items():
            if name not in cols:
                con.execute(f"ALTER TABLE lab_jobs ADD COLUMN {name} {ddl}")

    def _ensure_run_identity_columns(self, con: sqlite3.Connection) -> None:
        additions_by_table = {
            "runs": {
                "experiment_uuid": "TEXT",
                "run_uuid": "TEXT",
                "manifest_hash": "TEXT",
            },
            "live_runs": {
                "experiment_uuid": "TEXT",
                "run_uuid": "TEXT",
                "manifest_hash": "TEXT",
            },
        }
        for table, additions in additions_by_table.items():
            cols = {
                r["name"]
                for r in con.execute(f"PRAGMA table_info({table})").fetchall()
            }
            for name, ddl in additions.items():
                if name not in cols:
                    con.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")

    def _conn(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=30.0)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA busy_timeout=30000")
        return con

    def exists(
        self, suite: str, variant: str, dataset: str, seed: int, steps: int
    ) -> bool:
        with self._conn() as con:
            row = con.execute(
                "SELECT 1 FROM runs WHERE suite=? AND variant=? AND dataset=? "
                "AND seed=? AND steps=?",
                (suite, variant, dataset, seed, steps),
            ).fetchone()
        return row is not None

    def record(
        self,
        suite: str,
        variant: str,
        dataset: str,
        seed: int,
        steps: int,
        n_params: int,
        val_loss: float,
        val_ppl: float,
        val_bpc: float,
        wall_seconds: float,
        config: dict | None = None,
        run_uuid: str | None = None,
        experiment_uuid: str | None = None,
        manifest_hash: str | None = None,
        manifest: dict | None = None,
        finalize_manifest: bool = True,
    ) -> dict:
        if manifest is None and run_uuid is not None:
            existing_manifest = self.get_run_manifest(run_uuid)
            if existing_manifest is not None:
                manifest = existing_manifest["manifest"]
        if manifest is None:
            from .train.artifacts import build_record_manifest

            manifest = build_record_manifest(
                suite=suite,
                variant=variant,
                dataset=dataset,
                seed=seed,
                steps=steps,
                config=config,
                experiment_uuid=experiment_uuid,
                run_uuid=run_uuid,
            )
        manifest_run_uuid = str(manifest.get("run_uuid") or "")
        manifest_experiment_uuid = str(manifest.get("experiment_uuid") or "")
        manifest_identity_hash = str(manifest.get("manifest_hash") or "")
        if not manifest_run_uuid or not manifest_experiment_uuid or not manifest_identity_hash:
            raise ValueError("Result manifest is missing immutable identity fields.")
        if run_uuid is not None and str(run_uuid) != manifest_run_uuid:
            raise ValueError("run_uuid does not match the supplied result manifest.")
        if (
            experiment_uuid is not None
            and str(experiment_uuid) != manifest_experiment_uuid
        ):
            raise ValueError(
                "experiment_uuid does not match the supplied result manifest."
            )
        if manifest_hash is not None and str(manifest_hash) != manifest_identity_hash:
            raise ValueError("manifest_hash does not match the supplied manifest.")
        run_uuid = manifest_run_uuid
        experiment_uuid = manifest_experiment_uuid
        manifest_hash = manifest_identity_hash
        self.register_run_manifest(manifest)
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        config_json = json.dumps(
            {k: str(v) for k, v in (config or {}).items()},
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._conn() as con:
            con.execute("BEGIN IMMEDIATE")
            if run_uuid is not None:
                if experiment_uuid is None:
                    raise ValueError("experiment_uuid is required with run_uuid.")
                existing_result = con.execute(
                    "SELECT experiment_uuid, suite, variant, dataset, seed, steps, "
                    "n_params, val_loss, val_ppl, val_bpc, config_json, manifest_hash "
                    "FROM run_results WHERE run_uuid=?", (run_uuid,)
                ).fetchone()
                identity = (
                    experiment_uuid, suite, variant, dataset, int(seed), int(steps)
                )
                if existing_result is not None:
                    if tuple(existing_result[:6]) != identity:
                        raise ValueError(
                            "Conflicting immutable result identity for run_uuid "
                            f"{run_uuid}."
                        )
                    same_evidence = (
                        int(existing_result["n_params"]) == int(n_params)
                        and self._same_metric(existing_result["val_loss"], val_loss)
                        and self._same_metric(existing_result["val_ppl"], val_ppl)
                        and self._same_metric(existing_result["val_bpc"], val_bpc)
                        and existing_result["config_json"] == config_json
                        and existing_result["manifest_hash"] == manifest_hash
                    )
                    if not same_evidence:
                        raise ValueError(
                            "Conflicting final result evidence for immutable "
                            f"run_uuid {run_uuid}."
                        )
                    # Result publication can precede the queue's terminal
                    # transition. A recovery retry preserves the first wall
                    # time and evidence instead of rewriting it.
                else:
                    con.execute(
                        "INSERT INTO run_results "
                        "(run_uuid, experiment_uuid, ts, suite, variant, dataset, "
                        "seed, steps, n_params, val_loss, val_ppl, val_bpc, "
                        "wall_seconds, config_json, manifest_hash) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            run_uuid,
                            experiment_uuid,
                            now,
                            suite,
                            variant,
                            dataset,
                            seed,
                            steps,
                            n_params,
                            val_loss,
                            val_ppl,
                            val_bpc,
                            wall_seconds,
                            config_json,
                            manifest_hash,
                        ),
                    )
            legacy = con.execute(
                "SELECT run_uuid FROM runs WHERE suite=? AND variant=? AND dataset=? "
                "AND seed=? AND steps=?",
                (suite, variant, dataset, seed, steps),
            ).fetchone()
            legacy_values = (
                now,
                suite,
                variant,
                dataset,
                seed,
                steps,
                n_params,
                val_loss,
                val_ppl,
                val_bpc,
                wall_seconds,
                config_json,
                run_uuid,
                experiment_uuid,
                manifest_hash,
            )
            if legacy is None:
                con.execute(
                    "INSERT INTO runs "
                    "(ts, suite, variant, dataset, seed, steps, n_params, "
                    " val_loss, val_ppl, val_bpc, wall_seconds, config_json, "
                    " run_uuid, experiment_uuid, manifest_hash) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    legacy_values,
                )
            # A pre-M05 NULL-UUID row or another UUID-backed projection is
            # immutable historical evidence. Same-UUID retries are also a
            # no-op: canonical and projected evidence keep their first values.
            if finalize_manifest:
                con.execute(
                    "UPDATE run_manifests SET status='done', completed_step=?, "
                    "updated_ts=? "
                    "WHERE run_uuid=?",
                    (int(steps), now, run_uuid),
                )
        return {
            "experiment_uuid": experiment_uuid,
            "run_uuid": run_uuid,
            "manifest_hash": manifest_hash,
            "manifest": manifest,
        }

    def fetch(self, suite: str, dataset: str | None = None) -> list[dict]:
        query = "SELECT * FROM runs WHERE suite=?"
        args: list = [suite]
        if dataset is not None:
            query += " AND dataset=?"
            args.append(dataset)
        with self._conn() as con:
            rows = con.execute(query + " ORDER BY variant, seed", args).fetchall()
        return [dict(r) for r in rows]

    def get_run(
        self, suite: str, variant: str, dataset: str, seed: int, steps: int,
        run_uuid: str | None = None,
    ) -> dict | None:
        with self._conn() as con:
            if run_uuid is not None:
                row = con.execute(
                    "SELECT rr.*, r.id AS id FROM run_results AS rr LEFT JOIN "
                    "runs AS r ON r.run_uuid=rr.run_uuid AND r.suite=rr.suite "
                    "AND r.variant=rr.variant AND r.dataset=rr.dataset "
                    "AND r.seed=rr.seed AND r.steps=rr.steps WHERE rr.run_uuid=? "
                    "AND rr.suite=? AND rr.variant=? AND rr.dataset=? "
                    "AND rr.seed=? AND rr.steps=?",
                    (run_uuid, suite, variant, dataset, seed, steps),
                ).fetchone()
            else:
                row = con.execute(
                    "SELECT * FROM runs WHERE suite=? AND variant=? AND dataset=? "
                    "AND seed=? AND steps=?",
                    (suite, variant, dataset, seed, steps),
                ).fetchone()
        return dict(row) if row is not None else None

    # ---- live-run registry + per-step logging (own MLflow replacement) ----
    def start_run(self, run_key: str, run_name: str, suite: str, variant: str,
                  dataset: str, seed: int, total_steps: int,
                  config: dict | None = None, *, run_uuid: str | None = None,
                  experiment_uuid: str | None = None,
                  manifest: dict | None = None) -> None:
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        if run_uuid is not None and manifest is not None:
            self.register_run_manifest(manifest)
        with self._conn() as con:
            con.execute(
                "INSERT INTO live_runs (run_key, run_name, suite, "
                "variant, dataset, seed, total_steps, current_step, status, "
                "started_ts, updated_ts, config_json, run_uuid, experiment_uuid, "
                "manifest_hash) VALUES "
                "(?,?,?,?,?,?,?,0,'running',?,?,?,?,?,?) "
                "ON CONFLICT(run_key) DO UPDATE SET run_name=excluded.run_name, "
                "suite=excluded.suite, variant=excluded.variant, "
                "dataset=excluded.dataset, seed=excluded.seed, "
                "total_steps=excluded.total_steps, status='running', "
                "current_step=CASE WHEN live_runs.run_uuid=excluded.run_uuid "
                "THEN live_runs.current_step ELSE 0 END, "
                "started_ts=CASE WHEN live_runs.run_uuid=excluded.run_uuid "
                "THEN live_runs.started_ts ELSE excluded.started_ts END, "
                "updated_ts=excluded.updated_ts, config_json=excluded.config_json, "
                "run_uuid=excluded.run_uuid, "
                "experiment_uuid=excluded.experiment_uuid, "
                "manifest_hash=excluded.manifest_hash",
                (run_key, run_name, suite, variant, dataset, seed, total_steps,
                 now, now, json.dumps({k: str(v) for k, v in (config or {}).items()}),
                 run_uuid, experiment_uuid,
                 (manifest or {}).get("manifest_hash")))

    def log_step(self, run_key: str, step: int, metrics: dict[str, float],
                 train_loss: float | None = None,
                 val_ppl: float | None = None, *,
                 run_uuid: str | None = None) -> None:
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        with self._conn() as con:
            effective_uuid = run_uuid
            if effective_uuid is None:
                row = con.execute(
                    "SELECT run_uuid FROM live_runs WHERE run_key=?", (run_key,)
                ).fetchone()
                effective_uuid = row["run_uuid"] if row is not None else None
            if effective_uuid is None:
                # Legacy callers retain append-only behavior.  UUID-backed runs
                # use the canonical idempotent table below.
                con.executemany(
                    "INSERT INTO steps (run_key, step, name, value) VALUES (?,?,?,?)",
                    [(run_key, step, k, float(v)) for k, v in metrics.items()])
            else:
                for name, value in metrics.items():
                    numeric = float(value)
                    inserted = con.execute(
                        "INSERT INTO run_steps "
                        "(run_uuid, run_key, step, name, value, ts) "
                        "VALUES (?,?,?,?,?,?) ON CONFLICT(run_uuid, step, name) "
                        "DO NOTHING",
                        (
                            effective_uuid,
                            run_key,
                            int(step),
                            str(name),
                            numeric,
                            now,
                        ),
                    ).rowcount
                    if inserted == 0:
                        existing = con.execute(
                            "SELECT value FROM run_steps WHERE run_uuid=? AND step=? "
                            "AND name=?",
                            (effective_uuid, int(step), str(name)),
                        ).fetchone()
                        if existing is None:  # defensive transaction invariant
                            raise RuntimeError("Idempotent step row disappeared.")
                    else:
                        existing = None
                    if existing is not None and not self._same_metric(
                        existing["value"], numeric
                    ):
                        raise ValueError(
                            "Conflicting retry for step metric "
                            f"{effective_uuid}/{step}/{name}: "
                            f"{existing['value']} != {numeric}"
                        )
            if effective_uuid is None:
                con.execute(
                    "UPDATE live_runs SET current_step=MAX(COALESCE(current_step, 0), ?), "
                    "updated_ts=?, "
                    "last_train_loss=CASE WHEN COALESCE(current_step, 0)<=? THEN "
                    "COALESCE(?, last_train_loss) ELSE last_train_loss END, "
                    "last_val_ppl=CASE WHEN COALESCE(current_step, 0)<=? THEN "
                    "COALESCE(?, last_val_ppl) ELSE last_val_ppl END WHERE run_key=? "
                    "AND run_uuid IS NULL",
                    (step, now, step, train_loss, step, val_ppl, run_key),
                )
            else:
                con.execute(
                    "UPDATE live_runs SET current_step=MAX(COALESCE(current_step, 0), ?), "
                    "updated_ts=?, "
                    "last_train_loss=CASE WHEN COALESCE(current_step, 0)<=? THEN "
                    "COALESCE(?, last_train_loss) ELSE last_train_loss END, "
                    "last_val_ppl=CASE WHEN COALESCE(current_step, 0)<=? THEN "
                    "COALESCE(?, last_val_ppl) ELSE last_val_ppl END WHERE run_key=? "
                    "AND run_uuid=?",
                    (
                        step,
                        now,
                        step,
                        train_loss,
                        step,
                        val_ppl,
                        run_key,
                        effective_uuid,
                    ),
                )
            if effective_uuid is not None:
                con.execute(
                    "UPDATE run_manifests SET completed_step=MAX(completed_step, ?), "
                    "updated_ts=? WHERE run_uuid=?",
                    (int(step), now, effective_uuid),
                )

    @staticmethod
    def _same_metric(left, right) -> bool:
        if left is None:
            return right is None or math.isnan(float(right))
        if right is None:
            return math.isnan(float(left))
        return float(left) == float(right) or (
            math.isnan(float(left)) and math.isnan(float(right))
        )

    def finish_run(
        self, run_key: str, status: str = "done", *, run_uuid: str | None = None
    ) -> None:
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        with self._conn() as con:
            if run_uuid is None:
                con.execute(
                    "UPDATE live_runs SET status=?, updated_ts=? WHERE run_key=?",
                    (status, now, run_key),
                )
                con.execute(
                    "UPDATE run_manifests SET status=?, updated_ts=? WHERE run_uuid "
                    "IN (SELECT run_uuid FROM live_runs WHERE run_key=?)",
                    (status, now, run_key),
                )
            else:
                con.execute(
                    "UPDATE live_runs SET status=?, updated_ts=? WHERE run_key=? "
                    "AND run_uuid=?",
                    (status, now, run_key, run_uuid),
                )
                con.execute(
                    "UPDATE run_manifests SET status=?, updated_ts=? WHERE run_uuid=?",
                    (status, now, run_uuid),
                )

    def fetch_steps(self, run_key: str, run_uuid: str | None = None) -> list[dict]:
        with self._conn() as con:
            legacy = con.execute(
                "SELECT step, name, value FROM steps WHERE run_key=? "
                "ORDER BY step, id", (run_key,)).fetchall()
            effective_uuid = run_uuid
            if effective_uuid is None:
                active = con.execute(
                    "SELECT run_uuid FROM live_runs WHERE run_key=?", (run_key,)
                ).fetchone()
                effective_uuid = active["run_uuid"] if active is not None else None
            canonical = (
                con.execute(
                    "SELECT step, name, value FROM run_steps WHERE run_uuid=? "
                    "ORDER BY step, name", (effective_uuid,)
                ).fetchall()
                if effective_uuid is not None
                else []
            )
        if effective_uuid is None:
            # Preserve the pre-M05 append-only read contract, including
            # duplicate observations and insertion order.
            return [dict(row) for row in legacy]
        return [dict(row) for row in canonical]

    def register_run_manifest(self, manifest: dict) -> None:
        from .train.artifacts import validate_manifest

        manifest = validate_manifest(manifest)
        run_uuid = str(manifest.get("run_uuid") or "")
        experiment_uuid = str(manifest.get("experiment_uuid") or "")
        manifest_hash = str(manifest.get("manifest_hash") or "")
        if not run_uuid or not experiment_uuid or not manifest_hash:
            raise ValueError("Run manifest requires run/experiment UUIDs and hash.")
        body = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        with self._conn() as con:
            existing = con.execute(
                "SELECT manifest_hash, manifest_json FROM run_manifests "
                "WHERE run_uuid=?", (run_uuid,)
            ).fetchone()
            if existing is not None:
                if (
                    existing["manifest_hash"] != manifest_hash
                    or existing["manifest_json"] != body
                ):
                    raise ValueError(
                        f"Immutable DB manifest differs for run_uuid {run_uuid}."
                    )
                return
            con.execute(
                "INSERT INTO run_manifests "
                "(run_uuid, experiment_uuid, manifest_hash, manifest_json, "
                " status, completed_step, ts, updated_ts) "
                "VALUES (?,?,?,?, 'running', 0, ?, ?)",
                (run_uuid, experiment_uuid, manifest_hash, body, now, now),
            )

    def get_run_manifest(self, run_uuid: str) -> dict | None:
        with self._conn() as con:
            row = con.execute(
                "SELECT * FROM run_manifests WHERE run_uuid=?", (run_uuid,)
            ).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["manifest"] = json.loads(out["manifest_json"])
        return out

    def fetch_live_runs(self, status: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM live_runs", []
        if status:
            q += " WHERE status=?"; args.append(status)
        q += " ORDER BY updated_ts DESC"
        with self._conn() as con:
            return [dict(r) for r in con.execute(q, args).fetchall()]

    # ---- QLLM Lab: imported datasets and browser-launched jobs ----
    def upsert_lab_dataset(self, dataset: dict) -> None:
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        with self._conn() as con:
            con.execute(
                "INSERT INTO lab_datasets "
                "(name, source_type, source, split, text_column, corpus_path, "
                " n_rows, n_chars, preview, requested_revision, "
                " resolved_fingerprint, revision_applicable, row_limit, "
                " char_limit, byte_limit, rows_examined, n_bytes, sha256, "
                " truncated, truncation_reason, warnings_json, ts) VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(name) DO UPDATE SET "
                "source_type=excluded.source_type, source=excluded.source, "
                "split=excluded.split, text_column=excluded.text_column, "
                "corpus_path=excluded.corpus_path, n_rows=excluded.n_rows, "
                "n_chars=excluded.n_chars, preview=excluded.preview, "
                "requested_revision=excluded.requested_revision, "
                "resolved_fingerprint=excluded.resolved_fingerprint, "
                "revision_applicable=excluded.revision_applicable, "
                "row_limit=excluded.row_limit, char_limit=excluded.char_limit, "
                "byte_limit=excluded.byte_limit, rows_examined=excluded.rows_examined, "
                "n_bytes=excluded.n_bytes, sha256=excluded.sha256, "
                "truncated=excluded.truncated, "
                "truncation_reason=excluded.truncation_reason, "
                "warnings_json=excluded.warnings_json, ts=excluded.ts",
                (
                    dataset["name"],
                    dataset["source_type"],
                    dataset["source"],
                    dataset.get("split"),
                    dataset.get("text_column"),
                    dataset["corpus_path"],
                    dataset.get("n_rows"),
                    dataset.get("n_chars"),
                    dataset.get("preview"),
                    dataset.get("requested_revision"),
                    dataset.get("resolved_fingerprint"),
                    (
                        int(bool(dataset["revision_applicable"]))
                        if dataset.get("revision_applicable") is not None
                        else None
                    ),
                    dataset.get("row_limit"),
                    dataset.get("char_limit"),
                    dataset.get("byte_limit"),
                    dataset.get("rows_examined"),
                    dataset.get("n_bytes"),
                    dataset.get("sha256"),
                    int(bool(dataset.get("truncated", False))),
                    dataset.get("truncation_reason"),
                    json.dumps(dataset.get("warnings") or []),
                    now,
                ),
            )

    def fetch_lab_datasets(self) -> list[dict]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT * FROM lab_datasets ORDER BY ts DESC, name"
            ).fetchall()
        return [self._decode_lab_dataset(dict(r)) for r in rows]

    def get_lab_dataset(self, name: str) -> dict | None:
        with self._conn() as con:
            row = con.execute(
                "SELECT * FROM lab_datasets WHERE name=?", (name,)
            ).fetchone()
        return self._decode_lab_dataset(dict(row)) if row is not None else None

    @staticmethod
    def _decode_lab_dataset(row: dict) -> dict:
        try:
            warnings = json.loads(row.get("warnings_json") or "[]")
        except (json.JSONDecodeError, TypeError):
            warnings = []
        row["warnings"] = (
            [str(item) for item in warnings]
            if isinstance(warnings, list)
            else []
        )
        row["truncated"] = bool(row.get("truncated"))
        applicable = row.get("revision_applicable")
        if applicable is None:
            source = str(row.get("source") or "").lower()
            applicable = (
                row.get("source_type") == "huggingface"
                and not source.startswith(("http://", "https://", "hf://"))
            )
        row["revision_applicable"] = bool(applicable)
        return row

    def create_lab_job(self, job: dict) -> int:
        return self.create_lab_jobs([job])[0]

    @staticmethod
    def _insert_lab_job(
        con: sqlite3.Connection, job: dict, now: str
    ) -> int:
        cur = con.execute(
            "INSERT INTO lab_jobs "
            "(ts, updated_ts, status, preset_id, dataset_name, run_name, "
            " seed, steps, eval_every, run_key, error, config_json, "
            " group_id, parent_job_id, compare_to_job_id, device_target, "
            " comparison_role, experiment_uuid, run_uuid, parent_run_uuid, "
            " resume_from, checkpoint_path, best_checkpoint_path, "
            " artifact_dir, completed_step) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                now,
                now,
                job.get("status", "queued"),
                job["preset_id"],
                job["dataset_name"],
                job["run_name"],
                int(job["seed"]),
                int(job["steps"]),
                int(job["eval_every"]),
                job.get("run_key"),
                job.get("error"),
                json.dumps(job.get("config") or {}),
                job.get("group_id"),
                job.get("parent_job_id"),
                job.get("compare_to_job_id"),
                job.get("device_target", "auto"),
                job.get("comparison_role", "primary"),
                job["experiment_uuid"],
                job["run_uuid"],
                job.get("parent_run_uuid"),
                job.get("resume_from"),
                job.get("checkpoint_path"),
                job.get("best_checkpoint_path"),
                job.get("artifact_dir"),
                int(job.get("completed_step") or 0),
            ),
        )
        return int(cur.lastrowid)

    def create_lab_jobs(self, jobs: list[dict]) -> list[int]:
        """Create a batch atomically; UUIDs are assigned before persistence."""
        if not jobs:
            return []
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        prepared = []
        for job in jobs:
            item = dict(job)
            item["experiment_uuid"] = str(
                uuid.UUID(str(item.get("experiment_uuid") or uuid.uuid4()))
            )
            item["run_uuid"] = str(
                uuid.UUID(str(item.get("run_uuid") or uuid.uuid4()))
            )
            prepared.append(item)
        ids: list[int] = []
        with self._conn() as con:
            con.execute("BEGIN IMMEDIATE")
            for job in prepared:
                ids.append(self._insert_lab_job(con, job, now))
            if len(prepared) == 2 and all(
                bool(item.get("_pair_link")) for item in prepared
            ):
                primary_id, comparison_id = ids
                primary_config = dict(prepared[0].get("config") or {})
                comparison_config = dict(prepared[1].get("config") or {})
                if any(
                    str(key).startswith("lab.analogue.")
                    for key in primary_config
                ):
                    primary_config["lab.analogue.job_id"] = comparison_id
                if any(
                    str(key).startswith("lab.analogue.")
                    for key in comparison_config
                ):
                    comparison_config["lab.analogue.job_id"] = primary_id
                con.execute(
                    "UPDATE lab_jobs SET compare_to_job_id=?, config_json=?, "
                    "updated_ts=? WHERE id=?",
                    (
                        comparison_id,
                        json.dumps(primary_config),
                        now,
                        primary_id,
                    ),
                )
                con.execute(
                    "UPDATE lab_jobs SET parent_job_id=?, compare_to_job_id=?, "
                    "config_json=?, updated_ts=? WHERE id=?",
                    (
                        primary_id,
                        primary_id,
                        json.dumps(comparison_config),
                        now,
                        comparison_id,
                    ),
                )
        return ids

    def create_lab_job_pair(self, primary: dict, comparison: dict) -> tuple[int, int]:
        """Create a candidate/baseline pair in one transaction."""
        experiment_uuid = str(
            uuid.UUID(str(primary.get("experiment_uuid") or uuid.uuid4()))
        )
        primary = {
            **primary,
            "experiment_uuid": experiment_uuid,
            "_pair_link": True,
        }
        comparison = {
            **comparison,
            "experiment_uuid": experiment_uuid,
            "_pair_link": True,
        }
        primary_id, comparison_id = self.create_lab_jobs([primary, comparison])
        return primary_id, comparison_id

    def create_linked_lab_job(
        self,
        primary_job_id: int,
        comparison: dict,
        *,
        primary_config: dict,
        group_id: str,
    ) -> int:
        """Insert and link a post-hoc comparison in one transaction."""
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        item = dict(comparison)
        item["experiment_uuid"] = str(
            uuid.UUID(str(item.get("experiment_uuid") or uuid.uuid4()))
        )
        item["run_uuid"] = str(
            uuid.UUID(str(item.get("run_uuid") or uuid.uuid4()))
        )
        item["group_id"] = group_id
        item["parent_job_id"] = int(primary_job_id)
        item["compare_to_job_id"] = int(primary_job_id)
        comparison_config = dict(item.get("config") or {})
        if any(
            str(key).startswith("lab.analogue.") for key in comparison_config
        ):
            comparison_config["lab.analogue.job_id"] = int(primary_job_id)
        item["config"] = comparison_config
        with self._conn() as con:
            con.execute("BEGIN IMMEDIATE")
            primary = con.execute(
                "SELECT id FROM lab_jobs WHERE id=?", (int(primary_job_id),)
            ).fetchone()
            if primary is None:
                raise ValueError(f"Unknown primary lab job {primary_job_id}.")
            comparison_id = self._insert_lab_job(con, item, now)
            linked_primary_config = dict(primary_config)
            if any(
                str(key).startswith("lab.analogue.")
                for key in linked_primary_config
            ):
                linked_primary_config["lab.analogue.job_id"] = comparison_id
            changed = con.execute(
                "UPDATE lab_jobs SET group_id=?, compare_to_job_id=?, "
                "comparison_role='candidate', config_json=?, updated_ts=? "
                "WHERE id=?",
                (
                    group_id,
                    comparison_id,
                    json.dumps(linked_primary_config),
                    now,
                    int(primary_job_id),
                ),
            ).rowcount
            if changed != 1:  # defensive; SELECT above established existence
                raise RuntimeError("Primary lab job disappeared during linking.")
        return comparison_id

    def update_lab_job(self, job_id: int, **fields) -> None:
        if not fields:
            return
        fields["updated_ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        if "config" in fields:
            fields["config_json"] = json.dumps(fields.pop("config") or {})
        allowed = {
            "updated_ts", "status", "run_key", "error", "config_json",
            "run_name", "seed", "steps", "eval_every", "group_id",
            "parent_job_id", "compare_to_job_id", "device_target",
            "comparison_role",
            "experiment_uuid", "run_uuid", "parent_run_uuid",
            "manifest_hash", "manifest_json", "worker_id", "claimed_ts",
            "heartbeat_ts", "lease_expires_ts", "attempt_count",
            "recovery_count", "cancel_requested", "cancel_requested_ts",
            "resume_from", "checkpoint_path", "best_checkpoint_path",
            "artifact_dir", "completed_step",
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
        immutable = {
            key: value
            for key, value in updates.items()
            if key in {"experiment_uuid", "run_uuid", "manifest_hash", "manifest_json"}
        }
        if immutable:
            current = self.get_lab_job(job_id)
            if current is None:
                return
            for key, value in immutable.items():
                if current.get(key) is not None and current.get(key) != value:
                    raise ValueError(
                        f"Immutable lab job field '{key}' cannot be changed."
                    )
        set_sql = ", ".join(f"{k}=?" for k in updates)
        with self._conn() as con:
            con.execute(
                f"UPDATE lab_jobs SET {set_sql} WHERE id=?",
                [*updates.values(), job_id],
            )

    def fetch_lab_jobs(self, limit: int = 100) -> list[dict]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT * FROM lab_jobs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_lab_job(self, job_id: int) -> dict | None:
        with self._conn() as con:
            row = con.execute("SELECT * FROM lab_jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row is not None else None

    def get_lab_job_by_run_uuid(self, run_uuid: str) -> dict | None:
        with self._conn() as con:
            row = con.execute(
                "SELECT * FROM lab_jobs WHERE run_uuid=?", (run_uuid,)
            ).fetchone()
        return dict(row) if row is not None else None

    def requeue_lab_job_from_checkpoint(
        self,
        job_id: int,
        *,
        resume_from: str,
        checkpoint_path: str,
        completed_step: int,
        config: dict,
        artifact_dir: str | None = None,
    ) -> bool:
        with self._conn() as con:
            con.execute("BEGIN IMMEDIATE")
            changed = con.execute(
                "UPDATE lab_jobs SET status='queued', worker_id=NULL, "
                "claimed_ts=NULL, heartbeat_ts=NULL, lease_expires_ts=NULL, "
                "cancel_requested=0, cancel_requested_ts=NULL, resume_from=?, "
                "checkpoint_path=?, completed_step=?, config_json=?, "
                "artifact_dir=COALESCE(?, artifact_dir), error=NULL, updated_ts=? "
                "WHERE id=? AND status IN ('error','cancelled')",
                (
                    resume_from,
                    checkpoint_path,
                    int(completed_step),
                    json.dumps(config),
                    artifact_dir,
                    time.strftime("%Y-%m-%dT%H:%M:%S"),
                    int(job_id),
                ),
            ).rowcount
        return changed == 1

    # ---- durable SQLite-authoritative worker claims ----
    def claim_next_lab_job(
        self, worker_id: str, *, lease_seconds: float = 300.0
    ) -> dict | None:
        if not worker_id:
            raise ValueError("worker_id is required.")
        lease = float(lease_seconds)
        if not math.isfinite(lease) or lease <= 0:
            raise ValueError("lease_seconds must be finite and positive.")
        now = time.time()
        with self._conn() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT id FROM lab_jobs WHERE status='queued' "
                "AND COALESCE(cancel_requested, 0)=0 ORDER BY id LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            job_id = int(row["id"])
            changed = con.execute(
                "UPDATE lab_jobs SET status='running', worker_id=?, claimed_ts=?, "
                "heartbeat_ts=?, lease_expires_ts=?, attempt_count=attempt_count+1, "
                "experiment_uuid=COALESCE(experiment_uuid, ?), "
                "run_uuid=COALESCE(run_uuid, ?), updated_ts=?, error=NULL "
                "WHERE id=? AND status='queued' "
                "AND COALESCE(cancel_requested, 0)=0",
                (
                    worker_id,
                    now,
                    now,
                    now + lease,
                    str(uuid.uuid4()),
                    str(uuid.uuid4()),
                    time.strftime("%Y-%m-%dT%H:%M:%S"),
                    job_id,
                ),
            ).rowcount
            if changed != 1:
                return None
            self._update_job_run_status(
                con,
                job_id,
                "running",
                time.strftime("%Y-%m-%dT%H:%M:%S"),
            )
            claimed = con.execute(
                "SELECT * FROM lab_jobs WHERE id=?", (job_id,)
            ).fetchone()
            return dict(claimed)

    def claim_lab_job(
        self, job_id: int, worker_id: str, *, lease_seconds: float = 300.0
    ) -> dict | None:
        if not worker_id:
            raise ValueError("worker_id is required.")
        lease = float(lease_seconds)
        if not math.isfinite(lease) or lease <= 0:
            raise ValueError("lease_seconds must be finite and positive.")
        now = time.time()
        with self._conn() as con:
            con.execute("BEGIN IMMEDIATE")
            changed = con.execute(
                "UPDATE lab_jobs SET status='running', worker_id=?, claimed_ts=?, "
                "heartbeat_ts=?, lease_expires_ts=?, attempt_count=attempt_count+1, "
                "experiment_uuid=COALESCE(experiment_uuid, ?), "
                "run_uuid=COALESCE(run_uuid, ?), updated_ts=?, error=NULL "
                "WHERE id=? AND status='queued' "
                "AND COALESCE(cancel_requested, 0)=0",
                (
                    worker_id,
                    now,
                    now,
                    now + lease,
                    str(uuid.uuid4()),
                    str(uuid.uuid4()),
                    time.strftime("%Y-%m-%dT%H:%M:%S"),
                    int(job_id),
                ),
            ).rowcount
            if changed != 1:
                return None
            self._update_job_run_status(
                con,
                int(job_id),
                "running",
                time.strftime("%Y-%m-%dT%H:%M:%S"),
            )
            row = con.execute(
                "SELECT * FROM lab_jobs WHERE id=?", (int(job_id),)
            ).fetchone()
            return dict(row)

    def heartbeat_lab_job(
        self,
        job_id: int,
        worker_id: str,
        *,
        lease_seconds: float = 300.0,
        completed_step: int | None = None,
        checkpoint_path: str | None = None,
        best_checkpoint_path: str | None = None,
        artifact_dir: str | None = None,
        manifest: dict | None = None,
    ) -> bool:
        lease = float(lease_seconds)
        if not math.isfinite(lease) or lease <= 0:
            raise ValueError("lease_seconds must be finite and positive.")
        now = time.time()
        manifest_binding = (
            self._validated_job_manifest(manifest) if manifest is not None else None
        )
        updates = {
            "heartbeat_ts": now,
            "lease_expires_ts": now + lease,
            "updated_ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if completed_step is not None:
            updates["completed_step"] = int(completed_step)
        if checkpoint_path is not None:
            updates["checkpoint_path"] = checkpoint_path
        if best_checkpoint_path is not None:
            updates["best_checkpoint_path"] = best_checkpoint_path
        if artifact_dir is not None:
            updates["artifact_dir"] = artifact_dir
        if manifest_binding is not None:
            updates["manifest_hash"] = manifest_binding["manifest_hash"]
            updates["manifest_json"] = manifest_binding["manifest_json"]
        set_sql = ", ".join(f"{name}=?" for name in updates)
        with self._conn() as con:
            con.execute("BEGIN IMMEDIATE")
            current = con.execute(
                "SELECT status, worker_id, lease_expires_ts, experiment_uuid, "
                "run_uuid, manifest_hash, manifest_json FROM lab_jobs WHERE id=?",
                (int(job_id),),
            ).fetchone()
            if (
                current is None
                or current["status"] != "running"
                or current["worker_id"] != worker_id
                or current["lease_expires_ts"] is None
                or float(current["lease_expires_ts"]) < now
            ):
                return False
            if manifest_binding is not None:
                self._require_job_manifest_binding(current, manifest_binding)
            changed = con.execute(
                f"UPDATE lab_jobs SET {set_sql} WHERE id=? AND status='running' "
                "AND worker_id=? AND lease_expires_ts>=?",
                [*updates.values(), int(job_id), worker_id, now],
            ).rowcount
            if changed == 1:
                job = con.execute(
                    "SELECT run_uuid FROM lab_jobs WHERE id=?", (int(job_id),)
                ).fetchone()
                if job is not None and job["run_uuid"] is not None:
                    con.execute(
                        "UPDATE run_manifests SET "
                        "completed_step=MAX(completed_step, ?), "
                        "checkpoint_path=COALESCE(?, checkpoint_path), "
                        "best_checkpoint_path=COALESCE(?, best_checkpoint_path), "
                        "updated_ts=? WHERE run_uuid=?",
                        (
                            int(completed_step or 0),
                            checkpoint_path,
                            best_checkpoint_path,
                            updates["updated_ts"],
                            job["run_uuid"],
                        ),
                    )
        return changed == 1

    def prepare_claimed_lab_job(
        self,
        job_id: int,
        worker_id: str,
        *,
        run_key: str,
        config: dict,
        artifact_dir: str | None = None,
    ) -> bool:
        now = time.time()
        with self._conn() as con:
            changed = con.execute(
                "UPDATE lab_jobs SET run_key=?, config_json=?, "
                "artifact_dir=COALESCE(?, artifact_dir), error=NULL, "
                "updated_ts=? WHERE id=? AND status='running' AND worker_id=? "
                "AND lease_expires_ts>=?",
                (
                    run_key,
                    json.dumps(config),
                    artifact_dir,
                    time.strftime("%Y-%m-%dT%H:%M:%S"),
                    int(job_id),
                    worker_id,
                    now,
                ),
            ).rowcount
        return changed == 1

    def request_cancel_lab_job(self, job_id: int) -> dict | None:
        now = time.time()
        updated = time.strftime("%Y-%m-%dT%H:%M:%S")
        with self._conn() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT * FROM lab_jobs WHERE id=?", (int(job_id),)
            ).fetchone()
            if row is None:
                return None
            if row["status"] in ("done", "error", "cancelled"):
                return dict(row)
            if row["status"] == "queued":
                released = self._released_config(row["config_json"])
                con.execute(
                    "UPDATE lab_jobs SET status='cancelled', cancel_requested=1, "
                    "cancel_requested_ts=?, updated_ts=?, "
                    "config_json=?, error='Cancelled before start.' WHERE id=?",
                    (now, updated, released, int(job_id)),
                )
                self._update_job_run_status(
                    con, int(job_id), "cancelled", updated
                )
            else:
                con.execute(
                    "UPDATE lab_jobs SET cancel_requested=1, "
                    "cancel_requested_ts=?, updated_ts=? WHERE id=?",
                    (now, updated, int(job_id)),
                )
            current = con.execute(
                "SELECT * FROM lab_jobs WHERE id=?", (int(job_id),)
            ).fetchone()
            return dict(current)

    def lab_job_cancel_requested(self, job_id: int) -> bool:
        with self._conn() as con:
            row = con.execute(
                "SELECT cancel_requested FROM lab_jobs WHERE id=?", (int(job_id),)
            ).fetchone()
        return bool(row and row["cancel_requested"])

    def finish_claimed_lab_job(
        self,
        job_id: int,
        worker_id: str,
        *,
        status: str,
        error: str | None = None,
        config: dict | None = None,
        completed_step: int | None = None,
        checkpoint_path: str | None = None,
        best_checkpoint_path: str | None = None,
        artifact_dir: str | None = None,
        manifest: dict | None = None,
    ) -> bool:
        if status not in ("done", "error", "cancelled"):
            raise ValueError("Terminal status must be done, error, or cancelled.")
        manifest_binding = (
            self._validated_job_manifest(manifest) if manifest is not None else None
        )
        if config is None:
            current_job = self.get_lab_job(job_id)
            released_json = self._released_config(
                (current_job or {}).get("config_json")
            )
            try:
                parsed_released = json.loads(released_json)
            except json.JSONDecodeError:
                parsed_released = None
            if isinstance(parsed_released, dict):
                config = parsed_released
        updates: dict[str, object] = {
            "status": status,
            "error": error,
            "heartbeat_ts": time.time(),
            "lease_expires_ts": None,
            "updated_ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if config is not None:
            updates["config_json"] = json.dumps(config)
        if completed_step is not None:
            updates["completed_step"] = int(completed_step)
        if checkpoint_path is not None:
            updates["checkpoint_path"] = checkpoint_path
        if best_checkpoint_path is not None:
            updates["best_checkpoint_path"] = best_checkpoint_path
        if artifact_dir is not None:
            updates["artifact_dir"] = artifact_dir
        if manifest_binding is not None:
            updates["manifest_hash"] = manifest_binding["manifest_hash"]
            updates["manifest_json"] = manifest_binding["manifest_json"]
        set_sql = ", ".join(f"{name}=?" for name in updates)
        with self._conn() as con:
            con.execute("BEGIN IMMEDIATE")
            current = con.execute(
                "SELECT status, worker_id, lease_expires_ts, experiment_uuid, "
                "run_uuid, manifest_hash, manifest_json FROM lab_jobs WHERE id=?",
                (int(job_id),),
            ).fetchone()
            if current is None:
                return False
            if manifest_binding is not None:
                self._require_job_manifest_binding(current, manifest_binding)
            if current["status"] == status and current["worker_id"] == worker_id:
                return True
            if current["status"] != "running" or current["worker_id"] != worker_id:
                return False
            if (
                current["lease_expires_ts"] is None
                or float(current["lease_expires_ts"]) < time.time()
            ):
                return False
            changed = con.execute(
                f"UPDATE lab_jobs SET {set_sql} WHERE id=? AND status='running' "
                "AND worker_id=? AND lease_expires_ts>=?",
                [*updates.values(), int(job_id), worker_id, time.time()],
            ).rowcount
            if changed == 1:
                con.execute(
                    "UPDATE run_manifests SET status=?, "
                    "completed_step=MAX(completed_step, ?), "
                    "checkpoint_path=COALESCE(?, checkpoint_path), "
                    "best_checkpoint_path=COALESCE(?, best_checkpoint_path), "
                    "updated_ts=? WHERE run_uuid IN "
                    "(SELECT run_uuid FROM lab_jobs WHERE id=?)",
                    (
                        status,
                        int(completed_step or 0),
                        checkpoint_path,
                        best_checkpoint_path,
                        updates["updated_ts"],
                        int(job_id),
                    ),
                )
                con.execute(
                    "UPDATE live_runs SET status=?, updated_ts=? WHERE run_uuid "
                    "IN (SELECT run_uuid FROM lab_jobs WHERE id=?)",
                    (status, updates["updated_ts"], int(job_id)),
                )
            return changed == 1

    @staticmethod
    def _validated_job_manifest(manifest: dict) -> dict[str, str]:
        from .train.artifacts import validate_manifest

        normalized = validate_manifest(manifest)
        return {
            "experiment_uuid": str(normalized["experiment_uuid"]),
            "run_uuid": str(normalized["run_uuid"]),
            "manifest_hash": str(normalized["manifest_hash"]),
            "manifest_json": json.dumps(
                normalized, sort_keys=True, separators=(",", ":")
            ),
        }

    @staticmethod
    def _require_job_manifest_binding(
        job: sqlite3.Row, binding: dict[str, str]
    ) -> None:
        if job["experiment_uuid"] != binding["experiment_uuid"]:
            raise ValueError("Manifest experiment_uuid does not match the lab job.")
        if job["run_uuid"] != binding["run_uuid"]:
            raise ValueError("Manifest run_uuid does not match the lab job.")
        for name in ("manifest_hash", "manifest_json"):
            if job[name] is not None and job[name] != binding[name]:
                raise ValueError(
                    f"Immutable lab job {name} differs from the supplied manifest."
                )

    @staticmethod
    def _update_job_run_status(
        con: sqlite3.Connection, job_id: int, status: str, updated_ts: str
    ) -> None:
        con.execute(
            "UPDATE run_manifests SET status=?, updated_ts=? WHERE run_uuid "
            "IN (SELECT run_uuid FROM lab_jobs WHERE id=?)",
            (status, updated_ts, int(job_id)),
        )
        con.execute(
            "UPDATE live_runs SET status=?, updated_ts=? WHERE run_uuid "
            "IN (SELECT run_uuid FROM lab_jobs WHERE id=?)",
            (status, updated_ts, int(job_id)),
        )

    @staticmethod
    def _reconcile_job_run_recovery(
        con: sqlite3.Connection,
        job_id: int,
        *,
        status: str,
        updated_ts: str,
        completed_step: int,
        checkpoint_path: str | None,
    ) -> None:
        """Align advertised progress with the last durable checkpoint."""
        con.execute(
            "UPDATE run_manifests SET status=?, completed_step=?, "
            "checkpoint_path=?, updated_ts=? WHERE run_uuid IN "
            "(SELECT run_uuid FROM lab_jobs WHERE id=?)",
            (
                status,
                int(completed_step),
                checkpoint_path,
                updated_ts,
                int(job_id),
            ),
        )
        con.execute(
            "UPDATE live_runs SET status=?, "
            "last_train_loss=CASE WHEN COALESCE(current_step, 0)>? THEN NULL "
            "ELSE last_train_loss END, "
            "last_val_ppl=CASE WHEN COALESCE(current_step, 0)>? THEN NULL "
            "ELSE last_val_ppl END, current_step=?, updated_ts=? "
            "WHERE run_uuid IN (SELECT run_uuid FROM lab_jobs WHERE id=?)",
            (
                status,
                int(completed_step),
                int(completed_step),
                int(completed_step),
                updated_ts,
                int(job_id),
            ),
        )

    @staticmethod
    def _released_config(config_json: str | None) -> str:
        try:
            config = json.loads(config_json or "{}")
        except json.JSONDecodeError:
            return config_json if config_json is not None else ""
        if isinstance(config, dict):
            for key in list(config):
                if key.endswith("reservation.state"):
                    config[key] = "released"
            for key in list(config):
                if key.endswith("reservation.job_id"):
                    config[key] = None
        return json.dumps(config)

    def recover_stale_lab_jobs(
        self,
        *,
        now: float | None = None,
        checkpoint_validator: Callable[[str | None], bool] | None = None,
        checkpoint_resolver: Callable[[dict], dict | None] | None = None,
    ) -> dict:
        """Recover expired leases without silently restarting lost progress."""
        moment = float(time.time() if now is None else now)
        validator = checkpoint_validator or (
            lambda path: bool(path and Path(path).is_file() and Path(path).stat().st_size)
        )
        recovered: list[int] = []
        failed: list[int] = []
        cancelled: list[int] = []
        with self._conn() as con:
            con.execute("BEGIN IMMEDIATE")
            rows = con.execute(
                "SELECT * FROM lab_jobs WHERE status='running' AND "
                "(lease_expires_ts IS NULL OR lease_expires_ts<=?) ORDER BY id",
                (moment,),
            ).fetchall()
            for row in rows:
                job_id = int(row["id"])
                updated = time.strftime("%Y-%m-%dT%H:%M:%S")
                released = self._released_config(row["config_json"])
                if row["worker_id"] is None or row["claimed_ts"] is None:
                    con.execute(
                        "UPDATE lab_jobs SET status='error', lease_expires_ts=NULL, "
                        "config_json=?, updated_ts=?, error=? WHERE id=?",
                        (
                            released,
                            updated,
                            "Legacy running job has no durable lease/owner; "
                            "recovery cannot infer safe progress.",
                            job_id,
                        ),
                    )
                    self._update_job_run_status(con, job_id, "error", updated)
                    failed.append(job_id)
                    continue
                if row["cancel_requested"]:
                    con.execute(
                        "UPDATE lab_jobs SET status='cancelled', worker_id=NULL, "
                        "lease_expires_ts=NULL, config_json=?, updated_ts=?, "
                        "error='Cancelled while worker lease was stale.' WHERE id=?",
                        (released, updated, job_id),
                    )
                    self._update_job_run_status(
                        con, job_id, "cancelled", updated
                    )
                    cancelled.append(job_id)
                    continue
                completed = int(row["completed_step"] or 0)
                checkpoint = row["checkpoint_path"]
                resolution = (
                    checkpoint_resolver(dict(row))
                    if checkpoint_resolver is not None
                    else None
                )
                if resolution is not None:
                    checkpoint = resolution.get("path")
                    completed = int(resolution.get("completed_step") or 0)
                    recoverable = bool(
                        resolution.get("fresh") or checkpoint is not None
                    )
                else:
                    valid_checkpoint = bool(checkpoint and validator(checkpoint))
                    recoverable = (
                        completed == 0 or valid_checkpoint
                        if checkpoint_resolver is None
                        else False
                    )
                if recoverable:
                    con.execute(
                        "UPDATE lab_jobs SET status='queued', worker_id=NULL, "
                        "claimed_ts=NULL, heartbeat_ts=NULL, lease_expires_ts=NULL, "
                        "recovery_count=recovery_count+1, resume_from=?, "
                        "checkpoint_path=?, completed_step=?, "
                        "config_json=?, updated_ts=?, error=NULL WHERE id=?",
                        (
                            checkpoint,
                            checkpoint,
                            completed,
                            released,
                            updated,
                            job_id,
                        ),
                    )
                    self._reconcile_job_run_recovery(
                        con,
                        job_id,
                        status="queued",
                        updated_ts=updated,
                        completed_step=completed,
                        checkpoint_path=checkpoint,
                    )
                    recovered.append(job_id)
                else:
                    con.execute(
                        "UPDATE lab_jobs SET status='error', lease_expires_ts=NULL, "
                        "config_json=?, updated_ts=?, error=? WHERE id=?",
                        (
                            released,
                            updated,
                            "Stale worker lease after progress without a valid checkpoint.",
                            job_id,
                        ),
                    )
                    self._update_job_run_status(con, job_id, "error", updated)
                    failed.append(job_id)
        return {
            "requeued": recovered,
            "errored": failed,
            "cancelled": cancelled,
            "count": len(recovered) + len(failed) + len(cancelled),
        }

    # ---- editable model specs ----
    def create_model_spec(self, spec: dict) -> int:
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        with self._conn() as con:
            cur = con.execute(
                "INSERT INTO model_specs "
                "(ts, updated_ts, name, source, parent_id, version, notes, config_json, graph_json) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    now,
                    now,
                    spec["name"],
                    spec.get("source"),
                    spec.get("parent_id"),
                    int(spec.get("version") or 1),
                    spec.get("notes"),
                    json.dumps(spec["config"]),
                    json.dumps(spec.get("graph") or {}),
                ),
            )
            return int(cur.lastrowid)

    def update_model_spec(self, spec_id: int, **fields) -> None:
        if not fields:
            return
        fields["updated_ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        if "config" in fields:
            fields["config_json"] = json.dumps(fields.pop("config") or {})
        if "graph" in fields:
            fields["graph_json"] = json.dumps(fields.pop("graph") or {})
        allowed = {
            "updated_ts", "name", "source", "parent_id", "version", "notes",
            "config_json", "graph_json",
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
        set_sql = ", ".join(f"{k}=?" for k in updates)
        with self._conn() as con:
            con.execute(
                f"UPDATE model_specs SET {set_sql} WHERE id=?",
                [*updates.values(), spec_id],
            )

    def fetch_model_specs(self) -> list[dict]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT * FROM model_specs ORDER BY updated_ts DESC, id DESC"
            ).fetchall()
        return [self._decode_model_spec(dict(r)) for r in rows]

    def get_model_spec(self, spec_id: int) -> dict | None:
        with self._conn() as con:
            row = con.execute("SELECT * FROM model_specs WHERE id=?", (spec_id,)).fetchone()
        return self._decode_model_spec(dict(row)) if row is not None else None

    @staticmethod
    def _decode_model_spec(row: dict) -> dict:
        try:
            row["config"] = json.loads(row.get("config_json") or "{}")
        except json.JSONDecodeError:
            row["config"] = {}
        try:
            row["graph"] = json.loads(row.get("graph_json") or "{}")
        except json.JSONDecodeError:
            row["graph"] = {}
        return row

    # ---- studies: first-class research protocols over lab jobs ----
    def create_study(self, study: dict) -> int:
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        with self._conn() as con:
            cur = con.execute(
                "INSERT INTO studies "
                "(ts, updated_ts, name, research_question, task, description, "
                " dataset_names_json, candidate_preset_id, baseline_policy, "
                " control_preset_ids_json, seeds_json, sweep_json, status, group_id, protocol_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    now,
                    now,
                    study["name"],
                    study.get("research_question"),
                    study.get("task"),
                    study.get("description"),
                    json.dumps(study.get("dataset_names") or []),
                    study["candidate_preset_id"],
                    study.get("baseline_policy") or "analogue",
                    json.dumps(study.get("control_preset_ids") or []),
                    json.dumps(study.get("seeds") or []),
                    json.dumps(study.get("sweep") or {}),
                    study.get("status") or "draft",
                    study["group_id"],
                    json.dumps(study.get("protocol") or {}),
                ),
            )
            return int(cur.lastrowid)

    def update_study(self, study_id: int, **fields) -> None:
        if not fields:
            return
        fields["updated_ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        json_fields = {
            "dataset_names": "dataset_names_json",
            "control_preset_ids": "control_preset_ids_json",
            "seeds": "seeds_json",
            "sweep": "sweep_json",
            "protocol": "protocol_json",
        }
        for key, column in list(json_fields.items()):
            if key in fields:
                fields[column] = json.dumps(fields.pop(key) or ([] if key in {"dataset_names", "control_preset_ids", "seeds"} else {}))
        allowed = {
            "updated_ts", "name", "research_question", "task", "description",
            "dataset_names_json", "candidate_preset_id", "baseline_policy",
            "control_preset_ids_json", "seeds_json", "sweep_json",
            "status", "group_id", "protocol_json",
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
        set_sql = ", ".join(f"{k}=?" for k in updates)
        with self._conn() as con:
            con.execute(
                f"UPDATE studies SET {set_sql} WHERE id=?",
                [*updates.values(), study_id],
            )

    def fetch_studies(self) -> list[dict]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT * FROM studies ORDER BY updated_ts DESC, id DESC"
            ).fetchall()
        return [self._decode_study(dict(r)) for r in rows]

    def get_study(self, study_id: int) -> dict | None:
        with self._conn() as con:
            row = con.execute("SELECT * FROM studies WHERE id=?", (study_id,)).fetchone()
        return self._decode_study(dict(row)) if row is not None else None

    def add_study_job(self, study_id: int, job_id: int, role: str, sweep: dict | None = None) -> None:
        with self._conn() as con:
            con.execute(
                "INSERT OR REPLACE INTO study_jobs (study_id, job_id, role, sweep_json) "
                "VALUES (?,?,?,?)",
                (int(study_id), int(job_id), role, json.dumps(sweep or {})),
            )

    def fetch_study_jobs(self, study_id: int) -> list[dict]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT sj.*, j.* FROM study_jobs sj "
                "JOIN lab_jobs j ON j.id=sj.job_id "
                "WHERE sj.study_id=? ORDER BY j.id",
                (int(study_id),),
            ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            try:
                item["study_sweep"] = json.loads(item.get("sweep_json") or "{}")
            except json.JSONDecodeError:
                item["study_sweep"] = {}
            out.append(item)
        return out

    @staticmethod
    def _decode_study(row: dict) -> dict:
        for column, key, default in (
            ("dataset_names_json", "dataset_names", []),
            ("control_preset_ids_json", "control_preset_ids", []),
            ("seeds_json", "seeds", []),
            ("sweep_json", "sweep", {}),
            ("protocol_json", "protocol", {}),
        ):
            try:
                row[key] = json.loads(row.get(column) or json.dumps(default))
            except json.JSONDecodeError:
                row[key] = default
        return row

    def record_metrics(
        self, suite: str, variant: str, dataset: str, seed: int,
        metrics: dict[str, float],
    ) -> None:
        with self._conn() as con:
            for name, value in metrics.items():
                con.execute(
                    "INSERT OR REPLACE INTO metrics "
                    "(ts, suite, variant, dataset, seed, name, value) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (time.strftime("%Y-%m-%dT%H:%M:%S"), suite, variant,
                     dataset, seed, name, float(value)),
                )

    def fetch_metrics(self, suite: str, dataset: str | None = None) -> list[dict]:
        query = "SELECT * FROM metrics WHERE suite=?"
        args: list = [suite]
        if dataset is not None:
            query += " AND dataset=?"
            args.append(dataset)
        with self._conn() as con:
            rows = con.execute(
                query + " ORDER BY variant, seed, name", args
            ).fetchall()
        return [dict(r) for r in rows]

    def delete(self, suite: str, variant: str | None = None) -> int:
        query, args = "DELETE FROM runs WHERE suite=?", [suite]
        if variant is not None:
            query += " AND variant=?"
            args.append(variant)
        with self._conn() as con:
            cur = con.execute(query, args)
        return cur.rowcount
