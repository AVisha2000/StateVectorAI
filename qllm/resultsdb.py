"""Results database: one row per benchmark run, SQLite, stdlib-only.

MLflow (sqlite:///mlflow.db) remains the per-step tracker; this is the
clean cross-suite results table that makes comparison plots and
resumability trivial: a (suite, variant, dataset, seed, steps) key is
unique, so re-running a suite skips finished work and a crashed/timed-out
sweep continues where it stopped — on this machine or on a GPU box later.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

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
    comparison_role TEXT DEFAULT 'primary'
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
"""


class ResultsDB:
    def __init__(self, path: str | Path = "results/qllm_results.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as con:
            con.executescript(_SCHEMA)
            self._ensure_lab_job_columns(con)

    def _ensure_lab_job_columns(self, con: sqlite3.Connection) -> None:
        cols = {r["name"] for r in con.execute("PRAGMA table_info(lab_jobs)").fetchall()}
        additions = {
            "group_id": "TEXT",
            "parent_job_id": "INTEGER",
            "compare_to_job_id": "INTEGER",
            "device_target": "TEXT DEFAULT 'auto'",
            "comparison_role": "TEXT DEFAULT 'primary'",
        }
        for name, ddl in additions.items():
            if name not in cols:
                con.execute(f"ALTER TABLE lab_jobs ADD COLUMN {name} {ddl}")

    def _conn(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
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
    ) -> None:
        with self._conn() as con:
            con.execute(
                "INSERT OR REPLACE INTO runs "
                "(ts, suite, variant, dataset, seed, steps, n_params, "
                " val_loss, val_ppl, val_bpc, wall_seconds, config_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    time.strftime("%Y-%m-%dT%H:%M:%S"),
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
                    json.dumps({k: str(v) for k, v in (config or {}).items()}),
                ),
            )

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
        self, suite: str, variant: str, dataset: str, seed: int, steps: int
    ) -> dict | None:
        with self._conn() as con:
            row = con.execute(
                "SELECT * FROM runs WHERE suite=? AND variant=? AND dataset=? "
                "AND seed=? AND steps=?",
                (suite, variant, dataset, seed, steps),
            ).fetchone()
        return dict(row) if row is not None else None

    # ---- live-run registry + per-step logging (own MLflow replacement) ----
    def start_run(self, run_key: str, run_name: str, suite: str, variant: str,
                  dataset: str, seed: int, total_steps: int,
                  config: dict | None = None) -> None:
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        with self._conn() as con:
            con.execute(
                "INSERT OR REPLACE INTO live_runs (run_key, run_name, suite, "
                "variant, dataset, seed, total_steps, current_step, status, "
                "started_ts, updated_ts, config_json) VALUES "
                "(?,?,?,?,?,?,?,0,'running',?,?,?)",
                (run_key, run_name, suite, variant, dataset, seed, total_steps,
                 now, now, json.dumps({k: str(v) for k, v in (config or {}).items()})))

    def log_step(self, run_key: str, step: int, metrics: dict[str, float],
                 train_loss: float | None = None,
                 val_ppl: float | None = None) -> None:
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        with self._conn() as con:
            con.executemany(
                "INSERT INTO steps (run_key, step, name, value) VALUES (?,?,?,?)",
                [(run_key, step, k, float(v)) for k, v in metrics.items()])
            con.execute(
                "UPDATE live_runs SET current_step=?, updated_ts=?, "
                "last_train_loss=COALESCE(?, last_train_loss), "
                "last_val_ppl=COALESCE(?, last_val_ppl) WHERE run_key=?",
                (step, now, train_loss, val_ppl, run_key))

    def finish_run(self, run_key: str, status: str = "done") -> None:
        with self._conn() as con:
            con.execute(
                "UPDATE live_runs SET status=?, updated_ts=? WHERE run_key=?",
                (status, time.strftime("%Y-%m-%dT%H:%M:%S"), run_key))

    def fetch_steps(self, run_key: str) -> list[dict]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT step, name, value FROM steps WHERE run_key=? "
                "ORDER BY step", (run_key,)).fetchall()
        return [dict(r) for r in rows]

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
                "INSERT OR REPLACE INTO lab_datasets "
                "(name, source_type, source, split, text_column, corpus_path, "
                " n_rows, n_chars, preview, ts) VALUES (?,?,?,?,?,?,?,?,?,?)",
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
                    now,
                ),
            )

    def fetch_lab_datasets(self) -> list[dict]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT * FROM lab_datasets ORDER BY ts DESC, name"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_lab_dataset(self, name: str) -> dict | None:
        with self._conn() as con:
            row = con.execute(
                "SELECT * FROM lab_datasets WHERE name=?", (name,)
            ).fetchone()
        return dict(row) if row is not None else None

    def create_lab_job(self, job: dict) -> int:
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        with self._conn() as con:
            cur = con.execute(
                "INSERT INTO lab_jobs "
                "(ts, updated_ts, status, preset_id, dataset_name, run_name, "
                " seed, steps, eval_every, run_key, error, config_json, "
                " group_id, parent_job_id, compare_to_job_id, device_target, "
                " comparison_role) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
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
                ),
            )
            return int(cur.lastrowid)

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
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
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
