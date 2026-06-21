"""All dashboard SQL in one place: read-only views over the results DB.

Keeps the FastAPI layer thin and makes the data contract explicit. Every
function takes a ResultsDB and returns JSON-able dicts/lists.
"""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict

from ..resultsdb import ResultsDB


def suites_overview(db: ResultsDB) -> list[dict]:
    """One card per suite: counts + best (lowest) val_ppl + last activity."""
    with db._conn() as con:
        rows = con.execute(
            "SELECT suite, COUNT(*) n, COUNT(DISTINCT variant) variants, "
            "COUNT(DISTINCT dataset) datasets, MIN(val_ppl) best_ppl, "
            "MAX(ts) last_ts FROM runs GROUP BY suite ORDER BY last_ts DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def suite_detail(db: ResultsDB, suite: str, dataset: str | None = None) -> dict:
    """Leaderboard for a suite: per-variant aggregated stats across seeds,
    plus the list of datasets and any extra metrics available."""
    runs = db.fetch(suite, dataset)
    metrics = db.fetch_metrics(suite, dataset)
    # group runs by variant
    by_var: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        by_var[r["variant"]].append(r)
    # extra metrics keyed (variant, seed) -> {name: value}
    met_by_var: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for m in metrics:
        met_by_var[m["variant"]][m["name"]].append(m["value"])

    leaderboard = []
    for variant, rs in by_var.items():
        ppls = [r["val_ppl"] for r in rs if r["val_ppl"] is not None]
        entry = {
            "variant": variant,
            "n_runs": len(rs),
            "n_params": rs[0]["n_params"],
            "val_ppl_mean": st.mean(ppls) if ppls else None,
            "val_ppl_std": st.stdev(ppls) if len(ppls) > 1 else 0.0,
            "seeds": sorted({r["seed"] for r in rs}),
            "steps": sorted({r["steps"] for r in rs}),
        }
        for name, vals in met_by_var.get(variant, {}).items():
            entry[f"metric_{name}"] = st.mean(vals)
        leaderboard.append(entry)
    leaderboard.sort(key=lambda e: (e["val_ppl_mean"] is None,
                                    e["val_ppl_mean"] or 0))
    datasets = sorted({r["dataset"] for r in db.fetch(suite)})
    metric_names = sorted({m["name"] for m in metrics})
    return {"suite": suite, "datasets": datasets, "metric_names": metric_names,
            "leaderboard": leaderboard}


def all_runs(db: ResultsDB, suite: str | None = None) -> list[dict]:
    q = "SELECT * FROM runs"
    args: list = []
    if suite:
        q += " WHERE suite=?"; args.append(suite)
    q += " ORDER BY ts DESC"
    with db._conn() as con:
        return [dict(r) for r in con.execute(q, args).fetchall()]


def run_detail(db: ResultsDB, run_id: int) -> dict:
    with db._conn() as con:
        row = con.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    if row is None:
        return {}
    run = dict(row)
    run["config"] = json.loads(run.get("config_json") or "{}")
    # matching extra metrics (same suite/variant/dataset/seed)
    with db._conn() as con:
        mets = con.execute(
            "SELECT name, value FROM metrics WHERE suite=? AND variant=? "
            "AND dataset=? AND seed=?",
            (run["suite"], run["variant"], run["dataset"], run["seed"])
        ).fetchall()
    run["metrics"] = {m["name"]: m["value"] for m in mets}
    # per-step curve if logged under the canonical run_key
    run_key = (f"{run['suite']}/{run['variant']}/{run['dataset']}/"
               f"{run['seed']}/{run['steps']}")
    run["steps_curve"] = _curve(db, run_key)
    return run


def _curve(db: ResultsDB, run_key: str) -> dict:
    steps = db.fetch_steps(run_key)
    series: dict[str, list] = defaultdict(list)
    for s in steps:
        series[s["name"]].append({"step": s["step"], "value": s["value"]})
    return series


def live_runs(db: ResultsDB) -> list[dict]:
    return db.fetch_live_runs()


def live_curve(db: ResultsDB, run_key: str) -> dict:
    return _curve(db, run_key)
