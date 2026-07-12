"""All dashboard SQL in one place: read-only views over the results DB.

Keeps the FastAPI layer thin and makes the data contract explicit. Every
function takes a ResultsDB and returns JSON-able dicts/lists.
"""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict

from ..claims import get_claim, infer_claim_id
from ..research_protocol import two_stream_metric_contract
from ..research_protocol import normalize_seed_axes
from ..resultsdb import ResultsDB
from ._shared import curve as _curve
from .model_graph import uses_quantum_config
from .evidence import interpretation_warnings, run_resource_payload


def _contract_for_run(run: dict) -> dict | None:
    try:
        config = json.loads(run.get("config_json") or "{}")
    except json.JSONDecodeError:
        config = {}
    return two_stream_metric_contract(
        suite=str(run.get("suite") or ""),
        config=config,
    )


def _claim_fields(run: dict, config: dict | None = None) -> dict:
    config = config or {}
    claim_id = infer_claim_id(
        explicit=config.get("research.claim_id"),
        suite=run.get("suite"),
        preset_id=run.get("variant"),
    )
    claim = get_claim(claim_id) if claim_id else None
    contract = two_stream_metric_contract(
        suite=str(run.get("suite") or ""), config=config
    )
    seed_axes = config.get("research.seed_axes")
    if not isinstance(seed_axes, dict):
        seed_axes = normalize_seed_axes(
            int(run.get("seed", 0)),
            generator_seed=config.get("data.gen_seed"),
            data_kind=config.get("data.kind"),
            circuit_applicable=uses_quantum_config(config),
        )
    return {
        "claim_id": claim_id,
        "claim": claim,
        "metric_type": (
            (contract or {}).get("metric_type")
            or config.get("research.metric_type")
            or (claim or {}).get("metric_type")
        ),
        "seed_axes": seed_axes,
        "assessment_status": "unassigned" if claim_id is None else "descriptive",
    }


def suites_overview(db: ResultsDB) -> list[dict]:
    """One card per suite: counts + best (lowest) val_ppl + last activity."""
    with db._conn() as con:
        rows = con.execute(
            "SELECT suite, COUNT(*) n, COUNT(DISTINCT variant) variants, "
            "COUNT(DISTINCT dataset) datasets, COUNT(DISTINCT seed) independent_seeds, MIN(val_ppl) best_ppl, "
            "MAX(ts) last_ts FROM runs GROUP BY suite ORDER BY last_ts DESC"
        ).fetchall()
    suites = []
    for raw in rows:
        row = dict(raw)
        contract = two_stream_metric_contract(suite=row["suite"])
        row["metric_contract"] = contract
        if contract and contract["rerun_required"]:
            row["historical_best_ppl"] = row["best_ppl"]
            row["best_ppl"] = None
        row["interpretation_warnings"] = interpretation_warnings(
            single_seed=int(row.get("independent_seeds") or 0) == 1,
            metric_contract=contract,
            metric_type=(contract or {}).get("metric_type"),
            assessment_status=("rerun_required" if contract and contract.get("rerun_required") else None),
        )
        suites.append(row)
    return suites


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
        entry["interpretation_warnings"] = interpretation_warnings(
            single_seed=len(entry["seeds"]) == 1,
            metric_contract=two_stream_metric_contract(suite=suite),
            metric_type=(two_stream_metric_contract(suite=suite) or {}).get("metric_type"),
            assessment_status=(
                "rerun_required"
                if (two_stream_metric_contract(suite=suite) or {}).get("rerun_required")
                else None
            ),
        )
        for name, vals in met_by_var.get(variant, {}).items():
            entry[f"metric_{name}"] = st.mean(vals)
        leaderboard.append(entry)
    leaderboard.sort(key=lambda e: (e["val_ppl_mean"] is None,
                                    e["val_ppl_mean"] or 0))
    datasets = sorted({r["dataset"] for r in db.fetch(suite)})
    metric_names = sorted({m["name"] for m in metrics})
    contract = two_stream_metric_contract(suite=suite)
    return {
        "suite": suite,
        "datasets": datasets,
        "metric_names": metric_names,
        "leaderboard": leaderboard,
        "metric_contract": contract,
        "interpretation_warnings": [
            warning
            for row in leaderboard
            for warning in row.get("interpretation_warnings") or []
        ],
    }


def all_runs(db: ResultsDB, suite: str | None = None) -> list[dict]:
    q = "SELECT * FROM runs"
    args: list = []
    if suite:
        q += " WHERE suite=?"; args.append(suite)
    q += " ORDER BY ts DESC"
    with db._conn() as con:
        rows = [
            db.decode_result_row(dict(r))
            for r in con.execute(q, args).fetchall()
        ]
    for row in rows:
        row["metric_contract"] = _contract_for_run(row)
        try:
            config = json.loads(row.get("config_json") or "{}")
        except json.JSONDecodeError:
            config = {}
        row.update(_claim_fields(row, config))
        if row.get("run_uuid"):
            manifest_row = db.get_run_manifest(str(row["run_uuid"]))
            if manifest_row:
                row["manifest"] = manifest_row.get("manifest")
        row.update(run_resource_payload(row))
        row["interpretation_warnings"] = interpretation_warnings(
            single_seed=True,
            metric_contract=row.get("metric_contract"),
            metric_type=row.get("metric_type"),
            assessment_status=(
                "rerun_required"
                if (row.get("metric_contract") or {}).get("rerun_required")
                else row.get("assessment_status")
            ),
        )
    return rows


def run_detail(db: ResultsDB, run_id: int) -> dict:
    with db._conn() as con:
        row = con.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    if row is None:
        return {}
    run = db.decode_result_row(dict(row))
    run["config"] = json.loads(run.get("config_json") or "{}")
    run["metric_contract"] = two_stream_metric_contract(
        suite=run.get("suite", ""),
        config=run["config"],
    )
    run.update(_claim_fields(run, run["config"]))
    if run.get("run_uuid"):
        manifest_row = db.get_run_manifest(str(run["run_uuid"]))
        if manifest_row:
            run["manifest"] = manifest_row.get("manifest")
    run.update(run_resource_payload(run))
    run["interpretation_warnings"] = interpretation_warnings(
        single_seed=True,
        metric_contract=run.get("metric_contract"),
        metric_type=run.get("metric_type"),
        assessment_status=(
            "rerun_required"
            if (run.get("metric_contract") or {}).get("rerun_required")
            else run.get("assessment_status")
        ),
    )
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
    run["steps_curve"] = _curve(db, run_key, run.get("run_uuid"))
    return run


def live_runs(db: ResultsDB) -> list[dict]:
    return db.fetch_live_runs()


def live_curve(db: ResultsDB, run_key: str) -> dict:
    return _curve(db, run_key)
