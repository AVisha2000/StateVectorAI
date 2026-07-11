"""Research-map payloads for the Quantum Advantage cockpit."""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from ..research_protocol import two_stream_metric_contract
from ..resultsdb import ResultsDB
from .datasets import list_datasets
from .model_graph import model_family, uses_quantum_config


def _decode_config(row: dict) -> dict:
    try:
        return json.loads(row.get("config_json") or "{}")
    except json.JSONDecodeError:
        return {}


def _slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")


def _as_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value in (None, ""):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _role(config: dict, variant: str = "") -> str:
    if uses_quantum_config(config):
        family = model_family(config)
        if family.startswith("quantum") or family in {"qrnn"}:
            return "quantum"
        return "hybrid"
    lower = variant.lower()
    if "quantum" in lower or lower.startswith("q-"):
        return "quantum"
    if "hybrid" in lower or "two-stream" in lower:
        return "hybrid"
    return "classical"


def infer_research_context(
    *,
    suite: str = "",
    variant: str = "",
    dataset: str = "",
    config: dict | None = None,
) -> dict:
    """Infer domain/task metadata from local, inspectable run information."""
    config = config or {}
    text = " ".join(
        str(v).lower()
        for v in (
            suite,
            variant,
            dataset,
            config.get("data.kind"),
            config.get("model.arch"),
            config.get("model.encoder_kind"),
        )
        if v is not None
    )
    inferred_from = []

    if any(token in text for token in ("contextual", "parity")):
        domain, task, confidence = "Synthetic quantum data", "Contextual parity", 0.86
        inferred_from.append("contextual/parity keyword")
    elif any(token in text for token in ("interference", "cancellation", "seq_cancel")):
        domain, task, confidence = "Sequence memory", "Interference/cancellation", 0.82
        inferred_from.append("interference/cancellation keyword")
    elif any(token in text for token in ("ising", "monitored_ising", "quantum_seq")):
        domain, task, confidence = "Synthetic quantum data", "Quantum-generated sequence prediction", 0.9
        inferred_from.append("synthetic quantum data source")
    elif "markov" in text or "qrnn" in text or "memory" in text:
        domain, task, confidence = "Sequence memory", "Sequence memory", 0.78
        inferred_from.append("memory/recurrent keyword")
    elif any(token in text for token in ("qnlp", "text", "language", "default-text", "two-stream")):
        domain, task, confidence = "QNLP", "Language modelling", 0.74
        inferred_from.append("text/QNLP source")
    elif any(token in text for token in ("tabular", "regression", "classification")):
        domain, task, confidence = "Tabular ML", "Classification" if "classification" in text else "Regression", 0.65
        inferred_from.append("tabular task keyword")
    else:
        domain, task, confidence = "General ML", "Language modelling", 0.45
        inferred_from.append("fallback")

    return {
        "domain": domain,
        "domain_slug": _slug(domain),
        "task": task,
        "task_slug": _slug(task),
        "confidence": confidence,
        "inferred_from": inferred_from,
    }


def _resource_from_config(config: dict) -> dict:
    return {
        "n_qubits": _as_int(config.get("model.quantum.n_qubits")),
        "n_circuit_layers": _as_int(config.get("model.quantum.n_circuit_layers")),
        "shots": config.get("model.quantum.shots"),
        "backend": config.get("model.quantum.backend"),
        "resource_band": config.get("lab.resource.band"),
        "resource_score": config.get("lab.resource.score"),
    }


def _run_item(row: dict) -> dict:
    config = _decode_config(row)
    contract = two_stream_metric_contract(
        suite=row.get("suite", ""),
        config=config,
    )
    context = infer_research_context(
        suite=row.get("suite", ""),
        variant=row.get("variant", ""),
        dataset=row.get("dataset", ""),
        config=config,
    )
    family = model_family(config) if config else row.get("variant")
    return {
        "kind": "run",
        "id": row["id"],
        "suite": row["suite"],
        "variant": row["variant"],
        "dataset": row["dataset"],
        "seed": row["seed"],
        "steps": row["steps"],
        "n_params": row["n_params"],
        "val_loss": row["val_loss"],
        "val_ppl": row["val_ppl"],
        "val_bpc": row["val_bpc"],
        "wall_seconds": row["wall_seconds"],
        "model_family": family,
        "role": _role(config, row.get("variant", "")),
        "link": f"/run/{row['id']}",
        "context": context,
        "resource": _resource_from_config(config),
        "metric_contract": contract,
        "rerun_required": bool(contract and contract["rerun_required"]),
    }


def _job_item(row: dict) -> dict:
    config = _decode_config(row)
    contract = two_stream_metric_contract(suite="lab", config=config)
    context = infer_research_context(
        suite="lab",
        variant=row.get("preset_id", ""),
        dataset=row.get("dataset_name", ""),
        config=config,
    )
    return {
        "kind": "job",
        "id": row["id"],
        "status": row["status"],
        "run_name": row["run_name"],
        "preset_id": row["preset_id"],
        "dataset": row["dataset_name"],
        "seed": row["seed"],
        "steps": row["steps"],
        "group_id": row.get("group_id"),
        "comparison_role": row.get("comparison_role"),
        "compare_to_job_id": row.get("compare_to_job_id"),
        "model_family": model_family(config) if config else row.get("preset_id"),
        "role": _role(config, row.get("preset_id", "")),
        "link": f"/jobs/{row['id']}",
        "comparison_link": (
            f"/comparisons/{row['id']}" if row.get("compare_to_job_id") else None
        ),
        "context": context,
        "resource": _resource_from_config(config),
        "metric_contract": contract,
        "rerun_required": bool(contract and contract["rerun_required"]),
    }


def _all_run_items(db: ResultsDB) -> list[dict]:
    with db._conn() as con:
        rows = con.execute("SELECT * FROM runs ORDER BY ts DESC").fetchall()
    return [_run_item(dict(row)) for row in rows]


def _all_job_items(db: ResultsDB) -> list[dict]:
    return [_job_item(row) for row in db.fetch_lab_jobs(limit=500)]


def _job_variant(job: dict) -> str:
    if job.get("run_key"):
        parts = str(job["run_key"]).split("/")
        if len(parts) >= 2:
            return parts[1]
    config = _decode_config(job)
    q = config.get("lab.quantum_override.n_qubits")
    d = config.get("lab.quantum_override.n_circuit_layers")
    if q is not None and d is not None:
        return f"{job['preset_id']}-q{q}-d{d}"
    return job["preset_id"]


def _lab_final_run(db: ResultsDB, job: dict) -> dict | None:
    return db.get_run(
        "lab",
        _job_variant(job),
        job["dataset_name"],
        int(job["seed"]),
        int(job["steps"]),
    )


def explore_payload(db: ResultsDB) -> dict:
    runs = _all_run_items(db)
    jobs = _all_job_items(db)
    datasets = list_datasets(db)
    items = [*runs, *jobs]

    by_domain: dict[str, dict] = {}
    by_task: dict[tuple[str, str], dict] = {}
    by_dataset: dict[str, dict] = {}
    studies: dict[str, dict] = {}

    for item in items:
        context = item["context"]
        domain_key = context["domain_slug"]
        task_key = (domain_key, context["task_slug"])
        dataset_key = item["dataset"]

        domain = by_domain.setdefault(domain_key, {
            "name": context["domain"],
            "slug": domain_key,
            "tasks": set(),
            "datasets": set(),
            "runs": 0,
            "jobs": 0,
            "inferred_from": set(),
        })
        domain["tasks"].add(context["task"])
        domain["datasets"].add(dataset_key)
        domain["runs"] += 1 if item["kind"] == "run" else 0
        domain["jobs"] += 1 if item["kind"] == "job" else 0
        domain["inferred_from"].update(context["inferred_from"])

        task = by_task.setdefault(task_key, {
            "domain": context["domain"],
            "domain_slug": domain_key,
            "name": context["task"],
            "slug": context["task_slug"],
            "datasets": set(),
            "runs": 0,
            "jobs": 0,
        })
        task["datasets"].add(dataset_key)
        task["runs"] += 1 if item["kind"] == "run" else 0
        task["jobs"] += 1 if item["kind"] == "job" else 0

        dataset = by_dataset.setdefault(dataset_key, {
            "name": dataset_key,
            "domains": set(),
            "tasks": set(),
            "runs": 0,
            "jobs": 0,
            "best_val_ppl": None,
            "link": f"/explore/dataset/{dataset_key}",
        })
        dataset["domains"].add(context["domain"])
        dataset["tasks"].add(context["task"])
        dataset["runs"] += 1 if item["kind"] == "run" else 0
        dataset["jobs"] += 1 if item["kind"] == "job" else 0
        if (
            item["kind"] == "run"
            and item.get("val_ppl") is not None
            and not item.get("rerun_required")
        ):
            current = dataset["best_val_ppl"]
            dataset["best_val_ppl"] = item["val_ppl"] if current is None else min(current, item["val_ppl"])

        group_id = item.get("group_id") or f"{item.get('suite', 'run')}::{dataset_key}"
        study = studies.setdefault(group_id, {
            "id": group_id,
            "dataset": dataset_key,
            "domain": context["domain"],
            "task": context["task"],
            "runs": 0,
            "jobs": 0,
            "models": set(),
        })
        study["runs"] += 1 if item["kind"] == "run" else 0
        study["jobs"] += 1 if item["kind"] == "job" else 0
        study["models"].add(item.get("variant") or item.get("preset_id") or item.get("model_family"))

    def finish_domain(row):
        return {
            **row,
            "tasks": sorted(row["tasks"]),
            "datasets": sorted(row["datasets"]),
            "inferred_from": sorted(row["inferred_from"]),
        }

    def finish_task(row):
        return {**row, "datasets": sorted(row["datasets"])}

    def finish_dataset(row):
        return {
            **row,
            "domains": sorted(row["domains"]),
            "tasks": sorted(row["tasks"]),
        }

    def finish_study(row):
        return {**row, "models": sorted(v for v in row["models"] if v)}

    return {
        "domains": sorted((finish_domain(v) for v in by_domain.values()), key=lambda x: x["name"]),
        "tasks": sorted((finish_task(v) for v in by_task.values()), key=lambda x: (x["domain"], x["name"])),
        "datasets": sorted((finish_dataset(v) for v in by_dataset.values()), key=lambda x: x["name"]),
        "studies": sorted((finish_study(v) for v in studies.values()), key=lambda x: (x["dataset"], x["id"])),
        "runs": runs,
        "jobs": jobs,
        "registered_datasets": datasets,
    }


def domain_payload(db: ResultsDB, domain_slug: str) -> dict:
    payload = explore_payload(db)
    domain = next((d for d in payload["domains"] if d["slug"] == domain_slug), None)
    if domain is None:
        return {"available": False, "reason": "domain not found", "domain_slug": domain_slug}
    tasks = [t for t in payload["tasks"] if t["domain_slug"] == domain_slug]
    datasets = [d for d in payload["datasets"] if domain["name"] in d["domains"]]
    runs = [r for r in payload["runs"] if r["context"]["domain_slug"] == domain_slug]
    jobs = [j for j in payload["jobs"] if j["context"]["domain_slug"] == domain_slug]
    return {
        "available": True,
        "domain": domain,
        "tasks": tasks,
        "datasets": datasets,
        "runs": runs,
        "jobs": jobs,
    }


def _summary_pick(rows: list[dict], predicate) -> dict | None:
    candidates = [row for row in rows if predicate(row) and row.get("val_ppl") is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda row: row["val_ppl"])


def _summary_card(label: str, row: dict | None, note: str) -> dict:
    if row is None:
        return {"label": label, "available": False, "note": note}
    return {
        "label": label,
        "available": True,
        "model": row["model"],
        "role": row["role"],
        "val_ppl": row.get("val_ppl"),
        "wall_seconds": row.get("wall_seconds"),
        "n_params": row.get("n_params"),
        "resource": row.get("resource", {}),
        "verdict_label": row.get("verdict_label"),
        "link": row.get("link"),
        "note": note,
    }


def _run_result_row(row: dict, metrics_by_key: dict[tuple, dict]) -> dict:
    item = _run_item(row)
    contract = item.get("metric_contract") or {}
    metrics = metrics_by_key.get((row["suite"], row["variant"], row["dataset"], row["seed"]), {})
    return {
        "source": "run",
        "id": row["id"],
        "run_name": row["variant"],
        "model": row["variant"],
        "model_family": item["model_family"],
        "role": item["role"],
        "dataset": row["dataset"],
        "task": item["context"]["task"],
        "domain": item["context"]["domain"],
        "seed": row["seed"],
        "steps": row["steps"],
        "val_loss": row["val_loss"],
        "val_ppl": row["val_ppl"],
        "val_bpc": row["val_bpc"],
        "accuracy": metrics.get("accuracy"),
        "wall_seconds": row["wall_seconds"],
        "n_params": row["n_params"],
        "resource": item["resource"],
        "metric_contract": item.get("metric_contract"),
        "metric_type": contract.get("metric_type"),
        "rerun_required": bool(contract.get("rerun_required")),
        "verdict_label": None,
        "claim_level": None,
        "link": item["link"],
        "comparison_link": None,
        "inferred_from": item["context"]["inferred_from"],
    }


def _job_result_row(db: ResultsDB, job: dict) -> dict:
    item = _job_item(job)
    contract = item.get("metric_contract") or {}
    final = _lab_final_run(db, job) or {}
    verdict = {}
    resource_normalized = None
    if job.get("compare_to_job_id"):
        try:
            from .lab import comparison_research_payload

            comparison = comparison_research_payload(db, int(job["id"]))
            verdict = comparison.get("verdict") or {}
            resource_normalized = comparison.get("resource_normalized")
        except Exception:
            verdict = {}
    return {
        "source": "job",
        "id": job["id"],
        "run_name": job["run_name"],
        "model": job["preset_id"],
        "model_family": item["model_family"],
        "role": item["role"],
        "dataset": job["dataset_name"],
        "task": item["context"]["task"],
        "domain": item["context"]["domain"],
        "seed": job["seed"],
        "steps": job["steps"],
        "val_loss": final.get("val_loss"),
        "val_ppl": final.get("val_ppl"),
        "val_bpc": final.get("val_bpc"),
        "accuracy": None,
        "wall_seconds": final.get("wall_seconds"),
        "n_params": final.get("n_params"),
        "resource": item["resource"],
        "metric_contract": item.get("metric_contract"),
        "metric_type": contract.get("metric_type"),
        "rerun_required": bool(contract.get("rerun_required")),
        "verdict_label": verdict.get("label"),
        "claim_level": verdict.get("claim_level"),
        "resource_normalized": resource_normalized,
        "link": item["link"],
        "comparison_link": item["comparison_link"],
        "inferred_from": item["context"]["inferred_from"],
    }


def result_dashboard_payload(
    db: ResultsDB,
    *,
    dataset: str | None = None,
    task_slug: str | None = None,
    domain_slug: str | None = None,
) -> dict:
    """Focused result dashboard for a dataset or inferred task."""
    with db._conn() as con:
        run_rows = [dict(row) for row in con.execute("SELECT * FROM runs ORDER BY ts DESC").fetchall()]
    lab_jobs = db.fetch_lab_jobs(limit=500)
    lab_run_keys = {
        ("lab", _job_variant(job), job["dataset_name"], int(job["seed"]), int(job["steps"]))
        for job in lab_jobs
    }
    run_rows = [
        row for row in run_rows
        if (row["suite"], row["variant"], row["dataset"], int(row["seed"]), int(row["steps"])) not in lab_run_keys
    ]
    metrics_by_key: dict[tuple, dict] = defaultdict(dict)
    for metric in db.fetch_metrics("lab"):
        metrics_by_key[(metric["suite"], metric["variant"], metric["dataset"], metric["seed"])][metric["name"]] = metric["value"]
    for metric in [m for suite in {r["suite"] for r in run_rows if r.get("suite")} for m in db.fetch_metrics(suite)]:
        metrics_by_key[(metric["suite"], metric["variant"], metric["dataset"], metric["seed"])][metric["name"]] = metric["value"]

    rows = [_run_result_row(row, metrics_by_key) for row in run_rows]
    rows.extend(_job_result_row(db, job) for job in lab_jobs)

    if dataset is not None:
        rows = [row for row in rows if row["dataset"] == dataset]
    if task_slug is not None:
        rows = [row for row in rows if _slug(row["task"]) == task_slug]
    if domain_slug is not None:
        rows = [row for row in rows if _slug(row["domain"]) == domain_slug]

    if not rows:
        return {
            "available": False,
            "reason": "no matching runs or jobs",
            "dataset": dataset,
            "task_slug": task_slug,
            "domain_slug": domain_slug,
        }

    completed = [row for row in rows if row.get("val_ppl") is not None]
    current = [row for row in completed if not row.get("rerun_required")]
    champion = _summary_pick(current, lambda row: True)
    best_quantum = _summary_pick(current, lambda row: row["role"] in {"quantum", "hybrid"})
    best_classical = _summary_pick(current, lambda row: row["role"] == "classical")
    matched = _summary_pick(
        current,
        lambda row: "matched" in row["model"].lower() or "classical-matched" in row["model"].lower(),
    )
    frozen = _summary_pick(
        current,
        lambda row: any(token in row["model"].lower() for token in ("frozen", "random")),
    )
    promising = _summary_pick(
        current,
        lambda row: row["role"] in {"quantum", "hybrid"} and row.get("verdict_label") not in {"insufficient fairness", "no evidence"},
    ) or best_quantum
    evidence = next((row for row in current if row.get("verdict_label")), None)

    protocol_warnings = sorted({
        row["metric_contract"]["limitation"]
        for row in rows
        if row.get("rerun_required") and row.get("metric_contract")
    })

    summaries = [
        _summary_card("Champion model overall", champion, "Lowest validation perplexity in this slice."),
        _summary_card("Best quantum model", best_quantum, "Quantum and hybrid runs are shown with resource cost."),
        _summary_card("Best classical model", best_classical, "Classical control with the lowest validation perplexity."),
        _summary_card("Best parameter-matched classical baseline", matched, "Available when a matched baseline row exists."),
        _summary_card("Best frozen/random quantum control", frozen, "Available when random-feature or frozen controls exist."),
        _summary_card("Most promising quantum candidate", promising, "Cautious candidate; inspect fairness and cost before claiming advantage."),
        _summary_card("Current evidence label", evidence, "Uses stricter protocol verdicts when linked comparisons exist."),
    ]

    rows.sort(key=lambda row: (row.get("val_ppl") is None, row.get("val_ppl") or 0, row["model"]))
    return {
        "available": True,
        "dataset": dataset,
        "task_slug": task_slug,
        "domain_slug": domain_slug,
        "title": dataset or task_slug or domain_slug or "Results",
        "domains": sorted({row["domain"] for row in rows}),
        "tasks": sorted({row["task"] for row in rows}),
        "protocol_warnings": protocol_warnings,
        "summaries": summaries,
        "rows": rows,
    }
