"""Self-hosted dashboard API for the QLLM testbed.

Thin FastAPI layer over ResultsDB (the project's own SQLite store). Serves
JSON for the React frontend and the existing matplotlib plots as PNGs. No
MLflow dependency — per-step curves come from the `steps` table this
project writes directly. Run:

    uvicorn qllm.dashboard.server:app --reload --port 8000
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..resultsdb import ResultsDB
from . import queries as Q
from .datasets import import_hf_text_dataset, list_datasets
from .lab import (
    comparison_research_payload,
    enrich_job,
    lab_overview,
    scaling_test_payload,
    scaling_tests_overview,
)
from .model_graph import model_graph_from_config
from .model_specs import (
    create_spec,
    get_spec,
    list_specs,
    spec_diff,
    update_spec,
    validation_payload,
)
from .presets import list_presets
from .presets import build_preset
from .runner import ExperimentQueue
from .status import environment_status
from .workspace import comparison_payload, workspace_payload

DB_PATH = os.environ.get("QLLM_DB", "results/qllm_results.db")
RESULTS_DIR = Path(os.environ.get("QLLM_RESULTS", "results"))
FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"

app = FastAPI(title="QLLM Dashboard", version="0.1")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"])


def db() -> ResultsDB:
    return ResultsDB(DB_PATH)


QUEUE = ExperimentQueue(DB_PATH)


def _payload_error(exc: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "db": DB_PATH}


@app.get("/api/status")
def api_status() -> dict:
    return environment_status(FRONTEND_DIST)


@app.get("/api/presets")
def api_presets() -> list[dict]:
    return list_presets()


@app.get("/api/presets/{preset_id}/model-graph")
def api_preset_model_graph(preset_id: str) -> dict:
    try:
        return model_graph_from_config(build_preset(preset_id))
    except Exception as exc:
        raise _payload_error(exc) from exc


@app.get("/api/model-specs")
def api_model_specs() -> list[dict]:
    return list_specs(db())


@app.post("/api/model-specs")
def api_create_model_spec(payload: dict) -> dict:
    try:
        return create_spec(db(), payload)
    except Exception as exc:
        raise _payload_error(exc) from exc


@app.get("/api/model-specs/{spec_id}")
def api_model_spec(spec_id: int) -> dict:
    try:
        return get_spec(db(), spec_id)
    except Exception as exc:
        raise _payload_error(exc) from exc


@app.patch("/api/model-specs/{spec_id}")
def api_update_model_spec(spec_id: int, payload: dict) -> dict:
    try:
        return update_spec(db(), spec_id, payload)
    except Exception as exc:
        raise _payload_error(exc) from exc


@app.post("/api/model-specs/validate")
def api_validate_model_spec_payload(payload: dict) -> dict:
    try:
        return validation_payload(payload.get("config") or payload)
    except Exception as exc:
        raise _payload_error(exc) from exc


@app.post("/api/model-specs/{spec_id}/validate")
def api_validate_model_spec(spec_id: int) -> dict:
    try:
        return validation_payload(get_spec(db(), spec_id)["config"])
    except Exception as exc:
        raise _payload_error(exc) from exc


@app.get("/api/model-specs/{spec_id}/diff")
def api_model_spec_diff(spec_id: int, base: int | None = None) -> dict:
    try:
        return spec_diff(db(), spec_id, base)
    except Exception as exc:
        raise _payload_error(exc) from exc


@app.post("/api/model-specs/{spec_id}/jobs")
def api_create_model_spec_job(spec_id: int, payload: dict) -> dict:
    try:
        return QUEUE.submit_model_spec(
            spec_id=spec_id,
            dataset_name=payload.get("dataset_name", "default-text"),
            run_name=payload.get("run_name") or None,
            seed=int(payload.get("seed") or 0),
            steps=int(payload.get("steps") or 100),
            eval_every=int(payload.get("eval_every") or 20),
            device_target=payload.get("device_target") or "auto",
            batch_size=(
                int(payload["batch_size"]) if payload.get("batch_size") not in (None, "") else None
            ),
            seq_len=(
                int(payload["seq_len"]) if payload.get("seq_len") not in (None, "") else None
            ),
        )
    except Exception as exc:
        raise _payload_error(exc) from exc


@app.get("/api/lab/overview")
def api_lab_overview() -> dict:
    return lab_overview(db(), environment_status(FRONTEND_DIST))


@app.get("/api/scaling-tests")
def api_scaling_tests() -> list[dict]:
    return scaling_tests_overview(db())


@app.get("/api/scaling-tests/{group_id}")
def api_scaling_test(group_id: str) -> dict:
    payload = scaling_test_payload(db(), group_id)
    if not payload.get("available"):
        raise HTTPException(status_code=404, detail=payload.get("reason"))
    return payload


@app.get("/api/datasets")
def api_datasets() -> list[dict]:
    return list_datasets(db())


@app.post("/api/datasets/hf/import")
def api_import_hf(payload: dict) -> dict:
    try:
        return import_hf_text_dataset(
            db(),
            source=payload.get("source", ""),
            split=payload.get("split", "train"),
            text_column=payload.get("text_column", ""),
            display_name=payload.get("display_name") or None,
            row_limit=int(payload.get("row_limit") or 5000),
        )
    except Exception as exc:
        raise _payload_error(exc) from exc


@app.get("/api/jobs")
def api_jobs() -> list[dict]:
    return [enrich_job(j) for j in QUEUE.list()]


@app.get("/api/jobs/{job_id}")
def api_job(job_id: int) -> dict:
    job = QUEUE.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return enrich_job(job)


@app.post("/api/jobs")
def api_create_job(payload: dict) -> dict:
    try:
        return QUEUE.submit(
            preset_id=payload.get("preset_id", ""),
            dataset_name=payload.get("dataset_name", "default-text"),
            run_name=payload.get("run_name") or None,
            seed=int(payload.get("seed") or 0),
            steps=int(payload.get("steps") or 300),
            eval_every=int(payload.get("eval_every") or 50),
            device_target=payload.get("device_target") or "auto",
            queue_classical_comparison=bool(
                payload.get("queue_classical_comparison", False)
            ),
            quantum_overrides=payload.get("quantum_overrides") or None,
            batch_size=(
                int(payload["batch_size"]) if payload.get("batch_size") not in (None, "") else None
            ),
            seq_len=(
                int(payload["seq_len"]) if payload.get("seq_len") not in (None, "") else None
            ),
        )
    except Exception as exc:
        raise _payload_error(exc) from exc


@app.post("/api/jobs/sweep")
def api_create_scaling_sweep(payload: dict) -> dict:
    try:
        return QUEUE.submit_scaling_sweep(
            preset_id=payload.get("preset_id", ""),
            dataset_name=payload.get("dataset_name", "default-text"),
            run_name=payload.get("run_name") or None,
            seed=int(payload.get("seed") or 0),
            steps=int(payload.get("steps") or 100),
            eval_every=int(payload.get("eval_every") or 20),
            device_target=payload.get("device_target") or "gpu",
            qubits=[int(v) for v in payload.get("qubits", [])],
            depths=[int(v) for v in payload.get("depths", [])],
            batch_size=(
                int(payload["batch_size"]) if payload.get("batch_size") not in (None, "") else None
            ),
            seq_len=(
                int(payload["seq_len"]) if payload.get("seq_len") not in (None, "") else None
            ),
        )
    except Exception as exc:
        raise _payload_error(exc) from exc


@app.post("/api/jobs/{job_id}/cancel")
def api_cancel_job(job_id: int) -> dict:
    try:
        return QUEUE.cancel(job_id)
    except Exception as exc:
        raise _payload_error(exc) from exc


@app.get("/api/jobs/{job_id}/workspace")
def api_job_workspace(job_id: int) -> dict:
    payload = workspace_payload(db(), job_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="job not found")
    return payload


@app.get("/api/jobs/{job_id}/comparison")
def api_job_comparison(job_id: int) -> dict:
    return comparison_research_payload(db(), job_id)


@app.get("/api/jobs/{job_id}/model-graph")
def api_job_model_graph(job_id: int) -> dict:
    job = QUEUE.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return model_graph_from_config(job.get("config") or {})


@app.get("/api/suites")
def api_suites() -> list[dict]:
    return Q.suites_overview(db())


@app.get("/api/suite/{suite}")
def api_suite(suite: str, dataset: str | None = None) -> dict:
    return Q.suite_detail(db(), suite, dataset)


@app.get("/api/runs")
def api_runs(suite: str | None = None) -> list[dict]:
    return Q.all_runs(db(), suite)


@app.get("/api/run/{run_id}")
def api_run(run_id: int) -> dict:
    return Q.run_detail(db(), run_id)


@app.get("/api/live")
def api_live() -> list[dict]:
    return Q.live_runs(db())


@app.get("/api/live/{run_key:path}/curve")
def api_live_curve(run_key: str) -> dict:
    return Q.live_curve(db(), run_key)


@app.get("/api/plots")
def api_plots() -> list[str]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(p.name for p in RESULTS_DIR.glob("*.png"))


@app.get("/api/plot/{name}")
def api_plot(name: str):
    path = RESULTS_DIR / name
    if path.exists() and path.suffix == ".png":
        return FileResponse(path)
    return {"error": "not found"}


# serve the built React app, with SPA fallback so client-side routes
# (e.g. /suite/qnlp-v1) resolve to index.html instead of 404 on refresh
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"),
              name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        candidate = FRONTEND_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")
