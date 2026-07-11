"""Manual testing helpers for completed dashboard jobs."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from ..data.datasets import load_dataset
from ..data.text import sample_batch
from ..models.model import build_model
from ..resultsdb import ResultsDB
from ..train.loop import generate_outcome
from .analogues import config_from_flat_payload
from .security import resolve_data_path, resolve_within


def _decode_config(job: dict) -> dict:
    config = job.get("config")
    if config is not None:
        return config
    try:
        return json.loads(job.get("config_json") or "{}")
    except json.JSONDecodeError:
        return {}


def _artifact_dir(results_dir: str | Path, job: dict) -> Path:
    root = Path(results_dir).resolve()
    trusted = job.get("artifact_dir")
    if trusted:
        return resolve_within(root, trusted, label="persisted artifact directory")
    checkpoint = job.get("checkpoint_path")
    if checkpoint:
        safe_checkpoint = resolve_within(
            root, checkpoint, label="persisted checkpoint"
        )
        return resolve_within(
            root,
            safe_checkpoint.parent.parent,
            label="checkpoint artifact directory",
        )
    return resolve_within(root, str(job["run_name"]), label="legacy run artifact")


def model_test_payload(
    db: ResultsDB,
    job_id: int,
    results_dir: str | Path = "results",
    data_dir: str | Path = "data",
) -> dict:
    job = db.get_lab_job(job_id)
    if job is None:
        raise KeyError(f"Unknown job {job_id}")
    config = _decode_config(job)
    authorized_corpus = None
    data_error = None
    if config:
        try:
            configured = config_from_flat_payload(config)
            authorized_corpus = resolve_data_path(
                data_dir,
                configured.data.corpus_path,
                label="persisted dataset corpus path",
            )
        except (KeyError, TypeError, ValueError) as exc:
            data_error = str(exc)
    out_dir = _artifact_dir(results_dir, job)
    summary_path = resolve_within(
        out_dir, out_dir / "summary.json", label="summary artifact"
    )
    params_path = resolve_within(
        out_dir, out_dir / "params.msgpack", label="parameter artifact"
    )
    summary = None
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            summary = {"error": "summary.json is not valid JSON"}
    can_generate = (
        job.get("status") == "done"
        and params_path.exists()
        and bool(config)
        and data_error is None
        and (config.get("data.kind") in (None, "text"))
    )
    reasons = []
    if job.get("status") != "done":
        reasons.append("job is not complete")
    if not params_path.exists():
        reasons.append("params.msgpack artifact is missing")
    if not config:
        reasons.append("job config is missing")
    if config.get("data.kind") not in (None, "text"):
        reasons.append("manual text generation currently supports text datasets only")
    if data_error is not None:
        reasons.append(data_error)
    return {
        "job": {"id": job["id"], "run_name": job["run_name"], "status": job["status"]},
        "artifacts": {
            "directory": str(out_dir),
            "summary_path": str(summary_path),
            "params_path": str(params_path),
            "summary_exists": summary_path.exists(),
            "params_exists": params_path.exists(),
        },
        "summary": summary,
        "data": {
            "corpus_path": str(authorized_corpus) if authorized_corpus else None,
            "authorized": data_error is None and authorized_corpus is not None,
        },
        "supported_tests": {
            "summary_review": summary_path.exists(),
            "prompt_generation": can_generate,
        },
        "unsupported_reasons": reasons,
    }


def run_model_test(
    db: ResultsDB,
    job_id: int,
    payload: dict,
    results_dir: str | Path = "results",
    data_dir: str | Path = "data",
) -> dict:
    info = model_test_payload(db, job_id, results_dir, data_dir)
    if not info["supported_tests"]["prompt_generation"]:
        return {
            "ok": False,
            "supported": False,
            "status": "unsupported",
            "kind": "prompt_generation",
            "reason": "; ".join(info["unsupported_reasons"]) or "prompt generation is unavailable",
            "generated_text": None,
            "capabilities": info,
        }

    job = db.get_lab_job(job_id)
    config = _decode_config(job)
    cfg = config_from_flat_payload(config)
    cfg = dataclasses.replace(
        cfg,
        data=dataclasses.replace(
            cfg.data, corpus_path=str(info["data"]["corpus_path"])
        ),
    )
    ids, tokenizer = load_dataset(cfg.data)
    model, model_cfg = build_model(cfg.model, vocab_size=tokenizer.vocab_size)
    rng = np.random.default_rng(cfg.train.seed)
    sample = jnp.asarray(
        sample_batch(rng, ids, max(1, min(cfg.train.batch_size, 4)), cfg.train.seq_len)
    )
    init_params = model.init(jax.random.PRNGKey(cfg.train.seed), sample[:, :-1])["params"]
    params_path = Path(info["artifacts"]["params_path"])
    params = serialization.from_bytes(init_params, params_path.read_bytes())
    prompt = str(payload.get("prompt") or "\n")
    try:
        max_new_tokens = max(
            1, min(int(payload.get("max_new_tokens") or 120), 240)
        )
        temperature = float(payload.get("temperature") or 0.8)
        seed = int(payload.get("seed") or cfg.train.seed)
    except (TypeError, ValueError, OverflowError) as exc:
        return {
            "ok": False,
            "supported": False,
            "status": "unsupported",
            "kind": "prompt_generation",
            "reason": f"invalid generation setting: {exc}",
            "generated_text": None,
            "capabilities": info,
        }
    outcome = generate_outcome(
        model,
        params,
        tokenizer,
        model_cfg=model_cfg,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        seed=seed,
    )
    return {
        **outcome,
        "prompt": prompt,
        "settings": {**outcome["settings"], "arch": model_cfg.arch},
        "capabilities": info,
    }
