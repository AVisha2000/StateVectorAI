"""Editable model spec services for QLLM Lab."""
from __future__ import annotations

import dataclasses
from copy import deepcopy
from typing import Any

from ..config import ExperimentConfig, from_dict, to_flat_dict, validate_config
from ..resultsdb import ResultsDB
from .model_graph import model_graph_from_config
from .resources import quantum_resource_estimate


def config_payload(cfg: ExperimentConfig) -> dict:
    return dataclasses.asdict(cfg)


def cfg_from_payload(payload: dict) -> ExperimentConfig:
    return from_dict(payload)


def _quantum_template(qcfg: dict) -> list[dict]:
    n_qubits = int(qcfg.get("n_qubits", 4))
    depth = int(qcfg.get("n_circuit_layers", 2))
    ansatz = qcfg.get("ansatz", "reuploading")
    gates = [{"gate": "AngleEmbedding", "wires": list(range(n_qubits)), "trainable": False}]
    for layer in range(depth):
        if ansatz == "reuploading":
            gates.append({"gate": "AngleEmbedding", "layer": layer + 1, "wires": list(range(n_qubits)), "trainable": False})
        gates.append({"gate": "Rot", "layer": layer + 1, "wires": list(range(n_qubits)), "trainable": True})
        gates.append({"gate": "Entangle", "layer": layer + 1, "pattern": "strongly_entangling", "trainable": False})
    return gates


def _graph_with_circuits(cfg: ExperimentConfig) -> dict:
    graph = model_graph_from_config(cfg)
    qcfg = dataclasses.asdict(cfg.model.quantum)
    estimate = quantum_resource_estimate(cfg)
    for node in graph["nodes"]:
        if node["kind"] == "quantum":
            node["circuit"] = {
                "quantum": qcfg,
                "template": _quantum_template(qcfg),
                "resource": estimate,
            }
    return graph


def validation_payload(config: dict) -> dict:
    cfg = cfg_from_payload(config)
    errors = validate_config(cfg)
    warnings = quantum_resource_estimate(cfg)["advice"]
    graph = _graph_with_circuits(cfg)
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "resource": quantum_resource_estimate(cfg),
        "graph": graph,
        "flat_config": to_flat_dict(cfg),
    }


def create_spec(db: ResultsDB, payload: dict) -> dict:
    cfg_payload = payload.get("config") or {}
    validation = validation_payload(cfg_payload)
    if not validation["ok"]:
        raise ValueError("; ".join(validation["errors"]))
    spec_id = db.create_model_spec({
        "name": payload.get("name") or "Untitled model",
        "source": payload.get("source"),
        "parent_id": payload.get("parent_id"),
        "version": payload.get("version") or 1,
        "notes": payload.get("notes"),
        "config": cfg_payload,
        "graph": validation["graph"],
    })
    return get_spec(db, spec_id)


def update_spec(db: ResultsDB, spec_id: int, payload: dict) -> dict:
    spec = db.get_model_spec(spec_id)
    if spec is None:
        raise KeyError(f"Unknown model spec {spec_id}")
    updates: dict[str, Any] = {}
    if "config" in payload:
        validation = validation_payload(payload["config"])
        if not validation["ok"]:
            raise ValueError("; ".join(validation["errors"]))
        updates["config"] = payload["config"]
        updates["graph"] = validation["graph"]
    for key in ("name", "source", "parent_id", "version", "notes"):
        if key in payload:
            updates[key] = payload[key]
    db.update_model_spec(spec_id, **updates)
    return get_spec(db, spec_id)


def list_specs(db: ResultsDB) -> list[dict]:
    return [_public_spec(row) for row in db.fetch_model_specs()]


def get_spec(db: ResultsDB, spec_id: int) -> dict:
    row = db.get_model_spec(spec_id)
    if row is None:
        raise KeyError(f"Unknown model spec {spec_id}")
    return _public_spec(row)


def _public_spec(row: dict) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "source": row.get("source"),
        "parent_id": row.get("parent_id"),
        "version": row.get("version"),
        "notes": row.get("notes"),
        "ts": row.get("ts"),
        "updated_ts": row.get("updated_ts"),
        "config": row.get("config") or {},
        "graph": row.get("graph") or {},
    }


def diff_configs(current: dict, base: dict) -> list[dict]:
    changes: list[dict] = []

    def walk(a, b, path=""):
        keys = sorted(set((a or {}).keys()) | set((b or {}).keys()))
        for key in keys:
            av = (a or {}).get(key)
            bv = (b or {}).get(key)
            next_path = f"{path}.{key}" if path else key
            if isinstance(av, dict) and isinstance(bv, dict):
                walk(av, bv, next_path)
            elif av != bv:
                changes.append({"path": next_path, "before": bv, "after": av})

    walk(current, base)
    return changes


def spec_diff(db: ResultsDB, spec_id: int, base_id: int | None = None) -> dict:
    spec = get_spec(db, spec_id)
    if base_id:
        base = get_spec(db, base_id)
        base_config = base["config"]
    elif spec.get("parent_id"):
        base_config = get_spec(db, int(spec["parent_id"]))["config"]
    else:
        base_config = {}
    return {"changes": diff_configs(spec["config"], base_config)}


def draft_from_preset(preset: dict) -> dict:
    from .presets import build_preset

    return config_payload(build_preset(preset["id"]))


def clone_config(config: dict) -> dict:
    return deepcopy(config)
