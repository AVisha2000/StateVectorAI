"""Classical analogue discovery and job metadata helpers."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from typing import Any

from ..config import (
    BlockConfig,
    ExperimentConfig,
    ModelConfig,
    from_dict,
    to_flat_dict,
)
from ..resultsdb import ResultsDB
from ..registry import QUANTUM_ATTN_TYPES, QUANTUM_FFN_TYPES
from .model_graph import uses_quantum_config

QUANTUM_ATTN = frozenset(QUANTUM_ATTN_TYPES)
QUANTUM_FFN = frozenset(QUANTUM_FFN_TYPES)
DEFAULT_FAIRNESS_REQUIREMENTS = (
    "same_dataset",
    "same_seed",
    "same_steps",
    "same_eval_every",
    "same_train_split",
    "same_preprocessing",
    "same_batch_size",
    "same_sequence_length",
)


@dataclass(frozen=True)
class AnalogueSpec:
    kind: str
    analogue_type: str
    resolver: str
    label: str
    reason: str
    source_preset_id: str | None = None
    source_job_id: int | None = None
    analogue_preset_id: str | None = None
    analogue_model_spec_id: int | None = None
    config: ExperimentConfig | None = None
    fairness_requirements: tuple[str, ...] = DEFAULT_FAIRNESS_REQUIREMENTS
    known_limitations: tuple[str, ...] = ()

    def with_model_spec(self, spec_id: int) -> "AnalogueSpec":
        return dataclasses.replace(self, analogue_model_spec_id=int(spec_id))

    def to_payload(self, include_config: bool = False) -> dict:
        payload = {
            "kind": self.kind,
            "analogue_type": self.analogue_type,
            "resolver": self.resolver,
            "label": self.label,
            "reason": self.reason,
            "source_preset_id": self.source_preset_id,
            "source_job_id": self.source_job_id,
            "analogue_preset_id": self.analogue_preset_id,
            "analogue_model_spec_id": self.analogue_model_spec_id,
            "fairness_requirements": list(self.fairness_requirements),
            "known_limitations": list(self.known_limitations),
        }
        if include_config and self.config is not None:
            payload["config"] = dataclasses.asdict(self.config)
            payload["flat_config"] = to_flat_dict(self.config)
        return payload


def flat_to_nested_config(flat: dict[str, Any] | None) -> dict:
    """Convert dotted tracking config keys back into config sections."""
    out: dict[str, Any] = {}

    def assign(cursor: dict | list, parts: list[str], value: Any) -> None:
        part = parts[0]
        last = len(parts) == 1
        if part.isdigit():
            index = int(part)
            if not isinstance(cursor, list):
                raise TypeError("Numeric config path segment requires a list container.")
            while len(cursor) <= index:
                cursor.append(None)
            if last:
                cursor[index] = value
                return
            if cursor[index] is None:
                cursor[index] = [] if parts[1].isdigit() else {}
            assign(cursor[index], parts[1:], value)
            return

        if not isinstance(cursor, dict):
            raise TypeError("Named config path segment requires a dict container.")
        if last:
            cursor[part] = value
            return
        if part not in cursor or cursor[part] is None:
            cursor[part] = [] if parts[1].isdigit() else {}
        assign(cursor[part], parts[1:], value)

    for key, value in (flat or {}).items():
        if not isinstance(key, str) or "." not in key:
            continue
        assign(out, key.split("."), value)
    return out


def config_from_flat_payload(flat: dict[str, Any] | None) -> ExperimentConfig:
    sections = {"model", "train", "data", "tracking"}
    config_fields = {
        key: value
        for key, value in (flat or {}).items()
        if isinstance(key, str) and key.partition(".")[0] in sections
    }
    return from_dict(flat_to_nested_config(config_fields))


def _source_config_from_job(job: dict) -> dict:
    config = job.get("config")
    if config is not None:
        return config
    try:
        return json.loads(job.get("config_json") or "{}")
    except json.JSONDecodeError:
        return {}


def _swap_blocks(model: ModelConfig, changes: list[str]) -> tuple[BlockConfig, ...] | None:
    if model.blocks is None:
        return None
    blocks = []
    for index, block in enumerate(model.blocks):
        attn_type = block.attn_type
        ffn_type = block.ffn_type
        if attn_type in QUANTUM_ATTN:
            attn_type = "classical"
            changes.append(f"block {index + 1} quantum attention -> classical attention")
        if ffn_type in QUANTUM_FFN:
            ffn_type = "classical"
            changes.append(f"block {index + 1} quantum FFN -> classical FFN")
        blocks.append(BlockConfig(attn_type=attn_type, ffn_type=ffn_type, quantum=None))
    return tuple(blocks)


def classical_analogue_config(cfg: ExperimentConfig) -> tuple[ExperimentConfig, list[str], list[str]] | None:
    model = cfg.model
    updates: dict[str, Any] = {}
    changes: list[str] = []
    limitations: list[str] = [
        "Parameter matching is verified after training metrics are recorded.",
    ]

    if model.arch == "qrnn":
        updates["arch"] = "gru"
        updates["rnn_hidden"] = max(int(model.rnn_hidden or 0), int(model.d_model or 0), 16)
        changes.append("quantum recurrent memory -> GRU recurrent core")
        limitations.append("The recurrent analogue changes cell family while preserving the sequence task.")
    if model.embed_type == "quantum":
        updates["embed_type"] = "classical"
        changes.append("quantum embedding -> classical embedding")
    if model.encoder_kind == "quantum":
        updates["encoder_kind"] = "classical"
        changes.append("quantum sentence encoder -> classical sentence encoder")
    if model.attn_type in QUANTUM_ATTN:
        updates["attn_type"] = "classical"
        changes.append("quantum attention projection -> classical attention")
    if model.ffn_type in QUANTUM_FFN:
        updates["ffn_type"] = "classical"
        changes.append("quantum FFN -> classical FFN")
    if model.head_type == "interference":
        updates["head_type"] = "linear"
        changes.append("interference head -> linear head")

    blocks = _swap_blocks(model, changes)
    if blocks is not None:
        updates["blocks"] = blocks

    if not changes:
        return None

    analogue_model = dataclasses.replace(model, **updates)
    analogue = dataclasses.replace(
        cfg,
        model=analogue_model,
        tracking=dataclasses.replace(
            cfg.tracking,
            run_name=f"{cfg.tracking.run_name or 'model'}-classical-analogue",
            log_quantum_diagnostics=False,
        ),
    )
    return analogue, changes, limitations


def classical_analogue_for_config(
    cfg: ExperimentConfig,
    *,
    source_preset_id: str | None = None,
    source_job_id: int | None = None,
    label: str | None = None,
) -> AnalogueSpec | None:
    analogue = classical_analogue_config(cfg)
    if analogue is None:
        return None
    analogue_cfg, changes, limitations = analogue
    reason = (
        "Automatic component-swap analogue: "
        + "; ".join(changes)
        + ". Dataset, seed, training budget, preprocessing, batch size, and sequence length are preserved at queue time."
    )
    return AnalogueSpec(
        kind="classical_analogue",
        analogue_type="component_swap",
        resolver="automatic_component_swap",
        label=label or "Automatic classical analogue",
        reason=reason,
        source_preset_id=source_preset_id,
        source_job_id=source_job_id,
        config=analogue_cfg,
        known_limitations=tuple(dict.fromkeys(limitations)),
    )


def classical_analogue_for_preset(preset_id: str) -> AnalogueSpec | None:
    from .presets import build_preset, classical_twin_id, preset_meta

    twin_id = classical_twin_id(preset_id)
    if twin_id:
        twin = preset_meta(twin_id)
        return AnalogueSpec(
            kind="classical_analogue",
            analogue_type="component_swap",
            resolver="curated_twin",
            label=twin["label"],
            reason=(
                f"Uses curated classical twin '{twin_id}' from preset metadata. "
                "Queueing preserves dataset, seed, steps, eval cadence, preprocessing, batch size, and sequence length."
            ),
            source_preset_id=preset_id,
            analogue_preset_id=twin_id,
            config=build_preset(twin_id),
            known_limitations=(
                "Curated twin is the repository-defined fair comparison, not necessarily exact parameter matching before training.",
            ),
        )
    return classical_analogue_for_config(
        build_preset(preset_id),
        source_preset_id=preset_id,
    )


def analogue_config_fields(
    spec: AnalogueSpec | None,
    *,
    state: str | None = None,
    role: str | None = None,
    analogue_job_id: int | None = None,
) -> dict[str, Any]:
    if spec is None:
        return {}
    payload = spec.to_payload(include_config=False)
    fields: dict[str, Any] = {
        "lab.analogue.kind": payload["kind"],
        "lab.analogue.type": payload["analogue_type"],
        "lab.analogue.resolver": payload["resolver"],
        "lab.analogue.label": payload["label"],
        "lab.analogue.reason": payload["reason"],
        "lab.analogue.fairness_requirements": payload["fairness_requirements"],
        "lab.analogue.known_limitations": payload["known_limitations"],
    }
    optional = {
        "source_preset_id": payload.get("source_preset_id"),
        "source_job_id": payload.get("source_job_id"),
        "analogue_preset_id": payload.get("analogue_preset_id"),
        "analogue_model_spec_id": payload.get("analogue_model_spec_id"),
        "state": state,
        "role": role,
        "job_id": analogue_job_id,
    }
    for key, value in optional.items():
        if value is not None:
            fields[f"lab.analogue.{key}"] = value
    return fields


def _metadata_from_config(config: dict[str, Any]) -> dict:
    prefix = "lab.analogue."
    return {
        key.removeprefix(prefix): value
        for key, value in config.items()
        if isinstance(key, str) and key.startswith(prefix)
    }


def _state_for_status(status: str | None) -> str:
    if status == "error":
        return "failed"
    if status in {"queued", "running", "done", "cancelled"}:
        return status
    return status or "unknown"


def analogue_status_for_job(db: ResultsDB | None, job: dict) -> dict:
    config = _source_config_from_job(job)
    meta = _metadata_from_config(config)
    role = job.get("comparison_role") or "primary"
    compare_to = job.get("compare_to_job_id")

    if role == "baseline":
        return {
            "analogue_state": "baseline",
            "analogue_job_id": compare_to,
            "analogue_preset_id": meta.get("source_preset_id"),
            "analogue_type": meta.get("type"),
            "analogue": None,
            "comparison_group_id": job.get("group_id"),
        }

    if compare_to:
        other = db.get_lab_job(int(compare_to)) if db is not None else None
        other_status = _state_for_status(other.get("status") if other else None)
        other_preset = other.get("preset_id") if other else meta.get("analogue_preset_id")
        model_spec_id = None
        if isinstance(other_preset, str) and other_preset.startswith("model-spec:"):
            try:
                model_spec_id = int(other_preset.split(":", 1)[1])
            except ValueError:
                model_spec_id = None
        return {
            "analogue_state": other_status,
            "analogue_job_id": int(compare_to),
            "analogue_preset_id": None if model_spec_id else other_preset,
            "analogue_model_spec_id": model_spec_id or meta.get("analogue_model_spec_id"),
            "analogue_type": meta.get("type") or "component_swap",
            "analogue": {
                "kind": meta.get("kind", "classical_analogue"),
                "analogue_type": meta.get("type", "component_swap"),
                "resolver": meta.get("resolver"),
                "label": meta.get("label"),
                "reason": meta.get("reason"),
                "fairness_requirements": meta.get("fairness_requirements") or [],
                "known_limitations": meta.get("known_limitations") or [],
            },
            "comparison_group_id": job.get("group_id"),
        }

    if meta:
        return {
            "analogue_state": meta.get("state") or "missing",
            "analogue_job_id": None,
            "analogue_preset_id": meta.get("analogue_preset_id"),
            "analogue_model_spec_id": meta.get("analogue_model_spec_id"),
            "analogue_type": meta.get("type"),
            "analogue": {
                "kind": meta.get("kind", "classical_analogue"),
                "analogue_type": meta.get("type", "component_swap"),
                "resolver": meta.get("resolver"),
                "label": meta.get("label"),
                "reason": meta.get("reason"),
                "fairness_requirements": meta.get("fairness_requirements") or [],
                "known_limitations": meta.get("known_limitations") or [],
            },
            "comparison_group_id": job.get("group_id"),
        }

    try:
        preset_id = job.get("preset_id")
        if preset_id and not str(preset_id).startswith("model-spec:"):
            spec = classical_analogue_for_preset(str(preset_id))
            if spec:
                return {
                    "analogue_state": "missing",
                    "analogue_job_id": None,
                    "analogue_preset_id": spec.analogue_preset_id,
                    "analogue_model_spec_id": spec.analogue_model_spec_id,
                    "analogue_type": spec.analogue_type,
                    "analogue": spec.to_payload(include_config=False),
                    "comparison_group_id": job.get("group_id"),
                }
    except Exception:
        pass

    try:
        if uses_quantum_config(config):
            spec = classical_analogue_for_config(
                config_from_flat_payload(config),
                source_preset_id=job.get("preset_id"),
                source_job_id=job.get("id"),
            )
            if spec:
                return {
                    "analogue_state": "missing",
                    "analogue_job_id": None,
                    "analogue_preset_id": spec.analogue_preset_id,
                    "analogue_model_spec_id": spec.analogue_model_spec_id,
                    "analogue_type": spec.analogue_type,
                    "analogue": spec.to_payload(include_config=False),
                    "comparison_group_id": job.get("group_id"),
                }
    except Exception:
        pass
    return {
        "analogue_state": "none",
        "analogue_job_id": None,
        "analogue_preset_id": None,
        "analogue_type": None,
        "analogue": None,
        "comparison_group_id": job.get("group_id"),
    }
