"""Validated Atlas projection over the canonical research map."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict

from ..claims import list_claims


_DOCS_ROOT = Path(__file__).resolve().parents[2] / "docs"
DEFAULT_ONTOLOGY_PATH = _DOCS_ROOT / "ATLAS_ONTOLOGY.yaml"
DEFAULT_RESEARCH_MAP_PATH = _DOCS_ROOT / "RESEARCH_MAP.yaml"
ATLAS_VERDICT_KEY_PREFIX = "claim:"
ATLAS_VERDICT_SOURCE_KIND = "claim_projection"

_ONTOLOGY_KEYS = {
    "schema_version",
    "updated",
    "purpose",
    "relation_source",
    "domains",
}
_DOMAIN_KEYS = {"id", "label", "description", "cells"}
_CELL_KEYS = {
    "id",
    "area_id",
    "kind",
    "pipeline_stage",
    "quantum_resource",
    "advantage_target",
    "verdict_ref",
    "note",
}
_VERDICT_REF_KEYS = {"verdict_key", "source_kind", "source_id"}
_ATLAS_KINDS = {"head_to_head", "quantum_only", "suggested", "unexplored"}


class AtlasOntologyError(ValueError):
    """Raised when ontology metadata cannot safely project the research map."""


class AtlasVerdictRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict_key: str | None = None
    source_kind: str | None = None
    source_id: str | None = None


class AtlasCell(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    area_id: str
    label: str
    kind: Literal["head_to_head", "quantum_only", "suggested", "unexplored"]
    portfolio: str
    pipeline_stage: str
    quantum_resource: str
    advantage_target: str
    pipeline_stages: list[str]
    quantum_resources: list[str]
    advantage_targets: list[str]
    seed_status: str
    seed_claim_level: str
    seed_replication_status: str
    verdict_ref: AtlasVerdictRef | None = None
    note: str | None = None


class AtlasDomain(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    description: str
    cells: list[AtlasCell]


class AtlasRelation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    from_cell: str
    to_cell: str
    type: str


class AtlasOntologyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    source: Literal["backend-canonical"] = "backend-canonical"
    ontology_updated: str
    research_map_schema_version: int
    research_map_updated: str
    note: str
    claim_levels: list[str]
    replication_statuses: list[str]
    status_values: dict[str, str]
    domains: list[AtlasDomain]
    relations: list[AtlasRelation]


def atlas_ontology_response(
    ontology_path: str | Path = DEFAULT_ONTOLOGY_PATH,
    research_map_path: str | Path = DEFAULT_RESEARCH_MAP_PATH,
) -> AtlasOntologyResponse:
    """Join display-only grouping to map-owned evidence fields and relations."""
    ontology = _load_mapping(Path(ontology_path), label="Atlas ontology")
    research_map = _load_mapping(Path(research_map_path), label="research map")
    _reject_extra_keys(ontology, _ONTOLOGY_KEYS, label="Atlas ontology")

    if int(ontology.get("schema_version") or 0) != 1:
        raise AtlasOntologyError("Atlas ontology schema_version must be 1.")
    if ontology.get("relation_source") != "research_map":
        raise AtlasOntologyError(
            "Atlas relations must be sourced from RESEARCH_MAP.yaml."
        )

    claim_levels = _string_list(research_map.get("claim_levels"), "claim_levels")
    replication_statuses = _string_list(
        research_map.get("replication_statuses"),
        "replication_statuses",
    )
    status_values = _string_mapping(
        research_map.get("status_values"),
        "status_values",
    )
    dimensions = _mapping(research_map.get("dimensions"), "dimensions")
    pipeline_stages = set(
        _string_list(dimensions.get("pipeline_stages"), "pipeline_stages")
    )
    quantum_resources = set(
        _string_list(dimensions.get("quantum_resources"), "quantum_resources")
    )
    advantage_targets = set(
        _string_list(dimensions.get("advantage_targets"), "advantage_targets")
    )

    areas: dict[str, Mapping[str, Any]] = {}
    for raw_area in _mapping_list(research_map.get("areas"), "areas"):
        area_id = _required_text(raw_area.get("id"), "research area id")
        if area_id in areas:
            raise AtlasOntologyError(f"Duplicate research area id '{area_id}'.")
        _validate_area_integrity(
            raw_area,
            claim_levels=claim_levels,
            replication_statuses=replication_statuses,
            status_values=status_values,
        )
        areas[area_id] = raw_area

    domains: list[AtlasDomain] = []
    domain_ids: set[str] = set()
    cell_ids: set[str] = set()
    area_to_cell: dict[str, str] = {}
    for raw_domain in _mapping_list(ontology.get("domains"), "domains"):
        _reject_extra_keys(raw_domain, _DOMAIN_KEYS, label="Atlas domain")
        domain_id = _required_text(raw_domain.get("id"), "Atlas domain id")
        if domain_id in domain_ids:
            raise AtlasOntologyError(f"Duplicate Atlas domain id '{domain_id}'.")
        domain_ids.add(domain_id)
        cells: list[AtlasCell] = []
        for raw_cell in _mapping_list(raw_domain.get("cells"), f"{domain_id}.cells"):
            _reject_extra_keys(raw_cell, _CELL_KEYS, label="Atlas cell")
            cell = _build_cell(
                raw_cell,
                areas=areas,
                pipeline_stages=pipeline_stages,
                quantum_resources=quantum_resources,
                advantage_targets=advantage_targets,
            )
            if cell.id in cell_ids:
                raise AtlasOntologyError(f"Duplicate Atlas cell id '{cell.id}'.")
            if cell.area_id in area_to_cell:
                raise AtlasOntologyError(
                    f"Research area '{cell.area_id}' appears in more than one cell."
                )
            cell_ids.add(cell.id)
            area_to_cell[cell.area_id] = cell.id
            cells.append(cell)
        if not cells:
            raise AtlasOntologyError(f"Atlas domain '{domain_id}' has no cells.")
        domains.append(
            AtlasDomain(
                id=domain_id,
                label=_required_text(raw_domain.get("label"), f"{domain_id}.label"),
                description=_required_text(
                    raw_domain.get("description"),
                    f"{domain_id}.description",
                ),
                cells=cells,
            )
        )

    missing = sorted(set(areas) - set(area_to_cell))
    unknown = sorted(set(area_to_cell) - set(areas))
    if missing or unknown:
        raise AtlasOntologyError(
            f"Atlas area coverage must be exact; missing={missing}, unknown={unknown}."
        )

    relations: list[AtlasRelation] = []
    seen_relations: set[tuple[str, str, str]] = set()
    for raw_relation in _mapping_list(
        research_map.get("relations", []),
        "relations",
    ):
        from_area = _required_text(raw_relation.get("from"), "relation.from")
        to_area = _required_text(raw_relation.get("to"), "relation.to")
        relation_type = _required_text(raw_relation.get("type"), "relation.type")
        if from_area not in area_to_cell or to_area not in area_to_cell:
            raise AtlasOntologyError(
                f"Relation endpoint is not an Atlas area: {from_area} -> {to_area}."
            )
        key = (area_to_cell[from_area], area_to_cell[to_area], relation_type)
        if key in seen_relations:
            raise AtlasOntologyError(f"Duplicate Atlas relation {key!r}.")
        seen_relations.add(key)
        relations.append(
            AtlasRelation(
                from_cell=key[0],
                to_cell=key[1],
                type=key[2],
            )
        )

    return AtlasOntologyResponse(
        schema_version=int(ontology["schema_version"]),
        ontology_updated=str(ontology.get("updated") or ""),
        research_map_schema_version=int(research_map.get("schema_version") or 0),
        research_map_updated=str(research_map.get("updated") or ""),
        note=_required_text(ontology.get("purpose"), "Atlas ontology purpose"),
        claim_levels=claim_levels,
        replication_statuses=replication_statuses,
        status_values=status_values,
        domains=domains,
        relations=relations,
    )


def bind_atlas_verdict_refs(
    ontology: AtlasOntologyResponse,
    snapshots: Iterable[Mapping[str, Any]],
) -> AtlasOntologyResponse:
    """Bind cells to the newest snapshot that still matches the claim ledger."""
    claims = {claim["claim_id"]: claim for claim in list_claims()}
    latest_by_claim: dict[str, tuple[int, Mapping[str, Any]]] = {}
    for snapshot in snapshots:
        if not isinstance(snapshot, Mapping):
            continue
        snapshot_id = snapshot.get("id")
        claim_id = snapshot.get("claim_id")
        if (
            isinstance(snapshot_id, bool)
            or not isinstance(snapshot_id, int)
            or snapshot_id <= 0
            or not isinstance(claim_id, str)
        ):
            continue
        claim = claims.get(claim_id)
        if claim is None or any(
            snapshot.get(field) != expected
            for field, expected in (
                ("claim_level", claim["level"]),
                ("claim_status", claim["status"]),
                ("replication_status", claim["replication_status"]),
            )
        ):
            continue
        verdict_key = snapshot.get("verdict_key")
        source_kind = snapshot.get("source_kind")
        source_id = snapshot.get("source_id")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (verdict_key, source_kind, source_id)
        ):
            continue
        prior = latest_by_claim.get(claim_id)
        if prior is None or snapshot_id > prior[0]:
            latest_by_claim[claim_id] = (snapshot_id, snapshot)

    payload = ontology.model_dump()
    for domain in payload["domains"]:
        for cell in domain["cells"]:
            selected = latest_by_claim.get(cell["area_id"])
            if selected is None:
                cell["verdict_ref"] = None
                continue
            snapshot = selected[1]
            cell["verdict_ref"] = {
                "verdict_key": atlas_verdict_key(cell["area_id"]),
                "source_kind": ATLAS_VERDICT_SOURCE_KIND,
                "source_id": cell["area_id"],
            }
    return AtlasOntologyResponse(**payload)


def atlas_verdict_key(claim_id: str) -> str:
    """Return the stable projection key used to join one canonical claim."""
    return f"{ATLAS_VERDICT_KEY_PREFIX}{claim_id}"


def _build_cell(
    raw_cell: Mapping[str, Any],
    *,
    areas: Mapping[str, Mapping[str, Any]],
    pipeline_stages: set[str],
    quantum_resources: set[str],
    advantage_targets: set[str],
) -> AtlasCell:
    cell_id = _required_text(raw_cell.get("id"), "Atlas cell id")
    area_id = _required_text(raw_cell.get("area_id"), f"{cell_id}.area_id")
    area = areas.get(area_id)
    if area is None:
        raise AtlasOntologyError(f"Atlas cell '{cell_id}' has unknown area '{area_id}'.")
    pipeline_stage = _required_text(
        raw_cell.get("pipeline_stage"),
        f"{cell_id}.pipeline_stage",
    )
    quantum_resource = _required_text(
        raw_cell.get("quantum_resource"),
        f"{cell_id}.quantum_resource",
    )
    advantage_target = _required_text(
        raw_cell.get("advantage_target"),
        f"{cell_id}.advantage_target",
    )
    area_stages = _string_list(area.get("pipeline_stages"), f"{area_id}.pipeline_stages")
    area_resources = _string_list(
        area.get("quantum_resources"),
        f"{area_id}.quantum_resources",
    )
    area_targets = _string_list(
        area.get("advantage_targets"),
        f"{area_id}.advantage_targets",
    )
    _require_dimension(
        pipeline_stage,
        vocabulary=pipeline_stages,
        area_values=area_stages,
        label=f"{cell_id}.pipeline_stage",
    )
    _require_dimension(
        quantum_resource,
        vocabulary=quantum_resources,
        area_values=area_resources,
        label=f"{cell_id}.quantum_resource",
    )
    _require_dimension(
        advantage_target,
        vocabulary=advantage_targets,
        area_values=area_targets,
        label=f"{cell_id}.advantage_target",
    )
    verdict_ref = raw_cell.get("verdict_ref")
    if verdict_ref is not None:
        verdict_ref = _mapping(verdict_ref, f"{cell_id}.verdict_ref")
        _reject_extra_keys(
            verdict_ref,
            _VERDICT_REF_KEYS,
            label=f"{cell_id}.verdict_ref",
        )
        raise AtlasOntologyError(
            f"{cell_id}.verdict_ref is not supported until its claim binding "
            "can be validated against a canonical verdict source."
        )
    kind = _required_text(raw_cell.get("kind"), f"{cell_id}.kind")
    if kind not in _ATLAS_KINDS:
        raise AtlasOntologyError(
            f"{cell_id}.kind must be one of: {', '.join(sorted(_ATLAS_KINDS))}."
        )

    return AtlasCell(
        id=cell_id,
        area_id=area_id,
        label=_required_text(area.get("label"), f"{area_id}.label"),
        kind=kind,
        portfolio=_required_text(area.get("portfolio"), f"{area_id}.portfolio"),
        pipeline_stage=pipeline_stage,
        quantum_resource=quantum_resource,
        advantage_target=advantage_target,
        pipeline_stages=area_stages,
        quantum_resources=area_resources,
        advantage_targets=area_targets,
        seed_status=_required_text(area.get("status"), f"{area_id}.status"),
        seed_claim_level=_required_text(
            area.get("claim_level"),
            f"{area_id}.claim_level",
        ),
        seed_replication_status=_required_text(
            area.get("replication_status"),
            f"{area_id}.replication_status",
        ),
        verdict_ref=None,
        note=(
            str(raw_cell.get("note"))
            if raw_cell.get("note") is not None
            else str(area.get("present_conclusion") or "") or None
        ),
    )


def _validate_area_integrity(
    area: Mapping[str, Any],
    *,
    claim_levels: list[str],
    replication_statuses: list[str],
    status_values: Mapping[str, str],
) -> None:
    area_id = _required_text(area.get("id"), "research area id")
    status = _required_text(area.get("status"), f"{area_id}.status")
    claim_level = _required_text(area.get("claim_level"), f"{area_id}.claim_level")
    replication = _required_text(
        area.get("replication_status"),
        f"{area_id}.replication_status",
    )
    if status not in status_values:
        raise AtlasOntologyError(f"Unknown status '{status}' for area '{area_id}'.")
    if claim_level not in claim_levels:
        raise AtlasOntologyError(
            f"Unknown claim level '{claim_level}' for area '{area_id}'."
        )
    if replication not in replication_statuses:
        raise AtlasOntologyError(
            f"Unknown replication status '{replication}' for area '{area_id}'."
        )


def _require_dimension(
    value: str,
    *,
    vocabulary: set[str],
    area_values: list[str],
    label: str,
) -> None:
    if value not in vocabulary:
        raise AtlasOntologyError(f"{label} has unknown value '{value}'.")
    if value not in area_values:
        raise AtlasOntologyError(
            f"{label}='{value}' is not declared by its research-map area."
        )


def _load_mapping(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise AtlasOntologyError(f"Unable to load {label} '{path}'.") from exc
    return _mapping(payload, label)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AtlasOntologyError(f"{label} must be a mapping.")
    return value


def _mapping_list(value: Any, label: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise AtlasOntologyError(f"{label} must be a list.")
    return [_mapping(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item for item in value
    ):
        raise AtlasOntologyError(f"{label} must be a non-empty string list.")
    if len(value) != len(set(value)):
        raise AtlasOntologyError(f"{label} must not contain duplicates.")
    return list(value)


def _string_mapping(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value or not all(
        isinstance(key, str)
        and key
        and isinstance(item, str)
        and item
        for key, item in value.items()
    ):
        raise AtlasOntologyError(f"{label} must be a non-empty string mapping.")
    return dict(value)


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AtlasOntologyError(f"{label} must be a non-empty string.")
    return value.strip()


def _reject_extra_keys(
    value: Mapping[str, Any],
    allowed: set[str],
    *,
    label: str,
) -> None:
    extra = sorted(set(value) - allowed)
    if extra:
        raise AtlasOntologyError(f"{label} contains unsupported fields: {extra}.")


__all__ = [
    "ATLAS_VERDICT_KEY_PREFIX",
    "ATLAS_VERDICT_SOURCE_KIND",
    "AtlasOntologyError",
    "AtlasOntologyResponse",
    "atlas_verdict_key",
    "atlas_ontology_response",
    "bind_atlas_verdict_refs",
]
