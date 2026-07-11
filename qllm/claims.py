"""Strict read-only access to the canonical QLLM claim ledger.

The ledger is deliberately separate from run-time evidence.  Loading it never
writes YAML or promotes a claim; it only validates the checked-in research
contract against ``docs/RESEARCH_MAP.yaml`` and returns defensive copies.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import fnmatch
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Sequence, overload

import yaml


CLAIM_LEVELS = frozenset(
    {
        "untested",
        "correctness",
        "diagnostic",
        "mechanism",
        "paired_empirical",
        "scaling",
        "hardware",
        "practical",
        "formal",
    }
)
CLAIM_STATUSES = frozenset(
    {
        "untested",
        "supported",
        "contradicted",
        "inconclusive",
        "relabeled",
        "rerun_required",
        "blocked",
    }
)
REPLICATION_STATUSES = frozenset(
    {
        "none",
        "within_study_resampling",
        "single_task_instance",
        "multi_seed_single_instance",
        "multi_instance",
        "multi_hardware_calibration",
    }
)
METRIC_TYPES = frozenset(
    {
        "communication_complexity",
        "constrained_accuracy",
        "exact_likelihood",
        "excess_cross_entropy",
        "gradient_variance",
        "held_out_predictive_quality",
        "kernel_target_alignment",
        "logical_resource_estimate",
        "predictive_memory_to_target",
        "strict_autoregressive_next_token",
        "teacher_forced_side_information",
        "time_to_target",
        "validation_perplexity",
    }
)

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLAIMS_PATH = _ROOT / "research" / "claims.yaml"
RESEARCH_MAP_PATH = _ROOT / "docs" / "RESEARCH_MAP.yaml"

_CLAIM_KEYS = frozenset(
    {
        "claim_id",
        "research_area_id",
        "statement",
        "level",
        "status",
        "replication_status",
        "metric_type",
        "evidence",
        "contradictions",
        "limitations",
        "fairness_schema",
        "analogue_ladder",
        "analysis_settings",
        "next_decisive_test",
    }
)
_FAIRNESS_KEYS = frozenset(
    {
        "schema_id",
        "required_equal",
        "intentional_differences",
        "parameter_tolerance",
        "resource_requirements",
    }
)
_EVIDENCE_KEYS = frozenset({"source", "sections", "suites", "summary"})
_DIFFERENCE_KEYS = frozenset({"path", "reason"})
_ANALOGUE_KEYS = frozenset(
    {"rung_id", "required", "comparator", "match_mode", "limitation"}
)
_ANALYSIS_KEYS = frozenset(
    {
        "phase",
        "alpha",
        "target_power",
        "bootstrap_resamples",
        "bootstrap_seed",
        "sign_flip_draws",
        "sign_flip_seed",
        "minimum_confirmatory_pairs",
        "practical_equivalence_margin",
        "suites",
        "preset_ids",
    }
)
_ANALYSIS_PHASES = frozenset({"exploratory", "pilot", "confirmatory"})
_MATCH_MODES = frozenset(
    {
        "parameters",
        "memory",
        "circuit_evaluations",
        "state_preparations",
        "logical_resources",
        "communication",
    }
)
_ALLOWED_STATUS_BY_MAP_STATUS = {
    "negative": frozenset({"contradicted"}),
    "infrastructure": frozenset({"supported", "inconclusive"}),
    "methodology_only": frozenset({"supported", "inconclusive"}),
    "blocked": frozenset({"blocked", "relabeled", "rerun_required"}),
    "partial": frozenset({"supported", "inconclusive"}),
    "quantum_inspired": frozenset({"relabeled"}),
    "open": frozenset({"untested", "inconclusive", "blocked"}),
    "unexplored": frozenset({"untested"}),
}


class ClaimRegistryError(ValueError):
    """Raised when the checked-in claim ledger violates its schema."""


@dataclass(frozen=True)
class ClaimRegistry(Sequence[dict[str, Any]]):
    """Validated claims in canonical research-map order.

    Public accessors return defensive copies so callers cannot mutate the
    process-local registry and mistake that mutation for a ledger update.
    """

    schema_version: int
    _claims: tuple[dict[str, Any], ...]

    def __len__(self) -> int:
        return len(self._claims)

    @overload
    def __getitem__(self, index: int) -> dict[str, Any]: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[dict[str, Any], ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> dict[str, Any] | tuple[dict[str, Any], ...]:
        return deepcopy(self._claims[index])

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for claim in self._claims:
            yield deepcopy(claim)

    @property
    def claims(self) -> tuple[dict[str, Any], ...]:
        return tuple(deepcopy(self._claims))

    def as_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "claims": list(self.claims)}


def _fail(location: str, message: str) -> None:
    raise ClaimRegistryError(f"{location}: {message}")


def _expect_mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(location, "must be a mapping")
    return value


def _expect_exact_keys(value: dict[str, Any], expected: frozenset[str], location: str) -> None:
    missing = sorted(expected - value.keys())
    unexpected = sorted(value.keys() - expected)
    if missing or unexpected:
        _fail(location, f"keys mismatch; missing={missing}, unexpected={unexpected}")


def _expect_text(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(location, "must be a non-empty string")
    if value != value.strip():
        _fail(location, "must not have leading or trailing whitespace")
    return value


def _expect_string_list(
    value: Any,
    location: str,
    *,
    allow_empty: bool = True,
    sorted_values: bool = False,
) -> list[str]:
    if not isinstance(value, list) or (not allow_empty and not value):
        suffix = "non-empty " if not allow_empty else ""
        _fail(location, f"must be a {suffix}list")
    items = [_expect_text(item, f"{location}[{index}]") for index, item in enumerate(value)]
    if len(items) != len(set(items)):
        _fail(location, "must not contain duplicates")
    if sorted_values and items != sorted(items):
        _fail(location, "must be sorted for deterministic inference")
    return items


def _load_yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise ClaimRegistryError(f"cannot load {label} at {path}: {exc}") from exc
    return _expect_mapping(payload, label)


def _research_areas() -> tuple[list[str], dict[str, dict[str, Any]]]:
    payload = _load_yaml_mapping(RESEARCH_MAP_PATH, "research map")
    areas = payload.get("areas")
    if not isinstance(areas, list) or not areas:
        _fail("research map.areas", "must be a non-empty list")
    by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for index, raw in enumerate(areas):
        area = _expect_mapping(raw, f"research map.areas[{index}]")
        area_id = _expect_text(area.get("id"), f"research map.areas[{index}].id")
        if area_id in by_id:
            _fail("research map.areas", f"duplicate area ID {area_id!r}")
        order.append(area_id)
        by_id[area_id] = area
    return order, by_id


def _validate_evidence(
    value: Any,
    *,
    location: str,
    claim_level: str,
    area: dict[str, Any],
) -> None:
    if not isinstance(value, list):
        _fail(location, "must be a list")
    if not value and claim_level != "untested":
        _fail(location, "may be empty only for an untested claim")
    allowed_sections = set(area.get("results_sections") or [])
    for index, raw in enumerate(value):
        item_location = f"{location}[{index}]"
        item = _expect_mapping(raw, item_location)
        _expect_exact_keys(item, _EVIDENCE_KEYS, item_location)
        if item["source"] != "RESULTS.md":
            _fail(f"{item_location}.source", "must reference immutable RESULTS.md")
        sections = item["sections"]
        if not isinstance(sections, list) or not sections:
            _fail(f"{item_location}.sections", "must be a non-empty list")
        if any(isinstance(section, bool) or not isinstance(section, int) or section <= 0 for section in sections):
            _fail(f"{item_location}.sections", "must contain positive integers")
        if sections != sorted(set(sections)):
            _fail(f"{item_location}.sections", "must be unique and sorted")
        unknown = sorted(set(sections) - allowed_sections)
        if unknown:
            _fail(f"{item_location}.sections", f"not listed for research area: {unknown}")
        suites = _expect_string_list(
            item["suites"], f"{item_location}.suites", sorted_values=True
        )
        unknown_suites = sorted(set(suites) - set(area.get("suites") or []))
        if unknown_suites:
            _fail(
                f"{item_location}.suites",
                f"not listed for research area: {unknown_suites}",
            )
        _expect_text(item["summary"], f"{item_location}.summary")


def _validate_fairness(value: Any, location: str) -> None:
    schema = _expect_mapping(value, location)
    _expect_exact_keys(schema, _FAIRNESS_KEYS, location)
    _expect_text(schema["schema_id"], f"{location}.schema_id")
    required = _expect_string_list(
        schema["required_equal"], f"{location}.required_equal", allow_empty=False
    )
    differences = schema["intentional_differences"]
    if not isinstance(differences, list) or not differences:
        _fail(f"{location}.intentional_differences", "must be a non-empty list")
    paths: list[str] = []
    for index, raw in enumerate(differences):
        item_location = f"{location}.intentional_differences[{index}]"
        item = _expect_mapping(raw, item_location)
        _expect_exact_keys(item, _DIFFERENCE_KEYS, item_location)
        paths.append(_expect_text(item["path"], f"{item_location}.path"))
        _expect_text(item["reason"], f"{item_location}.reason")
    if len(paths) != len(set(paths)):
        _fail(f"{location}.intentional_differences", "paths must be unique")
    overlap = sorted({
        f"{required_path} <-> {difference_path}"
        for required_path in required
        for difference_path in paths
        if fnmatch.fnmatchcase(difference_path, required_path)
        or fnmatch.fnmatchcase(required_path, difference_path)
    })
    if overlap:
        _fail(location, f"paths cannot be both required-equal and intentionally different: {overlap}")
    tolerance = schema["parameter_tolerance"]
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not 0.0 <= float(tolerance) <= 1.0
    ):
        _fail(f"{location}.parameter_tolerance", "must be between 0 and 1")
    _expect_string_list(
        schema["resource_requirements"],
        f"{location}.resource_requirements",
        allow_empty=False,
    )


def _validate_analogue_ladder(value: Any, location: str) -> None:
    if not isinstance(value, list) or not value:
        _fail(location, "must be a non-empty list")
    rung_ids: list[str] = []
    for index, raw in enumerate(value):
        item_location = f"{location}[{index}]"
        item = _expect_mapping(raw, item_location)
        _expect_exact_keys(item, _ANALOGUE_KEYS, item_location)
        rung_ids.append(_expect_text(item["rung_id"], f"{item_location}.rung_id"))
        if not isinstance(item["required"], bool):
            _fail(f"{item_location}.required", "must be boolean")
        _expect_text(item["comparator"], f"{item_location}.comparator")
        if item["match_mode"] not in _MATCH_MODES:
            _fail(f"{item_location}.match_mode", f"must be one of {sorted(_MATCH_MODES)}")
        _expect_text(item["limitation"], f"{item_location}.limitation")
    if len(rung_ids) != len(set(rung_ids)):
        _fail(location, "rung IDs must be unique")


def _validate_analysis(value: Any, location: str) -> None:
    settings = _expect_mapping(value, location)
    _expect_exact_keys(settings, _ANALYSIS_KEYS, location)
    if settings["phase"] not in _ANALYSIS_PHASES:
        _fail(f"{location}.phase", f"must be one of {sorted(_ANALYSIS_PHASES)}")
    for key in ("alpha", "target_power"):
        number = settings[key]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not 0.0 < float(number) < 1.0:
            _fail(f"{location}.{key}", "must be between 0 and 1")
    for key in (
        "bootstrap_resamples",
        "bootstrap_seed",
        "sign_flip_draws",
        "sign_flip_seed",
        "minimum_confirmatory_pairs",
    ):
        number = settings[key]
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            _fail(f"{location}.{key}", "must be a non-negative integer")
    if settings["bootstrap_resamples"] <= 0 or settings["sign_flip_draws"] <= 0:
        _fail(location, "bootstrap_resamples and sign_flip_draws must be positive")
    if settings["minimum_confirmatory_pairs"] < 6:
        _fail(
            f"{location}.minimum_confirmatory_pairs",
            "must be at least 6; an exact two-sided sign-flip test cannot "
            "reach alpha=.05 with fewer pairs",
        )
    margin = settings["practical_equivalence_margin"]
    if margin is not None and (
        isinstance(margin, bool)
        or not isinstance(margin, (int, float))
        or float(margin) <= 0.0
    ):
        _fail(f"{location}.practical_equivalence_margin", "must be null or positive")
    _expect_string_list(
        settings["suites"], f"{location}.suites", sorted_values=True
    )
    _expect_string_list(
        settings["preset_ids"], f"{location}.preset_ids", sorted_values=True
    )


def _validate_claim(
    raw: Any,
    *,
    index: int,
    area: dict[str, Any],
) -> dict[str, Any]:
    location = f"claims[{index}]"
    claim = _expect_mapping(raw, location)
    _expect_exact_keys(claim, _CLAIM_KEYS, location)
    claim_id = _expect_text(claim["claim_id"], f"{location}.claim_id")
    area_id = _expect_text(claim["research_area_id"], f"{location}.research_area_id")
    if claim_id != area_id:
        _fail(location, "claim_id and research_area_id must be identical")
    if area_id != area["id"]:
        _fail(f"{location}.research_area_id", f"expected {area['id']!r}")
    _expect_text(claim["statement"], f"{location}.statement")
    if claim["level"] not in CLAIM_LEVELS:
        _fail(f"{location}.level", f"must be one of {sorted(CLAIM_LEVELS)}")
    if claim["level"] != area.get("claim_level"):
        _fail(f"{location}.level", f"must match research map level {area.get('claim_level')!r}")
    if claim["status"] not in CLAIM_STATUSES:
        _fail(f"{location}.status", f"must be one of {sorted(CLAIM_STATUSES)}")
    allowed_statuses = _ALLOWED_STATUS_BY_MAP_STATUS.get(area.get("status"), frozenset())
    if claim["status"] not in allowed_statuses:
        _fail(
            f"{location}.status",
            f"{claim['status']!r} is stronger than or incompatible with map status {area.get('status')!r}",
        )
    if claim_id == "two_stream_conditioning" and claim["status"] != "rerun_required":
        _fail(f"{location}.status", "two_stream_conditioning must remain rerun_required")
    if claim["replication_status"] not in REPLICATION_STATUSES:
        _fail(
            f"{location}.replication_status",
            f"must be one of {sorted(REPLICATION_STATUSES)}",
        )
    if claim["replication_status"] != area.get("replication_status"):
        _fail(
            f"{location}.replication_status",
            f"must match research map replication {area.get('replication_status')!r}",
        )
    if claim["metric_type"] not in METRIC_TYPES:
        _fail(f"{location}.metric_type", f"must be one of {sorted(METRIC_TYPES)}")
    _validate_evidence(
        claim["evidence"],
        location=f"{location}.evidence",
        claim_level=claim["level"],
        area=area,
    )
    _expect_string_list(claim["contradictions"], f"{location}.contradictions")
    _expect_string_list(
        claim["limitations"], f"{location}.limitations", allow_empty=False
    )
    _validate_fairness(claim["fairness_schema"], f"{location}.fairness_schema")
    _validate_analogue_ladder(claim["analogue_ladder"], f"{location}.analogue_ladder")
    _validate_analysis(claim["analysis_settings"], f"{location}.analysis_settings")
    _expect_text(claim["next_decisive_test"], f"{location}.next_decisive_test")
    return deepcopy(claim)


def _load_claim_registry_path(ledger_path: Path) -> ClaimRegistry:
    payload = _load_yaml_mapping(ledger_path, "claim ledger")
    _expect_exact_keys(payload, frozenset({"schema_version", "claims"}), "claim ledger")
    version = payload["schema_version"]
    if isinstance(version, bool) or version != 1:
        _fail("claim ledger.schema_version", "must equal integer 1")
    raw_claims = payload["claims"]
    if not isinstance(raw_claims, list):
        _fail("claim ledger.claims", "must be a list")

    canonical_order, areas = _research_areas()
    raw_ids: list[str] = []
    for index, raw in enumerate(raw_claims):
        mapping = _expect_mapping(raw, f"claims[{index}]")
        raw_ids.append(_expect_text(mapping.get("claim_id"), f"claims[{index}].claim_id"))
    if len(raw_ids) != len(set(raw_ids)):
        duplicates = sorted({claim_id for claim_id in raw_ids if raw_ids.count(claim_id) > 1})
        _fail("claim ledger.claims", f"duplicate claim IDs: {duplicates}")
    missing = sorted(set(canonical_order) - set(raw_ids))
    unexpected = sorted(set(raw_ids) - set(canonical_order))
    if missing or unexpected:
        _fail("claim ledger.claims", f"area coverage mismatch; missing={missing}, unexpected={unexpected}")
    if raw_ids != canonical_order:
        _fail("claim ledger.claims", "must follow docs/RESEARCH_MAP.yaml area order")

    claims = tuple(
        _validate_claim(raw, index=index, area=areas[canonical_order[index]])
        for index, raw in enumerate(raw_claims)
    )
    return ClaimRegistry(schema_version=1, _claims=claims)


@lru_cache(maxsize=1)
def _cached_default_registry() -> ClaimRegistry:
    return _load_claim_registry_path(DEFAULT_CLAIMS_PATH)


def load_claim_registry(path: str | Path | None = None) -> ClaimRegistry:
    """Load and strictly validate the canonical claim registry.

    The checked-in registry is cached after validation for dashboard reads;
    custom paths remain uncached so validation tests and audits always inspect
    the supplied file. Public accessors still return defensive claim copies.
    """
    if path is None:
        return _cached_default_registry()
    return _load_claim_registry_path(Path(path))


def list_claims(
    registry: ClaimRegistry | None = None,
    *,
    level: str | None = None,
    status: str | None = None,
) -> list[dict[str, Any]]:
    """Return claims in deterministic map order, optionally filtered."""
    if level is not None and level not in CLAIM_LEVELS:
        raise ValueError(f"unknown claim level {level!r}")
    if status is not None and status not in CLAIM_STATUSES:
        raise ValueError(f"unknown claim status {status!r}")
    source = load_claim_registry() if registry is None else registry
    return [
        claim
        for claim in source
        if (level is None or claim["level"] == level)
        and (status is None or claim["status"] == status)
    ]


def get_claim(
    claim_id: str,
    registry: ClaimRegistry | None = None,
) -> dict[str, Any]:
    """Return one claim by ID or raise ``KeyError`` for an unknown ID."""
    source = load_claim_registry() if registry is None else registry
    for claim in source:
        if claim["claim_id"] == claim_id:
            return claim
    raise KeyError(f"unknown claim_id {claim_id!r}")


def infer_claim_id(
    *,
    explicit: str | None = None,
    suite: str | None = None,
    preset_id: str | None = None,
    registry: ClaimRegistry | None = None,
) -> str | None:
    """Infer a claim only when the supplied selectors have one unique match.

    Explicit valid IDs take precedence.  Unknown selectors, conflicting
    selectors, and ambiguous suites deliberately return ``None`` rather than
    attaching a run to the wrong scientific claim.
    """
    source = load_claim_registry() if registry is None else registry
    known_ids = {claim["claim_id"] for claim in source}
    if explicit is not None:
        return explicit if explicit in known_ids else None

    selections: list[set[str]] = []
    if suite is not None:
        selections.append(
            {
                claim["claim_id"]
                for claim in source
                if suite in claim["analysis_settings"]["suites"]
            }
        )
    if preset_id is not None:
        selections.append(
            {
                claim["claim_id"]
                for claim in source
                if preset_id in claim["analysis_settings"]["preset_ids"]
            }
        )
    if not selections or any(not selection for selection in selections):
        return None
    matches = set.intersection(*selections)
    return next(iter(matches)) if len(matches) == 1 else None


__all__ = [
    "CLAIM_LEVELS",
    "CLAIM_STATUSES",
    "DEFAULT_CLAIMS_PATH",
    "METRIC_TYPES",
    "REPLICATION_STATUSES",
    "ClaimRegistry",
    "ClaimRegistryError",
    "get_claim",
    "infer_claim_id",
    "list_claims",
    "load_claim_registry",
]
