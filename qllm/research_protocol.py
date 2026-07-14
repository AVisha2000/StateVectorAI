"""Research protocol guards for quantum-vs-classical claims.

These helpers keep the project honest about the difference between a useful
smoke result, a fair paired benchmark, and evidence that is strong enough to
talk about advantage. They are deliberately lightweight: no SciPy dependency,
deterministic small-sample statistics, and plain dictionaries for dashboard
payloads.
"""
from __future__ import annotations

import dataclasses
import fnmatch
import hashlib
import itertools
import json
import math
import uuid
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Iterable, Mapping

import numpy as np

from . import registry as registry_module
from .config import ExperimentConfig, to_flat_dict
from .registry import (
    QUANTUM_ARCH_TYPES,
    QUANTUM_ATTN_TYPES,
    QUANTUM_FFN_TYPES,
)


TWO_STREAM_CAUSAL_PROTOCOL = "causal_prefix_v2"
TWO_STREAM_CAUSAL_SUITE = "two-stream-causal-v2"
HISTORICAL_TWO_STREAM_SUITES = frozenset({"two-stream-v1"})
MIN_PAIRED_EDGE_PAIRS = 6
SOLVER_COMPETITION_SCHEMA_ID = "solver_competition_v1"
SOLVER_COMPETITION_COMPARATOR_KIND = "best_in_class_solver"
SEED_AXIS_NAMES = (
    "generator",
    "split",
    "initialization",
    "minibatch",
    "circuit",
    "hardware_calibration",
)
_NORMALIZED_SEED_AXIS_METADATA = frozenset({
    "applicability",
    "assessment_status",
    "coupled_axes",
    "requested",
    "source",
    "sources",
    "supported",
    "unsupported_overrides",
})


def _integer_seed(value: object, path: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{path} must be an integer or null")
    return int(value)


def normalize_seed_axes(
    legacy_seed: int,
    *,
    generator_seed: int | None = None,
    data_kind: str | None = None,
    circuit_applicable: bool = True,
    minibatch_applicable: bool = True,
    explicit: Mapping[str, object] | None = None,
    reject_unsupported: bool = False,
) -> dict[str, Any]:
    """Describe the random axes actually controlled by today's scalar seeds.

    This is deliberately metadata, not a claim that the execution path can
    vary every axis independently.  The legacy training seed currently couples
    initialization, minibatch order, and applicable circuit initialization.
    """
    legacy = _integer_seed(legacy_seed, "legacy_seed")
    if legacy is None:  # defensive: the public argument is intentionally non-null
        raise ValueError("legacy_seed must be an integer")
    generated = bool(data_kind and data_kind != "text")
    actual = {
        "generator": (
            _integer_seed(generator_seed, "generator_seed") if generated else None
        ),
        "split": None,
        "initialization": legacy,
        "minibatch": legacy if minibatch_applicable else None,
        "circuit": legacy if circuit_applicable else None,
        "hardware_calibration": None,
    }
    explicit_values = dict(explicit or {})
    unknown = sorted(
        set(explicit_values)
        - set(SEED_AXIS_NAMES)
        - {"legacy_seed"}
        - _NORMALIZED_SEED_AXIS_METADATA
    )
    if unknown:
        raise ValueError("unknown seed axis/axes: " + ", ".join(unknown))
    normalized_input = bool(
        set(explicit_values) & _NORMALIZED_SEED_AXIS_METADATA
    )
    requested_values: Mapping[str, object]
    if normalized_input:
        raw_requested = explicit_values.get("requested") or {}
        if not isinstance(raw_requested, Mapping):
            raise ValueError("seed_axes.requested must be a mapping")
        requested_values = raw_requested
    else:
        requested_values = explicit_values
    unknown_requested = sorted(
        set(requested_values) - set(SEED_AXIS_NAMES) - {"legacy_seed"}
    )
    if unknown_requested:
        raise ValueError(
            "unknown requested seed axis/axes: " + ", ".join(unknown_requested)
        )
    requested: dict[str, int | None] = {}
    for axis in SEED_AXIS_NAMES:
        if axis in requested_values:
            requested[axis] = _integer_seed(
                requested_values[axis], f"seed_axes.{axis}"
            )
    if "legacy_seed" in requested_values:
        requested_legacy = _integer_seed(
            requested_values["legacy_seed"], "seed_axes.legacy_seed"
        )
        if requested_legacy != legacy:
            requested["legacy_seed"] = requested_legacy

    unsupported = sorted(
        axis for axis, value in requested.items()
        if value != (legacy if axis == "legacy_seed" else actual[axis])
    )
    if reject_unsupported and unsupported:
        raise ValueError(
            "independent seed-axis overrides are not supported by the current "
            "execution path: " + ", ".join(unsupported)
        )
    applicability = {
        "generator": generated,
        "split": False,
        "initialization": True,
        "minibatch": bool(minibatch_applicable),
        "circuit": bool(circuit_applicable),
        "hardware_calibration": False,
    }
    sources = {
        "generator": "data.gen_seed" if generated else "not_applicable",
        "split": "deterministic_split",
        "initialization": "train.seed",
        "minibatch": "train.seed" if minibatch_applicable else "not_applicable",
        "circuit": "train.seed" if circuit_applicable else "not_applicable",
        "hardware_calibration": "not_applicable",
    }
    return {
        "legacy_seed": legacy,
        **actual,
        "applicability": applicability,
        "sources": sources,
        "source": "legacy_scalar" if not explicit else "explicit_metadata",
        "coupled_axes": [
            axis for axis in ("initialization", "minibatch", "circuit")
            if applicability[axis]
        ],
        "supported": not unsupported,
        "assessment_status": (
            "supported_legacy_coupling"
            if not unsupported
            else "unsupported_independent_override"
        ),
        "unsupported_overrides": unsupported,
        "requested": requested,
    }


def _flat_or_nested(config: dict, path: str):
    if path in config:
        return config[path]
    current = config
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def two_stream_metric_contract(
    *,
    suite: str = "",
    config: dict | None = None,
) -> dict[str, str | bool] | None:
    """Return the evidence contract for current or historical two-stream runs.

    ``two-stream-v1`` used one full-window summary for every position.  Those
    rows remain valid records of a teacher-forced side-information probe, but
    they cannot be compared with strict autoregressive next-token metrics.
    New dashboard jobs carry an explicit protocol marker; an unmarked
    two-stream job is conservatively treated as historical.
    """
    config = config or {}
    arch = _flat_or_nested(config, "model.arch")
    marker = (
        _flat_or_nested(config, "lab.two_stream_protocol")
        or _flat_or_nested(config, "research.two_stream_protocol")
    )
    historical = suite in HISTORICAL_TWO_STREAM_SUITES or (
        arch == "two_stream"
        and suite != TWO_STREAM_CAUSAL_SUITE
        and marker != TWO_STREAM_CAUSAL_PROTOCOL
    )
    if historical:
        return {
            "metric_type": "teacher_forced_side_information",
            "protocol": "full_window_v1",
            "protocol_status": "rerun_required",
            "rerun_required": True,
            "strict_autoregressive": False,
            "limitation": (
                "Historical two-stream results used a full-window encoder "
                "summary. They are teacher-forced side-information metrics, "
                "not strict autoregressive evidence, and require a causal rerun."
            ),
        }
    if suite == TWO_STREAM_CAUSAL_SUITE or arch == "two_stream":
        return {
            "metric_type": "strict_autoregressive_next_token",
            "protocol": TWO_STREAM_CAUSAL_PROTOCOL,
            "protocol_status": "current",
            "rerun_required": False,
            "strict_autoregressive": True,
            "limitation": (
                "Causal-prefix metrics are not evidence of an encoder edge "
                "without matched controls and adequate paired replication."
            ),
        }
    return None


@dataclass(frozen=True)
class PairedStats:
    """Summary of paired candidate-vs-baseline improvements.

    ``improvements`` are positive when the candidate is better. For lower-is-
    better metrics such as perplexity this means ``baseline - candidate``.
    """

    n_pairs: int
    mean_improvement: float
    median_improvement: float
    ci_low: float
    ci_high: float
    win_rate: float
    p_value: float
    effect_size: float | None
    significant: bool
    effect_size_status: str = "defined"
    bootstrap_seed: int = 0
    bootstrap_resamples: int = 0
    ci_method: str = "paired_bootstrap_percentile_mean"
    sign_flip_method: str = "exact"
    sign_flip_seed: int | None = None
    sign_flip_draws: int = 0
    pilot_only: bool = False

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _as_float_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("paired comparisons require at least one score")
    if not np.isfinite(arr).all():
        raise ValueError("paired scores must be finite")
    return arr


def paired_improvements(
    candidate_scores: Iterable[float],
    baseline_scores: Iterable[float],
    *,
    lower_is_better: bool = True,
) -> np.ndarray:
    """Return per-pair improvements, positive when the candidate is better."""
    candidate = _as_float_array(candidate_scores)
    baseline = _as_float_array(baseline_scores)
    if candidate.shape != baseline.shape:
        raise ValueError("candidate and baseline scores must have the same length")
    return baseline - candidate if lower_is_better else candidate - baseline


def sign_flip_test(
    improvements: Iterable[float],
    *,
    max_exact: int = 16,
    seed: int = 0,
    draws: int = 20_000,
) -> dict[str, int | float | str | None]:
    """Return a two-sided paired randomisation test with method metadata."""
    diffs = _as_float_array(improvements)
    if isinstance(max_exact, bool) or not isinstance(max_exact, int) or max_exact < 0:
        raise ValueError("max_exact must be a non-negative integer")
    seed = _integer_seed(seed, "sign_flip_seed")
    if seed is None:
        raise ValueError("sign_flip_seed must be an integer")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise ValueError("sign_flip_draws must be a positive integer")
    observed = abs(float(diffs.mean()))
    if len(diffs) <= max_exact:
        extreme = total = 0
        for signs in itertools.product((-1.0, 1.0), repeat=len(diffs)):
            total += 1
            if abs(float((diffs * np.asarray(signs)).mean())) >= observed - 1e-15:
                extreme += 1
        return {
            "p_value": float(extreme / total),
            "method": "exact",
            "seed": None,
            "draws": total,
            "extreme": extreme,
        }

    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(draws, len(diffs)))
    means = np.abs((signs * diffs).mean(axis=1))
    extreme = int(np.count_nonzero(means >= observed - 1e-15))
    return {
        "p_value": float((extreme + 1) / (draws + 1)),
        "method": "monte_carlo",
        "seed": seed,
        "draws": draws,
        "extreme": extreme,
    }


def sign_flip_p_value(
    improvements: Iterable[float],
    *,
    max_exact: int = 16,
    seed: int = 0,
    draws: int = 20_000,
) -> float:
    """Backward-compatible scalar wrapper around :func:`sign_flip_test`."""
    return float(
        sign_flip_test(
            improvements, max_exact=max_exact, seed=seed, draws=draws
        )["p_value"]
    )


def paired_stats(
    candidate_scores: Iterable[float],
    baseline_scores: Iterable[float],
    *,
    lower_is_better: bool = True,
    alpha: float = 0.05,
    bootstrap_seed: int = 0,
    bootstrap_resamples: int = 20_000,
    sign_flip_seed: int = 0,
    max_exact: int = 16,
    sign_flip_draws: int = 20_000,
) -> PairedStats:
    """Compute deterministic paired inference for matched benchmark runs."""
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    bootstrap_seed = _integer_seed(bootstrap_seed, "bootstrap_seed")
    if bootstrap_seed is None:
        raise ValueError("bootstrap_seed must be an integer")
    if (
        isinstance(bootstrap_resamples, bool)
        or not isinstance(bootstrap_resamples, int)
        or bootstrap_resamples < 1
    ):
        raise ValueError("bootstrap_resamples must be a positive integer")
    improvements = paired_improvements(
        candidate_scores, baseline_scores, lower_is_better=lower_is_better
    )
    if len(improvements) == 1:
        ci_low = ci_high = float(improvements[0])
    else:
        rng = np.random.default_rng(bootstrap_seed)
        indices = rng.integers(
            0, len(improvements), size=(bootstrap_resamples, len(improvements))
        )
        bootstrap_means = improvements[indices].mean(axis=1)
        ci_low, ci_high = np.quantile(
            bootstrap_means, [alpha / 2, 1 - alpha / 2]
        )
    std = float(improvements.std(ddof=1)) if len(improvements) > 1 else 0.0
    if std > np.finfo(np.float64).eps:
        effect: float | None = float(improvements.mean() / std)
        effect_status = "defined"
    else:
        effect = None
        effect_status = (
            "undefined_zero_variance_nonzero_mean"
            if improvements.mean() != 0
            else "undefined_zero_variance_zero_mean"
        )
    sign_flip = sign_flip_test(
        improvements,
        max_exact=max_exact,
        seed=sign_flip_seed,
        draws=sign_flip_draws,
    )
    p = float(sign_flip["p_value"])
    return PairedStats(
        n_pairs=int(len(improvements)),
        mean_improvement=float(improvements.mean()),
        median_improvement=float(np.median(improvements)),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        win_rate=float(np.mean(improvements > 0)),
        p_value=float(p),
        effect_size=effect,
        significant=bool(p <= alpha and ci_low > 0.0),
        effect_size_status=effect_status,
        bootstrap_seed=bootstrap_seed,
        bootstrap_resamples=bootstrap_resamples,
        sign_flip_method=str(sign_flip["method"]),
        sign_flip_seed=(
            None if sign_flip["seed"] is None else int(sign_flip["seed"])
        ),
        sign_flip_draws=int(sign_flip["draws"]),
        pilot_only=bool(len(improvements) <= 3),
    )


def practical_equivalence(
    stats: PairedStats | Mapping[str, Any],
    *,
    margin: float,
) -> dict[str, float | str | bool]:
    """Assess practical equivalence from the paired mean-effect interval."""
    if not isinstance(margin, (int, float)) or isinstance(margin, bool):
        raise ValueError("equivalence margin must be a finite positive number")
    margin = float(margin)
    if not math.isfinite(margin) or margin <= 0.0:
        raise ValueError("equivalence margin must be a finite positive number")
    payload = stats.as_dict() if isinstance(stats, PairedStats) else dict(stats)
    low = float(payload["ci_low"])
    high = float(payload["ci_high"])
    equivalent = low >= -margin and high <= margin
    if equivalent:
        status = "equivalent"
    elif low > margin:
        status = "candidate_meaningfully_better"
    elif high < -margin:
        status = "baseline_meaningfully_better"
    else:
        status = "inconclusive"
    return {
        "status": status,
        "equivalent": equivalent,
        "margin": margin,
        "ci_low": low,
        "ci_high": high,
        "confidence_interval_method": str(
            payload.get("ci_method") or "paired_bootstrap_percentile_mean"
        ),
    }


def paired_power_plan(
    improvements: Iterable[float],
    *,
    smallest_useful_effect: float,
    alpha: float = 0.05,
    power: float = 0.8,
    desired_half_width: float | None = None,
) -> dict[str, int | float | str | bool | None]:
    """Plan paired confirmation size from pilot sample variance.

    This normal approximation is a planning aid, not evidence.  Exact
    sign-flip resolution imposes a six-pair floor at alpha=.05.
    """
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if not 0.0 < float(power) < 1.0:
        raise ValueError("power must be between 0 and 1")
    effect = float(smallest_useful_effect)
    if not math.isfinite(effect) or effect <= 0.0:
        raise ValueError("smallest_useful_effect must be finite and positive")
    if desired_half_width is not None:
        desired_half_width = float(desired_half_width)
        if not math.isfinite(desired_half_width) or desired_half_width <= 0.0:
            raise ValueError("desired_half_width must be finite and positive")
    diffs = _as_float_array(improvements)
    observed = int(len(diffs))
    minimum_testable = max(
        MIN_PAIRED_EDGE_PAIRS,
        int(math.ceil(1.0 - math.log2(float(alpha)))),
    )
    if observed < 2:
        return {
            "status": "insufficient_pilot",
            "observed_pairs": observed,
            "pilot_std": None,
            "smallest_useful_effect": effect,
            "alpha": float(alpha),
            "target_power": float(power),
            "desired_half_width": desired_half_width,
            "minimum_testable_pairs": minimum_testable,
            "recommended_pairs": None,
            "adequately_powered": False,
        }
    pilot_std = float(diffs.std(ddof=1))
    if pilot_std <= np.finfo(np.float64).eps:
        return {
            "status": "zero_variance_pilot",
            "observed_pairs": observed,
            "pilot_std": 0.0,
            "smallest_useful_effect": effect,
            "alpha": float(alpha),
            "target_power": float(power),
            "desired_half_width": desired_half_width,
            "minimum_testable_pairs": minimum_testable,
            "recommended_pairs": minimum_testable,
            "adequately_powered": False,
        }
    normal = NormalDist()
    z_alpha = normal.inv_cdf(1.0 - float(alpha) / 2.0)
    z_power = normal.inv_cdf(float(power))
    effect_pairs = int(math.ceil(((z_alpha + z_power) * pilot_std / effect) ** 2))
    width_pairs = 0
    if desired_half_width is not None:
        width_pairs = int(
            math.ceil((z_alpha * pilot_std / desired_half_width) ** 2)
        )
    recommended = max(minimum_testable, effect_pairs, width_pairs, 2)
    return {
        "status": "adequately_powered" if observed >= recommended else "underpowered",
        "observed_pairs": observed,
        "pilot_std": pilot_std,
        "smallest_useful_effect": effect,
        "alpha": float(alpha),
        "target_power": float(power),
        "desired_half_width": desired_half_width,
        "minimum_testable_pairs": minimum_testable,
        "effect_required_pairs": effect_pairs,
        "precision_required_pairs": width_pairs or None,
        "recommended_pairs": recommended,
        "adequately_powered": bool(observed >= recommended),
    }


_MISSING = object()
_OPERATIONAL_PATTERNS = (
    "tracking.*",
    "lab.analogue.*",
    "lab.artifact_dir",
    "lab.config_snapshot_version",
    "lab.quantum_override.*",
    "lab.resource.*",
    "lab.reservation.*",
    "lab.submission.*",
    "lab.study_cell.*",
    "lab.train_override.*",
    "job.run_name",
    "job.id",
    "job.comparison_role",
)
_DEFAULT_REQUIRED_EQUAL = (
    "job.dataset_name",
    "job.seed",
    "job.steps",
    "job.eval_every",
    "job.device_target",
    "train.batch_size",
    "train.seq_len",
    "train.lr",
    "train.weight_decay",
    "train.grad_clip",
    "train.eval_batches",
    "data.*",
    "seed_axes.generator",
    "seed_axes.split",
    "seed_axes.initialization",
    "seed_axes.minibatch",
)


def _flatten_mapping(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for raw_key in sorted(value, key=str):
        item = value[raw_key]
        key = str(raw_key)
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, Mapping) and "." not in key:
            out.update(_flatten_mapping(item, path))
        elif isinstance(item, (list, tuple)):
            if item and all(isinstance(entry, Mapping) for entry in item):
                for index, entry in enumerate(item):
                    out.update(_flatten_mapping(entry, f"{path}.{index}"))
            else:
                out[path] = list(item)
        else:
            out[path] = item
    return out


def _job_config(job: Mapping[str, Any]) -> dict[str, Any]:
    config = job.get("config")
    if isinstance(config, Mapping):
        return dict(config)
    raw = job.get("config_json")
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw or "{}")
            return decoded if isinstance(decoded, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _uses_quantum_config(config: Mapping[str, Any]) -> bool:
    flat = _flatten_mapping(config)
    if flat.get("model.encoder_kind") == "quantum":
        return True
    if flat.get("model.embed_type") == "quantum":
        return True
    if str(flat.get("model.arch") or "") in QUANTUM_ARCH_TYPES:
        return True
    if str(flat.get("model.attn_type") or "") in QUANTUM_ATTN_TYPES:
        return True
    if str(flat.get("model.ffn_type") or "") in QUANTUM_FFN_TYPES:
        return True
    blocks = flat.get("model.blocks")
    if isinstance(blocks, (list, tuple)):
        for block in blocks:
            if not isinstance(block, Mapping):
                continue
            if str(block.get("attn_type") or "") in QUANTUM_ATTN_TYPES:
                return True
            if str(block.get("ffn_type") or "") in QUANTUM_FFN_TYPES:
                return True
    return any(
        str(value).startswith("quantum")
        for key, value in flat.items()
        if key.endswith((".attn_type", ".ffn_type"))
    )


def _comparison_values(item: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    job = item.get("job", item)
    if not isinstance(job, Mapping):
        job = {}
    config = _job_config(job)
    flat = _flatten_mapping(config)
    values = {
        key: value
        for key, value in flat.items()
        if key not in {"research.seed_axes", "lab.seed_axes"}
        and not key.startswith(("research.seed_axes.", "lab.seed_axes."))
    }
    for key in ("id", "run_name", "dataset_name", "seed", "steps", "eval_every", "comparison_role"):
        values[f"job.{key}"] = job.get(key, _MISSING)
    values["job.device_target"] = job.get("device_target") or "auto"
    explicit_axes = (
        config.get("research.seed_axes")
        or config.get("lab.seed_axes")
        or _flat_or_nested(config, "research.seed_axes")
    )
    axes = normalize_seed_axes(
        int(job.get("seed", 0)),
        generator_seed=_flat_or_nested(config, "data.gen_seed"),
        data_kind=_flat_or_nested(config, "data.kind"),
        circuit_applicable=_uses_quantum_config(config),
        explicit=explicit_axes if isinstance(explicit_axes, Mapping) else None,
    )
    for axis in SEED_AXIS_NAMES:
        values[f"seed_axes.{axis}"] = axes[axis]
        values[f"seed_axes.applicability.{axis}"] = axes["applicability"][axis]
        if axis in axes["requested"]:
            values[f"seed_axes.requested.{axis}"] = axes["requested"][axis]
    values["seed_axes.legacy_seed"] = axes["legacy_seed"]
    values["seed_axes.supported"] = axes["supported"]
    return values, axes


def _matches_any(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, str(pattern)) for pattern in patterns)


def _category(path: str) -> str:
    if path.startswith("seed_axes."):
        return "seed"
    if path.startswith("data.") or path.startswith("job.dataset"):
        return "data"
    if path.startswith("train.") or path.startswith("job.steps") or path.startswith("job.eval"):
        return "training"
    if path.startswith("model."):
        return "model"
    if path.startswith("job.device"):
        return "resource"
    if _matches_any(path, _OPERATIONAL_PATTERNS):
        return "operational"
    return "protocol"


def evaluate_fairness(
    candidate: Mapping[str, Any] | None,
    baseline: Mapping[str, Any] | None,
    *,
    schema: Mapping[str, Any] | None = None,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Compare the complete normalized protocol and retain every mismatch."""
    schema = dict(schema or {})
    schema_id = str(schema.get("schema_id") or "default_controlled_comparison_v1")
    required = tuple(schema.get("required_equal") or _DEFAULT_REQUIRED_EQUAL)
    if schema_id == SOLVER_COMPETITION_SCHEMA_ID:
        mismatch = {
            "path": "fairness_schema.schema_id",
            "candidate": schema_id,
            "baseline": schema_id,
            "candidate_missing": False,
            "baseline_missing": False,
            "category": "protocol",
            "requirement": "dedicated_solver_competition_evaluator",
            "allowed": False,
            "allowlist_reason": None,
            "reason": (
                "solver_competition_v1 must be evaluated with "
                "evaluate_solver_competition, not controlled-pair fairness"
            ),
        }
        return {
            "complete": False,
            "valid": False,
            "schema_id": schema_id,
            "requirements": list(required),
            "intentional_differences": list(
                schema.get("intentional_differences") or []
            ),
            "schema_errors": [mismatch],
            "mismatches": [mismatch],
            "allowed_mismatches": [],
            "disallowed_mismatches": [mismatch],
            "fairness_mismatches": [mismatch],
            "same_dataset": False,
            "same_seed": False,
            "same_steps": False,
            "same_eval_interval": False,
            "same_device_target": False,
            "same_training_budget": False,
            "same_preprocessing": False,
            "matched_config_fields": {},
            "role_validation": False,
            "parameter_delta_ratio": None,
            "seed_axes": None,
            "evaluator": "controlled_pair",
        }
    intentional_entries = schema.get("intentional_differences") or []
    intentional: dict[str, str] = {}
    schema_errors: list[dict[str, Any]] = []
    if not isinstance(intentional_entries, (list, tuple)):
        schema_errors.append({
            "path": "fairness_schema.intentional_differences",
            "reason": "intentional_differences must be a list",
        })
        intentional_entries = []
    for index, entry in enumerate(intentional_entries):
        path = (
            str(entry.get("path") or "").strip()
            if isinstance(entry, Mapping)
            else ""
        )
        reason = (
            str(entry.get("reason") or "").strip()
            if isinstance(entry, Mapping)
            else ""
        )
        if not path or not reason:
            schema_errors.append({
                "path": f"fairness_schema.intentional_differences[{index}]",
                "reason": "every intentional difference needs a non-empty path and reason",
            })
            continue
        intentional[path] = reason

    if not candidate or not baseline:
        missing_pair = {
            "path": "comparison.pair",
            "candidate": bool(candidate),
            "baseline": bool(baseline),
            "category": "protocol",
            "requirement": "linked_candidate_and_baseline",
            "allowed": False,
            "allowlist_reason": None,
            "reason": "candidate and baseline are both required",
        }
        return {
            "complete": False,
            "valid": False,
            "schema_id": schema_id,
            "requirements": list(required),
            "intentional_differences": list(intentional_entries),
            "mismatches": [missing_pair],
            "allowed_mismatches": [],
            "disallowed_mismatches": [missing_pair],
            "fairness_mismatches": [missing_pair],
            "same_dataset": False,
            "same_seed": False,
            "same_steps": False,
            "same_eval_interval": False,
            "same_device_target": False,
            "same_training_budget": False,
            "same_preprocessing": False,
            "matched_config_fields": {},
            "role_validation": False,
            "parameter_delta_ratio": None,
            "seed_axes": None,
        }

    cvalues, caxes = _comparison_values(candidate)
    bvalues, baxes = _comparison_values(baseline)
    cjob = candidate.get("job", candidate)
    bjob = baseline.get("job", baseline)
    cfinal = candidate.get("final_run") if isinstance(candidate, Mapping) else None
    bfinal = baseline.get("final_run") if isinstance(baseline, Mapping) else None
    complete = bool(cfinal and bfinal)
    role_validation = (
        cjob.get("comparison_role") == "candidate"
        and bjob.get("comparison_role") == "baseline"
    )
    union = sorted(set(cvalues) | set(bvalues))
    required_paths: set[str] = set()
    missing_patterns: list[str] = []
    for pattern in required:
        matches = [path for path in union if fnmatch.fnmatchcase(path, str(pattern))]
        if matches:
            required_paths.update(matches)
        else:
            missing_patterns.append(str(pattern))

    mismatches: list[dict[str, Any]] = []
    for path in union:
        cvalue = cvalues.get(path, _MISSING)
        bvalue = bvalues.get(path, _MISSING)
        required_path = path in required_paths
        missing = cvalue is _MISSING or bvalue is _MISSING
        if not missing and cvalue == bvalue:
            continue
        operational = _matches_any(path, _OPERATIONAL_PATTERNS)
        allow_pattern = next(
            (pattern for pattern in intentional if fnmatch.fnmatchcase(path, pattern)),
            None,
        )
        allowed = bool(operational or allow_pattern)
        reason = (
            "required value is missing"
            if missing and required_path
            else "documented intentional difference"
            if allow_pattern
            else "operational identity difference"
            if operational
            else "undocumented protocol mismatch"
        )
        mismatches.append({
            "path": path,
            "candidate": None if cvalue is _MISSING else cvalue,
            "baseline": None if bvalue is _MISSING else bvalue,
            "candidate_missing": cvalue is _MISSING,
            "baseline_missing": bvalue is _MISSING,
            "category": _category(path),
            "requirement": path if required_path else None,
            "allowed": allowed,
            "allowlist_reason": intentional.get(allow_pattern) if allow_pattern else None,
            "reason": reason,
        })
    for pattern in missing_patterns:
        mismatches.append({
            "path": pattern,
            "candidate": None,
            "baseline": None,
            "candidate_missing": True,
            "baseline_missing": True,
            "category": _category(pattern),
            "requirement": pattern,
            "allowed": False,
            "allowlist_reason": None,
            "reason": "required field pattern is absent from both protocols",
        })
    for error in schema_errors:
        mismatches.append({
            "path": error["path"],
            "candidate": None,
            "baseline": None,
            "candidate_missing": False,
            "baseline_missing": False,
            "category": "protocol",
            "requirement": "valid_intentional_difference_allowlist",
            "allowed": False,
            "allowlist_reason": None,
            "reason": error["reason"],
        })
    if require_complete and not complete:
        mismatches.append({
            "path": "comparison.complete",
            "candidate": bool(cfinal),
            "baseline": bool(bfinal),
            "candidate_missing": not bool(cfinal),
            "baseline_missing": not bool(bfinal),
            "category": "protocol",
            "requirement": "completed_pair",
            "allowed": False,
            "allowlist_reason": None,
            "reason": "incomplete pairs are never fair",
        })
    if not role_validation:
        mismatches.append({
            "path": "job.comparison_role",
            "candidate": cjob.get("comparison_role"),
            "baseline": bjob.get("comparison_role"),
            "candidate_missing": cjob.get("comparison_role") is None,
            "baseline_missing": bjob.get("comparison_role") is None,
            "category": "protocol",
            "requirement": "candidate_baseline_roles",
            "allowed": False,
            "allowlist_reason": None,
            "reason": "linked jobs must have candidate and baseline roles",
        })
    mismatches.sort(key=lambda row: (str(row["path"]), str(row["reason"])))
    allowed_mismatches = [row for row in mismatches if row["allowed"]]
    disallowed_mismatches = [row for row in mismatches if not row["allowed"]]
    cparams = (cfinal or candidate).get("n_params")
    bparams = (bfinal or baseline).get("n_params")
    ratio = None
    if cparams is not None and bparams not in (None, 0):
        ratio = (float(cparams) - float(bparams)) / max(abs(float(bparams)), 1.0)
    matched = {
        path: not any(row["path"] == path and not row["allowed"] for row in mismatches)
        for path in sorted(required_paths)
        if path.startswith(("train.", "data."))
    }
    training_ok = all(
        not (row["category"] == "training" and not row["allowed"])
        for row in mismatches
    )
    data_ok = all(
        not (row["category"] == "data" and not row["allowed"])
        for row in mismatches
    )
    return {
        "complete": complete,
        "valid": not disallowed_mismatches,
        "schema_id": schema_id,
        "requirements": list(required),
        "intentional_differences": list(intentional_entries),
        "schema_errors": schema_errors,
        "mismatches": mismatches,
        "allowed_mismatches": allowed_mismatches,
        "disallowed_mismatches": disallowed_mismatches,
        "fairness_mismatches": mismatches,
        "same_dataset": cjob.get("dataset_name") == bjob.get("dataset_name"),
        "same_seed": int(cjob.get("seed", -1)) == int(bjob.get("seed", -2)),
        "same_steps": int(cjob.get("steps", -1)) == int(bjob.get("steps", -2)),
        "same_eval_interval": int(cjob.get("eval_every", -1)) == int(bjob.get("eval_every", -2)),
        "same_device_target": (cjob.get("device_target") or "auto") == (bjob.get("device_target") or "auto"),
        "same_training_budget": training_ok,
        "same_preprocessing": data_ok,
        "matched_config_fields": matched,
        "role_validation": role_validation,
        "parameter_delta_ratio": ratio,
        "seed_axes": {"candidate": caxes, "baseline": baxes},
    }


def _solver_nested(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _solver_hash(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Solver competition budgets must be finite canonical JSON."
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def solver_budget_contract_hash(budget: Mapping[str, Any]) -> str:
    """Hash a prespecified solver budget without trusting a supplied hash."""
    if not isinstance(budget, Mapping):
        raise ValueError("Solver competition budget must be a mapping.")
    payload = dict(budget)
    payload.pop("contract_hash", None)
    return _solver_hash(payload)


def solver_configuration_hash(configuration: Mapping[str, Any]) -> str:
    """Hash the exact solver configuration selected by bounded search."""
    if not isinstance(configuration, Mapping):
        raise ValueError("Solver configuration must be a mapping.")
    payload = dict(configuration)
    if any(not isinstance(key, str) for key in payload):
        raise ValueError("Solver configuration keys must be strings.")
    expected_keys = {"algorithm_family", "hyperparameters"}
    if set(payload) != expected_keys:
        raise ValueError(
            "Solver configuration must contain exactly algorithm_family and "
            "hyperparameters."
        )
    algorithm_family = payload["algorithm_family"]
    if not isinstance(algorithm_family, str) or not algorithm_family.strip():
        raise ValueError("Solver algorithm_family must be a non-empty string.")
    hyperparameters = payload["hyperparameters"]
    if not isinstance(hyperparameters, Mapping):
        raise ValueError("Solver hyperparameters must be a mapping.")
    return _solver_hash(
        {
            "algorithm_family": algorithm_family,
            "hyperparameters": dict(hyperparameters),
        }
    )


def _solver_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _solver_uuid(value: object) -> bool:
    try:
        return str(uuid.UUID(str(value))) == str(value)
    except (ValueError, TypeError, AttributeError):
        return False


def _solver_resource_entry(
    run: Mapping[str, Any], field: str
) -> Mapping[str, Any] | None:
    resources = run.get("resources")
    if not isinstance(resources, Mapping):
        return None
    entry = resources.get(field)
    return entry if isinstance(entry, Mapping) else None


def _solver_number(value: object, *, integer: bool = False) -> float | int | None:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    if integer:
        if not number.is_integer():
            return None
        return int(number)
    return number


def evaluate_solver_competition(
    runs: Iterable[Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any],
    schema: Mapping[str, Any],
    search_ledgers: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Validate solver-competition admission without performing inference.

    Problem instances are the independent axis. Solver/model seeds are nested
    observations and can never inflate the independent-pair count.
    """
    violations: list[dict[str, Any]] = []

    def reject(path: str, reason: str, **detail: Any) -> None:
        violations.append({"path": path, "reason": reason, **detail})

    def string_list(value: object, path: str) -> list[str]:
        if not isinstance(value, list):
            reject(path, "must be a list of non-empty strings")
            return []
        normalized: list[str] = []
        for index, item in enumerate(value):
            if not isinstance(item, str) or not item.strip():
                reject(
                    f"{path}[{index}]",
                    "must be a non-empty string",
                )
                continue
            normalized.append(item)
        return normalized

    if not isinstance(schema, Mapping):
        reject("fairness_schema", "fairness schema must be a mapping")
        schema = {}
    else:
        schema = dict(schema)
    if not isinstance(protocol, Mapping):
        reject("protocol", "solver competition protocol must be a mapping")
        protocol = {}
    else:
        protocol = dict(protocol)
    schema_id = str(schema.get("schema_id") or "")
    comparator_kind = str(schema.get("comparator_kind") or "")
    if schema_id != SOLVER_COMPETITION_SCHEMA_ID:
        reject(
            "fairness_schema.schema_id",
            "the dedicated evaluator accepts solver_competition_v1 only",
            observed=schema_id or None,
        )
    if comparator_kind != SOLVER_COMPETITION_COMPARATOR_KIND:
        reject(
            "fairness_schema.comparator_kind",
            "unsupported solver comparator kind",
            observed=comparator_kind or None,
        )
    if schema.get("pairing_axis") != "problem_instance":
        reject(
            "fairness_schema.pairing_axis",
            "solver competitions must pair on immutable problem instances",
        )
    if schema.get("nested_seed_axis") != "model_seed":
        reject(
            "fairness_schema.nested_seed_axis",
            "model seeds must remain nested within problem instances",
        )
    if protocol.get("analysis_mode") != "solver_competition":
        reject(
            "protocol.analysis_mode",
            "solver competition admission requires analysis_mode='solver_competition'",
        )
    if protocol.get("schema_id") != SOLVER_COMPETITION_SCHEMA_ID:
        reject(
            "protocol.schema_id",
            "protocol and fairness schema IDs must match",
        )
    if protocol.get("comparator_kind") != SOLVER_COMPETITION_COMPARATOR_KIND:
        reject(
            "protocol.comparator_kind",
            "protocol comparator kind is unsupported or missing",
        )
    aggregation_rule = protocol.get("within_instance_aggregation")
    aggregation_valid = aggregation_rule == "best_within_total_budget"
    if not aggregation_valid:
        reject(
            "protocol.within_instance_aggregation",
            "within-instance selection must equal 'best_within_total_budget'",
            observed=aggregation_rule,
        )

    allowed_optimum_roles = string_list(
        schema.get("allowed_optimum_roles"),
        "fairness_schema.allowed_optimum_roles",
    )

    raw_instances = protocol.get("problem_instances")
    instances: list[dict[str, Any]] = []
    if not isinstance(raw_instances, list) or not raw_instances:
        reject(
            "protocol.problem_instances",
            "at least one immutable problem instance is required",
        )
    else:
        seen_instance_ids: set[str] = set()
        seen_instance_hashes: set[str] = set()
        for index, raw in enumerate(raw_instances):
            path = f"protocol.problem_instances[{index}]"
            if not isinstance(raw, Mapping):
                reject(path, "problem instance must be a mapping")
                continue
            raw_instance_id = raw.get("instance_id")
            instance_id = (
                raw_instance_id.strip()
                if isinstance(raw_instance_id, str)
                else ""
            )
            raw_instance_hash = raw.get("instance_hash")
            instance_hash = (
                raw_instance_hash.strip()
                if isinstance(raw_instance_hash, str)
                else ""
            )
            raw_energy_units = raw.get("energy_units")
            energy_units = (
                raw_energy_units.strip()
                if isinstance(raw_energy_units, str)
                else ""
            )
            identity_valid = True
            if not instance_id:
                reject(f"{path}.instance_id", "instance_id is required")
                identity_valid = False
            if not _solver_sha256(instance_hash):
                reject(
                    f"{path}.instance_hash",
                    "instance_hash must be a lowercase SHA-256 digest",
                )
                identity_valid = False
            if instance_id in seen_instance_ids:
                reject(f"{path}.instance_id", "problem instances must be unique")
                identity_valid = False
            if instance_hash in seen_instance_hashes:
                reject(
                    f"{path}.instance_hash",
                    "problem instance hashes must be unique",
                )
                identity_valid = False
            if not energy_units:
                reject(f"{path}.energy_units", "energy_units is required")

            raw_optimum = raw.get("optimum_reference")
            if not isinstance(raw_optimum, Mapping):
                reject(
                    f"{path}.optimum_reference",
                    "each problem instance requires its own optimum reference",
                )
                raw_optimum = {}
            raw_role = raw_optimum.get("role")
            optimum_role = raw_role if isinstance(raw_role, str) else None
            if optimum_role not in allowed_optimum_roles:
                reject(
                    f"{path}.optimum_reference.role",
                    "optimum role is unsupported by the fairness schema",
                    observed=raw_role,
                )
            if optimum_role == "best_known_optimum":
                reject(
                    f"{path}.optimum_reference.role",
                    "best-known optima require gap_to_best_known, not exact energy error",
                )
            optimum_value = _solver_number(raw_optimum.get("value"))
            if optimum_value is None:
                reject(
                    f"{path}.optimum_reference.value",
                    "optimum value must be finite",
                )
            raw_reference_id = raw_optimum.get("reference_id")
            reference_id = (
                raw_reference_id.strip()
                if isinstance(raw_reference_id, str)
                else ""
            )
            if not reference_id:
                reject(
                    f"{path}.optimum_reference.reference_id",
                    "optimum reference_id is required",
                )
            raw_optimum_units = raw_optimum.get("units")
            optimum_units = (
                raw_optimum_units.strip()
                if isinstance(raw_optimum_units, str)
                else ""
            )
            if not optimum_units:
                reject(
                    f"{path}.optimum_reference.units",
                    "optimum units are required",
                )
            elif energy_units and optimum_units != energy_units:
                reject(
                    f"{path}.optimum_reference.units",
                    "optimum units must match the problem energy units",
                )

            if identity_valid:
                seen_instance_ids.add(instance_id)
                seen_instance_hashes.add(instance_hash)
                instances.append(
                    {
                        "instance_id": instance_id,
                        "instance_hash": instance_hash,
                        "energy_units": energy_units,
                        "optimum_reference": {
                            "reference_id": reference_id,
                            "role": optimum_role,
                            "value": optimum_value,
                            "units": optimum_units,
                        },
                    }
                )

    raw_seeds = protocol.get("model_seeds")
    model_seeds: list[int] = []
    if not isinstance(raw_seeds, list) or not raw_seeds:
        reject("protocol.model_seeds", "at least one model seed is required")
    else:
        for index, raw_seed in enumerate(raw_seeds):
            seed = _solver_number(raw_seed, integer=True)
            if seed is None or int(seed) < 0:
                reject(
                    f"protocol.model_seeds[{index}]",
                    "model seeds must be non-negative integers",
                )
                continue
            model_seeds.append(int(seed))
        if len(model_seeds) != len(set(model_seeds)):
            reject("protocol.model_seeds", "model seeds must be unique")
        model_seeds = list(dict.fromkeys(model_seeds))

    budget = protocol.get("budget")
    budget_hash: str | None = None
    if not isinstance(budget, Mapping):
        reject("protocol.budget", "a prespecified budget mapping is required")
        budget = {}
    else:
        try:
            budget_hash = solver_budget_contract_hash(budget)
        except ValueError as exc:
            reject("protocol.budget", str(exc))
        if budget.get("prespecified") is not True:
            reject(
                "protocol.budget.prespecified",
                "solver budgets must be prespecified",
            )
        claimed_hash = budget.get("contract_hash")
        if claimed_hash is not None and claimed_hash != budget_hash:
            reject(
                "protocol.budget.contract_hash",
                "caller-provided budget hash does not match the server-derived hash",
                observed=claimed_hash,
                expected=budget_hash,
            )

    required_budget_fields = string_list(
        schema.get("equal_budget_fields"),
        "fairness_schema.equal_budget_fields",
    )
    for field in required_budget_fields:
        if _solver_nested(budget, str(field)) is _MISSING:
            reject(
                f"protocol.budget.{field}",
                "required equal-budget field is missing",
            )
    budget_seeds = _solver_nested(budget, "evaluation.model_seeds")
    if budget_seeds is not _MISSING:
        if not isinstance(budget_seeds, list):
            reject(
                "protocol.budget.evaluation.model_seeds",
                "budget model seeds must be a list",
            )
        elif budget_seeds != model_seeds:
            reject(
                "protocol.budget.evaluation.model_seeds",
                "budget model seeds must exactly match the nested study seeds",
            )
    numeric_budget_fields = {
        "search.max_trials": True,
        "search.max_objective_evaluations": True,
        "search.max_wall_seconds": False,
        "evaluation.max_objective_evaluations_per_seed": True,
        "evaluation.max_wall_seconds_per_seed": False,
        "evaluation.shots_per_objective": True,
    }
    budget_numbers: dict[str, float | int] = {}
    for field, integer in numeric_budget_fields.items():
        raw_value = _solver_nested(budget, field)
        value = None if raw_value is _MISSING else _solver_number(
            raw_value, integer=integer
        )
        if value is None or float(value) <= 0.0:
            reject(
                f"protocol.budget.{field}",
                "budget ceilings must be positive finite numbers",
            )
        else:
            budget_numbers[field] = value
    target_metric = _solver_nested(budget, "evaluation.target_metric")
    if target_metric != "ground_state_energy_error":
        reject(
            "protocol.budget.evaluation.target_metric",
            "this claim admits ground_state_energy_error only",
        )
    search_target_metric = (
        target_metric if isinstance(target_metric, str) else None
    )
    target_value = _solver_nested(budget, "evaluation.target_value")
    target_number = (
        None
        if target_value is _MISSING
        else _solver_number(target_value)
    )
    if target_number is None or float(target_number) <= 0.0:
        reject(
            "protocol.budget.evaluation.target_value",
            "target value must be a positive finite energy-error tolerance",
        )
    termination_rule = _solver_nested(budget, "evaluation.termination_rule")
    if not isinstance(termination_rule, str) or not termination_rule.strip():
        reject(
            "protocol.budget.evaluation.termination_rule",
            "an identical prespecified termination rule is required",
        )

    declarations: dict[str, Mapping[str, Any]] = {}
    forbidden_reference_solvers = {
        "exact_diagonalization",
        "best_product_state",
    }
    for role, key, computation_kind in (
        ("candidate", "candidate_solver", "quantum"),
        ("comparator", "comparator_solver", "classical"),
    ):
        declaration = protocol.get(key)
        if not isinstance(declaration, Mapping):
            reject(f"protocol.{key}", "registered solver declaration is required")
            declaration = {}
        declarations[role] = declaration
        for field in (
            "solver_id",
            "solver_version",
            "runner_id",
            "runner_version",
        ):
            value = declaration.get(field)
            if not isinstance(value, str) or not value.strip():
                reject(
                    f"protocol.{key}.{field}",
                    "solver declarations require a non-empty identity",
                )
        solver_id = declaration.get("solver_id")
        if isinstance(solver_id, str) and solver_id in forbidden_reference_solvers:
            reject(
                f"protocol.{key}.solver_id",
                "static oracle and product-state references are not solver runs",
            )
        if declaration.get("computation_kind") != computation_kind:
            reject(
                f"protocol.{key}.computation_kind",
                f"{role} computation_kind must be {computation_kind!r}",
            )
        registration = registry_module.solver_runner_registration(
            runner_id=declaration.get("runner_id"),
            runner_version=declaration.get("runner_version"),
            solver_id=declaration.get("solver_id"),
            solver_version=declaration.get("solver_version"),
        )
        if registration is None:
            reject(
                f"protocol.{key}",
                "solver runner identity is absent from the canonical registry",
            )
        else:
            if registration.get("task_type") != "ground_state":
                reject(
                    f"protocol.{key}",
                    "canonical solver registration is not a ground-state runner",
                )
            if registration.get("computation_kind") != computation_kind:
                reject(
                    f"protocol.{key}.computation_kind",
                    "canonical solver registration has the wrong computation kind",
                )
            if not (
                registration.get("registration_status") == "comparison_eligible"
                and registration.get("comparison_eligible") is True
            ):
                reject(
                    f"protocol.{key}",
                    "canonical solver registration is not comparison eligible",
                )

    search_violation_start = len(violations)
    if search_ledgers is None:
        search_rows: list[object] = []
    elif isinstance(search_ledgers, (Mapping, str, bytes)):
        reject(
            "search_ledgers",
            "search ledgers must be an iterable of ledger mappings",
        )
        search_rows = []
    else:
        try:
            search_rows = list(search_ledgers)
        except TypeError:
            reject(
                "search_ledgers",
                "search ledgers must be an iterable of ledger mappings",
            )
            search_rows = []

    observed_search: dict[str, list[Mapping[str, Any]]] = {}
    search_summaries: dict[str, dict[str, Any]] = {}
    seen_trial_ids: set[str] = set()
    for index, raw_ledger in enumerate(search_rows):
        path = f"search_ledgers[{index}]"
        if not isinstance(raw_ledger, Mapping):
            reject(path, "search ledger must be a mapping")
            continue
        raw_role = raw_ledger.get("role")
        role = raw_role if isinstance(raw_role, str) else ""
        if role not in {"candidate", "comparator"}:
            reject(f"{path}.role", "search ledger role is unsupported")
        else:
            observed_search.setdefault(role, []).append(raw_ledger)

        declaration = declarations.get(role, {})
        ledger_solver = (
            raw_ledger.get("solver")
            if isinstance(raw_ledger.get("solver"), Mapping)
            else {}
        )
        if not isinstance(raw_ledger.get("solver"), Mapping):
            reject(f"{path}.solver", "search ledger solver identity is required")
        for field in (
            "solver_id",
            "solver_version",
            "runner_id",
            "runner_version",
        ):
            value = ledger_solver.get(field)
            if not isinstance(value, str) or not value.strip():
                reject(
                    f"{path}.solver.{field}",
                    "search ledger solver identity must be a non-empty string",
                )
            if value != declaration.get(field):
                reject(
                    f"{path}.solver.{field}",
                    "search ledger does not match its prespecified solver",
                    observed=value,
                    expected=declaration.get(field),
                )
        if raw_ledger.get("budget_contract_hash") != budget_hash:
            reject(
                f"{path}.budget_contract_hash",
                "search ledger budget does not match the server-derived competition budget",
                observed=raw_ledger.get("budget_contract_hash"),
                expected=budget_hash,
            )
        if raw_ledger.get("selection_rule") != aggregation_rule:
            reject(
                f"{path}.selection_rule",
                "search selection rule must match within_instance_aggregation",
                observed=raw_ledger.get("selection_rule"),
                expected=aggregation_rule,
            )
        if raw_ledger.get("selection_metric") != search_target_metric:
            reject(
                f"{path}.selection_metric",
                "search selection metric must match the evaluation target metric",
                observed=raw_ledger.get("selection_metric"),
                expected=search_target_metric,
            )

        trials = raw_ledger.get("trials")
        if not isinstance(trials, list) or not trials:
            reject(f"{path}.trials", "at least one observed search trial is required")
            trials = []
        max_trials = budget_numbers.get("search.max_trials")
        if max_trials is not None and len(trials) > int(max_trials):
            reject(
                f"{path}.trials",
                "search exceeded the prespecified trial ceiling",
                observed=len(trials),
                maximum=int(max_trials),
            )

        total_objective_evaluations = 0
        total_wall_seconds = 0.0
        trial_metric_values: dict[str, float] = {}
        trial_configuration_hashes: dict[str, str] = {}
        ledger_trial_ids: set[str] = set()
        for trial_index, raw_trial in enumerate(trials):
            trial_path = f"{path}.trials[{trial_index}]"
            if not isinstance(raw_trial, Mapping):
                reject(trial_path, "search trial must be a mapping")
                continue
            raw_trial_id = raw_trial.get("trial_id")
            trial_id = raw_trial_id if isinstance(raw_trial_id, str) else ""
            if not trial_id.strip():
                reject(f"{trial_path}.trial_id", "trial_id is required")
            elif trial_id in ledger_trial_ids or trial_id in seen_trial_ids:
                reject(
                    f"{trial_path}.trial_id",
                    "search trial identities must be globally unique",
                )
            else:
                ledger_trial_ids.add(trial_id)
                seen_trial_ids.add(trial_id)
            if raw_trial.get("status") != "done":
                reject(
                    f"{trial_path}.status",
                    "only completed search trials may select a solver configuration",
                )
            objective_evaluations = _solver_number(
                raw_trial.get("objective_evaluations"), integer=True
            )
            if objective_evaluations is None or int(objective_evaluations) <= 0:
                reject(
                    f"{trial_path}.objective_evaluations",
                    "search objective evaluations must be a positive integer",
                )
            else:
                total_objective_evaluations += int(objective_evaluations)
            wall_seconds = _solver_number(raw_trial.get("wall_seconds"))
            if wall_seconds is None or float(wall_seconds) <= 0.0:
                reject(
                    f"{trial_path}.wall_seconds",
                    "search wall time must be positive and finite",
                )
            else:
                total_wall_seconds += float(wall_seconds)
            selection_value = _solver_number(
                raw_trial.get("selection_metric_value")
            )
            if selection_value is None or float(selection_value) < 0.0:
                reject(
                    f"{trial_path}.selection_metric_value",
                    "selection metric value must be finite and non-negative",
                )
            elif trial_id:
                trial_metric_values[trial_id] = float(selection_value)

            raw_configuration = raw_trial.get("configuration")
            derived_configuration_hash: str | None = None
            try:
                derived_configuration_hash = solver_configuration_hash(
                    raw_configuration
                )
            except ValueError as exc:
                reject(f"{trial_path}.configuration", str(exc))
            reported_configuration_hash = raw_trial.get("configuration_hash")
            if not _solver_sha256(reported_configuration_hash):
                reject(
                    f"{trial_path}.configuration_hash",
                    "configuration_hash must be a lowercase SHA-256 digest",
                )
            elif reported_configuration_hash != derived_configuration_hash:
                reject(
                    f"{trial_path}.configuration_hash",
                    "configuration hash does not match the server-derived search configuration",
                    observed=reported_configuration_hash,
                    expected=derived_configuration_hash,
                )
            elif trial_id:
                trial_configuration_hashes[trial_id] = reported_configuration_hash

        max_search_objectives = budget_numbers.get(
            "search.max_objective_evaluations"
        )
        if (
            max_search_objectives is not None
            and total_objective_evaluations > int(max_search_objectives)
        ):
            reject(
                f"{path}.trials",
                "search exceeded the aggregate objective-evaluation ceiling",
                observed=total_objective_evaluations,
                maximum=int(max_search_objectives),
            )
        max_search_wall = budget_numbers.get("search.max_wall_seconds")
        if (
            max_search_wall is not None
            and total_wall_seconds > float(max_search_wall)
        ):
            reject(
                f"{path}.trials",
                "search exceeded the aggregate wall-time ceiling",
                observed=total_wall_seconds,
                maximum=float(max_search_wall),
            )

        raw_selected_trial_id = raw_ledger.get("selected_trial_id")
        selected_trial_id = (
            raw_selected_trial_id
            if isinstance(raw_selected_trial_id, str)
            else ""
        )
        if selected_trial_id not in ledger_trial_ids:
            reject(
                f"{path}.selected_trial_id",
                "selected_trial_id must identify one completed ledger trial",
                observed=raw_selected_trial_id,
            )
        elif (
            selected_trial_id in trial_metric_values
            and trial_metric_values
            and trial_metric_values[selected_trial_id]
            > min(trial_metric_values.values()) + 1e-12
        ):
            reject(
                f"{path}.selected_trial_id",
                "best_within_total_budget must select the lowest observed energy error",
            )
        selected_configuration_hash = trial_configuration_hashes.get(
            selected_trial_id
        )
        if selected_trial_id and selected_configuration_hash is None:
            reject(
                f"{path}.selected_trial_id",
                "selected search trial must have a valid configuration identity",
            )

        if role in {"candidate", "comparator"}:
            search_summaries[role] = {
                "trial_count": len(trials),
                "objective_evaluations": total_objective_evaluations,
                "wall_seconds": total_wall_seconds,
                "selected_trial_id": selected_trial_id or None,
                "selected_configuration_hash": selected_configuration_hash,
            }

    for role in ("candidate", "comparator"):
        role_ledgers = observed_search.get(role, [])
        if not role_ledgers:
            reject(
                "search_ledgers.missing_role",
                "each solver role requires exactly one observed search ledger",
                role=role,
            )
        elif len(role_ledgers) != 1:
            reject(
                "search_ledgers.duplicate_role",
                "each solver role requires exactly one observed search ledger",
                role=role,
                observed=len(role_ledgers),
            )
    search_complete = (
        all(len(observed_search.get(role, [])) == 1 for role in ("candidate", "comparator"))
        and aggregation_valid
        and len(violations) == search_violation_start
    )

    expected_cells = {
        (instance["instance_id"], role, seed)
        for instance in instances
        for role in ("candidate", "comparator")
        for seed in model_seeds
    }
    observed: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {}
    if runs is None:
        run_rows: list[object] = []
    elif isinstance(runs, (Mapping, str, bytes)):
        reject("runs", "solver results must be an iterable of mappings")
        run_rows = []
    else:
        try:
            run_rows = list(runs)
        except TypeError:
            reject("runs", "solver results must be an iterable of mappings")
            run_rows = []
    required_present = string_list(
        schema.get("required_present"),
        "fairness_schema.required_present",
    )
    resource_fields = string_list(
        schema.get("resource_requirements"),
        "fairness_schema.resource_requirements",
    )
    seen_run_uuids: set[str] = set()
    seen_manifest_hashes: set[str] = set()
    instances_by_id = {instance["instance_id"]: instance for instance in instances}

    for index, run in enumerate(run_rows):
        path = f"runs[{index}]"
        if not isinstance(run, Mapping):
            reject(path, "solver result must be a mapping")
            continue
        normalized = dict(run)
        normalized["budget"] = {"contract_hash": run.get("budget_contract_hash")}
        for field in required_present:
            value = _solver_nested(normalized, str(field))
            if value is _MISSING or value is None or value == "":
                reject(
                    f"{path}.{field}",
                    "required solver-result identity is missing",
                )
        run_uuid = run.get("run_uuid")
        if not _solver_uuid(run_uuid):
            reject(f"{path}.run_uuid", "run_uuid must identify an executed run")
        elif run_uuid in seen_run_uuids:
            reject(
                f"{path}.run_uuid",
                "executed run identities must be unique across grid cells",
            )
        else:
            seen_run_uuids.add(run_uuid)
        manifest_hash = run.get("manifest_hash")
        if not _solver_sha256(manifest_hash):
            reject(
                f"{path}.manifest_hash",
                "manifest_hash must be a lowercase SHA-256 digest",
            )
        elif manifest_hash in seen_manifest_hashes:
            reject(
                f"{path}.manifest_hash",
                "immutable run manifests must be unique across grid cells",
            )
        else:
            seen_manifest_hashes.add(manifest_hash)

        problem = run.get("problem") if isinstance(run.get("problem"), Mapping) else {}
        solver = run.get("solver") if isinstance(run.get("solver"), Mapping) else {}
        raw_role = solver.get("role")
        role = raw_role if isinstance(raw_role, str) else ""
        seed_value = _solver_number(solver.get("model_seed"), integer=True)
        raw_instance_id = problem.get("instance_id")
        instance_id = raw_instance_id if isinstance(raw_instance_id, str) else ""
        seed = int(seed_value) if seed_value is not None else -1
        cell = (instance_id, role, seed)
        observed.setdefault(cell, []).append(run)

        if run.get("status") != "done":
            reject(f"{path}.status", "only completed solver runs fill a grid cell")
        if role not in {"candidate", "comparator"}:
            reject(f"{path}.solver.role", "solver role is unsupported")
        declaration = declarations.get(role, {})
        for field in (
            "solver_id",
            "solver_version",
            "runner_id",
            "runner_version",
            "computation_kind",
        ):
            if solver.get(field) != declaration.get(field):
                reject(
                    f"{path}.solver.{field}",
                    "solver result does not match its prespecified declaration",
                    observed=solver.get(field),
                    expected=declaration.get(field),
                )
        canonical_registration = registry_module.solver_runner_registration(
            runner_id=solver.get("runner_id"),
            runner_version=solver.get("runner_version"),
            solver_id=solver.get("solver_id"),
            solver_version=solver.get("solver_version"),
        )
        if canonical_registration is None:
            reject(
                f"{path}.solver",
                "solver runner identity is absent from the canonical registry",
            )
        elif not (
            canonical_registration.get("task_type") == "ground_state"
            and canonical_registration.get("computation_kind")
            == solver.get("computation_kind")
            and canonical_registration.get("registration_status")
            == "comparison_eligible"
            and canonical_registration.get("comparison_eligible") is True
        ):
            reject(
                f"{path}.solver",
                "canonical solver registration is not comparison eligible",
            )
        if (
            canonical_registration is None
            or solver.get("registration_status")
            != canonical_registration.get("registration_status")
        ):
            reject(
                f"{path}.solver.registration_status",
                "reported registration status does not match the canonical registry",
            )
        solver_id = solver.get("solver_id")
        if isinstance(solver_id, str) and solver_id in forbidden_reference_solvers:
            reject(
                f"{path}.solver.solver_id",
                "static oracle and product-state references are not solver runs",
            )
        selected_search = search_summaries.get(role, {})
        selected_trial_id = solver.get("selected_trial_id")
        if (
            not isinstance(selected_trial_id, str)
            or selected_trial_id != selected_search.get("selected_trial_id")
        ):
            reject(
                f"{path}.solver.selected_trial_id",
                "evaluation run must identify the selected search trial",
                observed=selected_trial_id,
                expected=selected_search.get("selected_trial_id"),
            )
        run_configuration = {
            "algorithm_family": solver.get("algorithm_family"),
            "hyperparameters": solver.get("hyperparameters"),
        }
        derived_configuration_hash: str | None = None
        try:
            derived_configuration_hash = solver_configuration_hash(
                run_configuration
            )
        except ValueError as exc:
            reject(f"{path}.solver", str(exc))
        reported_configuration_hash = solver.get("configuration_hash")
        if not _solver_sha256(reported_configuration_hash):
            reject(
                f"{path}.solver.configuration_hash",
                "configuration_hash must be a lowercase SHA-256 digest",
            )
        elif reported_configuration_hash != derived_configuration_hash:
            reject(
                f"{path}.solver.configuration_hash",
                "evaluation configuration hash is not server-derived from the run",
                observed=reported_configuration_hash,
                expected=derived_configuration_hash,
            )
        if reported_configuration_hash != selected_search.get(
            "selected_configuration_hash"
        ):
            reject(
                f"{path}.solver.configuration_hash",
                "evaluation configuration does not match the selected search trial",
                observed=reported_configuration_hash,
                expected=selected_search.get("selected_configuration_hash"),
            )
        for field in ("code_hash", "environment_hash"):
            if not _solver_sha256(solver.get(field)):
                reject(
                    f"{path}.solver.{field}",
                    f"{field} must be a lowercase SHA-256 digest",
                )

        expected_instance = instances_by_id.get(instance_id)
        if expected_instance is None:
            reject(
                f"{path}.problem.instance_id",
                "run uses an undeclared problem instance",
            )
        else:
            if problem.get("instance_hash") != expected_instance["instance_hash"]:
                reject(
                    f"{path}.problem.instance_hash",
                    "instance hash does not match the immutable protocol identity",
                )
            if problem.get("energy_units") != expected_instance["energy_units"]:
                reject(
                    f"{path}.problem.energy_units",
                    "problem energy units do not match the protocol instance",
                )
        if seed not in model_seeds:
            reject(
                f"{path}.solver.model_seed",
                "run uses an undeclared nested model seed",
            )
        if run.get("budget_contract_hash") != budget_hash:
            reject(
                f"{path}.budget_contract_hash",
                "run budget does not match the server-derived competition budget",
                observed=run.get("budget_contract_hash"),
                expected=budget_hash,
            )

        optimum = run.get("optimum") if isinstance(run.get("optimum"), Mapping) else {}
        expected_optimum = (
            expected_instance.get("optimum_reference", {})
            if expected_instance is not None
            else {}
        )
        for field in ("reference_id", "role", "value", "units"):
            if optimum.get(field) != expected_optimum.get(field):
                reject(
                    f"{path}.optimum.{field}",
                    "run optimum reference does not match its protocol instance",
                )

        resource_entries: dict[str, Mapping[str, Any]] = {}
        for field in resource_fields:
            entry = _solver_resource_entry(run, str(field))
            if entry is None:
                reject(
                    f"{path}.resources.{field}",
                    "required resource entry is missing",
                )
                continue
            resource_entries[str(field)] = entry
            if not isinstance(entry.get("status"), str) or not str(
                entry.get("status")
            ).strip():
                reject(
                    f"{path}.resources.{field}.status",
                    "resource status must be explicit",
                )

        objective_entry = resource_entries.get("objective_evaluations", {})
        objective_status = objective_entry.get("status")
        if not isinstance(objective_status, str) or objective_status not in {
            "measured",
            "derived",
        }:
            reject(
                f"{path}.resources.objective_evaluations.status",
                "objective evaluations must be measured or derived",
            )
        objective_calls = _solver_number(
            objective_entry.get("value"), integer=True
        )
        if objective_calls is None or int(objective_calls) < 0:
            reject(
                f"{path}.resources.objective_evaluations.value",
                "objective evaluations must be a non-negative integer",
            )
        elif (
            "evaluation.max_objective_evaluations_per_seed" in budget_numbers
            and int(objective_calls)
            > int(
                budget_numbers["evaluation.max_objective_evaluations_per_seed"]
            )
        ):
            reject(
                f"{path}.resources.objective_evaluations.value",
                "run exceeded the prespecified objective-evaluation ceiling",
            )
        wall_entry = resource_entries.get("wall_seconds", {})
        if wall_entry.get("status") != "measured":
            reject(
                f"{path}.resources.wall_seconds.status",
                "wall time must be measured",
            )
        wall_seconds = _solver_number(wall_entry.get("value"))
        if wall_seconds is None or float(wall_seconds) <= 0.0:
            reject(
                f"{path}.resources.wall_seconds.value",
                "wall time must be positive and finite",
            )
        elif (
            "evaluation.max_wall_seconds_per_seed" in budget_numbers
            and float(wall_seconds)
            > float(budget_numbers["evaluation.max_wall_seconds_per_seed"])
        ):
            reject(
                f"{path}.resources.wall_seconds.value",
                "run exceeded the prespecified wall-time ceiling",
            )
        state_entry = resource_entries.get("state_preparations", {})
        state_status = state_entry.get("status")
        if not isinstance(state_status, str) or state_status not in {
            "measured",
            "derived",
        }:
            reject(
                f"{path}.resources.state_preparations.status",
                "state preparations must be measured or derived",
            )
        state_preparations = _solver_number(
            state_entry.get("value"), integer=True
        )
        if state_preparations is None or int(state_preparations) < 0:
            reject(
                f"{path}.resources.state_preparations.value",
                "state preparations must be a non-negative integer",
            )
        representation = resource_entries.get("state_representation", {}).get(
            "value"
        )
        if not isinstance(representation, str) or not representation.strip():
            reject(
                f"{path}.resources.state_representation.value",
                "state representation must be declared",
            )
        representation_status = resource_entries.get(
            "state_representation", {}
        ).get("status")
        if (
            not isinstance(representation_status, str)
            or representation_status not in {"configured", "measured"}
        ):
            reject(
                f"{path}.resources.state_representation.status",
                "state representation must be configured or measured",
            )

        shots_entry = resource_entries.get("total_shots", {})
        computation_kind = solver.get("computation_kind")
        if computation_kind == "quantum":
            shots_status = shots_entry.get("status")
            if not isinstance(shots_status, str) or shots_status not in {
                "measured",
                "derived",
            }:
                reject(
                    f"{path}.resources.total_shots.status",
                    "quantum shot accounting must be measured or derived",
                )
            shots = _solver_number(shots_entry.get("value"), integer=True)
            if shots is None or int(shots) <= 0:
                reject(
                    f"{path}.resources.total_shots.value",
                    "quantum solver comparisons require finite positive shots",
                )
            elif (
                objective_calls is not None
                and "evaluation.shots_per_objective" in budget_numbers
                and int(shots)
                > int(objective_calls)
                * int(budget_numbers["evaluation.shots_per_objective"])
            ):
                reject(
                    f"{path}.resources.total_shots.value",
                    "run exceeded its prespecified shot ceiling",
                )
        elif computation_kind == "classical":
            if not (
                shots_entry.get("value") is None
                and shots_entry.get("status") == "not_applicable"
            ):
                reject(
                    f"{path}.resources.total_shots",
                    "classical solver shots must be explicitly not_applicable",
                )
        else:
            reject(
                f"{path}.solver.computation_kind",
                "computation_kind must be quantum or classical",
            )

        peak_entry = resource_entries.get("peak_memory_bytes", {})
        peak_value = peak_entry.get("value")
        if peak_value is not None:
            peak_number = _solver_number(peak_value, integer=True)
            if peak_number is None or int(peak_number) < 0:
                reject(
                    f"{path}.resources.peak_memory_bytes.value",
                    "peak memory must be non-negative or explicitly unmeasured",
                )
        elif peak_entry.get("status") != "unmeasured":
            reject(
                f"{path}.resources.peak_memory_bytes.status",
                "missing peak memory must be labeled unmeasured",
            )

        outcome = run.get("outcome") if isinstance(run.get("outcome"), Mapping) else {}
        energy_error = _solver_number(outcome.get("energy_error"))
        if energy_error is None or float(energy_error) < 0.0:
            reject(
                f"{path}.outcome.energy_error",
                "energy error must be finite and non-negative",
            )
        reached_target = outcome.get("reached_target")
        censored = outcome.get("censored")
        if not isinstance(reached_target, bool) or not isinstance(censored, bool):
            reject(
                f"{path}.outcome",
                "reached_target and censored must be explicit booleans",
            )
        elif reached_target == censored:
            reject(
                f"{path}.outcome",
                "successful outcomes are uncensored and failures are censored",
            )
        if (
            energy_error is not None
            and target_number is not None
            and isinstance(reached_target, bool)
            and reached_target != (float(energy_error) <= float(target_number))
        ):
            reject(
                f"{path}.outcome.reached_target",
                "target flag disagrees with the prespecified energy tolerance",
            )
        time_to_target = outcome.get("time_to_target_seconds")
        if reached_target is True:
            time_number = _solver_number(time_to_target)
            if time_number is None or float(time_number) < 0.0:
                reject(
                    f"{path}.outcome.time_to_target_seconds",
                    "successful runs require a finite non-negative time to target",
                )
        elif time_to_target is not None:
            reject(
                f"{path}.outcome.time_to_target_seconds",
                "censored runs must not invent a time-to-target value",
            )

    missing_cells = sorted(expected_cells - set(observed))
    unexpected_cells = sorted(set(observed) - expected_cells)
    duplicate_cells = sorted(
        cell for cell, rows in observed.items() if len(rows) != 1
    )
    incomplete_cells = sorted(
        cell
        for cell, rows in observed.items()
        if len(rows) == 1 and rows[0].get("status") != "done"
    )
    for cell in missing_cells:
        reject(
            "grid.missing_cell",
            "complete-case analysis is forbidden; every declared cell is required",
            cell=list(cell),
        )
    for cell in unexpected_cells:
        reject(
            "grid.unexpected_cell",
            "run does not belong to the prespecified grid",
            cell=list(cell),
        )
    for cell in duplicate_cells:
        reject(
            "grid.duplicate_cell",
            "each instance/role/seed cell must contain exactly one run",
            cell=list(cell),
        )
    for cell in incomplete_cells:
        reject(
            "grid.incomplete_cell",
            "failed, cancelled, queued, and running cells disable the whole grid",
            cell=list(cell),
        )

    required_equal = string_list(
        schema.get("required_equal"),
        "fairness_schema.required_equal",
    )
    for instance in instances:
        for seed in model_seeds:
            candidate_rows = observed.get(
                (instance["instance_id"], "candidate", seed), []
            )
            comparator_rows = observed.get(
                (instance["instance_id"], "comparator", seed), []
            )
            if len(candidate_rows) != 1 or len(comparator_rows) != 1:
                continue
            candidate_view = dict(candidate_rows[0])
            candidate_view["budget"] = {
                "contract_hash": candidate_rows[0].get("budget_contract_hash")
            }
            comparator_view = dict(comparator_rows[0])
            comparator_view["budget"] = {
                "contract_hash": comparator_rows[0].get("budget_contract_hash")
            }
            for field in required_equal:
                candidate_value = _solver_nested(candidate_view, str(field))
                comparator_value = _solver_nested(comparator_view, str(field))
                if (
                    candidate_value is _MISSING
                    or comparator_value is _MISSING
                    or candidate_value != comparator_value
                ):
                    reject(
                        f"pairs.{instance['instance_id']}.{seed}.{field}",
                        "solver pair violates a required-equal field",
                        candidate=(
                            None if candidate_value is _MISSING else candidate_value
                        ),
                        comparator=(
                            None if comparator_value is _MISSING else comparator_value
                        ),
                    )

    grid_complete = bool(expected_cells) and not (
        missing_cells or unexpected_cells or duplicate_cells or incomplete_cells
    )
    complete = grid_complete and search_complete
    violations.sort(
        key=lambda row: (
            str(row.get("path") or ""),
            str(row.get("reason") or ""),
            str(row.get("cell") or ""),
        )
    )
    valid = complete and not violations
    instance_results = []
    for instance in instances:
        instance_cells = {
            (instance["instance_id"], role, seed)
            for role in ("candidate", "comparator")
            for seed in model_seeds
        }
        instance_violations = [
            row
            for row in violations
            if instance["instance_id"] in str(row)
            or tuple(row.get("cell") or ()) in instance_cells
        ]
        instance_results.append(
            {
                **instance,
                "nested_model_seeds": list(model_seeds),
                "expected_cells": len(instance_cells),
                "complete": all(
                    len(observed.get(cell, [])) == 1
                    and observed[cell][0].get("status") == "done"
                    for cell in instance_cells
                ),
                "valid": valid and not instance_violations,
                "violations": instance_violations,
            }
        )

    return {
        "schema_id": schema_id,
        "comparator_kind": comparator_kind,
        "pairing_axis": "problem_instance",
        "nested_seed_axis": "model_seed",
        "within_instance_aggregation": (
            aggregation_rule
            if aggregation_rule == "best_within_total_budget"
            else None
        ),
        "complete": complete,
        "grid_complete": grid_complete,
        "search_complete": search_complete,
        "valid": valid,
        "admission_valid": valid,
        "assessment_status": "admission_valid" if valid else "blocked",
        "budget_contract_hash": budget_hash,
        "expected_independent_pairs": len(instances),
        "n_pairs": len(instances) if valid else 0,
        "nested_model_seed_count": len(model_seeds),
        "expected_cell_count": len(expected_cells),
        "observed_run_count": len(run_rows),
        "observed_search_ledger_count": len(search_rows),
        "search_summaries": search_summaries,
        "missing_cells": [list(cell) for cell in missing_cells],
        "unexpected_cells": [list(cell) for cell in unexpected_cells],
        "duplicate_cells": [list(cell) for cell in duplicate_cells],
        "incomplete_cells": [list(cell) for cell in incomplete_cells],
        "instance_results": instance_results,
        "violations": violations,
        "paired_stats": None,
        "aggregate_available": False,
        "comparative_inference_enabled": False,
        "claim_eligible": False,
        "reason": (
            "solver competition admission is valid; inferential execution remains disabled"
            if valid
            else "solver competition admission failed closed"
        ),
        "limitations": [
            "Model seeds are nested observations; only problem instances are independent pairs.",
            "This contract slice validates admission only and never produces a solver-edge or quantum-advantage score.",
        ],
    }


def protocol_fairness(
    candidate: dict,
    baseline: dict,
    *,
    require_same_seed: bool = True,
) -> dict[str, Any]:
    """Return the original lightweight fairness contract unchanged.

    New claim-bearing paths use :func:`evaluate_fairness`.  This adapter keeps
    callers that intentionally compare aggregate, unpaired rows compatible and
    does not invent candidate/baseline role requirements they never supplied.
    """
    cjob = candidate.get("job", candidate)
    bjob = baseline.get("job", baseline)
    cparams = (candidate.get("final_run") or candidate).get("n_params")
    bparams = (baseline.get("final_run") or baseline).get("n_params")
    ratio = None
    if cparams is not None and bparams:
        ratio = (float(cparams) - float(bparams)) / max(abs(float(bparams)), 1.0)
    return {
        "same_dataset": cjob.get("dataset_name") == bjob.get("dataset_name"),
        "same_seed": (
            int(cjob.get("seed", -1)) == int(bjob.get("seed", -2))
            if require_same_seed
            else True
        ),
        "same_steps": int(cjob.get("steps", -1)) == int(bjob.get("steps", -2)),
        "same_eval_interval": int(cjob.get("eval_every", -1))
        == int(bjob.get("eval_every", -2)),
        "same_device_target": (cjob.get("device_target") or "auto")
        == (bjob.get("device_target") or "auto"),
        "parameter_delta_ratio": ratio,
    }


_RESOURCE_MODE_FIELDS = {
    "parameters": ("n_params",),
    "memory": ("peak_memory_bytes",),
    "circuit_evaluations": ("circuit_calls",),
    "state_preparations": ("state_preparations",),
    "logical_resources": ("logical_qubits", "logical_t_count"),
    "communication": ("communication_bits", "communication_rounds"),
}


def _resource_value(side: Mapping[str, Any] | None, field: str) -> Any:
    if not side:
        return None
    job = side.get("job") if isinstance(side.get("job"), Mapping) else {}
    final = (
        side.get("final_run")
        if isinstance(side.get("final_run"), Mapping)
        else {}
    )
    containers = [
        final,
        final.get("resources") if isinstance(final.get("resources"), Mapping) else {},
        side.get("resources") if isinstance(side.get("resources"), Mapping) else {},
        job.get("resources") if isinstance(job.get("resources"), Mapping) else {},
    ]
    config = _job_config(job)
    for container in containers:
        if field in container:
            return container[field]
    for key in (field, f"resources.{field}", f"lab.resource.{field}"):
        value = _flat_or_nested(config, key)
        if value is not None:
            return value
    return None


def _resource_value_valid(field: str, value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return False
        if field in {"wall_seconds", "n_params"}:
            return number > 0.0
        return number >= 0.0
    return isinstance(value, str) and bool(value.strip())


def evaluate_analogue_ladder(
    *,
    candidate: Mapping[str, Any] | None,
    baseline: Mapping[str, Any] | None,
    fairness: Mapping[str, Any] | None = None,
    controls: Iterable[Mapping[str, Any]] = (),
    claim: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assess a claim-specific analogue ladder without inventing matches."""
    claim = dict(claim or {})
    fairness_schema = claim.get("fairness_schema") or {}
    if fairness_schema.get("schema_id") == SOLVER_COMPETITION_SCHEMA_ID:
        return {
            "rungs": [],
            "required_complete": False,
            "missing_required": ["dedicated_solver_competition_evaluator"],
            "parameter_tolerance": None,
            "resource_requirements": list(
                fairness_schema.get("resource_requirements") or []
            ),
            "valid": False,
            "evaluator": "controlled_component_analogue",
            "reason": (
                "solver_competition_v1 cannot be evaluated as a controlled "
                "component analogue ladder"
            ),
        }
    tolerance = float(fairness_schema.get("parameter_tolerance", 0.10))
    requested = {
        str(item.get("id") or item.get("rung_id")): dict(item)
        for item in (claim.get("analogue_ladder") or [])
        if isinstance(item, Mapping) and (item.get("id") or item.get("rung_id"))
    }

    available = bool(candidate and baseline)
    fair = bool((fairness or {}).get("valid"))
    candidate_job = (candidate or {}).get("job", candidate or {})
    baseline_job = (baseline or {}).get("job", baseline or {})
    candidate_quantum = bool(
        isinstance(candidate_job, Mapping)
        and _uses_quantum_config(_job_config(candidate_job))
    )
    baseline_quantum = bool(
        isinstance(baseline_job, Mapping)
        and _uses_quantum_config(_job_config(baseline_job))
    )
    comparator_identity_valid = candidate_quantum and not baseline_quantum
    cparams = _resource_value(candidate, "n_params")
    bparams = _resource_value(baseline, "n_params")
    parameter_ratio = None
    parameter_values_valid = (
        _resource_value_valid("n_params", cparams)
        and _resource_value_valid("n_params", bparams)
    )
    if parameter_values_valid:
        parameter_ratio = (float(cparams) - float(bparams)) / max(
            abs(float(bparams)), 1.0
        )
    parameter_status = (
        "unknown"
        if cparams is None or bparams is None
        else "not_met"
        if not parameter_values_valid
        else "met"
        if abs(float(parameter_ratio)) <= tolerance
        else "not_met"
    )

    resource_spec = requested.get("resource_accounting", {})
    match_mode = str(resource_spec.get("match_mode") or "parameters")
    resource_fields = list(dict.fromkeys([
        *(str(item) for item in fairness_schema.get("resource_requirements") or ("n_params", "wall_seconds")),
        *_RESOURCE_MODE_FIELDS.get(match_mode, ()),
    ]))
    resource_values = {
        role: {field: _resource_value(side, field) for field in resource_fields}
        for role, side in (("candidate", candidate), ("baseline", baseline))
    }
    missing_resources = sorted(
        f"{role}.{field}"
        for role, values in resource_values.items()
        for field, value in values.items()
        if value is None
    )
    invalid_resources = sorted(
        f"{role}.{field}"
        for role, values in resource_values.items()
        for field, value in values.items()
        if value is not None and not _resource_value_valid(field, value)
    )
    resource_status = (
        "unknown"
        if missing_resources
        else "not_met"
        if invalid_resources
        else "met"
    )

    control_rows = list(controls)
    frozen = [
        row for row in control_rows
        if "frozen" in str(
            row.get("preset_id")
            or row.get("comparison_role")
            or row.get("study_role")
            or ""
        ).lower()
        or "random" in str(row.get("preset_id") or "").lower()
    ]
    strong = [
        row for row in control_rows
        if row not in frozen
        and str(row.get("study_role") or "") in {"control", "analogue"}
    ]
    frozen_met = any(
        bool(row.get("final_run"))
        and bool((row.get("control_match") or {}).get("valid"))
        for row in frozen
    )
    strong_met = any(
        bool(row.get("final_run"))
        and bool((row.get("control_match") or {}).get("valid"))
        for row in strong
    )

    base_rungs = [
        {
            "id": "linked_component_analogue",
            "label": "Linked component analogue",
            "status": (
                "met"
                if available and fair and comparator_identity_valid
                else "not_met"
                if available
                else "unknown"
            ),
            "detail": {
                "linked": available,
                "fair": fair,
                "candidate_uses_quantum": candidate_quantum,
                "baseline_uses_quantum": baseline_quantum,
                "classical_analogue_identity": comparator_identity_valid,
            },
        },
        {
            "id": "parameter_match",
            "label": "Parameter match",
            "status": parameter_status,
            "detail": {
                "candidate_n_params": cparams,
                "baseline_n_params": bparams,
                "relative_delta": parameter_ratio,
                "tolerance": tolerance,
            },
        },
        {
            "id": "resource_accounting",
            "label": "Resource accounting",
            "status": resource_status,
            "detail": {
                "required_fields": resource_fields,
                "values": resource_values,
                "missing_fields": missing_resources,
                "invalid_fields": invalid_resources,
                "match_mode": match_mode,
                "match_assessment": "reported_not_thresholded",
            },
        },
        {
            "id": "frozen_random_control",
            "label": "Frozen/random quantum control",
            "status": "met" if frozen_met else "unknown",
            "detail": f"{len(frozen)} matching control job(s); a completed result is required",
        },
        {
            "id": "strong_classical_challenger",
            "label": "Strong classical challenger",
            "status": "met" if strong_met else "unknown",
            "detail": f"{len(strong)} matching control job(s); a completed result is required",
        },
    ]
    rungs: list[dict[str, Any]] = []
    for rung in base_rungs:
        spec = requested.get(rung["id"], {})
        rungs.append({
            **rung,
            "required": bool(spec.get("required", False)),
            "comparator": spec.get("comparator"),
            "match_mode": spec.get("match_mode"),
            "limitation": spec.get("limitation"),
        })
    for rung_id, spec in requested.items():
        if any(row["id"] == rung_id for row in rungs):
            continue
        rungs.append({
            "id": rung_id,
            "label": str(spec.get("label") or rung_id.replace("_", " ").title()),
            "status": "unknown",
            "detail": "claim-specific rung has no completed matching evidence",
            "required": bool(spec.get("required", False)),
            "comparator": spec.get("comparator"),
            "match_mode": spec.get("match_mode"),
            "limitation": spec.get("limitation"),
        })
    missing = [
        row["id"] for row in rungs
        if row.get("required") and row.get("status") != "met"
    ]
    return {
        "rungs": rungs,
        "required_complete": not missing,
        "missing_required": missing,
        "parameter_tolerance": tolerance,
        "resource_requirements": resource_fields,
    }


def _coerce_stats(value: PairedStats | Mapping[str, Any]) -> PairedStats:
    if isinstance(value, PairedStats):
        return value
    fields = {field.name for field in dataclasses.fields(PairedStats)}
    return PairedStats(**{key: item for key, item in value.items() if key in fields})


def with_legacy_assessment_alias(
    assessment: Mapping[str, Any],
) -> dict[str, Any]:
    """Add the deprecated dashboard ``claim_level`` assessment alias.

    Canonical claim-ledger levels never pass through this helper. It exists
    only while older dashboard consumers migrate to ``assessment_level``.
    """
    payload = dict(assessment)
    if "assessment_level" in payload and "claim_level" not in payload:
        payload["claim_level"] = payload["assessment_level"]
    return payload


def classify_claim(
    *,
    fairness: dict,
    paired: PairedStats | dict | None = None,
    single_delta: float | None = None,
    min_pairs: int = 3,
    equivalence: Mapping[str, Any] | None = None,
    power: Mapping[str, Any] | None = None,
    analogue_ladder: Mapping[str, Any] | None = None,
    dequantization_gap: float | None = None,
    hardware_gap: float | None = None,
    metric_name: str = "validation metric",
) -> dict[str, Any]:
    """Classify the strength of a result without overstating it."""
    required = [
        "same_dataset",
        "same_steps",
        "same_eval_interval",
        "same_device_target",
    ]
    if "same_seed" in fairness:
        required.append("same_seed")
    if "role_validation" in fairness:
        required.append("role_validation")
    if not all(bool(fairness.get(k)) for k in required):
        return {
            "label": "insufficient fairness",
            "assessment_level": "invalid",
            "assessment_status": "invalid",
            "reason": "protocol fields do not match",
            "metric": metric_name,
        }
    if fairness.get("valid") is False:
        return {
            "label": "insufficient fairness",
            "assessment_level": "invalid",
            "assessment_status": "invalid",
            "reason": "one or more undisclosed protocol mismatches remain",
            "metric": metric_name,
        }

    if paired is None:
        if single_delta is None:
            return {
                "label": "incomplete",
                "assessment_level": "incomplete",
                "assessment_status": "incomplete",
                "reason": f"{metric_name} is unavailable",
                "metric": metric_name,
            }
        if single_delta > 0:
            return {
                "label": "single-run candidate better",
                "assessment_level": "anecdote",
                "assessment_status": "smoke",
                "reason": "one fair pair is useful smoke evidence, not an advantage claim",
                "metric": metric_name,
            }
        if single_delta < 0:
            return {
                "label": "single-run baseline better",
                "assessment_level": "no evidence",
                "assessment_status": "negative",
                "reason": "baseline is better on this fair pair",
                "metric": metric_name,
            }
        return {
            "label": "single-run tie",
            "assessment_level": "no evidence",
            "assessment_status": "equivalent_or_inconclusive",
            "reason": "candidate and baseline tie on this fair pair",
            "metric": metric_name,
        }

    stats = _coerce_stats(paired)
    edge_minimum = max(int(min_pairs), MIN_PAIRED_EDGE_PAIRS)
    if (
        stats.n_pairs >= edge_minimum
        and not stats.pilot_only
        and equivalence
        and equivalence.get("equivalent")
    ):
        return {
            "label": "practically equivalent",
            "assessment_level": "no evidence",
            "assessment_status": "equivalent",
            "reason": "the paired mean-effect interval is fully inside the predeclared negligible range",
            "n_pairs": stats.n_pairs,
            "metric": metric_name,
        }
    if stats.mean_improvement <= 0.0:
        return {
            "label": "no evidence",
            "assessment_level": "none",
            "assessment_status": "negative",
            "reason": "paired mean improvement is not positive",
            "n_pairs": stats.n_pairs,
            "metric": metric_name,
        }
    if stats.n_pairs < edge_minimum or stats.pilot_only:
        return {
            "label": "paired smoke only",
            "assessment_level": "smoke",
            "assessment_status": "pilot_only" if stats.n_pairs <= 3 else "underpowered",
            "reason": f"needs at least {edge_minimum} paired runs before paired empirical claims",
            "n_pairs": stats.n_pairs,
            "minimum_pairs": edge_minimum,
            "metric": metric_name,
        }
    if power and not bool(power.get("adequately_powered")):
        return {
            "label": "underpowered paired result",
            "assessment_level": "smoke",
            "assessment_status": "underpowered",
            "reason": "observed pairs do not meet the pilot-variance power plan",
            "n_pairs": stats.n_pairs,
            "recommended_pairs": power.get("recommended_pairs"),
            "metric": metric_name,
        }
    if not stats.significant:
        return {
            "label": "positive but not significant",
            "assessment_level": "weak",
            "assessment_status": "inconclusive",
            "reason": "paired improvement is positive but confidence interval or p-value is weak",
            "n_pairs": stats.n_pairs,
            "metric": metric_name,
        }
    if dequantization_gap is not None and dequantization_gap <= 0.0:
        return {
            "label": "matched by classical surrogate",
            "assessment_level": "quantum-inspired",
            "assessment_status": "classically_explained",
            "reason": "an architecture-aware classical surrogate matches or beats the candidate",
            "n_pairs": stats.n_pairs,
            "metric": metric_name,
        }
    if hardware_gap is not None and hardware_gap > stats.mean_improvement:
        return {
            "label": "hardware gap exceeds edge",
            "assessment_level": "fragile",
            "assessment_status": "fragile",
            "reason": "hardware/noise degradation is larger than the measured edge",
            "n_pairs": stats.n_pairs,
            "metric": metric_name,
        }
    if equivalence is None:
        return {
            "label": "missing practical threshold",
            "assessment_level": "weak",
            "assessment_status": "blocked",
            "reason": "a predeclared practical-equivalence margin is required before an empirical-edge assessment",
            "n_pairs": stats.n_pairs,
            "metric": metric_name,
        }
    if equivalence.get("status") != "candidate_meaningfully_better":
        return {
            "label": "positive but below practical threshold",
            "assessment_level": "weak",
            "assessment_status": "inconclusive",
            "reason": "statistical support does not clear the predeclared practical margin",
            "n_pairs": stats.n_pairs,
            "metric": metric_name,
        }
    if power is None:
        return {
            "label": "missing power plan",
            "assessment_level": "weak",
            "assessment_status": "blocked",
            "reason": "pilot-variance power planning is required before an empirical-edge assessment",
            "n_pairs": stats.n_pairs,
            "metric": metric_name,
        }
    if not bool(power.get("adequately_powered")):
        return {
            "label": "underpowered paired result",
            "assessment_level": "smoke",
            "assessment_status": "underpowered",
            "reason": "observed pairs do not meet the pilot-variance power plan",
            "n_pairs": stats.n_pairs,
            "recommended_pairs": power.get("recommended_pairs"),
            "metric": metric_name,
        }
    if analogue_ladder is None or not bool(analogue_ladder.get("required_complete")):
        return {
            "label": "incomplete analogue ladder",
            "assessment_level": "weak",
            "assessment_status": "blocked",
            "reason": "required classical/control analogue evidence is missing or unresolved",
            "missing_analogue_rungs": list(
                (analogue_ladder or {}).get("missing_required") or ["not_assessed"]
            ),
            "n_pairs": stats.n_pairs,
            "metric": metric_name,
        }
    return {
        "label": "paired empirical edge",
        "assessment_level": "empirical",
        "assessment_status": "paired_empirical_edge",
        "reason": "fair paired runs show a statistically supported candidate edge",
        "n_pairs": stats.n_pairs,
        "metric": metric_name,
    }


def resource_ledger_from_config(cfg: ExperimentConfig) -> dict[str, int | float | str | None]:
    """Minimal resource ledger required by the research protocol."""
    q = cfg.model.quantum
    flat = to_flat_dict(cfg)
    token_calls = int(cfg.train.batch_size * cfg.train.seq_len)
    state_dim = int(2 ** int(q.n_qubits)) if q is not None else 0
    quantum_components = [
        name
        for name, value in {
            "embedding": cfg.model.embed_type == "quantum",
            "attention": cfg.model.attn_type in QUANTUM_ATTN_TYPES,
            "ffn": cfg.model.ffn_type in QUANTUM_FFN_TYPES,
            "recurrent": cfg.model.arch in QUANTUM_ARCH_TYPES,
            "two_stream": (
                cfg.model.arch == "two_stream"
                and cfg.model.encoder_kind == "quantum"
            ),
        }.items()
        if value
    ]
    return {
        "n_qubits": int(q.n_qubits) if q is not None else 0,
        "n_circuit_layers": int(q.n_circuit_layers) if q is not None else 0,
        "ansatz": q.ansatz if q is not None else "none",
        "backend": q.backend if q is not None else "none",
        "device": q.device if q is not None else None,
        "shots": q.shots if q is not None else None,
        "state_dim": state_dim,
        "batch_size": int(cfg.train.batch_size),
        "seq_len": int(cfg.train.seq_len),
        "steps": int(cfg.train.steps),
        "eval_every": int(cfg.train.eval_every),
        "token_calls_per_step": token_calls,
        "quantum_components": ",".join(quantum_components) or "none",
        "config_key_count": len(flat),
    }


def resource_normalized_delta(
    *,
    candidate_metric: float,
    baseline_metric: float,
    candidate_wall_seconds: float | None,
    baseline_wall_seconds: float | None,
    lower_is_better: bool = True,
) -> dict[str, float | None]:
    """Score improvement and improvement per extra wall second."""
    improvement = (
        baseline_metric - candidate_metric
        if lower_is_better
        else candidate_metric - baseline_metric
    )
    if candidate_wall_seconds is None or baseline_wall_seconds is None:
        per_extra_second = None
    else:
        extra = candidate_wall_seconds - baseline_wall_seconds
        per_extra_second = improvement / extra if extra > 0 else None
    return {
        "improvement": float(improvement),
        "candidate_wall_seconds": (
            None if candidate_wall_seconds is None else float(candidate_wall_seconds)
        ),
        "baseline_wall_seconds": (
            None if baseline_wall_seconds is None else float(baseline_wall_seconds)
        ),
        "improvement_per_extra_second": (
            None if per_extra_second is None else float(per_extra_second)
        ),
    }
