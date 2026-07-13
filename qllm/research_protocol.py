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
import itertools
import json
import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Iterable, Mapping

import numpy as np

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
        "minibatch": legacy,
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
        "minibatch": True,
        "circuit": bool(circuit_applicable),
        "hardware_calibration": False,
    }
    sources = {
        "generator": "data.gen_seed" if generated else "not_applicable",
        "split": "deterministic_split",
        "initialization": "train.seed",
        "minibatch": "train.seed",
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
