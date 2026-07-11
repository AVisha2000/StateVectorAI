"""Research protocol guards for quantum-vs-classical claims.

These helpers keep the project honest about the difference between a useful
smoke result, a fair paired benchmark, and evidence that is strong enough to
talk about advantage. They are deliberately lightweight: no SciPy dependency,
deterministic small-sample statistics, and plain dictionaries for dashboard
payloads.
"""
from __future__ import annotations

import dataclasses
import itertools
import math
from dataclasses import dataclass
from typing import Iterable

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
    effect_size: float
    significant: bool

    def as_dict(self) -> dict[str, float | int | bool]:
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


def sign_flip_p_value(improvements: Iterable[float], *, max_exact: int = 16) -> float:
    """Two-sided paired randomisation p-value via sign flips.

    For small n this enumerates all sign assignments exactly. For larger n it
    uses a deterministic Monte Carlo approximation so dashboards and tests stay
    reproducible without pulling in SciPy.
    """
    diffs = _as_float_array(improvements)
    observed = abs(float(diffs.mean()))
    if observed == 0.0:
        return 1.0
    if len(diffs) <= max_exact:
        extreme = total = 0
        for signs in itertools.product((-1.0, 1.0), repeat=len(diffs)):
            total += 1
            if abs(float((diffs * np.asarray(signs)).mean())) >= observed - 1e-15:
                extreme += 1
        return extreme / total

    rng = np.random.default_rng(0)
    n_draws = 20_000
    signs = rng.choice((-1.0, 1.0), size=(n_draws, len(diffs)))
    means = np.abs((signs * diffs).mean(axis=1))
    return float((np.count_nonzero(means >= observed - 1e-15) + 1) / (n_draws + 1))


def paired_stats(
    candidate_scores: Iterable[float],
    baseline_scores: Iterable[float],
    *,
    lower_is_better: bool = True,
    alpha: float = 0.05,
) -> PairedStats:
    """Compute deterministic paired statistics for matched benchmark runs."""
    improvements = paired_improvements(
        candidate_scores, baseline_scores, lower_is_better=lower_is_better
    )
    if len(improvements) == 1:
        ci_low = ci_high = float(improvements[0])
    else:
        ci_low, ci_high = np.quantile(improvements, [alpha / 2, 1 - alpha / 2])
    std = float(improvements.std(ddof=1)) if len(improvements) > 1 else 0.0
    effect = (
        float(improvements.mean() / std)
        if std > 0
        else (math.inf if improvements.mean() > 0 else -math.inf if improvements.mean() < 0 else 0.0)
    )
    p = sign_flip_p_value(improvements)
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
    )


def protocol_fairness(
    candidate: dict,
    baseline: dict,
    *,
    require_same_seed: bool = True,
) -> dict[str, bool | float | None]:
    """Fairness checks shared by dashboard payloads and offline reports."""
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
        "same_eval_interval": int(cjob.get("eval_every", -1)) == int(
            bjob.get("eval_every", -2)
        ),
        "same_device_target": (cjob.get("device_target") or "auto")
        == (bjob.get("device_target") or "auto"),
        "parameter_delta_ratio": ratio,
    }


def classify_claim(
    *,
    fairness: dict,
    paired: PairedStats | dict | None = None,
    single_delta: float | None = None,
    min_pairs: int = 3,
    dequantization_gap: float | None = None,
    hardware_gap: float | None = None,
    metric_name: str = "validation metric",
) -> dict[str, str | bool | float | int | None]:
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
            "claim_level": "invalid",
            "reason": "protocol fields do not match",
            "metric": metric_name,
        }

    if paired is None:
        if single_delta is None:
            return {
                "label": "incomplete",
                "claim_level": "incomplete",
                "reason": f"{metric_name} is unavailable",
                "metric": metric_name,
            }
        if single_delta > 0:
            return {
                "label": "single-run candidate better",
                "claim_level": "anecdote",
                "reason": "one fair pair is useful smoke evidence, not an advantage claim",
                "metric": metric_name,
            }
        if single_delta < 0:
            return {
                "label": "single-run baseline better",
                "claim_level": "no evidence",
                "reason": "baseline is better on this fair pair",
                "metric": metric_name,
            }
        return {
            "label": "single-run tie",
            "claim_level": "no evidence",
            "reason": "candidate and baseline tie on this fair pair",
            "metric": metric_name,
        }

    stats = paired if isinstance(paired, PairedStats) else PairedStats(**paired)
    if stats.n_pairs < min_pairs:
        return {
            "label": "paired smoke only",
            "claim_level": "smoke",
            "reason": f"needs at least {min_pairs} paired runs before evidence claims",
            "n_pairs": stats.n_pairs,
            "metric": metric_name,
        }
    if stats.mean_improvement <= 0.0:
        return {
            "label": "no evidence",
            "claim_level": "none",
            "reason": "paired mean improvement is not positive",
            "n_pairs": stats.n_pairs,
            "metric": metric_name,
        }
    if not stats.significant:
        return {
            "label": "positive but not significant",
            "claim_level": "weak",
            "reason": "paired improvement is positive but confidence interval or p-value is weak",
            "n_pairs": stats.n_pairs,
            "metric": metric_name,
        }
    if dequantization_gap is not None and dequantization_gap <= 0.0:
        return {
            "label": "matched by classical surrogate",
            "claim_level": "quantum-inspired",
            "reason": "an architecture-aware classical surrogate matches or beats the candidate",
            "n_pairs": stats.n_pairs,
            "metric": metric_name,
        }
    if hardware_gap is not None and hardware_gap > stats.mean_improvement:
        return {
            "label": "hardware gap exceeds edge",
            "claim_level": "fragile",
            "reason": "hardware/noise degradation is larger than the measured edge",
            "n_pairs": stats.n_pairs,
            "metric": metric_name,
        }
    return {
        "label": "paired empirical edge",
        "claim_level": "empirical",
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
