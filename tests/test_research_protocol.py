from __future__ import annotations

import math

import pytest

from benchmarks.two_stream_probe import build_parser, validate_suite
from qllm.research_protocol import (
    TWO_STREAM_CAUSAL_PROTOCOL,
    TWO_STREAM_CAUSAL_SUITE,
    classify_claim,
    paired_stats,
    resource_ledger_from_config,
    resource_normalized_delta,
    two_stream_metric_contract,
)


FAIR = {
    "same_dataset": True,
    "same_seed": True,
    "same_steps": True,
    "same_eval_interval": True,
    "same_device_target": True,
    "role_validation": True,
}


def test_single_run_is_downgraded_to_anecdote():
    verdict = classify_claim(
        fairness=FAIR,
        single_delta=0.4,
        metric_name="validation perplexity",
    )
    assert verdict["label"] == "single-run candidate better"
    assert verdict["claim_level"] == "anecdote"


def test_unfair_protocol_rejects_claim():
    unfair = dict(FAIR, same_seed=False)
    verdict = classify_claim(fairness=unfair, single_delta=1.0)
    assert verdict["label"] == "insufficient fairness"
    assert verdict["claim_level"] == "invalid"


def test_paired_stats_can_support_empirical_edge():
    stats = paired_stats(
        candidate_scores=[0.8] * 6,
        baseline_scores=[1.0] * 6,
        lower_is_better=True,
    )
    assert stats.n_pairs == 6
    assert stats.mean_improvement > 0
    assert stats.significant is True

    verdict = classify_claim(fairness=FAIR, paired=stats, min_pairs=3)
    assert verdict["label"] == "paired empirical edge"
    assert verdict["claim_level"] == "empirical"


def test_dequantization_gap_downgrades_paired_edge():
    stats = paired_stats(
        candidate_scores=[0.8] * 6,
        baseline_scores=[1.0] * 6,
        lower_is_better=True,
    )
    verdict = classify_claim(
        fairness=FAIR,
        paired=stats,
        min_pairs=3,
        dequantization_gap=-0.01,
    )
    assert verdict["label"] == "matched by classical surrogate"
    assert verdict["claim_level"] == "quantum-inspired"


def test_resource_ledger_covers_required_quantum_fields(tiny_quantum_cfg):
    ledger = resource_ledger_from_config(tiny_quantum_cfg)
    assert ledger["n_qubits"] == tiny_quantum_cfg.model.quantum.n_qubits
    assert ledger["n_circuit_layers"] == tiny_quantum_cfg.model.quantum.n_circuit_layers
    assert ledger["shots"] is None
    assert ledger["token_calls_per_step"] == (
        tiny_quantum_cfg.train.batch_size * tiny_quantum_cfg.train.seq_len
    )
    assert ledger["state_dim"] == 2 ** tiny_quantum_cfg.model.quantum.n_qubits


def test_resource_normalized_delta_reports_extra_cost_efficiency():
    rep = resource_normalized_delta(
        candidate_metric=2.5,
        baseline_metric=3.0,
        candidate_wall_seconds=5.0,
        baseline_wall_seconds=3.0,
        lower_is_better=True,
    )
    assert rep["improvement"] == 0.5
    assert rep["improvement_per_extra_second"] == 0.25

    no_extra = resource_normalized_delta(
        candidate_metric=2.5,
        baseline_metric=3.0,
        candidate_wall_seconds=2.0,
        baseline_wall_seconds=3.0,
    )
    assert no_extra["improvement"] == 0.5
    assert no_extra["improvement_per_extra_second"] is None
    assert math.isfinite(no_extra["improvement"])


def test_historical_two_stream_metrics_require_a_causal_rerun():
    contract = two_stream_metric_contract(suite="two-stream-v1")
    assert contract == {
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


def test_two_stream_jobs_need_an_explicit_causal_protocol_marker():
    unmarked = two_stream_metric_contract(
        suite="lab",
        config={"model.arch": "two_stream"},
    )
    assert unmarked["rerun_required"] is True

    current = two_stream_metric_contract(
        suite="lab",
        config={
            "model": {"arch": "two_stream"},
            "lab": {"two_stream_protocol": TWO_STREAM_CAUSAL_PROTOCOL},
        },
    )
    assert current["metric_type"] == "strict_autoregressive_next_token"
    assert current["rerun_required"] is False

    causal_suite = two_stream_metric_contract(suite=TWO_STREAM_CAUSAL_SUITE)
    assert causal_suite["protocol"] == TWO_STREAM_CAUSAL_PROTOCOL
    assert causal_suite["strict_autoregressive"] is True

    assert two_stream_metric_contract(suite="qnlp-v1") is None


def test_two_stream_benchmark_defaults_to_causal_and_rejects_v1():
    assert build_parser().parse_args([]).suite == TWO_STREAM_CAUSAL_SUITE
    with pytest.raises(ValueError, match="immutable full-window"):
        validate_suite("two-stream-v1")
