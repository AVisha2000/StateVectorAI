from __future__ import annotations

from copy import deepcopy
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from benchmarks.two_stream_probe import build_parser, validate_suite
import qllm.claims as claims_module
from qllm.claims import (
    ClaimRegistryError,
    get_claim,
    infer_claim_id,
    load_claim_registry,
)
from qllm.research_protocol import (
    TWO_STREAM_CAUSAL_PROTOCOL,
    TWO_STREAM_CAUSAL_SUITE,
    evaluate_analogue_ladder,
    evaluate_fairness,
    classify_claim,
    normalize_seed_axes,
    paired_power_plan,
    paired_stats,
    practical_equivalence,
    resource_ledger_from_config,
    resource_normalized_delta,
    sign_flip_test,
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

ROOT = Path(__file__).resolve().parents[1]


def test_historical_two_stream_results_match_metric_contract():
    results = (ROOT / "RESULTS.md").read_text(encoding="utf-8")
    section = results.split("## 20.", maxsplit=1)[1]

    for required in (
        "`teacher_forced_side_information`",
        "`rerun_required`",
        "supports no strict autoregressive conclusion",
        "`two-stream-causal-v2`",
        "results/two_stream.png",
        "9.29 ± 0.37",
        "9.64 ± 0.44",
        "9.56 ± 0.22",
        "9.70 ± 0.19",
        "9.91 ± 0.13",
        "10.01 ± 0.11",
        "0.35 lower perplexity",
        "9.01 vs 9.35",
        "theta_x=0.75",
        "1.072",
        "1.093",
    ):
        assert required in section
    for superseded in (
        "the FIRST config",
        "SUGGESTIVE LEAN",
        "closest thing to an exception",
    ):
        assert superseded not in section


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
    improvements = [0.18, 0.19, 0.20, 0.21, 0.22, 0.20]
    stats = paired_stats(
        candidate_scores=[1.0 - value for value in improvements],
        baseline_scores=[1.0] * 6,
        lower_is_better=True,
    )
    assert stats.n_pairs == 6
    assert stats.mean_improvement > 0
    assert stats.significant is True

    equivalence = practical_equivalence(stats, margin=0.1)
    power = paired_power_plan(
        improvements, smallest_useful_effect=0.1
    )
    assert equivalence["status"] == "candidate_meaningfully_better"
    assert power["adequately_powered"] is True
    verdict = classify_claim(
        fairness=FAIR,
        paired=stats,
        min_pairs=3,
        equivalence=equivalence,
        power=power,
        analogue_ladder={"required_complete": True, "missing_required": []},
    )
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


def test_paired_bootstrap_and_sign_flip_are_deterministic():
    candidate = [0.79, 0.82, 0.76, 0.81, 0.75, 0.78]
    baseline = [1.01, 1.00, 1.04, 0.99, 1.02, 1.03]
    first = paired_stats(
        candidate,
        baseline,
        bootstrap_seed=17,
        bootstrap_resamples=2_000,
    )
    second = paired_stats(
        candidate,
        baseline,
        bootstrap_seed=17,
        bootstrap_resamples=2_000,
    )
    assert first.as_dict() == second.as_dict()
    assert first.ci_method == "paired_bootstrap_percentile_mean"
    assert first.ci_low > 0
    assert first.sign_flip_method == "exact"
    assert first.sign_flip_draws == 64

    diffs = [0.1 + index / 100 for index in range(17)]
    mc1 = sign_flip_test(diffs, max_exact=8, seed=41, draws=2_000)
    mc2 = sign_flip_test(diffs, max_exact=8, seed=41, draws=2_000)
    assert mc1 == mc2
    assert mc1["method"] == "monte_carlo"
    assert mc1["seed"] == 41
    assert mc1["draws"] == 2_000


def test_equivalence_power_and_three_pair_pilot_guard():
    equivalent_stats = paired_stats(
        [1.001, 0.999, 1.002, 0.998, 1.000, 1.001],
        [1.0] * 6,
        bootstrap_resamples=2_000,
    )
    equivalence = practical_equivalence(equivalent_stats, margin=0.01)
    assert equivalence["status"] == "equivalent"
    assert equivalence["equivalent"] is True

    power = paired_power_plan(
        [0.05, 0.15, 0.08, 0.12], smallest_useful_effect=0.1
    )
    assert power["recommended_pairs"] >= 6
    assert power["status"] in {"underpowered", "adequately_powered"}
    assert paired_power_plan(
        [0.2], smallest_useful_effect=0.1
    )["status"] == "insufficient_pilot"
    assert paired_power_plan(
        [0.2, 0.2, 0.2], smallest_useful_effect=0.1
    )["status"] == "zero_variance_pilot"

    forged = replace(
        paired_stats([0.5] * 6, [1.0] * 6),
        n_pairs=3,
        significant=True,
        pilot_only=False,
    )
    verdict = classify_claim(fairness=FAIR, paired=forged, min_pairs=1)
    assert verdict["assessment_status"] == "pilot_only"
    assert verdict["label"] != "paired empirical edge"

    negative = paired_stats([1.2, 1.1, 1.3], [1.0, 1.0, 1.0])
    negative_verdict = classify_claim(fairness=FAIR, paired=negative)
    assert negative_verdict["label"] == "no evidence"
    assert negative_verdict["assessment_status"] == "negative"

    slightly_worse = paired_stats([1.001] * 6, [1.0] * 6)
    equivalent = practical_equivalence(slightly_worse, margin=0.01)
    equivalent_verdict = classify_claim(
        fairness=FAIR,
        paired=slightly_worse,
        equivalence=equivalent,
    )
    assert equivalent_verdict["label"] == "practically equivalent"
    assert equivalent_verdict["assessment_status"] == "equivalent"


def test_empirical_edge_requires_all_predeclared_gates_and_json_safe_stats():
    stats = paired_stats([0.8] * 6, [1.0] * 6)
    assert stats.effect_size is None
    assert stats.effect_size_status == "undefined_zero_variance_nonzero_mean"
    json.dumps(stats.as_dict(), allow_nan=False)

    verdict = classify_claim(fairness=FAIR, paired=stats)
    assert verdict["label"] == "missing practical threshold"
    assert verdict["label"] != "paired empirical edge"


def test_legacy_seed_axes_disclose_coupling_and_reject_false_overrides():
    axes = normalize_seed_axes(
        7,
        generator_seed=3,
        data_kind="monitored_ising",
        circuit_applicable=True,
    )
    assert axes["generator"] == 3
    assert axes["split"] is None
    assert axes["initialization"] == axes["minibatch"] == axes["circuit"] == 7
    assert axes["coupled_axes"] == ["initialization", "minibatch", "circuit"]
    assert axes["source"] == "legacy_scalar"

    unsupported = normalize_seed_axes(7, explicit={"minibatch": 8})
    assert unsupported["supported"] is False
    assert unsupported["assessment_status"] == "unsupported_independent_override"
    with pytest.raises(ValueError, match="independent seed-axis overrides"):
        normalize_seed_axes(
            7, explicit={"minibatch": 8}, reject_unsupported=True
        )

    round_trip = normalize_seed_axes(7, explicit=axes)
    assert round_trip["supported"] is True
    assert round_trip["requested"] == {}


def _comparison_fixture(*, candidate_config, baseline_config, complete=True):
    def side(role, config, params):
        return {
            "job": {
                "id": 1 if role == "candidate" else 2,
                "run_name": role,
                "dataset_name": "synthetic",
                "seed": 5,
                "steps": 10,
                "eval_every": 2,
                "device_target": "cpu",
                "comparison_role": role,
                "config": config,
            },
            "final_run": (
                {"n_params": params, "wall_seconds": 2.0}
                if complete else None
            ),
        }
    return side("candidate", candidate_config, 101), side(
        "baseline", baseline_config, 100
    )


def test_submission_identity_metadata_is_an_allowed_operational_mismatch():
    candidate, baseline = _comparison_fixture(
        candidate_config={
            "train.lr": 0.1,
            "lab.submission.comparison_mode": "paired",
        },
        baseline_config={
            "train.lr": 0.1,
            "lab.submission.comparison_mode": "single",
        },
    )
    report = evaluate_fairness(
        candidate,
        baseline,
        schema={"required_equal": ["train.lr"]},
    )
    mismatch = next(
        row
        for row in report["mismatches"]
        if row["path"] == "lab.submission.comparison_mode"
    )
    assert mismatch["allowed"] is True
    assert mismatch["category"] == "operational"
    assert report["valid"] is True


def test_claim_specific_fairness_retains_allowed_and_disallowed_mismatches():
    candidate, baseline = _comparison_fixture(
        candidate_config={
            "model.ffn_type": "quantum",
            "train.lr": 0.01,
            "data.kind": "monitored_ising",
            "data.gen_seed": 1,
            "data.provenance.sha256": "candidate",
            "research.seed_axes": {"split": 9},
        },
        baseline_config={
            "model.ffn_type": "classical",
            "train.lr": 0.02,
            "data.kind": "monitored_ising",
            "data.gen_seed": 2,
            "data.provenance.sha256": "baseline",
        },
    )
    report = evaluate_fairness(
        candidate,
        baseline,
        schema={
            "schema_id": "ffn_test",
            "required_equal": [
                "job.dataset_name", "job.seed", "job.steps",
                "job.eval_every", "job.device_target", "train.lr", "data.*",
                "seed_axes.*",
            ],
            "intentional_differences": [{
                "path": "model.ffn_type",
                "reason": "Named quantum/classical component swap.",
            }],
        },
    )
    by_path = {item["path"]: item for item in report["mismatches"]}
    assert by_path["model.ffn_type"]["allowed"] is True
    assert by_path["model.ffn_type"]["allowlist_reason"]
    assert by_path["train.lr"]["allowed"] is False
    assert by_path["data.gen_seed"]["allowed"] is False
    assert by_path["data.provenance.sha256"]["allowed"] is False
    assert by_path["seed_axes.requested.split"]["allowed"] is False
    assert report["valid"] is False
    assert report["fairness_mismatches"] == report["mismatches"]

    incomplete_candidate, complete_baseline = _comparison_fixture(
        candidate_config={"train.lr": 0.1},
        baseline_config={"train.lr": 0.1},
        complete=False,
    )
    incomplete = evaluate_fairness(
        incomplete_candidate,
        complete_baseline,
        schema={"required_equal": ["train.lr"]},
    )
    assert incomplete["complete"] is False
    assert incomplete["valid"] is False

    reasonless = evaluate_fairness(
        candidate,
        baseline,
        schema={
            "required_equal": ["job.dataset_name"],
            "intentional_differences": [{"path": "train.lr", "reason": ""}],
        },
    )
    assert reasonless["valid"] is False
    assert reasonless["schema_errors"]


def test_per_block_quantum_detection_and_narrow_component_allowlist():
    candidate, baseline = _comparison_fixture(
        candidate_config={
            "model": {
                "d_model": 16,
                "blocks": [{"ffn_type": "quantum", "attn_type": "classical"}],
            },
            "data.kind": "text",
            "train.batch_size": 4,
            "train.eval_batches": 2,
            "train.grad_clip": 1.0,
            "train.lr": 0.1,
            "train.seq_len": 8,
            "train.weight_decay": 0.01,
        },
        baseline_config={
            "model": {
                "d_model": 16,
                "blocks": [{"ffn_type": "classical", "attn_type": "classical"}],
            },
            "data.kind": "text",
            "train.batch_size": 4,
            "train.eval_batches": 2,
            "train.grad_clip": 1.0,
            "train.lr": 0.1,
            "train.seq_len": 8,
            "train.weight_decay": 0.01,
        },
    )
    claim = get_claim("variational_component_swaps")
    report = evaluate_fairness(
        candidate, baseline, schema=claim["fairness_schema"]
    )
    assert report["seed_axes"]["candidate"]["applicability"]["circuit"] is True
    assert report["seed_axes"]["baseline"]["applicability"]["circuit"] is False
    assert report["valid"] is True

    baseline["job"]["config"]["model"]["d_model"] = 32
    capacity_drift = evaluate_fairness(
        candidate, baseline, schema=claim["fairness_schema"]
    )
    assert capacity_drift["valid"] is False
    assert any(
        row["path"] == "model.d_model" and not row["allowed"]
        for row in capacity_drift["mismatches"]
    )


def test_analogue_ladder_parameter_boundary_and_missing_resources_controls():
    candidate, baseline = _comparison_fixture(
        candidate_config={"model.ffn_type": "quantum", "train.lr": 0.1},
        baseline_config={"model.ffn_type": "classical", "train.lr": 0.1},
    )
    fairness = {"valid": True}
    claim = {
        "fairness_schema": {"parameter_tolerance": 0.01},
        "analogue_ladder": [
            {"id": "linked_component_analogue", "required": True},
            {"id": "parameter_match", "required": True},
            {"id": "resource_accounting", "required": True},
            {"id": "frozen_random_control", "required": True},
        ],
    }
    ladder = evaluate_analogue_ladder(
        candidate=candidate,
        baseline=baseline,
        fairness=fairness,
        claim=claim,
    )
    status = {row["id"]: row["status"] for row in ladder["rungs"]}
    assert status["parameter_match"] == "met"
    assert status["resource_accounting"] == "met"
    assert status["frozen_random_control"] == "unknown"
    assert ladder["required_complete"] is False

    with_control = evaluate_analogue_ladder(
        candidate=candidate,
        baseline=baseline,
        fairness=fairness,
        controls=[{
            "preset_id": "quantum-frozen",
            "status": "done",
            "study_role": "control",
            "final_run": {"n_params": 101, "wall_seconds": 2.0},
            "control_match": {"valid": True},
        }],
        claim=claim,
    )
    assert with_control["required_complete"] is True

    unmatched = evaluate_analogue_ladder(
        candidate=candidate,
        baseline=baseline,
        fairness=fairness,
        controls=[{
            "preset_id": "quantum-frozen",
            "final_run": {"n_params": 101, "wall_seconds": 2.0},
        }],
        claim=claim,
    )
    assert unmatched["required_complete"] is False

    quantum_copy = deepcopy(candidate)
    invalid_identity = evaluate_analogue_ladder(
        candidate=candidate,
        baseline=quantum_copy,
        fairness={"valid": True},
        claim=claim,
    )
    linked = next(
        row for row in invalid_identity["rungs"]
        if row["id"] == "linked_component_analogue"
    )
    assert linked["status"] == "not_met"
    assert linked["detail"]["baseline_uses_quantum"] is True


def test_analogue_ladder_preserves_contract_and_requires_mode_resources():
    candidate, baseline = _comparison_fixture(
        candidate_config={"train.lr": 0.1},
        baseline_config={"train.lr": 0.1},
    )
    claim = {
        "fairness_schema": {
            "parameter_tolerance": 0.1,
            "resource_requirements": ["n_params", "wall_seconds"],
        },
        "analogue_ladder": [{
            "rung_id": "resource_accounting",
            "required": True,
            "comparator": "paired resource ledgers",
            "match_mode": "circuit_evaluations",
            "limitation": "Circuit work must be recorded.",
        }],
    }
    missing = evaluate_analogue_ladder(
        candidate=candidate, baseline=baseline, fairness={"valid": True}, claim=claim
    )
    rung = next(row for row in missing["rungs"] if row["id"] == "resource_accounting")
    assert rung["status"] == "unknown"
    assert rung["comparator"] == "paired resource ledgers"
    assert "candidate.circuit_calls" in rung["detail"]["missing_fields"]

    candidate["final_run"]["circuit_calls"] = 100
    baseline["final_run"]["circuit_calls"] = 0
    complete = evaluate_analogue_ladder(
        candidate=candidate, baseline=baseline, fairness={"valid": True}, claim=claim
    )
    rung = next(row for row in complete["rungs"] if row["id"] == "resource_accounting")
    assert rung["status"] == "met"
    candidate["final_run"]["wall_seconds"] = 0.0
    invalid = evaluate_analogue_ladder(
        candidate=candidate, baseline=baseline, fairness={"valid": True}, claim=claim
    )
    rung = next(row for row in invalid["rungs"] if row["id"] == "resource_accounting")
    assert rung["status"] == "not_met"
    assert "candidate.wall_seconds" in rung["detail"]["invalid_fields"]


def test_claim_registry_is_complete_defensive_cached_and_fail_closed(tmp_path, monkeypatch):
    claims_module._cached_default_registry.cache_clear()
    calls = []
    original = claims_module._load_yaml_mapping

    def counted(path, label):
        calls.append((path, label))
        return original(path, label)

    monkeypatch.setattr(claims_module, "_load_yaml_mapping", counted)
    try:
        registry = load_claim_registry()
        assert len(registry) == 19
        assert load_claim_registry() is registry
        assert len(calls) == 2
    finally:
        claims_module._cached_default_registry.cache_clear()

    registry = load_claim_registry()
    first = registry[0]
    first["statement"] = "mutated"
    assert get_claim(first["claim_id"])["statement"] != "mutated"
    assert infer_claim_id(explicit="variational_component_swaps") == "variational_component_swaps"
    assert infer_claim_id(suite="qrnn-landscape-v1") is None

    payload = registry.as_dict()
    payload["claims"][0]["evidence"][0]["suites"] = ["fabricated-suite"]
    invalid_path = tmp_path / "claims.yaml"
    invalid_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ClaimRegistryError, match="not listed for research area"):
        load_claim_registry(invalid_path)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "enum", "reason", "overlap"])
def test_claim_registry_rejects_invalid_schema_mutations(tmp_path, mutation):
    payload = deepcopy(load_claim_registry().as_dict())
    if mutation == "duplicate":
        payload["claims"][1]["claim_id"] = payload["claims"][0]["claim_id"]
    elif mutation == "missing":
        payload["claims"][0].pop("next_decisive_test")
    elif mutation == "enum":
        payload["claims"][0]["status"] = "promoted"
    elif mutation == "reason":
        payload["claims"][0]["fairness_schema"]["intentional_differences"][0]["reason"] = ""
    else:
        payload["claims"][0]["fairness_schema"]["intentional_differences"].append({
            "path": "data.gen_seed",
            "reason": "must not override required data equality",
        })
    path = tmp_path / f"{mutation}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ClaimRegistryError):
        load_claim_registry(path)
