from __future__ import annotations

import pytest

from qllm import registry
from qllm.dashboard import evidence, lab, runner, studies
from qllm.dashboard.workspace import comparison_payload
from qllm.resultsdb import ResultsDB


CUSTOM_METRIC = "ground_state_energy_error"
CUSTOM_SPEC = {
    "lower_is_better": True,
    "units": "hartree",
    "pairable": True,
    "extraction_key": "energy_error",
    "comparator_class": "classical_solver",
}
HIGHER_METRIC = "success_probability"
HIGHER_SPEC = {
    "lower_is_better": False,
    "units": "probability",
    "pairable": True,
    "extraction_key": "success_probability",
    "comparator_class": "classical_solver",
}
UNPAIRABLE_METRIC = "descriptive_runtime"
UNPAIRABLE_SPEC = {
    "lower_is_better": True,
    "units": "seconds",
    "pairable": False,
    "extraction_key": "wall_seconds",
    "comparator_class": "none",
}
FAIRNESS = {
    "complete": True,
    "valid": True,
    "same_dataset": True,
    "same_seed": True,
    "same_steps": True,
    "same_eval_interval": True,
    "same_device_target": True,
    "role_validation": True,
    "mismatches": [],
    "disallowed_mismatches": [],
}


def _install_metric(
    monkeypatch: pytest.MonkeyPatch, metric_type: str, spec: dict
) -> None:
    monkeypatch.setattr(
        registry,
        "METRIC_TYPES",
        {**registry.METRIC_TYPES, metric_type: spec},
    )


def _paired_jobs(db: ResultsDB, metric_type: str) -> tuple[dict, dict]:
    common_config = {
        "model.arch": "transformer",
        "model.attn_type": "classical",
        "model.embed_type": "classical",
        "model.encoder_kind": "none",
        "model.head_type": "linear",
        "data.kind": "text",
        "train.seed": 0,
        "train.steps": 1,
        "train.eval_every": 1,
        "train.batch_size": 1,
        "train.seq_len": 4,
        "research.metric_type": metric_type,
    }
    candidate_id, baseline_id = db.create_lab_job_pair(
        {
            "status": "done",
            "preset_id": "quantum-ffn-4q",
            "dataset_name": "default-text",
            "run_name": "metric-candidate",
            "seed": 0,
            "steps": 1,
            "eval_every": 1,
            "device_target": "cpu",
            "comparison_role": "primary",
            "config": {**common_config, "model.ffn_type": "quantum"},
        },
        {
            "status": "done",
            "preset_id": "classical-small",
            "dataset_name": "default-text",
            "run_name": "metric-baseline",
            "seed": 0,
            "steps": 1,
            "eval_every": 1,
            "device_target": "cpu",
            "comparison_role": "baseline",
            "config": {**common_config, "model.ffn_type": "classical"},
        },
    )
    candidate = db.get_lab_job(candidate_id)
    baseline = db.get_lab_job(baseline_id)
    candidate["config"] = ResultsDB._lab_job_config(candidate)
    baseline["config"] = ResultsDB._lab_job_config(baseline)
    return candidate, baseline


def test_comparison_admission_and_extraction_share_the_metric_contract(monkeypatch):
    _install_metric(monkeypatch, CUSTOM_METRIC, CUSTOM_SPEC)
    monkeypatch.setattr(lab, "fairness_flags", lambda *_args, **_kwargs: FAIRNESS)
    payload = {
        "available": True,
        "deltas": {"val_ppl": 99.0, "energy_error": -0.2},
        "candidate": {
            "final_run": {
                "val_ppl": 100.0,
                "energy_error": 0.8,
                "primary_metric_name": "energy_error",
                "primary_metric_value": 0.8,
                "wall_seconds": 3.0,
            }
        },
        "baseline": {
            "final_run": {
                "val_ppl": 1.0,
                "energy_error": 1.0,
                "primary_metric_name": "energy_error",
                "primary_metric_value": 1.0,
                "wall_seconds": 2.0,
            }
        },
    }

    verdict = lab.verdict_for_comparison(payload, CUSTOM_METRIC)
    normalized = lab._resource_normalized_for_payload(payload, CUSTOM_METRIC)
    payload.update(
        {
            "verdict": verdict,
            "fairness": FAIRNESS,
            "resource_normalized": normalized,
            "metric_type": CUSTOM_METRIC,
            "metric_contract": {},
            "claim_id": "synthetic-energy",
        }
    )
    ladder = evidence.comparison_evidence_ladder(payload)

    assert verdict["label"] == "single-run candidate better"
    assert verdict["metric"] == "ground state energy error"
    assert normalized["improvement"] == pytest.approx(0.2)
    assert ladder["label"] == "promising run"
    steps = {row["key"]: row for row in ladder["steps"]}
    assert steps["metric_supported"]["ok"] is True
    assert steps["run_level_improvement"]["ok"] is True


def test_unpairable_metric_fails_closed_even_when_val_ppl_improves(monkeypatch):
    _install_metric(monkeypatch, UNPAIRABLE_METRIC, UNPAIRABLE_SPEC)
    monkeypatch.setattr(lab, "fairness_flags", lambda *_args, **_kwargs: FAIRNESS)
    payload = {
        "available": True,
        "deltas": {"val_ppl": -99.0, "wall_seconds": -1.0},
        "candidate": {"final_run": {"val_ppl": 1.0, "wall_seconds": 1.0}},
        "baseline": {"final_run": {"val_ppl": 100.0, "wall_seconds": 2.0}},
        "fairness": FAIRNESS,
        "metric_type": UNPAIRABLE_METRIC,
        "metric_contract": {},
        "claim_id": "synthetic-runtime",
    }
    verdict = lab.verdict_for_comparison(payload, UNPAIRABLE_METRIC)
    payload["verdict"] = verdict
    ladder = evidence.comparison_evidence_ladder(payload)

    assert verdict["label"] == "unsupported metric"
    assert ladder["label"] == "unsupported metric"
    steps = {row["key"]: row for row in ladder["steps"]}
    assert steps["metric_supported"]["ok"] is False
    assert steps["run_level_improvement"]["ok"] is False


def test_higher_is_better_direction_reaches_both_evidence_ladders(monkeypatch):
    _install_metric(monkeypatch, HIGHER_METRIC, HIGHER_SPEC)
    monkeypatch.setattr(lab, "fairness_flags", lambda *_args, **_kwargs: FAIRNESS)
    comparison = {
        "available": True,
        "deltas": {"val_ppl": 99.0, "success_probability": 0.1},
        "candidate": {
            "final_run": {
                "val_ppl": 100.0,
                "success_probability": 0.8,
                "primary_metric_name": "success_probability",
                "primary_metric_value": 0.8,
            }
        },
        "baseline": {
            "final_run": {
                "val_ppl": 1.0,
                "success_probability": 0.7,
                "primary_metric_name": "success_probability",
                "primary_metric_value": 0.7,
            }
        },
        "fairness": FAIRNESS,
        "metric_type": HIGHER_METRIC,
        "metric_contract": {},
        "claim_id": "synthetic-success",
    }
    verdict = lab.verdict_for_comparison(comparison, HIGHER_METRIC)
    comparison["verdict"] = verdict
    comparison_ladder = evidence.comparison_evidence_ladder(comparison)
    monkeypatch.setattr(
        studies,
        "comparison_research_payload",
        lambda _db, job_id: {
            **comparison,
            "claim_id": None,
            "fairness_mismatches": [],
            "candidate": {
                "job": {"dataset_name": "toy", "seed": job_id - 1},
                "final_run": {
                    "val_ppl": 100.0,
                    "success_probability": 0.8,
                    "primary_metric_name": "success_probability",
                    "primary_metric_value": 0.8,
                },
            },
            "baseline": {
                "job": {"dataset_name": "toy", "seed": job_id - 1},
                "final_run": {
                    "val_ppl": 1.0,
                    "success_probability": 0.7,
                    "primary_metric_name": "success_probability",
                    "primary_metric_value": 0.7,
                },
            },
        },
    )
    monkeypatch.setattr(
        studies,
        "evaluate_analogue_ladder",
        lambda **_kwargs: {
            "required_complete": True,
            "missing_required": [],
            "rungs": [],
        },
    )
    jobs = [
        {
            "id": seed + 1,
            "study_role": "candidate",
            "seed": seed,
            "dataset_name": "toy",
            "study_sweep": {},
            "uses_quantum": True,
            "seed_axes": {},
        }
        for seed in range(6)
    ]
    study = studies._evidence_for_jobs(
        object(), jobs, {"metric_type": HIGHER_METRIC}
    )

    assert verdict["label"] == "single-run candidate better"
    comparison_steps = {row["key"]: row for row in comparison_ladder["steps"]}
    study_steps = {row["key"]: row for row in study["ladder"]}
    assert comparison_steps["run_level_improvement"]["ok"] is True
    assert study["mean_delta"] == pytest.approx(0.1)
    assert study["mean_delta_val_ppl"] is None
    assert study_steps["metric_supported"]["ok"] is True
    assert study_steps["candidate_better"]["ok"] is True


def test_study_inference_never_writes_non_ppl_values_to_ppl_fields(monkeypatch):
    _install_metric(monkeypatch, CUSTOM_METRIC, CUSTOM_SPEC)
    payload = {
        "available": True,
        "fairness": FAIRNESS,
        "fairness_mismatches": [],
        "metric_contract": {},
        "metric_type": CUSTOM_METRIC,
        "claim_id": None,
        "deltas": {"val_ppl": 99.0, "energy_error": -0.2},
        "candidate": {
            "job": {"dataset_name": "toy", "seed": 0},
            "final_run": {
                "val_ppl": 100.0,
                "energy_error": 0.8,
                "primary_metric_name": "energy_error",
                "primary_metric_value": 0.8,
            },
        },
        "baseline": {
            "job": {"dataset_name": "toy", "seed": 0},
            "final_run": {
                "val_ppl": 1.0,
                "energy_error": 1.0,
                "primary_metric_name": "energy_error",
                "primary_metric_value": 1.0,
            },
        },
    }
    monkeypatch.setattr(
        studies, "comparison_research_payload", lambda *_args, **_kwargs: payload
    )
    monkeypatch.setattr(
        studies,
        "evaluate_analogue_ladder",
        lambda **_kwargs: {
            "required_complete": True,
            "missing_required": [],
            "rungs": [],
        },
    )
    jobs = [
        {
            "id": 1,
            "study_role": "candidate",
            "seed": 0,
            "dataset_name": "toy",
            "study_sweep": {},
            "uses_quantum": True,
            "seed_axes": {},
        }
    ]

    evidence = studies._evidence_for_jobs(
        object(), jobs, {"metric_type": CUSTOM_METRIC}
    )

    comparison = evidence["comparisons"][0]
    analysis = evidence["analyses"][0]
    assert comparison["metric_key"] == "energy_error"
    assert comparison["metric_delta"] == pytest.approx(-0.2)
    assert comparison["delta_val_ppl"] is None
    assert analysis["wins"] == 1
    assert analysis["paired_stats"]["mean_improvement"] == pytest.approx(0.2)
    assert analysis["mean_delta"] == pytest.approx(-0.2)
    assert analysis["mean_delta_val_ppl"] is None
    assert evidence["mean_delta"] == pytest.approx(-0.2)
    assert evidence["mean_delta_val_ppl"] is None


def test_sequence_queue_rejects_registered_non_sequence_extraction(monkeypatch):
    _install_metric(monkeypatch, CUSTOM_METRIC, CUSTOM_SPEC)

    with pytest.raises(ValueError, match="metric-specific sibling runner"):
        runner._sequence_primary_metric_type(CUSTOM_METRIC)

    assert (
        runner._sequence_primary_metric_type("time_to_target")
        == "strict_autoregressive_next_token"
    )


def test_persisted_primary_metric_flows_into_registry_comparison(
    monkeypatch, tmp_path
):
    _install_metric(monkeypatch, CUSTOM_METRIC, CUSTOM_SPEC)
    monkeypatch.setattr(lab, "fairness_flags", lambda *_args, **_kwargs: FAIRNESS)
    db = ResultsDB(tmp_path / "primary-comparison.db")
    candidate, baseline = _paired_jobs(db, CUSTOM_METRIC)

    for job, energy_error in ((candidate, 0.8), (baseline, 1.0)):
        db.record(
            "lab",
            job["preset_id"],
            job["dataset_name"],
            int(job["seed"]),
            int(job["steps"]),
            100,
            None,
            None,
            None,
            1.0,
            config=job["config"],
            run_uuid=job["run_uuid"],
            experiment_uuid=job["experiment_uuid"],
            primary_metric_type=CUSTOM_METRIC,
            metric_values={"energy_error": energy_error},
        )

    payload = comparison_payload(db, int(candidate["id"]))
    verdict = lab.verdict_for_comparison(payload, CUSTOM_METRIC)

    assert payload["candidate"]["final_run"]["primary_metric_name"] == (
        "energy_error"
    )
    assert payload["candidate"]["final_run"]["energy_error"] == pytest.approx(
        0.8
    )
    assert payload["deltas"]["energy_error"] == pytest.approx(-0.2)
    assert payload["deltas"]["val_ppl"] is None
    assert verdict["label"] == "single-run candidate better"


def test_comparison_refuses_mismatched_persisted_primary_names(
    monkeypatch, tmp_path
):
    _install_metric(monkeypatch, CUSTOM_METRIC, CUSTOM_SPEC)
    monkeypatch.setattr(lab, "fairness_flags", lambda *_args, **_kwargs: FAIRNESS)
    db = ResultsDB(tmp_path / "mismatched-primary.db")
    candidate, baseline = _paired_jobs(db, CUSTOM_METRIC)
    db.record(
        "lab",
        candidate["preset_id"],
        candidate["dataset_name"],
        0,
        1,
        100,
        None,
        None,
        None,
        1.0,
        config=candidate["config"],
        run_uuid=candidate["run_uuid"],
        experiment_uuid=candidate["experiment_uuid"],
        primary_metric_type=CUSTOM_METRIC,
        metric_values={"energy_error": 0.8},
    )
    db.record(
        "lab",
        baseline["preset_id"],
        baseline["dataset_name"],
        0,
        1,
        100,
        1.0,
        2.0,
        1.2,
        1.0,
        config=baseline["config"],
        run_uuid=baseline["run_uuid"],
        experiment_uuid=baseline["experiment_uuid"],
    )

    payload = comparison_payload(db, int(candidate["id"]))
    verdict = lab.verdict_for_comparison(payload, CUSTOM_METRIC)

    assert payload["deltas"]["energy_error"] is None
    assert verdict["label"] == "needs review"


def test_val_ppl_key_collision_cannot_bypass_persisted_name_gate(
    monkeypatch, tmp_path
):
    _install_metric(monkeypatch, CUSTOM_METRIC, CUSTOM_SPEC)
    monkeypatch.setattr(lab, "fairness_flags", lambda *_args, **_kwargs: FAIRNESS)
    db = ResultsDB(tmp_path / "val-ppl-name-mismatch.db")
    candidate, baseline = _paired_jobs(db, "validation_perplexity")
    db.record(
        "lab",
        candidate["preset_id"],
        candidate["dataset_name"],
        0,
        1,
        100,
        1.0,
        1.0,
        1.0,
        2.0,
        config=candidate["config"],
        run_uuid=candidate["run_uuid"],
        experiment_uuid=candidate["experiment_uuid"],
        primary_metric_type=CUSTOM_METRIC,
        metric_values={"energy_error": 0.8},
    )
    db.record(
        "lab",
        baseline["preset_id"],
        baseline["dataset_name"],
        0,
        1,
        100,
        1.0,
        2.0,
        1.2,
        1.0,
        config=baseline["config"],
        run_uuid=baseline["run_uuid"],
        experiment_uuid=baseline["experiment_uuid"],
    )

    payload = comparison_payload(db, int(candidate["id"]))
    verdict = lab.verdict_for_comparison(payload, "validation_perplexity")
    normalized = lab._resource_normalized_for_payload(
        payload, "validation_perplexity"
    )
    payload.update(
        {
            "verdict": verdict,
            "fairness": FAIRNESS,
            "resource_normalized": normalized,
            "metric_type": "validation_perplexity",
            "metric_contract": {},
            "claim_id": "synthetic-ppl-mismatch",
        }
    )
    ladder = evidence.comparison_evidence_ladder(payload)
    monkeypatch.setattr(
        studies,
        "comparison_research_payload",
        lambda *_args, **_kwargs: payload,
    )
    study = studies._evidence_for_jobs(
        object(),
        [
            {
                **candidate,
                "study_role": "candidate",
                "study_sweep": {},
                "uses_quantum": True,
                "seed_axes": {},
            }
        ],
        {"metric_type": "validation_perplexity"},
    )

    assert payload["deltas"]["val_ppl"] is None
    assert verdict["label"] == "needs review"
    assert normalized is None
    assert next(
        row for row in ladder["steps"] if row["key"] == "run_level_improvement"
    )["ok"] is False
    assert study["analyses"] == []
    assert study["comparisons"][0]["metric_delta"] is None
