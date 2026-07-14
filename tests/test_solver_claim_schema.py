"""Regression contract for the ground-state solver competition schema."""
from __future__ import annotations

from copy import deepcopy

import pytest
import yaml

import qllm.claims as claims_module
from qllm.claims import ClaimRegistryError, get_claim, load_claim_registry


def _write_registry(tmp_path, payload):
    path = tmp_path / "claims.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _solver_payload():
    return deepcopy(load_claim_registry().as_dict())


def _research_map_payload():
    return yaml.safe_load(
        claims_module.RESEARCH_MAP_PATH.read_text(encoding="utf-8")
    )


def _write_research_map(tmp_path, payload):
    path = tmp_path / "research-map.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_ground_state_claim_declares_strict_solver_competition_shape():
    claim = get_claim("vqe_ground_state_optimization")

    assert claim["level"] == "untested"
    assert claim["status"] == "untested"
    assert claim["replication_status"] == "none"
    assert claim["evidence"] == []
    assert claim["fairness_schema"] == {
        "schema_id": "solver_competition_v1",
        "comparator_kind": "best_in_class_solver",
        "pairing_axis": "problem_instance",
        "nested_seed_axis": "model_seed",
        "required_equal": [
            "problem.instance_id",
            "problem.instance_hash",
            "problem.energy_units",
            "optimum.reference_id",
            "optimum.role",
            "optimum.value",
            "optimum.units",
            "budget.contract_hash",
        ],
        "required_present": [
            "run_uuid",
            "manifest_hash",
            "solver.solver_id",
            "solver.solver_version",
            "solver.runner_id",
            "solver.runner_version",
            "solver.algorithm_family",
            "solver.hyperparameters",
            "solver.selected_trial_id",
            "solver.configuration_hash",
            "solver.code_hash",
            "solver.environment_hash",
            "solver.role",
            "solver.model_seed",
            "solver.computation_kind",
            "solver.registration_status",
        ],
        "equal_budget_fields": [
            "search.max_trials",
            "search.max_objective_evaluations",
            "search.max_wall_seconds",
            "evaluation.model_seeds",
            "evaluation.max_objective_evaluations_per_seed",
            "evaluation.max_wall_seconds_per_seed",
            "evaluation.shots_per_objective",
            "evaluation.target_metric",
            "evaluation.target_value",
            "evaluation.termination_rule",
        ],
        "intentional_differences": [
            {
                "path": "solver.solver_id",
                "reason": "Best-in-class competition intentionally compares distinct registered solvers.",
            },
            {
                "path": "solver.solver_version",
                "reason": "Each solver version is declared rather than forced to match.",
            },
            {
                "path": "solver.algorithm_family",
                "reason": "Solver families may differ under the same prespecified competition budget.",
            },
            {
                "path": "solver.hyperparameters",
                "reason": "Each arm may use its selected configuration within an equal search budget.",
            },
        ],
        "resource_requirements": [
            "objective_evaluations",
            "wall_seconds",
            "state_preparations",
            "total_shots",
            "peak_memory_bytes",
            "state_representation",
        ],
        "allowed_optimum_roles": ["best_known_optimum", "certified_optimum"],
    }
    assert "parameter_tolerance" not in claim["fairness_schema"]


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_comparator",
        "unknown_pairing_axis",
        "unknown_seed_axis",
        "list_comparator",
        "list_pairing_axis",
        "list_seed_axis",
        "missing_declaration",
        "missing_required_path",
        "budget_path_drift",
        "parameter_tolerance",
        "unknown_optimum_role",
    ],
)
def test_solver_schema_rejects_malformed_variants(tmp_path, mutation):
    payload = _solver_payload()
    schema = payload["claims"][-1]["fairness_schema"]
    if mutation == "unknown_comparator":
        schema["comparator_kind"] = "product_state_reference"
    elif mutation == "unknown_pairing_axis":
        schema["pairing_axis"] = "run"
    elif mutation == "unknown_seed_axis":
        schema["nested_seed_axis"] = "circuit_seed"
    elif mutation == "list_comparator":
        schema["comparator_kind"] = ["best_in_class_solver"]
    elif mutation == "list_pairing_axis":
        schema["pairing_axis"] = ["problem_instance"]
    elif mutation == "list_seed_axis":
        schema["nested_seed_axis"] = ["model_seed"]
    elif mutation == "missing_declaration":
        schema.pop("required_present")
    elif mutation == "missing_required_path":
        schema["required_present"].pop()
    elif mutation == "budget_path_drift":
        schema["equal_budget_fields"][-1] = "evaluation.unbounded_rule"
    elif mutation == "parameter_tolerance":
        schema["parameter_tolerance"] = 0.01
    else:
        schema["allowed_optimum_roles"] = ["oracle"]

    with pytest.raises(ClaimRegistryError):
        load_claim_registry(_write_registry(tmp_path, payload))


@pytest.mark.parametrize(
    "field",
    [
        "level",
        "status",
        "replication_status",
        "task_type",
        "metric_type",
        "analogue_match_mode",
        "analysis_phase",
    ],
)
def test_list_valued_claim_enums_raise_registry_error(tmp_path, field):
    payload = _solver_payload()
    claim = payload["claims"][-1]
    if field == "analogue_match_mode":
        claim["analogue_ladder"][0]["match_mode"] = ["logical_resources"]
    elif field == "analysis_phase":
        claim["analysis_settings"]["phase"] = ["exploratory"]
    else:
        claim[field] = [claim[field]]

    with pytest.raises(ClaimRegistryError):
        load_claim_registry(_write_registry(tmp_path, payload))


def test_mixed_mapping_key_types_raise_registry_error(tmp_path):
    payload = _solver_payload()
    payload["claims"][-1][1] = "unexpected"

    with pytest.raises(ClaimRegistryError, match="mapping keys must be strings"):
        load_claim_registry(_write_registry(tmp_path, payload))


@pytest.mark.parametrize(
    "mutation",
    ["status", "results_sections", "suites", "mixed_mapping_key"],
)
def test_malformed_research_map_shapes_raise_registry_error(
    tmp_path, monkeypatch, mutation
):
    claims_payload = _solver_payload()
    research_map = _research_map_payload()
    area = research_map["areas"][-1]
    if mutation == "status":
        area["status"] = [area["status"]]
    elif mutation == "results_sections":
        area["results_sections"] = [[1]]
    elif mutation == "suites":
        area["suites"] = [["solver-suite"]]
    else:
        area[1] = "unexpected"
    monkeypatch.setattr(
        claims_module,
        "RESEARCH_MAP_PATH",
        _write_research_map(tmp_path, research_map),
    )

    with pytest.raises(ClaimRegistryError):
        load_claim_registry(_write_registry(tmp_path, claims_payload))


def test_solver_schema_task_and_metric_are_coupled_exactly(tmp_path):
    payload = _solver_payload()
    qml_claim = next(
        claim
        for claim in payload["claims"]
        if claim["claim_id"] == "variational_component_swaps"
    )
    qml_schema = deepcopy(qml_claim["fairness_schema"])
    qml_claim["fairness_schema"] = deepcopy(payload["claims"][-1]["fairness_schema"])
    with pytest.raises(ClaimRegistryError, match="require task_type='ground_state'"):
        load_claim_registry(_write_registry(tmp_path, payload))

    payload = _solver_payload()
    payload["claims"][-1]["fairness_schema"] = qml_schema
    with pytest.raises(ClaimRegistryError, match="require solver_competition_v1"):
        load_claim_registry(_write_registry(tmp_path, payload))

    payload = _solver_payload()
    payload["claims"][-1]["metric_type"] = "validation_perplexity"
    with pytest.raises(
        ClaimRegistryError,
        match="require metric_type='ground_state_energy_error'",
    ):
        load_claim_registry(_write_registry(tmp_path, payload))

    payload = _solver_payload()
    qml_claim = next(
        claim
        for claim in payload["claims"]
        if claim["claim_id"] == "variational_component_swaps"
    )
    qml_claim["metric_type"] = "ground_state_energy_error"
    with pytest.raises(ClaimRegistryError, match="require task_type='ground_state'"):
        load_claim_registry(_write_registry(tmp_path, payload))


def test_qml_schemas_remain_legacy_and_solver_slice_does_not_promote_claims():
    qml_claim = get_claim("variational_component_swaps")
    solver_claim = get_claim("vqe_ground_state_optimization")

    assert qml_claim["fairness_schema"]["schema_id"] == "controlled_component_ablation_v1"
    assert "parameter_tolerance" in qml_claim["fairness_schema"]
    assert solver_claim["level"] == "untested"
    assert solver_claim["status"] == "untested"
    assert solver_claim["replication_status"] == "none"
    assert solver_claim["evidence"] == []
