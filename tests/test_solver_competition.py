"""Fail-closed admission tests for solver_competition_v1."""
from __future__ import annotations

import copy
import hashlib
import json
import uuid
from types import MappingProxyType

import pytest

import qllm.registry as registry
from qllm.claims import get_claim
from qllm.research_protocol import (
    evaluate_analogue_ladder,
    evaluate_fairness,
    evaluate_solver_competition,
    solver_budget_contract_hash,
    solver_configuration_hash,
)
from qllm.registry import metric_type_spec


_INSTANCES = [
    {
        "instance_id": "tfim-a",
        "instance_hash": "a" * 64,
        "energy_units": "dimensionless",
        "optimum_reference": {
            "reference_id": "exact_diagonalization:tfim-a",
            "role": "certified_optimum",
            "value": -2.23606797749979,
            "units": "dimensionless",
        },
    },
    {
        "instance_id": "tfim-b",
        "instance_hash": "b" * 64,
        "energy_units": "dimensionless",
        "optimum_reference": {
            "reference_id": "exact_diagonalization:tfim-b",
            "role": "certified_optimum",
            "value": -2.5,
            "units": "dimensionless",
        },
    },
]
_SEEDS = [0, 1, 2]


@pytest.fixture
def comparison_registry(monkeypatch):
    registrations = dict(registry.SOLVER_RUNNERS)
    registrations.update(
        {
            (
                "ground-state-vqe",
                "1",
                "qllm-finite-shot-vqe",
                "1",
            ): MappingProxyType(
                {
                    "task_type": "ground_state",
                    "computation_kind": "quantum",
                    "registration_status": "comparison_eligible",
                    "comparison_eligible": True,
                }
            ),
            (
                "ground-state-classical",
                "1",
                "registered-classical-variational",
                "1",
            ): MappingProxyType(
                {
                    "task_type": "ground_state",
                    "computation_kind": "classical",
                    "registration_status": "comparison_eligible",
                    "comparison_eligible": True,
                }
            ),
        }
    )
    monkeypatch.setattr(
        registry,
        "SOLVER_RUNNERS",
        MappingProxyType(registrations),
    )


def _schema():
    return get_claim("vqe_ground_state_optimization")["fairness_schema"]


def _protocol():
    return {
        "analysis_mode": "solver_competition",
        "schema_id": "solver_competition_v1",
        "comparator_kind": "best_in_class_solver",
        "candidate_solver": {
            "solver_id": "qllm-finite-shot-vqe",
            "solver_version": "1",
            "runner_id": "ground-state-vqe",
            "runner_version": "1",
            "computation_kind": "quantum",
        },
        "comparator_solver": {
            "solver_id": "registered-classical-variational",
            "solver_version": "1",
            "runner_id": "ground-state-classical",
            "runner_version": "1",
            "computation_kind": "classical",
        },
        "problem_instances": copy.deepcopy(_INSTANCES),
        "model_seeds": list(_SEEDS),
        "budget": {
            "prespecified": True,
            "search": {
                "max_trials": 1,
                "max_objective_evaluations": 1000,
                "max_wall_seconds": 60.0,
            },
            "evaluation": {
                "model_seeds": list(_SEEDS),
                "max_objective_evaluations_per_seed": 100,
                "max_wall_seconds_per_seed": 10.0,
                "shots_per_objective": 1000,
                "target_metric": "ground_state_energy_error",
                "target_value": 0.001,
                "termination_rule": "first_target_or_any_limit",
            },
        },
        "within_instance_aggregation": "best_within_total_budget",
    }


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _configuration(role: str):
    return {
        "algorithm_family": (
            "variational_quantum"
            if role == "candidate"
            else "classical_variational"
        ),
        "hyperparameters": {"declared": True},
    }


def _run(protocol, instance, role: str, seed: int):
    declaration = protocol[
        "candidate_solver" if role == "candidate" else "comparator_solver"
    ]
    computation_kind = declaration["computation_kind"]
    energy_error = 0.0008 if role == "candidate" else 0.0009
    configuration = _configuration(role)
    return {
        "status": "done",
        "run_uuid": str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{instance['instance_id']}:{role}:{seed}",
            )
        ),
        "manifest_hash": _digest(
            f"manifest:{instance['instance_id']}:{role}:{seed}"
        ),
        "problem": {
            "instance_id": instance["instance_id"],
            "instance_hash": instance["instance_hash"],
            "energy_units": instance["energy_units"],
        },
        "solver": {
            **declaration,
            "code_hash": _digest(f"code:{role}"),
            "environment_hash": _digest(f"environment:{role}"),
            "role": role,
            "model_seed": seed,
            "registration_status": "comparison_eligible",
            **configuration,
            "selected_trial_id": f"{role}-trial-0",
            "configuration_hash": solver_configuration_hash(configuration),
        },
        "optimum": copy.deepcopy(instance["optimum_reference"]),
        "budget_contract_hash": solver_budget_contract_hash(protocol["budget"]),
        "outcome": {
            "energy_error": energy_error,
            "reached_target": True,
            "time_to_target_seconds": 4.0,
            "censored": False,
        },
        "resources": {
            "objective_evaluations": {"value": 8, "status": "measured"},
            "wall_seconds": {"value": 4.0, "status": "measured"},
            "state_preparations": {
                "value": 8000 if computation_kind == "quantum" else 0,
                "status": "derived" if computation_kind == "quantum" else "measured",
            },
            "total_shots": (
                {"value": 8000, "status": "derived"}
                if computation_kind == "quantum"
                else {"value": None, "status": "not_applicable"}
            ),
            "peak_memory_bytes": {"value": None, "status": "unmeasured"},
            "state_representation": {
                "value": (
                    "finite_shot_circuit"
                    if computation_kind == "quantum"
                    else "classical_state_vector"
                ),
                "status": "configured",
            },
        },
    }


def _grid(protocol):
    return [
        _run(protocol, instance, role, seed)
        for instance in protocol["problem_instances"]
        for role in ("candidate", "comparator")
        for seed in protocol["model_seeds"]
    ]


def _search_ledgers(protocol):
    ledgers = []
    for role, declaration_key, metric_value in (
        ("candidate", "candidate_solver", 0.002),
        ("comparator", "comparator_solver", 0.003),
    ):
        declaration = protocol[declaration_key]
        trial_id = f"{role}-trial-0"
        configuration = _configuration(role)
        ledgers.append(
            {
                "role": role,
                "solver": {
                    field: copy.deepcopy(declaration[field])
                    for field in (
                        "solver_id",
                        "solver_version",
                        "runner_id",
                        "runner_version",
                    )
                },
                "budget_contract_hash": solver_budget_contract_hash(
                    protocol["budget"]
                ),
                "selection_rule": protocol["within_instance_aggregation"],
                "selection_metric": protocol["budget"]["evaluation"][
                    "target_metric"
                ],
                "trials": [
                    {
                        "trial_id": trial_id,
                        "status": "done",
                        "objective_evaluations": 8,
                        "wall_seconds": 4.0,
                        "selection_metric_value": metric_value,
                        "configuration": configuration,
                        "configuration_hash": solver_configuration_hash(
                            configuration
                        ),
                    }
                ],
                "selected_trial_id": trial_id,
            }
        )
    return ledgers


def _evaluate(protocol, *, runs=None, schema=None, search_ledgers=None):
    return evaluate_solver_competition(
        _grid(protocol) if runs is None else runs,
        protocol=protocol,
        schema=_schema() if schema is None else schema,
        search_ledgers=(
            _search_ledgers(protocol)
            if search_ledgers is None
            else search_ledgers
        ),
    )


def _all_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _all_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_keys(item)


def test_complete_grid_pairs_instances_and_never_seeds_or_scores(
    comparison_registry,
):
    protocol = _protocol()
    result = _evaluate(protocol)

    assert result["complete"] is True
    assert result["valid"] is True
    assert result["admission_valid"] is True
    assert result["expected_independent_pairs"] == 2
    assert result["n_pairs"] == 2
    assert result["nested_model_seed_count"] == 3
    assert result["expected_cell_count"] == 12
    assert result["search_complete"] is True
    assert result["observed_search_ledger_count"] == 2
    assert result["paired_stats"] is None
    assert result["aggregate_available"] is False
    assert result["comparative_inference_enabled"] is False
    assert result["claim_eligible"] is False
    forbidden = {
        "advantage_score",
        "composite_score",
        "composite_advantage_score",
    }
    assert forbidden.isdisjoint(_all_keys(result))


def test_production_registry_keeps_solver_comparison_disabled():
    protocol = _protocol()

    result = _evaluate(protocol)

    assert result["valid"] is False
    assert result["comparative_inference_enabled"] is False
    assert any(
        row["path"] in {
            "protocol.candidate_solver",
            "protocol.comparator_solver",
        }
        and "canonical registry" in row["reason"]
        for row in result["violations"]
    )


def test_self_attested_renamed_solver_is_not_registered(
    comparison_registry,
):
    protocol = _protocol()
    protocol["candidate_solver"]["solver_id"] = "renamed-quantum-solver"

    result = _evaluate(protocol)

    assert result["valid"] is False
    assert any(
        row["path"] == "protocol.candidate_solver"
        and "canonical registry" in row["reason"]
        for row in result["violations"]
    )
    assert result["paired_stats"] is None


def test_ground_state_energy_error_remains_nonpairable_after_schema_declaration():
    spec = metric_type_spec("ground_state_energy_error")

    assert spec["extraction_key"] == "energy_error"
    assert spec["pairable"] is False


def test_missing_nested_seed_disables_the_entire_grid(comparison_registry):
    protocol = _protocol()
    runs = _grid(protocol)
    runs.pop()

    result = _evaluate(protocol, runs=runs)

    assert result["complete"] is False
    assert result["valid"] is False
    assert result["n_pairs"] == 0
    assert result["missing_cells"]
    assert result["paired_stats"] is None
    assert result["aggregate_available"] is False


def test_duplicate_cell_disables_the_entire_grid(comparison_registry):
    protocol = _protocol()
    runs = _grid(protocol)
    runs.append(copy.deepcopy(runs[0]))

    result = _evaluate(protocol, runs=runs)

    assert result["complete"] is False
    assert result["valid"] is False
    assert result["duplicate_cells"]
    assert result["paired_stats"] is None


@pytest.mark.parametrize(
    ("mutation", "path_fragment"),
    [
        ("instance_hash", "problem_instances[1].instance_hash"),
        ("run_uuid", "run_uuid"),
        ("manifest_hash", "manifest_hash"),
        ("instance_optimum", "optimum.reference_id"),
    ],
)
def test_independent_instance_and_run_identities_are_unique(
    comparison_registry, mutation, path_fragment
):
    protocol = _protocol()
    runs = _grid(protocol)
    if mutation == "instance_hash":
        protocol["problem_instances"][1]["instance_hash"] = protocol[
            "problem_instances"
        ][0]["instance_hash"]
        runs = _grid(protocol)
    elif mutation == "run_uuid":
        runs[1]["run_uuid"] = runs[0]["run_uuid"]
    elif mutation == "manifest_hash":
        runs[1]["manifest_hash"] = runs[0]["manifest_hash"]
    else:
        runs[-1]["optimum"] = copy.deepcopy(
            protocol["problem_instances"][0]["optimum_reference"]
        )

    result = _evaluate(protocol, runs=runs)

    assert result["valid"] is False
    assert result["n_pairs"] == 0
    assert any(
        path_fragment in row["path"] for row in result["violations"]
    )


@pytest.mark.parametrize(
    ("mutation", "path_fragment"),
    [
        ("missing_role", "missing_role"),
        ("duplicate_role", "duplicate_role"),
        ("trial_ceiling", ".trials"),
        ("objective_ceiling", ".trials"),
        ("wall_ceiling", ".trials"),
        ("selected_trial", "selected_trial_id"),
        ("aggregation", "within_instance_aggregation"),
        ("selection_rule", "selection_rule"),
        ("not_best", "selected_trial_id"),
    ],
)
def test_observed_search_ledgers_enforce_total_budget_and_selection(
    comparison_registry, mutation, path_fragment
):
    protocol = _protocol()
    if mutation == "aggregation":
        protocol["within_instance_aggregation"] = "mean_over_trials"
    if mutation == "not_best":
        protocol["budget"]["search"]["max_trials"] = 2
    ledgers = _search_ledgers(protocol)
    if mutation == "missing_role":
        ledgers.pop()
    elif mutation == "duplicate_role":
        duplicate = copy.deepcopy(ledgers[0])
        duplicate["trials"][0]["trial_id"] = "candidate-trial-duplicate"
        duplicate["selected_trial_id"] = "candidate-trial-duplicate"
        ledgers.append(duplicate)
    elif mutation == "trial_ceiling":
        extra = copy.deepcopy(ledgers[0]["trials"][0])
        extra["trial_id"] = "candidate-trial-1"
        ledgers[0]["trials"].append(extra)
    elif mutation == "objective_ceiling":
        ledgers[0]["trials"][0]["objective_evaluations"] = 1001
    elif mutation == "wall_ceiling":
        ledgers[0]["trials"][0]["wall_seconds"] = 61.0
    elif mutation == "selected_trial":
        ledgers[0]["selected_trial_id"] = "missing-trial"
    elif mutation == "selection_rule":
        ledgers[0]["selection_rule"] = "last_trial"
    elif mutation == "not_best":
        extra = copy.deepcopy(ledgers[0]["trials"][0])
        extra["trial_id"] = "candidate-trial-1"
        extra["selection_metric_value"] = 0.001
        ledgers[0]["trials"].append(extra)

    result = _evaluate(protocol, search_ledgers=ledgers)

    assert result["valid"] is False
    assert result["search_complete"] is False
    assert result["paired_stats"] is None
    assert any(
        path_fragment in row["path"] for row in result["violations"]
    )


@pytest.mark.parametrize(
    ("mutation", "path_fragment"),
    [
        ("selected_trial", "solver.selected_trial_id"),
        ("unobserved_configuration", "solver.configuration_hash"),
        ("stale_run_hash", "solver.configuration_hash"),
        ("missing_algorithm", "solver.algorithm_family"),
        ("missing_hyperparameters", "solver.hyperparameters"),
        ("stale_trial_hash", "trials[0].configuration_hash"),
        ("search_budget_hash", "budget_contract_hash"),
    ],
)
def test_evaluation_runs_are_bound_to_selected_search_configuration(
    comparison_registry, mutation, path_fragment
):
    protocol = _protocol()
    runs = _grid(protocol)
    ledgers = _search_ledgers(protocol)
    if mutation == "selected_trial":
        runs[0]["solver"]["selected_trial_id"] = "candidate-trial-1"
    elif mutation == "unobserved_configuration":
        configuration = {
            "algorithm_family": "variational_quantum",
            "hyperparameters": {"declared": False},
        }
        runs[0]["solver"].update(configuration)
        runs[0]["solver"]["configuration_hash"] = solver_configuration_hash(
            configuration
        )
    elif mutation == "stale_run_hash":
        runs[0]["solver"]["hyperparameters"] = {"declared": False}
    elif mutation == "missing_algorithm":
        runs[0]["solver"].pop("algorithm_family")
    elif mutation == "missing_hyperparameters":
        runs[0]["solver"].pop("hyperparameters")
    elif mutation == "stale_trial_hash":
        ledgers[0]["trials"][0]["configuration"]["hyperparameters"] = {
            "declared": False
        }
    else:
        ledgers[0]["budget_contract_hash"] = "f" * 64

    result = _evaluate(
        protocol,
        runs=runs,
        search_ledgers=ledgers,
    )

    assert result["valid"] is False
    assert result["n_pairs"] == 0
    assert result["paired_stats"] is None
    assert any(
        path_fragment in row["path"] for row in result["violations"]
    )


@pytest.mark.parametrize(
    ("mutation", "path_fragment"),
    [
        ("budget_hash", "budget_contract_hash"),
        ("instance_hash", "problem.instance_hash"),
        ("missing_version", "solver.solver_version"),
        ("resource_overrun", "objective_evaluations.value"),
        ("analytic_shots", "total_shots.value"),
        ("classical_shots_missing", "total_shots"),
        ("failed_cell", "status"),
    ],
)
def test_solver_grid_contract_drift_fails_closed(
    comparison_registry, mutation, path_fragment
):
    protocol = _protocol()
    runs = _grid(protocol)
    target = (
        next(
            row
            for row in runs
            if row["solver"]["computation_kind"] == "classical"
        )
        if mutation == "classical_shots_missing"
        else runs[0]
    )
    if mutation == "budget_hash":
        target["budget_contract_hash"] = "f" * 64
    elif mutation == "instance_hash":
        target["problem"]["instance_hash"] = "f" * 64
    elif mutation == "missing_version":
        target["solver"]["solver_version"] = ""
    elif mutation == "resource_overrun":
        target["resources"]["objective_evaluations"]["value"] = 101
    elif mutation == "analytic_shots":
        target["resources"]["total_shots"] = {
            "value": None,
            "status": "not_applicable",
        }
    elif mutation == "classical_shots_missing":
        target["resources"]["total_shots"] = {
            "value": None,
            "status": "unmeasured",
        }
    else:
        target["status"] = "cancelled"

    result = _evaluate(protocol, runs=runs)

    assert result["valid"] is False
    assert result["paired_stats"] is None
    assert any(
        path_fragment in row["path"] for row in result["violations"]
    )


@pytest.mark.parametrize("reference_solver", ["exact_diagonalization", "best_product_state"])
def test_static_references_cannot_masquerade_as_solver_runs(
    comparison_registry, reference_solver
):
    protocol = _protocol()
    protocol["comparator_solver"]["solver_id"] = reference_solver
    runs = _grid(protocol)

    result = _evaluate(protocol, runs=runs)

    assert result["valid"] is False
    assert any(
        row["path"].endswith("solver.solver_id")
        and "not solver runs" in row["reason"]
        for row in result["violations"]
    )


def test_best_known_optimum_cannot_be_relabelled_as_exact_energy_error(
    comparison_registry,
):
    protocol = _protocol()
    protocol["problem_instances"][0]["optimum_reference"][
        "role"
    ] = "best_known_optimum"
    runs = _grid(protocol)

    result = _evaluate(protocol, runs=runs)

    assert result["valid"] is False
    assert any(
        "gap_to_best_known" in row["reason"] for row in result["violations"]
    )
    assert result["paired_stats"] is None


def test_solver_schema_fails_closed_in_controlled_comparison_evaluators():
    schema = _schema()
    fairness = evaluate_fairness({}, {}, schema=schema)
    ladder = evaluate_analogue_ladder(
        candidate={},
        baseline={},
        fairness={},
        claim={"fairness_schema": schema},
    )

    assert fairness["valid"] is False
    assert fairness["evaluator"] == "controlled_pair"
    assert fairness["disallowed_mismatches"][0]["requirement"] == (
        "dedicated_solver_competition_evaluator"
    )
    assert ladder["valid"] is False
    assert ladder["required_complete"] is False
    assert ladder["parameter_tolerance"] is None


def test_persisted_malformed_solver_contract_returns_blocked_not_exception(
    comparison_registry,
):
    protocol = _protocol()
    protocol["budget"]["evaluation"]["model_seeds"] = 3
    schema = copy.deepcopy(_schema())
    schema["comparator_kind"] = "unsupported"

    result = _evaluate(protocol, runs=_grid(_protocol()), schema=schema)

    assert result["valid"] is False
    assert result["assessment_status"] == "blocked"
    assert result["paired_stats"] is None
    assert any(
        row["path"] == "fairness_schema.comparator_kind"
        for row in result["violations"]
    )
    assert any(
        row["path"] == "protocol.budget.evaluation.model_seeds"
        for row in result["violations"]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "list_optimum_role",
        "list_solver_id",
        "list_allowed_roles",
        "scalar_required_present",
    ],
)
def test_unhashable_or_wrong_shape_persisted_values_fail_closed(
    comparison_registry, mutation
):
    protocol = _protocol()
    schema = copy.deepcopy(_schema())
    if mutation == "list_optimum_role":
        protocol["problem_instances"][0]["optimum_reference"]["role"] = [
            "certified_optimum"
        ]
    elif mutation == "list_solver_id":
        protocol["candidate_solver"]["solver_id"] = ["qllm-finite-shot-vqe"]
    elif mutation == "list_allowed_roles":
        schema["allowed_optimum_roles"] = [["certified_optimum"]]
    else:
        schema["required_present"] = 3

    result = _evaluate(protocol, schema=schema)

    assert result["valid"] is False
    assert result["assessment_status"] == "blocked"
    assert result["paired_stats"] is None
    assert result["violations"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "resource_field",
    [
        "objective_evaluations",
        "wall_seconds",
        "state_preparations",
        "total_shots",
        "peak_memory_bytes",
        "state_representation",
    ],
)
def test_list_valued_resource_statuses_fail_closed(
    comparison_registry, resource_field
):
    protocol = _protocol()
    runs = _grid(protocol)
    runs[0]["resources"][resource_field]["status"] = ["measured"]

    result = _evaluate(protocol, runs=runs)

    assert result["valid"] is False
    assert result["assessment_status"] == "blocked"
    assert result["paired_stats"] is None
    assert any(
        row["path"] == f"runs[0].resources.{resource_field}.status"
        for row in result["violations"]
    )
    json.dumps(result, allow_nan=False)


def test_budget_hash_is_server_derived_and_rejects_nonfinite_values():
    budget = _protocol()["budget"]
    expected = solver_budget_contract_hash(budget)
    budget["contract_hash"] = "caller-controlled"

    assert solver_budget_contract_hash(budget) == expected
    budget["evaluation"]["target_value"] = float("nan")
    with pytest.raises(ValueError, match="finite canonical JSON"):
        solver_budget_contract_hash(budget)
