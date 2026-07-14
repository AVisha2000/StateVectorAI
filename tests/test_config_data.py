"""Config system and data pipeline tests."""
from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from qllm.config import (
    ProblemConfig,
    QuantumConfig,
    from_dict,
    load_yaml,
    to_flat_dict,
    two_stream_position_count,
    validate_config,
)
from qllm.data.text import CharTokenizer, load_corpus, sample_batch, train_val_split
from qllm.registry import (
    ANSATZ_TYPES,
    ARCH_TYPES,
    ATTN_TYPES,
    BACKEND_TYPES,
    CONDITION_TYPES,
    DATASET_KINDS,
    FFN_TYPES,
    METRIC_TYPES,
    READOUT_TYPES,
    TASK_TYPES,
    metric_type_spec,
    supported_choices_payload,
)


BOUNDARY_SAFE_DATASET_CALLERS = (
    "benchmarks/memory_sweep.py",
    "benchmarks/planted_qrnn.py",
    "benchmarks/resonance_search.py",
    "benchmarks/seq_interference_probe.py",
    "benchmarks/model_report.py",
    "qllm/dashboard/model_tests.py",
)


def _ground_state_payload(
    *, quantum: dict | None = None, steps: int = 100
) -> dict:
    quantum_payload = {
        "n_qubits": 2,
        "n_circuit_layers": 2,
        "ansatz": "hardware_efficient",
        "backend": "pennylane",
        "device": "default.qubit",
        "diff_method": "backprop",
        "shots": None,
        "trainable": True,
    }
    quantum_payload.update(quantum or {})
    return {
        "problem": {
            "task_type": "ground_state",
            "instance_id": "tfim-2q-open-j1-h1",
        },
        "model": {"quantum": quantum_payload},
        "train": {"steps": steps},
    }

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_from_dict_nested_quantum():
    cfg = from_dict(
        {
            "model": {
                "d_model": 32,
                "ffn_type": "quantum",
                "quantum": {"n_qubits": 6, "ansatz": "hardware_efficient"},
            }
        }
    )
    assert cfg.model.d_model == 32
    assert cfg.model.quantum.n_qubits == 6
    assert cfg.model.quantum.ansatz == "hardware_efficient"
    # untouched defaults survive
    assert cfg.train.steps == 200


def test_unknown_key_raises():
    with pytest.raises(KeyError):
        from_dict({"model": {"not_a_field": 1}})


def test_unknown_top_level_section_raises():
    with pytest.raises(KeyError, match="Unknown config section.*runtime"):
        from_dict({"model": {}, "runtime": {"device": "cpu"}})


@pytest.mark.parametrize(
    "payload",
    [
        {"model": []},
        {"model": {"quantum": []}},
        {"model": {"blocks": {}}},
    ],
)
def test_falsey_non_mapping_sections_are_rejected(payload):
    with pytest.raises(TypeError):
        from_dict(payload)


def test_yaml_roundtrip(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "model:\n  d_model: 24\n  quantum:\n    n_qubits: 3\n"
        "train:\n  lr: 0.0005\n"
    )
    cfg = load_yaml(path)
    assert cfg.model.d_model == 24
    assert cfg.model.quantum.n_qubits == 3
    assert cfg.train.lr == pytest.approx(5e-4)


def test_flat_dict_keys():
    flat = to_flat_dict(from_dict({}))
    assert "model.quantum.n_qubits" in flat
    assert "train.lr" in flat
    assert "tracking.experiment" in flat
    assert flat["problem.task_type"] == "sequence_modeling"
    assert flat["problem.instance_id"] is None


def test_configs_are_hashable():
    cfg = from_dict({})
    hash(cfg)  # frozen dataclasses must be hashable (Flax static fields)
    assert cfg.problem == ProblemConfig()


def test_problem_config_roundtrip_and_canonical_choices():
    assert TASK_TYPES == (
        "sequence_modeling",
        "ground_state",
        "combinatorial_optimization",
    )
    payload = {
        "problem": {
            "task_type": "ground_state",
            "instance_id": "ising-chain-8",
        }
    }
    cfg = from_dict(payload)
    assert cfg.problem == ProblemConfig(**payload["problem"])
    assert to_flat_dict(cfg)["problem.instance_id"] == "ising-chain-8"
    assert supported_choices_payload()["task_type"] == list(TASK_TYPES)


def test_config_choices_response_requires_canonical_task_types():
    from qllm.dashboard.config_choices import validate_config_choices

    payload = supported_choices_payload()
    payload["quantum_default"] = dataclasses.asdict(QuantumConfig())
    assert validate_config_choices(payload)["task_type"] == list(TASK_TYPES)
    payload.pop("task_type")
    with pytest.raises(ValueError, match="task_type"):
        validate_config_choices(payload)


def test_problem_config_accepts_registered_ground_state_task():
    assert validate_config(from_dict(_ground_state_payload())) == []


def test_problem_config_accepts_opaque_combinatorial_task_identity():
    problem = {
        "task_type": "combinatorial_optimization",
        "instance_id": "maxcut-cube",
    }
    assert validate_config(from_dict({"problem": problem})) == []


def test_problem_config_rejects_unregistered_ground_state_instance():
    payload = _ground_state_payload()
    payload["problem"]["instance_id"] = "ising-chain-8"
    errors = validate_config(from_dict(payload))
    assert any("registered ground-state instance" in error for error in errors)


@pytest.mark.parametrize(
    ("quantum", "steps", "expected"),
    [
        ({"n_qubits": 3}, 100, "must match the registered"),
        ({"shots": 10}, 100, "shots must be null"),
        ({"trainable": False}, 100, "trainable must be true"),
        ({"ansatz": "reuploading"}, 100, "hardware_efficient"),
        ({"n_circuit_layers": 9}, 100, "must be at most 8"),
        (
            {"backend": "tensorcircuit", "device": "statevector"},
            100,
            "requires model.quantum.backend='pennylane'",
        ),
        (
            {"diff_method": "parameter-shift"},
            100,
            "diff_method must be 'backprop'",
        ),
        ({}, 1001, "train.steps must be at most 1000"),
    ],
)
def test_ground_state_config_is_bounded_to_initial_diagnostic_slice(
    quantum, steps, expected
):
    errors = validate_config(
        from_dict(_ground_state_payload(quantum=quantum, steps=steps))
    )
    assert expected in "\n".join(errors)


@pytest.mark.parametrize(
    ("problem", "expected"),
    [
        (
            {"task_type": "sequence_modeling", "instance_id": "text-1"},
            "problem.instance_id must be null",
        ),
        (
            {"task_type": "ground_state"},
            "problem.instance_id must be provided",
        ),
        (
            {"task_type": "combinatorial_optimization"},
            "problem.instance_id must be provided",
        ),
    ],
)
def test_problem_config_rejects_required_forbidden_and_empty_values(problem, expected):
    assert expected in "\n".join(validate_config(from_dict({"problem": problem})))


@pytest.mark.parametrize(
    "instance_id",
    ["", "UPPER", "with space", "../escape", "path/name", "x" * 129, 7],
)
def test_problem_config_rejects_malformed_instance_ids(instance_id):
    errors = validate_config(from_dict({
        "problem": {"task_type": "ground_state", "instance_id": instance_id}
    }))
    assert any("ASCII registry key matching" in error for error in errors)


def test_problem_config_rejects_unrecognized_problem_fields():
    with pytest.raises(KeyError, match="hamiltonian_id"):
        from_dict({"problem": {"hamiltonian_id": "tfim-open"}})


def test_problem_config_rejects_unknown_task_without_secondary_task_errors():
    errors = validate_config(from_dict({"problem": {"task_type": "unknown"}}))
    assert any("problem.task_type must be one of" in error for error in errors)
    assert not any("must be provided when problem.task_type" in error for error in errors)


def test_quantum_diagnostic_display_preserves_unavailable_evidence():
    from qllm.train.loop import _format_quantum_diagnostics_for_display

    formatted = _format_quantum_diagnostics_for_display({
        "grad_var_mean": 0.0125,
        "meyer_wallach_q": None,
        "availability": {
            "meyer_wallach_q": {
                "status": "unsupported",
                "reason": "state access is unavailable",
            },
        },
    })
    assert formatted["grad_var_mean"] == "1.250e-02"
    assert formatted["meyer_wallach_q"] is None
    assert formatted["availability"]["meyer_wallach_q"]["status"] == "unsupported"


def test_shared_validation_uses_canonical_registries():
    assert validate_config(from_dict({})) == []
    assert "two_stream" in ARCH_TYPES
    assert {"quantum_proj", "quantum_qkv"} <= set(ATTN_TYPES)
    assert {"quantum", "quantum_linear", "lowrank"} <= set(FFN_TYPES)
    assert set(DATASET_KINDS) == {
        "text", "monitored_ising", "markov_control", "contextual",
        "interference", "seq_cancellation",
    }
    assert "ising" in ANSATZ_TYPES
    assert {"pennylane", "tensorcircuit", "tensorcircuit_mps"} == set(BACKEND_TYPES)
    assert {"z", "zz"} == set(READOUT_TYPES)
    assert {"film", "token", "bias"} == set(CONDITION_TYPES)
    payload = supported_choices_payload()
    assert payload["architecture"] == list(ARCH_TYPES)
    assert payload["circuit_ansatz"] == ["hardware_efficient", "reuploading"]
    assert set(payload["metric_types"]) == {
        "ground_state_energy_error",
        "strict_autoregressive_next_token",
        "validation_perplexity",
    }
    assert payload["metric_types"]["ground_state_energy_error"] == {
        "lower_is_better": True,
        "units": "problem_energy_units",
        "pairable": False,
        "extraction_key": "energy_error",
        "comparator_class": "exact_reference_diagnostic",
    }
    assert payload["ground_state_instances"][0]["instance_id"] == (
        "tfim-2q-open-j1-h1"
    )
    assert payload["ground_state_instances"][0]["classical_references"][0][
        "role"
    ] == "oracle"
    assert payload["metric_types"]["validation_perplexity"] == {
        "lower_is_better": True,
        "units": "ppl",
        "pairable": True,
        "extraction_key": "val_ppl",
        "comparator_class": "matched_control",
    }
    assert metric_type_spec("validation_perplexity", require_pairable=True) == (
        METRIC_TYPES["validation_perplexity"]
    )
    assert metric_type_spec("time_to_target", require_pairable=True) is None
    with pytest.raises(TypeError):
        METRIC_TYPES["validation_perplexity"]["extraction_key"] = "energy_error"


@pytest.mark.parametrize(
    ("quantum", "expected"),
    [
        (
            {"backend": "tensorcircuit", "device": "default.qubit"},
            "requires device='statevector'",
        ),
        (
            {
                "backend": "tensorcircuit",
                "device": "statevector",
                "diff_method": "parameter-shift",
            },
            "supports only diff_method='backprop'",
        ),
        (
            {"backend": "pennylane", "shots": 100},
            "finite-shot execution does not support diff_method='backprop'",
        ),
    ],
)
def test_config_validation_rejects_backend_semantic_mismatches(
    quantum, expected
):
    errors = validate_config(from_dict({"model": {"quantum": quantum}}))
    assert any(expected in error for error in errors)


def test_config_validation_accepts_dense_tensorcircuit_mode():
    cfg = from_dict({
        "model": {
            "quantum": {
                "backend": "tensorcircuit",
                "device": "statevector",
            },
        },
    })
    assert validate_config(cfg) == []


def test_validation_reports_numeric_and_semantic_errors():
    cfg = from_dict({
        "model": {
            "arch": "transformer",
            "d_model": 10,
            "n_heads": 3,
            "max_seq_len": 8,
            "quantum": {"ansatz": "ising", "n_qubits": 0},
        },
        "train": {"steps": 0, "seq_len": 16, "lr": -0.1},
        "data": {
            "kind": "monitored_ising",
            "gen_qubits": 2,
            "gen_measured": 2,
            "gen_sequences": 1,
            "gen_len": 16,
        },
    })
    errors = validate_config(cfg)
    joined = "\n".join(errors)
    assert "model.d_model must be divisible by model.n_heads" in joined
    assert "model.quantum.ansatz='ising'" in joined
    assert "train.steps must be a positive integer" in joined
    assert "train.seq_len must be <= model.max_seq_len" in joined
    assert "data.gen_measured must be smaller than data.gen_qubits" in joined
    assert "data.gen_sequences must be at least 2" in joined
    assert "data.gen_len must be greater than train.seq_len" in joined


def test_two_stream_internal_position_count_contract():
    assert two_stream_position_count(8, "classical", "token") == 16
    assert two_stream_position_count(8, "quantum", "token") == 16
    assert two_stream_position_count(8, "none", "token") == 8
    assert two_stream_position_count(8, "classical", "film") == 8
    assert two_stream_position_count(8, "classical", "bias") == 8


def test_two_stream_token_capacity_validation_uses_expanded_length():
    invalid = from_dict({
        "model": {
            "arch": "two_stream",
            "encoder_kind": "classical",
            "condition": "token",
            "max_seq_len": 15,
        },
        "train": {"seq_len": 8},
    })
    message = "\n".join(validate_config(invalid))
    assert "internal positional capacity" in message
    assert "2 * train.seq_len" in message
    assert "required 16, got 15" in message

    exact = from_dict({
        "model": {
            "arch": "two_stream",
            "encoder_kind": "classical",
            "condition": "token",
            "max_seq_len": 16,
        },
        "train": {"seq_len": 8},
    })
    assert validate_config(exact) == []


@pytest.mark.parametrize(
    ("encoder_kind", "condition"),
    [("none", "token"), ("classical", "film"), ("quantum", "bias")],
)
def test_two_stream_non_expanded_modes_use_real_token_length(
    encoder_kind, condition
):
    cfg = from_dict({
        "model": {
            "arch": "two_stream",
            "encoder_kind": encoder_kind,
            "condition": condition,
            "max_seq_len": 8,
        },
        "train": {"seq_len": 8},
    })
    assert validate_config(cfg) == []


def test_classical_config_can_omit_quantum_and_recurrent_dims_are_not_transformer_constraints():
    cfg = from_dict({
        "model": {
            "arch": "gru",
            "quantum": None,
            "d_model": 10,
            "n_heads": 3,
            "max_seq_len": 8,
        },
        "train": {"seq_len": 16},
    })
    assert validate_config(cfg) == []


def test_recurrent_models_reject_transformer_only_components():
    cfg = from_dict({
        "model": {"arch": "gru", "attn_type": "quantum_proj"},
    })
    assert "architecture-specific components" in "\n".join(validate_config(cfg))


def test_validation_rejects_non_integer_train_seed():
    cfg = from_dict({"train": {"seed": 1.5}})
    assert "train.seed must be an integer" in "\n".join(validate_config(cfg))


def test_validation_reports_all_supported_dataset_kinds():
    cfg = from_dict({"data": {"kind": "unknown"}})
    message = "\n".join(validate_config(cfg))
    for kind in DATASET_KINDS:
        assert kind in message


def test_validation_handles_non_string_registry_values_without_crashing():
    cfg = from_dict({
        "model": {"arch": ["transformer"]},
        "data": {"kind": ["text"]},
    })
    message = "\n".join(validate_config(cfg))
    assert "model.arch must be one of" in message
    assert "data.kind must be one of" in message


def test_ising_ansatz_is_qrnn_only():
    qrnn = from_dict({
        "model": {"arch": "qrnn", "quantum": {"ansatz": "ising"}},
        "data": {"kind": "monitored_ising"},
    })
    assert validate_config(qrnn) == []


def test_repository_configs_pass_shared_validation():
    for path in sorted(Path("configs").glob("*.yaml")):
        assert validate_config(load_yaml(path)) == [], path


def test_train_cli_rejects_shared_validation_errors_before_fit(
    monkeypatch, tmp_path
):
    import scripts.train as train_script

    path = tmp_path / "invalid.yaml"
    path.write_text("train:\n  steps: 0\n", encoding="utf-8")
    monkeypatch.setattr(
        train_script,
        "fit",
        lambda _cfg: pytest.fail("fit must not run for invalid CLI config"),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["train.py", "--config", str(path), "--sample", "0"],
    )

    with pytest.raises(SystemExit) as exc_info:
        train_script.main()

    assert exc_info.value.code == 2


def test_direct_backend_factory_rejects_unregistered_readout_before_import():
    from qllm.quantum.backends import get_expval_circuit

    with pytest.raises(ValueError, match="Unknown readout"):
        get_expval_circuit(
            "tensorcircuit",
            "statevector",
            "backprop",
            None,
            2,
            1,
            "reuploading",
            "not-a-readout",
        )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def test_load_corpus_fallback():
    text = load_corpus("__definitely_missing__", synthetic_fallback=True)
    assert len(text) > 1000


def test_load_corpus_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_corpus("__definitely_missing__", synthetic_fallback=False)


def test_tokenizer_roundtrip(tiny_text):
    tok = CharTokenizer(tiny_text)
    ids = tok.encode(tiny_text[:100])
    assert tok.decode(ids) == tiny_text[:100]
    assert ids.dtype == np.int32
    assert ids.max() < tok.vocab_size


@settings(max_examples=25, deadline=None)
@given(st.text(alphabet=st.sampled_from(sorted(set("hello quantum world! "))), min_size=1, max_size=64))
def test_tokenizer_roundtrip_property(sample):
    tok = CharTokenizer("hello quantum world! ")
    assert tok.decode(tok.encode(sample)) == sample


def test_split_and_batch(tiny_text):
    tok = CharTokenizer(tiny_text)
    ids = tok.encode(tiny_text)
    train, val = train_val_split(ids, 0.1)
    assert len(train) + len(val) == len(ids)
    assert len(val) >= 1

    rng = np.random.default_rng(0)
    batch = sample_batch(rng, train, batch_size=4, seq_len=8)
    assert batch.shape == (4, 9)  # seq_len + 1 for next-token targets
    assert batch.dtype == np.int32

    # determinism under a fixed seed
    again = sample_batch(np.random.default_rng(0), train, 4, 8)
    np.testing.assert_array_equal(batch, again)


def test_trajectory_split_and_batches_never_cross_boundaries():
    trajectories = np.stack([
        np.arange(20, dtype=np.int32) + 100 * row for row in range(5)
    ])
    train, val = train_val_split(trajectories, 0.4)
    assert train.shape == (3, 20)
    assert val.shape == (2, 20)
    assert set(train[:, 0]).isdisjoint(set(val[:, 0]))

    batch = sample_batch(np.random.default_rng(3), train, batch_size=32, seq_len=8)
    assert batch.shape == (32, 9)
    assert np.all(np.diff(batch, axis=1) == 1)
    assert np.all(batch[:, -1] // 100 == batch[:, 0] // 100)


@pytest.mark.parametrize("relative_path", BOUNDARY_SAFE_DATASET_CALLERS)
def test_synthetic_capable_callers_use_boundary_safe_dataset_api(relative_path):
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / relative_path).read_text(encoding="utf-8"))
    dataset_imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.endswith("data.datasets")
        for alias in node.names
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "load_dataset_bundle" in dataset_imports, relative_path
    assert "load_dataset" not in dataset_imports, relative_path
    assert "load_dataset_bundle" in called_names, relative_path


def test_caller_style_seeded_train_and_val_batches_preserve_trajectory_identity():
    trajectories = np.stack([
        np.arange(24, dtype=np.int32) + 100 * row for row in range(6)
    ])
    train, val = train_val_split(trajectories, 0.34)

    train_batch = sample_batch(
        np.random.default_rng(17), train, batch_size=32, seq_len=8
    )
    val_batch = sample_batch(
        np.random.default_rng(23), val, batch_size=32, seq_len=8
    )

    assert train.shape == (4, 24)
    assert val.shape == (2, 24)
    assert np.all(train_batch[:, -1] // 100 == train_batch[:, 0] // 100)
    assert np.all(val_batch[:, -1] // 100 == val_batch[:, 0] // 100)
    assert set(train_batch[:, 0] // 100).isdisjoint(
        set(val_batch[:, 0] // 100)
    )
