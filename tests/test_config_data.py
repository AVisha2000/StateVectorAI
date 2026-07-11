"""Config system and data pipeline tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from qllm.config import (
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
    READOUT_TYPES,
    supported_choices_payload,
)

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


def test_configs_are_hashable():
    cfg = from_dict({}).model
    hash(cfg)  # frozen dataclasses must be hashable (Flax static fields)


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
    assert {"pennylane", "tensorcircuit"} == set(BACKEND_TYPES)
    assert {"z", "zz"} == set(READOUT_TYPES)
    assert {"film", "token", "bias"} == set(CONDITION_TYPES)
    payload = supported_choices_payload()
    assert payload["architecture"] == list(ARCH_TYPES)
    assert payload["circuit_ansatz"] == ["hardware_efficient", "reuploading"]


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
