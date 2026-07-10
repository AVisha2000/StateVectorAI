"""Config system and data pipeline tests."""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from qllm.config import from_dict, load_yaml, to_flat_dict
from qllm.data.text import CharTokenizer, load_corpus, sample_batch, train_val_split

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
