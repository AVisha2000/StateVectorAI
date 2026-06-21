"""Quantum sequence data: generators, controls, dispatch, screening."""
from __future__ import annotations

import dataclasses

import numpy as np

from qllm.config import DataConfig
from qllm.data.datasets import load_dataset
from qllm.data.quantum_seq import (
    IdentityTokenizer,
    markov_control_sequences,
    monitored_ising_sequences,
)
from qllm.quantum.advantage import (
    classical_kernel_family,
    engineered_labels,
    geometric_difference,
    model_complexity,
    normalize_trace,
    quantum_fidelity_kernel,
    screen_sequence_dataset,
)
from qllm.train.loop import fit


def _gen(seed=0, **kw):
    defaults = dict(
        n_qubits=4, n_measured=2, n_sequences=4, seq_len=128, seed=seed
    )
    defaults.update(kw)
    return monitored_ising_sequences(**defaults)


def test_generator_shapes_vocab_determinism():
    ids, vocab = _gen()
    assert vocab == 4
    assert ids.shape == (4 * 128,)
    assert ids.dtype == np.int32
    assert ids.min() >= 0 and ids.max() < vocab
    ids2, _ = _gen()
    np.testing.assert_array_equal(ids, ids2)  # seeded determinism
    ids3, _ = _gen(seed=1)
    assert not np.array_equal(ids, ids3)


def test_generator_nontrivial_statistics():
    """Chaotic dynamics should produce non-degenerate, non-uniform-iid tokens."""
    ids, vocab = _gen(n_sequences=8, seq_len=512)
    counts = np.bincount(ids, minlength=vocab)
    assert (counts > 0).all(), "some token never occurs"
    assert counts.max() / counts.sum() < 0.95, "degenerate (constant) output"


def test_markov_control_matches_kgrams():
    ids, vocab = _gen(n_sequences=8, seq_len=1024)
    twin = markov_control_sequences(ids, vocab, order=2, seed=1)
    assert len(twin) == len(ids)
    assert twin.max() < vocab

    def bigram_dist(x):
        d = np.zeros((vocab, vocab))
        np.add.at(d, (x[:-1], x[1:]), 1.0)
        return d / d.sum()

    tv = 0.5 * np.abs(bigram_dist(ids) - bigram_dist(twin)).sum()
    assert tv < 0.1, f"bigram total-variation too large: {tv:.3f}"


def test_identity_tokenizer():
    tok = IdentityTokenizer(4)
    ids = tok.encode([0, 3, 1])
    assert tok.vocab_size == 4
    assert ids.dtype == np.int32
    assert tok.decode(ids) == "0 3 1"
    assert tok.stoi == {}


def test_load_dataset_dispatch():
    base = DataConfig(
        kind="monitored_ising",
        gen_qubits=4,
        gen_measured=1,
        gen_sequences=2,
        gen_len=64,
    )
    ids, tok = load_dataset(base)
    assert tok.vocab_size == 2
    assert len(ids) == 128

    ctrl = dataclasses.replace(base, kind="markov_control", markov_order=2)
    ids_c, tok_c = load_dataset(ctrl)
    assert tok_c.vocab_size == 2 and len(ids_c) == 128
    assert not np.array_equal(ids, ids_c)

    text_ids, text_tok = load_dataset(DataConfig(corpus_path="__missing__"))
    assert text_tok.vocab_size > 2 and len(text_ids) > 100


def test_model_complexity_prefers_own_kernel():
    """Labels engineered from the quantum kernel must have s_Q < s_C."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-1.5, 1.5, size=(80, 4))
    K_q = normalize_trace(quantum_fidelity_kernel(X, n_layers=2, seed=0))
    K_c = normalize_trace(classical_kernel_family(X)["rbf_1.0"])
    _, v = geometric_difference(K_c, K_q)
    y = engineered_labels(K_q, v)
    assert model_complexity(K_q, y) < model_complexity(K_c, y)


def test_screen_sequence_dataset_runs():
    ids, vocab = _gen(n_sequences=8, seq_len=512)
    rep = screen_sequence_dataset(ids, vocab, n_qubits=4, n_samples=80, seed=0)
    assert np.isfinite(rep.g_min) and rep.g_min > 0
    assert rep.s_quantum > 0 and rep.s_classical_best > 0
    assert np.isfinite(rep.s_ratio)


def test_fit_on_quantum_data(tiny_classical_cfg, tmp_path):
    cfg = dataclasses.replace(
        tiny_classical_cfg,
        data=DataConfig(
            kind="monitored_ising",
            gen_qubits=4,
            gen_measured=2,
            gen_sequences=2,
            gen_len=256,
        ),
    )
    res = fit(cfg, verbose=False, out_dir=tmp_path)
    assert np.isfinite(res["summary"]["val_loss"])
    assert res["tokenizer"].vocab_size == 4
