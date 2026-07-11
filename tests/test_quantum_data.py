"""Quantum sequence data: generators, controls, dispatch, screening."""
from __future__ import annotations

import dataclasses
from types import MappingProxyType

import numpy as np
import pytest

from qllm.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from qllm.data.datasets import data_config_hash, load_dataset, load_dataset_bundle
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


def test_markov_control_does_not_create_cross_trajectory_contexts():
    ids = np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.int32)
    twin = markov_control_sequences(ids, vocab_size=2, order=2, seed=4,
                                    smoothing=0.0)
    assert twin.shape == ids.shape
    np.testing.assert_array_equal(twin, ids)


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


def test_synthetic_bundle_retains_trajectory_boundaries():
    cfg = DataConfig(
        kind="monitored_ising", gen_qubits=3, gen_measured=1,
        gen_sequences=4, gen_len=24, gen_seed=17,
    )
    bundle = load_dataset_bundle(cfg)
    assert bundle.is_trajectory_data
    assert bundle.ids.shape == (4, 24)
    assert bundle.sequence_shape == (4, 24)
    assert bundle.boundaries == (0, 24, 48, 72, 96)
    assert bundle.sampler_policy == "within_trajectory"
    assert bundle.metadata["kind"] == "monitored_ising"
    assert isinstance(bundle.metadata["config"], MappingProxyType)
    assert bundle.provenance["generator"] == "monitored_ising_sequences"
    assert len(bundle.config_hash) == len(bundle.content_hash) == 64
    flat, _ = load_dataset(cfg)
    np.testing.assert_array_equal(flat, bundle.ids.reshape(-1))


def test_full_config_hash_avoids_legacy_rounded_angle_collision(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = DataConfig(
        kind="monitored_ising", gen_qubits=2, gen_measured=1,
        gen_sequences=2, gen_len=12, gen_theta_zz=0.123441,
        gen_theta_x=0.4,
    )
    other = dataclasses.replace(base, gen_theta_zz=0.123449)
    assert f"{base.gen_theta_zz:.4f}" == f"{other.gen_theta_zz:.4f}"
    assert data_config_hash(base) != data_config_hash(other)
    first = load_dataset_bundle(base)
    second = load_dataset_bundle(other)
    assert first.config_hash != second.config_hash
    caches = list((tmp_path / "results/.data_cache").glob("monitored_ising_*.npz"))
    assert len(caches) == 2


def test_ambiguous_rounded_legacy_cache_is_not_reused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = DataConfig(
        kind="monitored_ising", gen_qubits=2, gen_measured=1,
        gen_sequences=2, gen_len=12, gen_theta_zz=0.123441,
        gen_theta_x=0.4,
    )
    cache_dir = tmp_path / "results/.data_cache"
    cache_dir.mkdir(parents=True)
    legacy_name = (
        "monitored_ising_q2_m1_s2_l12_zz0.1234_x0.4000_spt1_seed0_k3.npz"
    )
    np.savez(cache_dir / legacy_name, ids=np.zeros(24, dtype=np.int32), vocab=2)

    bundle = load_dataset_bundle(cfg)

    assert bundle.provenance["cache_identity"] == "config_sha256"
    assert bundle.provenance["source"] == "generated"
    assert (cache_dir / f"monitored_ising_{data_config_hash(cfg)}.npz").exists()


def test_text_content_hash_includes_tokenizer_semantics(tmp_path):
    first_path = tmp_path / "first.txt"
    second_path = tmp_path / "second.txt"
    first_path.write_text("abab", encoding="utf-8")
    second_path.write_text("xyxy", encoding="utf-8")

    first = load_dataset_bundle(DataConfig(corpus_path=str(first_path)))
    second = load_dataset_bundle(DataConfig(corpus_path=str(second_path)))

    np.testing.assert_array_equal(first.ids, second.ids)
    assert first.content_hash != second.content_hash


def test_legacy_synthetic_cache_remains_readable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = DataConfig(
        kind="interference", ctx_observables=8,
        gen_sequences=2, gen_len=12, gen_seed=7,
    )
    expected = np.arange(24, dtype=np.int32) % 8
    cache_dir = tmp_path / "results/.data_cache"
    cache_dir.mkdir(parents=True)
    np.savez(
        cache_dir / "interference_v8_s2_l12_seed7.npz",
        ids=expected,
        vocab=8,
    )
    bundle = load_dataset_bundle(cfg)
    np.testing.assert_array_equal(bundle.ids.reshape(-1), expected)
    assert bundle.provenance["cache_identity"] == "legacy_key"
    assert bundle.provenance["legacy_identity_unverified"] is True


def test_hashed_cache_detects_content_tampering(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = DataConfig(
        kind="interference", ctx_observables=8,
        gen_sequences=2, gen_len=12, gen_seed=11,
    )
    generated = load_dataset_bundle(cfg)
    assert generated.provenance["content_identity_verified"] is True
    path = tmp_path / "results/.data_cache" / f"interference_{data_config_hash(cfg)}.npz"
    with np.load(path, allow_pickle=False) as loaded:
        payload = {name: np.array(loaded[name], copy=True) for name in loaded.files}
    payload["ids"][0] = (int(payload["ids"][0]) + 1) % int(payload["vocab"])
    np.savez(path, **payload)

    with pytest.raises(ValueError, match="identity mismatch"):
        load_dataset_bundle(cfg)


def test_fit_rejects_invalid_config_before_dataset_load(monkeypatch, tmp_path):
    import qllm.train.loop as train_loop

    cfg = ExperimentConfig(train=TrainConfig(steps=0))
    monkeypatch.setattr(
        train_loop,
        "load_dataset_bundle",
        lambda _cfg: pytest.fail("dataset loading must not run"),
    )
    with pytest.raises(ValueError, match="train.steps"):
        train_loop.fit(cfg, verbose=False, out_dir=tmp_path)


def test_fit_validates_runtime_qrnn_vocabulary_before_model_init(
    monkeypatch, tmp_path
):
    import qllm.train.loop as train_loop

    corpus = tmp_path / "three-symbols.txt"
    corpus.write_text("abc" * 100, encoding="utf-8")
    cfg = ExperimentConfig(
        model=ModelConfig(arch="qrnn"),
        train=TrainConfig(steps=1, batch_size=1, seq_len=4),
        data=DataConfig(corpus_path=str(corpus)),
    )
    monkeypatch.setattr(
        train_loop,
        "build_model",
        lambda *_args, **_kwargs: pytest.fail("model initialization must not run"),
    )

    with pytest.raises(ValueError, match="QRNN vocabulary size must be a power of two"):
        train_loop.fit(cfg, verbose=False, out_dir=tmp_path)


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
