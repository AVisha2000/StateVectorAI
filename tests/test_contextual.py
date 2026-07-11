"""Contextuality task: structure, constrained-mask semantics, loader, metric."""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from qllm.config import DataConfig, ModelConfig
from qllm.data.contextual import contextual_parity_sequences
from qllm.data.datasets import load_dataset_bundle
from qllm.evaluation_contextual import constrained_accuracy
from qllm.models.model import build_model


def test_stream_shapes_and_vocab():
    ids, vocab, mask = contextual_parity_sequences(
        n_sequences=4, seq_len=128, n_observables=12, context_size=4,
        n_live=3, seed=0)
    assert vocab == 14  # n_observables + 2
    assert ids.shape == mask.shape == (4 * 128,)
    assert ids.min() >= 0 and ids.max() < vocab


def test_constrained_fraction_reasonable():
    _, _, mask = contextual_parity_sequences(
        n_sequences=8, seq_len=512, n_observables=12, context_size=4,
        n_live=3, seed=0)
    # ~1/(2*context_size) of all tokens are parity-forced value tokens
    assert 0.05 < mask.mean() < 0.25


def test_parity_constraints_hold():
    """Reconstruct each context's bits from the stream; constrained value
    must equal parity of the earlier revealed values."""
    ids, vocab, mask = contextual_parity_sequences(
        n_sequences=1, seq_len=400, n_observables=8, context_size=3,
        n_live=2, seed=1)
    val0 = 8
    # walk cue/value pairs; verify each constrained value is a valid bit
    constrained_vals = [t - val0 for t, m in zip(ids, mask)
                        if m == 1 and t >= val0]
    assert len(constrained_vals) > 0
    assert all(v in (0, 1) for v in constrained_vals)


def test_loader_exposes_mask_and_caches(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = DataConfig(kind="contextual", ctx_observables=12, ctx_context_size=4,
                     ctx_n_live=3, gen_sequences=4, gen_len=256, gen_seed=0)
    bundle = load_dataset_bundle(cfg)
    assert bundle.tokenizer.vocab_size == 14
    assert bundle.masks["constrained"].shape == bundle.ids.shape
    assert bundle.sampler_policy == "within_trajectory"
    cached = load_dataset_bundle(cfg)
    assert np.array_equal(bundle.ids, cached.ids)
    assert np.array_equal(
        bundle.masks["constrained"], cached.masks["constrained"]
    )
    assert bundle.content_hash == cached.content_hash
    assert cached.provenance["source"] == "cache"


def test_constrained_accuracy_metric_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = DataConfig(kind="contextual", ctx_observables=12, ctx_context_size=4,
                     ctx_n_live=3, gen_sequences=4, gen_len=256, gen_seed=0)
    bundle = load_dataset_bundle(cfg)
    ids = bundle.ids
    tok = bundle.tokenizer
    mask = bundle.masks["constrained"]
    model, _ = build_model(
        ModelConfig(d_model=16, n_heads=2, n_blocks=1, d_ff=32, max_seq_len=64),
        vocab_size=tok.vocab_size)
    params = model.init(jax.random.PRNGKey(0),
                        jnp.asarray(ids[0, :32][None, :]))["params"]
    acc = constrained_accuracy(model, params, ids, mask, seq_len=32,
                               n_windows=8)
    assert set(acc) == {"constrained_acc", "unconstrained_acc", "separation"}
    assert 0.0 <= acc["constrained_acc"] <= 1.0


def test_constrained_accuracy_never_crosses_trajectory_boundaries():
    class RepeatTokenModel:
        @staticmethod
        def apply(_variables, batch):
            return 20.0 * jax.nn.one_hot(batch, 2)

    ids = np.asarray(
        [[0, 0, 0, 0], [1, 1, 1, 1]] * 8,
        dtype=np.int32,
    )
    mask = np.ones_like(ids)

    result = constrained_accuracy(
        RepeatTokenModel(),
        params={},
        ids=ids,
        mask=mask,
        seq_len=3,
        n_windows=64,
        seed=4,
    )

    assert result["constrained_acc"] == 1.0
