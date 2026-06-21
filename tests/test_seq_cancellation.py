"""Sequential cancellation task: structure, density knob, loader."""
from __future__ import annotations

import numpy as np

from qllm.config import DataConfig
from qllm.data.datasets import load_dataset
from qllm.data.seq_cancellation import seq_cancellation


def test_shapes_and_vocab():
    ids, V = seq_cancellation(n_sequences=4, seq_len=256, vocab_size=16,
                              parity_window=3, density=0.25, seed=0)
    assert V == 16
    assert ids.shape == (4 * 256,)
    assert ids.min() >= 0 and ids.max() < 16


def test_density_reduces_entropy_monotonically():
    import math
    from collections import Counter

    def H(d):
        ids, V = seq_cancellation(n_sequences=8, seq_len=1024, vocab_size=16,
                                  parity_window=3, density=d, seed=0)
        c = Counter(ids.tolist())
        return -sum(n / len(ids) * math.log2(n / len(ids)) for n in c.values())

    # higher density => more steps where the tail is forbidden => lower entropy
    assert H(0.5) < H(0.0) + 1e-6


def test_loader_and_cache():
    cfg = DataConfig(kind="seq_cancellation", ctx_observables=16,
                     ctx_context_size=3, seq_cancel_density=0.25,
                     gen_sequences=4, gen_len=256, gen_seed=0)
    ids, tok = load_dataset(cfg)
    assert tok.vocab_size == 16
    ids2, _ = load_dataset(cfg)
    assert np.array_equal(ids, ids2)
