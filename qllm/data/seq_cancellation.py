"""Sequential cancellation task: does interference compound across positions?

The v0.11 head-only probe showed interference expresses single-step XOR
cancellation that positive mixtures cannot match at equal params. The
open question: does that gap SURVIVE and GROW in a sequence model, or is
it a toy-task artifact? Here the allowed next-token set at each position
depends on a running parity of recent tokens, so cancellation structure
recurs every step. Cancellation DENSITY (fraction of vocab in the
conflict tail) is a knob: density 0 -> pure union task (no interference
needed); higher density -> more probability mass governed by cancellation.

allowed(t) at step i:
  base group              always
  group_a (if parity_a)   parity_a = XOR of last L tokens' low bit
  group_b (if parity_b)   parity_b = XOR of last L tokens' (bit 1)
  conflict tail           iff parity_a XOR parity_b   (the cancellation)
Token drawn uniformly from the allowed set. Perplexity floor = mean log
|allowed set|; a head that cannot express the cancellation overcovers the
tail and pays excess CE that accrues over the sequence.
"""
from __future__ import annotations

import numpy as np


def seq_cancellation(
    n_sequences: int = 64,
    seq_len: int = 2048,
    vocab_size: int = 16,
    parity_window: int = 3,
    density: float = 0.25,
    seed: int = 0,
) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    V = vocab_size
    tail = max(int(round(density * V)), 1)
    rest = V - tail
    g = max(rest // 3, 1)
    base = np.arange(0, g)
    ga = np.arange(g, 2 * g)
    gb = np.arange(2 * g, 3 * g)
    conflict = np.arange(rest, V)  # the cancellation-governed tail

    def allowed(pa: int, pb: int) -> np.ndarray:
        s = [base]
        if pa:
            s.append(ga)
        if pb:
            s.append(gb)
        if pa ^ pb:
            s.append(conflict)
        return np.concatenate(s)

    out = np.empty((n_sequences, seq_len), dtype=np.int32)
    for s in range(n_sequences):
        toks = list(rng.integers(0, V, size=parity_window))
        while len(toks) < seq_len:
            recent = toks[-parity_window:]
            pa = int(sum(t & 1 for t in recent) % 2)
            pb = int(sum((t >> 1) & 1 for t in recent) % 2)
            toks.append(int(rng.choice(allowed(pa, pb))))
        out[s] = np.asarray(toks[:seq_len], dtype=np.int32)
    return out.reshape(-1), V
