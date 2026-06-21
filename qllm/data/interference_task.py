"""Interference task: next-token sets defined by CANCELLATION, not union.

Designed to separate a coherent (interference) head from any positive-
mixture head IN A SINGLE OUTPUT LAYER. Each position carries two context
features read from the recent tokens, a in {0,1} and b in {0,1}. The set of
ALLOWED next tokens is governed by an exclusion rule:

  base allowed set A0 (always allowed)
  feature a, if on, ADDS group Ga
  feature b, if on, ADDS group Gb
  BUT the overlap that a and b would both contribute (Ga ∩ Gb, the
  "conflict" group C) is FORBIDDEN exactly when a==b==1.

So token t in C is allowed when exactly one of a,b is on, and forbidden when
both are on: allowed(t in C) = a XOR b. A positive mixture over hypotheses
keyed by a and by b can only ADD the contributions, so it necessarily allows
C whenever either is on — it cannot implement the XOR cancellation in one
layer. A coherent head can place opposite-sign amplitudes on C for the two
branches so they cancel when both fire.

The model sees a length-context of tokens; a and b are simple functions of
the previous two tokens (parity bits), so a sufficiently expressive BODY can
compute them — the bottleneck we test is purely the HEAD's ability to turn
(a,b) into the cancellation pattern. Token stream is i.i.d. over the allowed
set at each step given (a,b), so perplexity is governed by |allowed set|.
"""
from __future__ import annotations

import numpy as np


def interference_sequences(
    n_sequences: int = 64,
    seq_len: int = 2048,
    vocab_size: int = 16,
    seed: int = 0,
) -> tuple[np.ndarray, int]:
    """i.i.d.-per-step tokens drawn from an (a,b)-dependent allowed set with
    XOR cancellation on a conflict group. Returns (ids, vocab_size)."""
    rng = np.random.default_rng(seed)
    V = vocab_size
    # partition the vocab into base / Ga-only / Gb-only / conflict
    q = V // 4
    base = np.arange(0, q)
    ga_only = np.arange(q, 2 * q)
    gb_only = np.arange(2 * q, 3 * q)
    conflict = np.arange(3 * q, V)

    def allowed_set(a: int, b: int) -> np.ndarray:
        s = [base]
        if a:
            s.append(ga_only)
        if b:
            s.append(gb_only)
        # conflict group allowed iff exactly one of a,b is on (XOR)
        if a ^ b:
            s.append(conflict)
        return np.concatenate(s)

    all_ids = np.empty((n_sequences, seq_len), dtype=np.int32)
    for s in range(n_sequences):
        toks = list(rng.integers(0, V, size=2))  # seed two tokens
        while len(toks) < seq_len:
            a = int(toks[-1] % 2)
            b = int(toks[-2] % 2)
            allowed = allowed_set(a, b)
            toks.append(int(rng.choice(allowed)))
        all_ids[s] = np.asarray(toks[:seq_len], dtype=np.int32)
    return all_ids.reshape(-1), V
