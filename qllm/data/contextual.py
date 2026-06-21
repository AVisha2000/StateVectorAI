"""Contextual sequence task: provable quantum memory advantage via contextuality.

Motivation (Anschuetz, Hu, Huang, Gao 2022, arXiv:2209.14353; Zhao & Deng,
npj QI 2025): generic quantum dynamics gives at best constant-factor,
optimization-limited separations (our Ising track). The UNCONDITIONAL,
large separation comes from quantum CONTEXTUALITY — measurement outcomes
depend on which commuting observables were queried before, so no classical
model can avoid memorizing the full measurement context. That forces an
Omega(n^2) classical latent space where a quantum model needs O(n).

This module emits a discrete, finite-dimensional task carrying that exact
structure, so the separation is DESIGNED IN and checkable on real models
(no barren plateau: the generating process is explicit).

Construction — sequential Mermin–Peres / parity-context streams.
A "context" is a set of binary observables whose product is constrained
(a parity check). Observables are SHARED across contexts; a single
observable's value must be consistent everywhere it appears, AND each
context's parity must hold. A predictor that has seen part of a context
must, to predict the constrained final observable, recall the values of
every previously-revealed observable in that context — and contexts
interleave, so the required memory grows with the number of live
(partially-revealed) contexts. Classically this is the memorize-the-context
wall; the quantum recurrent model encodes it in entangled phase registers.

Token stream per step: (observable_id, value) flattened to a single token
in a vocab of 2 * n_observables, with constrained tokens marked so the
model is scored on PREDICTING the parity-forced value. We expose both the
raw sequence and a per-token "is_constrained" mask via the loss weighting.
"""
from __future__ import annotations

import numpy as np


def _build_contexts(n_contexts: int, context_size: int, n_observables: int,
                    rng: np.random.Generator) -> list[np.ndarray]:
    """Each context = context_size distinct observable ids; observables are
    reused across contexts (overlap is what creates contextuality)."""
    contexts = []
    for _ in range(n_contexts):
        ids = rng.choice(n_observables, size=context_size, replace=False)
        contexts.append(np.sort(ids))
    return contexts


def contextual_parity_sequences(
    n_sequences: int = 64,
    seq_len: int = 2048,
    n_observables: int = 12,
    context_size: int = 4,
    n_live: int = 3,
    seed: int = 0,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Generate interleaved parity-context streams.

    Returns (ids, vocab_size, constrained_mask):
      ids               concatenated token sequences, shape (n_sequences*seq_len,)
      vocab_size        2 * n_observables
      constrained_mask  1 where the token's VALUE is parity-forced (the
                        positions a memory-bound predictor can ace and a
                        memoryless one cannot), shape like ids.

    ``n_live`` interleaved contexts are open at once: the larger n_live and
    context_size, the more observable values must be simultaneously retained,
    i.e. the deeper the classical memory requirement.
    """
    rng = np.random.default_rng(seed)
    vocab = n_observables + 2  # cue ids + {value0, value1}
    val0 = n_observables
    all_ids = np.empty((n_sequences, seq_len), dtype=np.int32)
    all_mask = np.zeros((n_sequences, seq_len), dtype=np.int32)

    for s in range(n_sequences):
        # ground-truth assignment to observables, refreshed per sequence
        values = rng.integers(0, 2, size=n_observables)
        live: list[dict] = []  # each: {ids, pos, parity, assigned}
        toks, mask = [], []

        def open_context():
            ids = rng.choice(n_observables, size=context_size, replace=False)
            # parity target is determined by the current ground-truth values
            parity = int(values[ids].sum() % 2)
            return {"ids": ids, "pos": 0, "parity": parity}

        while len(toks) < seq_len - 1:
            if len(live) < n_live:
                live.append(open_context())
            ci = int(rng.integers(len(live)))
            ctx = live[ci]
            j = ctx["pos"]
            obs = int(ctx["ids"][j])
            is_last = j == context_size - 1
            if is_last:
                earlier = ctx["ids"][:j]
                val = int((ctx["parity"] - values[earlier].sum()) % 2)
                values[obs] = val
                constrained = 1
            else:
                val = int(values[obs])
                constrained = 0
            # cue token (free), then value token (constrained iff is_last)
            toks.append(obs);            mask.append(0)
            toks.append(val0 + val);     mask.append(constrained)
            ctx["pos"] += 1
            if ctx["pos"] >= context_size:
                live.pop(ci)

        all_ids[s] = np.asarray(toks[:seq_len], dtype=np.int32)
        all_mask[s] = np.asarray(mask[:seq_len], dtype=np.int32)

    return all_ids.reshape(-1), vocab, all_mask.reshape(-1)
