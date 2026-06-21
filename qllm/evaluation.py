"""Model-quality battery: what makes a trained sequence model "good".

Validation perplexity alone hides the interesting structure. This module
adds the measurements that matter for the quantum-vs-classical question:

- ``markov_baseline_ppl``: perplexity of an order-k smoothed Markov model
  fit on train — the short-memory floor. ``memory gain`` = how far below
  it a model gets = the long-range structure it actually exploits.
- ``sample_ids`` + ``generative_report``: sample from the trained model
  (seeded with a real validation prefix) and compare the GENERATED
  sequence's statistics to held-out truth: k-gram total-variation
  distance and conditional entropy. A model can have good ppl yet
  generate distributionally wrong sequences; this catches it.
- ``calibration``: expected calibration error + NLL of next-token
  predictions — are the probabilities honest?
- ``conditional_entropy``: H(next | previous k), the information-floor
  instrument used throughout the data work, now in the library.

Everything is model-agnostic (tokens -> logits contract), so the same
battery scores transformers, hybrids, the QRNN, and GRUs.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Information-theoretic floors
# ---------------------------------------------------------------------------


def conditional_entropy(ids: np.ndarray, vocab: int, k: int) -> float:
    """Empirical H(next | previous k) in bits/token."""
    ids = np.asarray(ids, dtype=np.int64)
    ctx = np.zeros(len(ids) - k, dtype=np.int64)
    for j in range(k):
        ctx = ctx * vocab + ids[j : len(ids) - k + j]
    nxt = ids[k:]
    joint: dict[tuple[int, int], int] = {}
    for c, x in zip(ctx, nxt):
        key = (int(c), int(x))
        joint[key] = joint.get(key, 0) + 1
    ctx_tot: dict[int, int] = {}
    for (c, _), n in joint.items():
        ctx_tot[c] = ctx_tot.get(c, 0) + n
    total = len(nxt)
    return -sum(
        n / total * math.log2(n / ctx_tot[c]) for (c, _), n in joint.items()
    )


def markov_baseline_ppl(
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    vocab: int,
    order: int,
    smoothing: float = 0.5,
) -> float:
    """Perplexity of an order-k smoothed Markov model (train -> val)."""
    train_ids = np.asarray(train_ids, dtype=np.int64)
    val_ids = np.asarray(val_ids, dtype=np.int64)

    counts: dict[tuple, np.ndarray] = {}
    for i in range(len(train_ids) - order):
        ctx = tuple(train_ids[i : i + order])
        if ctx not in counts:
            counts[ctx] = np.full(vocab, smoothing)
        counts[ctx][train_ids[i + order]] += 1

    uniform_logp = -math.log(vocab)
    nll = 0.0
    n = len(val_ids) - order
    for i in range(n):
        ctx = tuple(val_ids[i : i + order])
        c = counts.get(ctx)
        if c is None:
            nll -= uniform_logp
        else:
            nll -= math.log(c[val_ids[i + order]] / c.sum())
    return math.exp(nll / max(n, 1))


def kgram_tv_distance(
    a: np.ndarray, b: np.ndarray, vocab: int, k: int
) -> float:
    """Total-variation distance between k-gram distributions of two corpora."""

    def dist(ids):
        ids = np.asarray(ids, dtype=np.int64)
        idx = np.zeros(len(ids) - k + 1, dtype=np.int64)
        for j in range(k):
            idx = idx * vocab + ids[j : len(ids) - k + 1 + j]
        d = np.bincount(idx, minlength=vocab**k).astype(float)
        return d / d.sum()

    return float(0.5 * np.abs(dist(a) - dist(b)).sum())


def autocorr_distance(a: np.ndarray, b: np.ndarray, max_lag: int = 10) -> float:
    """L2 distance between autocorrelation profiles of two token sequences
    (tokens centered; lags 1..max_lag). Captures temporal structure that
    k-gram histograms miss."""

    def prof(ids):
        x = np.asarray(ids, dtype=np.float64)
        x = x - x.mean()
        v = (x * x).mean() + 1e-12
        return np.array(
            [(x[: -lag] * x[lag:]).mean() / v for lag in range(1, max_lag + 1)]
        )

    return float(np.linalg.norm(prof(a) - prof(b)))


# ---------------------------------------------------------------------------
# Generative output inspection
# ---------------------------------------------------------------------------


def sample_ids(
    model,
    params,
    vocab: int,
    prompt_ids: np.ndarray,
    n_tokens: int = 1500,
    context_len: int = 64,
    temperature: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Autoregressively sample token ids (fixed-shape window, single trace)."""
    ids = [int(i) for i in np.asarray(prompt_ids, dtype=np.int64)]
    key = jax.random.PRNGKey(seed)

    @jax.jit
    def step_fn(window, t_index, key):
        logits = model.apply({"params": params}, window[None])[0]
        key, sub = jax.random.split(key)
        nxt = jax.random.categorical(sub, logits[t_index] / temperature)
        return nxt, key

    for _ in range(n_tokens):
        window = ids[-context_len:]
        padded = np.zeros(context_len, dtype=np.int32)
        padded[: len(window)] = window
        nxt, key = step_fn(jnp.asarray(padded), jnp.asarray(len(window) - 1), key)
        ids.append(int(nxt))
    return np.asarray(ids[len(prompt_ids):], dtype=np.int32)


def generative_report(
    model,
    params,
    val_ids: np.ndarray,
    vocab: int,
    n_tokens: int = 1500,
    context_len: int = 64,
    k: int = 4,
    seed: int = 0,
    n_prompts: int = 24,
) -> dict[str, float]:
    """Sample from the model and compare statistics against held-out truth.

    Uses MANY short rollouts from distinct real-data prompts rather than
    one long rollout: a single trajectory can absorb into a degenerate
    basin (exposure bias) and then measures the basin, not the model.
    The absorption tendency itself is reported as ``gen_max_runlen_frac``.
    """
    rng = np.random.default_rng(seed)
    per = max(n_tokens // n_prompts, 8)
    chunks = []
    for j in range(n_prompts):
        start = int(rng.integers(0, len(val_ids) - context_len - 1))
        prompt = val_ids[start : start + context_len]
        chunks.append(
            sample_ids(model, params, vocab, prompt, n_tokens=per,
                       context_len=context_len, seed=seed + 7919 * j)
        )
    gen = np.concatenate(chunks)
    runs = np.diff(np.flatnonzero(
        np.concatenate(([True], np.diff(gen) != 0, [True]))
    ))
    max_runlen_frac = float(runs.max() / len(gen)) if len(runs) else 1.0
    ref = val_ids[: min(len(val_ids), 20_000)]
    out = {
        f"gen_tv_{k}gram": kgram_tv_distance(gen, ref, vocab, k),
        "gen_cond_entropy_k3": conditional_entropy(gen, vocab, 3),
        "true_cond_entropy_k3": conditional_entropy(ref, vocab, 3),
    }
    out["gen_entropy_gap_k3"] = (
        out["gen_cond_entropy_k3"] - out["true_cond_entropy_k3"]
    )
    out["gen_max_runlen_frac"] = max_runlen_frac
    out["gen_autocorr_dist"] = autocorr_distance(gen, ref)
    return out


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibration(
    model,
    params,
    ids: np.ndarray,
    vocab: int,
    batch_size: int = 16,
    seq_len: int = 64,
    n_batches: int = 8,
    n_bins: int = 10,
    seed: int = 1234,
) -> dict[str, float]:
    """Expected calibration error + NLL of next-token predictions."""
    from .data.text import sample_batch

    rng = np.random.default_rng(seed)
    confs, hits, nlls = [], [], []

    @jax.jit
    def forward(batch):
        return model.apply({"params": params}, batch[:, :-1])

    for _ in range(n_batches):
        batch = jnp.asarray(sample_batch(rng, ids, batch_size, seq_len))
        logits = forward(batch)
        probs = jax.nn.softmax(logits, axis=-1)
        targets = np.asarray(batch[:, 1:])
        p = np.asarray(probs).reshape(-1, vocab)
        t = targets.reshape(-1)
        conf = p.max(axis=-1)
        pred = p.argmax(axis=-1)
        confs.append(conf)
        hits.append((pred == t).astype(float))
        nlls.append(-np.log(p[np.arange(len(t)), t] + 1e-12))

    conf = np.concatenate(confs)
    hit = np.concatenate(hits)
    nll = float(np.concatenate(nlls).mean())

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(hit[mask].mean() - conf[mask].mean())
    return {"ece": float(ece), "nll": nll, "accuracy": float(hit.mean())}
