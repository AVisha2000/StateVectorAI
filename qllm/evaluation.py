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


def _trajectory_rows(ids: np.ndarray, *, dtype) -> np.ndarray:
    """Return a 2-D view where every row is an independent trajectory."""
    array = np.asarray(ids, dtype=dtype)
    if array.ndim == 1:
        return array[None, :]
    if array.ndim == 2:
        return array
    raise ValueError("token ids must be a 1-D stream or 2-D trajectory matrix")


# ---------------------------------------------------------------------------
# Information-theoretic floors
# ---------------------------------------------------------------------------


def conditional_entropy(ids: np.ndarray, vocab: int, k: int) -> float:
    """Empirical H(next | previous k) in bits/token.

    A 2-D input represents independent trajectories. Contexts never cross
    rows, and the pooled estimate weights rows by their number of valid
    next-token observations.
    """
    joint: dict[tuple[int, int], int] = {}
    total = 0
    for row in _trajectory_rows(ids, dtype=np.int64):
        n = len(row) - k
        if n <= 0:
            continue
        ctx = np.zeros(n, dtype=np.int64)
        for j in range(k):
            ctx = ctx * vocab + row[j : n + j]
        for c, x in zip(ctx, row[k:], strict=True):
            key = (int(c), int(x))
            joint[key] = joint.get(key, 0) + 1
        total += n
    ctx_tot: dict[int, int] = {}
    for (c, _), n in joint.items():
        ctx_tot[c] = ctx_tot.get(c, 0) + n
    if total == 0:
        return 0.0
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
    """Perplexity of an order-k smoothed Markov model (train -> val).

    For trajectory matrices, fitting and scoring use only within-row
    transitions. Validation observations are pooled by count.
    """

    counts: dict[tuple, np.ndarray] = {}
    for row in _trajectory_rows(train_ids, dtype=np.int64):
        for i in range(max(len(row) - order, 0)):
            ctx = tuple(row[i : i + order])
            if ctx not in counts:
                counts[ctx] = np.full(vocab, smoothing)
            counts[ctx][row[i + order]] += 1

    uniform_logp = -math.log(vocab)
    nll = 0.0
    n = 0
    for row in _trajectory_rows(val_ids, dtype=np.int64):
        row_observations = max(len(row) - order, 0)
        for i in range(row_observations):
            ctx = tuple(row[i : i + order])
            c = counts.get(ctx)
            if c is None:
                nll -= uniform_logp
            else:
                nll -= math.log(c[row[i + order]] / c.sum())
        n += row_observations
    return math.exp(nll / max(n, 1))


def kgram_tv_distance(
    a: np.ndarray, b: np.ndarray, vocab: int, k: int
) -> float:
    """Total-variation distance between within-trajectory k-gram distributions."""

    def dist(ids):
        d = np.zeros(vocab**k, dtype=float)
        for row in _trajectory_rows(ids, dtype=np.int64):
            n = len(row) - k + 1
            if n <= 0:
                continue
            idx = np.zeros(n, dtype=np.int64)
            for j in range(k):
                idx = idx * vocab + row[j : n + j]
            d += np.bincount(idx, minlength=vocab**k)
        return d / d.sum()

    return float(0.5 * np.abs(dist(a) - dist(b)).sum())


def autocorr_distance(a: np.ndarray, b: np.ndarray, max_lag: int = 10) -> float:
    """L2 distance between within-trajectory autocorrelation profiles.

    Tokens are centered over the whole corpus, then every lag is pooled over
    its valid within-row pairs. No lagged product crosses a row boundary.
    """

    def prof(ids):
        x = _trajectory_rows(ids, dtype=np.float64)
        x = x - x.mean()
        v = (x * x).mean() + 1e-12
        values = []
        for lag in range(1, max_lag + 1):
            valid = x.shape[1] - lag
            values.append(
                (x[:, :valid] * x[:, lag:]).mean() / v
                if valid > 0
                else np.nan
            )
        return np.asarray(values)

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
    val_ids = np.asarray(val_ids, dtype=np.int64)
    rows = _trajectory_rows(val_ids, dtype=np.int64)
    rng = np.random.default_rng(seed)
    per = max(n_tokens // n_prompts, 8)
    chunks = []
    for j in range(n_prompts):
        if val_ids.ndim == 1:
            row = rows[0]
        else:
            row = rows[int(rng.integers(0, len(rows)))]
        start = int(rng.integers(0, len(row) - context_len - 1))
        prompt = row[start : start + context_len]
        chunks.append(
            sample_ids(model, params, vocab, prompt, n_tokens=per,
                       context_len=context_len, seed=seed + 7919 * j)
        )
    gen = np.stack(chunks)
    max_run = 0
    for row in gen:
        runs = np.diff(np.flatnonzero(
            np.concatenate(([True], np.diff(row) != 0, [True]))
        ))
        if len(runs):
            max_run = max(max_run, int(runs.max()))
    max_runlen_frac = float(max_run / gen.size) if gen.size else 1.0
    if val_ids.ndim == 1:
        ref = val_ids[: min(len(val_ids), 20_000)]
    elif val_ids.shape[1] > 20_000:
        ref = val_ids[:1, :20_000]
    else:
        n_rows = max(1, 20_000 // val_ids.shape[1])
        ref = val_ids[: min(len(val_ids), n_rows)]
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
