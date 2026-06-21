#!/usr/bin/env python3
"""Head-only expressivity probe: coherent interference vs positive mixture.

NOVEL claim (quantum at the OUTPUT, not the memory): a coherent head
  p(t) = | sum_h c_h a_h(t) |^2                (complex, squared AFTER sum)
expresses multiplicative-with-CANCELLATION token constraints that a
positive-mixture head
  p(t) = sum_h w_h softmax_h(t)                 (w,softmax >= 0, only ADDS)
cannot represent at matched capacity, in a SINGLE output layer.

We isolate the head: inputs are purely LINEAR encodings of k binary context
features; the target allowed-token distribution applies an XOR-style
cancellation on conflict groups (allowed iff an ODD number of features
fire). The body cannot precompute conjunctions (there is no body), so the
head's expressivity is the only variable. We sweep the number of features k
(cancellation arity) and report excess cross-entropy over the exact entropy
floor; interference-K vs mixture-2K are parameter-matched.

Usage: python benchmarks/interference_head_probe.py --features 1 2 3 --steps 2000
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

import jax, jax.numpy as jnp, numpy as np, optax

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qllm.quantum.interference_head import (  # noqa: E402
    InterferenceHead, LinearHead, MixtureHead)
from qllm.resultsdb import ResultsDB  # noqa: E402


def make_data(k_features: int, vocab: int, n: int, seed: int):
    rng = np.random.default_rng(seed)
    feats = rng.integers(0, 2, size=(n, k_features))
    g = vocab // (k_features + 1)  # base + one conflict group per feature
    def dist(row):
        s = list(range(0, g))  # base always allowed
        for j in range(k_features):
            if row[j]:
                s += list(range((j + 1) * g, (j + 2) * g))
        # global conflict tail allowed iff ODD number of features fire
        if row.sum() % 2 == 1:
            s += list(range((k_features + 1) * g, vocab))
        p = np.zeros(vocab); p[s] = 1.0 / len(s); return p
    P = np.stack([dist(r) for r in feats])
    X = np.concatenate([feats, np.ones((n, 1))], axis=1).astype(np.float32)
    return jnp.asarray(X), jnp.asarray(P)


def fit_head(Head, X, P, vocab, steps, lr=5e-3, **kw):
    m = Head(vocab_size=vocab, **kw)
    p = m.init(jax.random.PRNGKey(0), X[:2])["params"]
    npar = sum(int(np.prod(v.shape)) for v in jax.tree_util.tree_leaves(p))
    opt = optax.adam(lr); st = opt.init(p)

    @jax.jit
    def step(p, st):
        def loss(p):
            lg = m.apply({"params": p}, X)
            logp = lg - jax.nn.logsumexp(lg, -1, keepdims=True)
            return -(P * logp).sum(-1).mean()
        l, g = jax.value_and_grad(loss)(p)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l

    l = None
    for _ in range(steps):
        p, st, l = step(p, st)
    return float(l), npar


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", default="interference-head-v1")
    ap.add_argument("--features", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--vocab", type=int, default=16)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    db = ResultsDB()
    heads = {
        "linear": (LinearHead, {}),
        "mixture-2": (MixtureHead, dict(n_hypotheses=2)),
        "mixture-4": (MixtureHead, dict(n_hypotheses=4)),
        "mixture-8": (MixtureHead, dict(n_hypotheses=8)),
        "interference-1": (InterferenceHead, dict(n_hypotheses=1)),
        "interference-2": (InterferenceHead, dict(n_hypotheses=2)),
        "interference-4": (InterferenceHead, dict(n_hypotheses=4)),
    }
    for k in args.features:
        dataset = f"xor-k{k}-v{args.vocab}"
        for seed in args.seeds:
            X, P = make_data(k, args.vocab, args.n, seed)
            floor = float(-(P * jnp.log(P + 1e-12)).sum(-1).mean())
            for name, (Head, kw) in heads.items():
                if db.exists(args.suite, name, dataset, seed, args.steps):
                    continue
                ce, npar = fit_head(Head, X, P, args.vocab, args.steps, **kw)
                db.record(suite=args.suite, variant=name, dataset=dataset,
                          seed=seed, steps=args.steps, n_params=npar,
                          val_loss=ce, val_ppl=float(np.exp(ce)),
                          val_bpc=ce / np.log(2), wall_seconds=0.0)
                db.record_metrics(args.suite, name, dataset, seed,
                                  {"excess_ce": ce - floor, "floor": floor})
        # summary
        import statistics as st
        rows = db.fetch(args.suite, dataset)
        mx = {(r["variant"]): [] for r in rows}
        met = {(x["variant"], x["seed"]): x["value"]
               for x in db.fetch_metrics(args.suite, dataset)
               if x["name"] == "excess_ce"}
        for r in rows:
            if (r["variant"], r["seed"]) in met:
                mx[r["variant"]].append(met[(r["variant"], r["seed"])])
        print(f"\n=== k={k} features (cancellation arity), vocab={args.vocab} ===")
        for name in sorted(mx, key=lambda n: st.mean(mx[n]) if mx[n] else 9):
            if not mx[name]:
                continue
            params = next(r["n_params"] for r in rows if r["variant"] == name)
            sd = st.stdev(mx[name]) if len(mx[name]) > 1 else 0.0
            print(f"  {name:16s} params={params:4d} "
                  f"excess_CE={st.mean(mx[name]):+.4f} ± {sd:.4f}")


if __name__ == "__main__":
    main()
