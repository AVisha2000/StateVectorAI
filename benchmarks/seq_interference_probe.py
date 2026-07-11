#!/usr/bin/env python3
"""Sequence-level interference probe with a FROZEN random body.

Tests whether the interference head's single-step expressivity edge
(v0.11) survives and compounds in a sequence model. A fixed random GRU
maps the token stream to per-position features (body NOT trained, so head
expressivity is the only variable); we train each head on the sequential
cancellation task and measure excess perplexity over the entropy floor as
cancellation DENSITY rises. If interference compounds, its gap over
param-matched mixtures should widen with density; if it's a toy artifact,
the gap stays flat/vanishes. Honest either way.

Usage: python benchmarks/seq_interference_probe.py --density 0.0 0.25 0.5 --seeds 0 1
"""
from __future__ import annotations

import argparse, math, sys
from pathlib import Path

import jax, jax.numpy as jnp, numpy as np, optax
from flax import linen as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qllm.config import DataConfig  # noqa: E402
from qllm.data.datasets import load_dataset_bundle  # noqa: E402
from qllm.data.text import sample_batch, train_val_split  # noqa: E402
from qllm.evaluation import conditional_entropy  # noqa: E402
from qllm.quantum.interference_head import (  # noqa: E402
    InterferenceHead, LinearHead, MixtureHead)
from qllm.resultsdb import ResultsDB  # noqa: E402


class FrozenBody(nn.Module):
    vocab: int
    hidden: int = 48

    @nn.compact
    def __call__(self, tokens):
        x = nn.Embed(self.vocab, self.hidden, name="embed")(tokens)
        return nn.RNN(nn.GRUCell(self.hidden), name="gru")(x)


HEADS = {
    "linear": (LinearHead, {}),
    "mixture-2": (MixtureHead, dict(n_hypotheses=2)),
    "mixture-4": (MixtureHead, dict(n_hypotheses=4)),
    "mixture-8": (MixtureHead, dict(n_hypotheses=8)),
    "interference-1": (InterferenceHead, dict(n_hypotheses=1)),
    "interference-2": (InterferenceHead, dict(n_hypotheses=2)),
    "interference-4": (InterferenceHead, dict(n_hypotheses=4)),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", default="seq-interference-v1")
    ap.add_argument("--density", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    ap.add_argument("--vocab", type=int, default=16)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seq-len", type=int, default=64)
    args = ap.parse_args()

    db = ResultsDB()
    for dens in args.density:
        dcfg = DataConfig(kind="seq_cancellation", ctx_observables=args.vocab,
                          ctx_context_size=args.window, seq_cancel_density=dens,
                          gen_sequences=64, gen_len=2048, gen_seed=0)
        bundle = load_dataset_bundle(dcfg)
        tok = bundle.tokenizer
        train_ids, val_ids = train_val_split(bundle.ids, 0.1)
        floor_bits = conditional_entropy(val_ids, tok.vocab_size, args.window)
        floor_ppl = float(2 ** floor_bits)
        dataset = f"seqcancel-d{dens:.2f}"

        for seed in args.seeds:
            # fixed random body params (seed-dependent, NEVER trained)
            body = FrozenBody(vocab=tok.vocab_size, hidden=args.hidden)
            bkey = jax.random.PRNGKey(1000 + seed)
            dummy = jnp.zeros((1, args.seq_len), jnp.int32)
            body_params = body.init(bkey, dummy)["params"]

            def featurize(batch):
                return body.apply({"params": body_params}, batch)

            for name, (Head, kw) in HEADS.items():
                if db.exists(args.suite, name, dataset, seed, args.steps):
                    continue
                head = Head(vocab_size=tok.vocab_size, **kw)
                hkey = jax.random.PRNGKey(seed)
                feat_dim = args.hidden
                hp = head.init(hkey, jnp.zeros((1, args.seq_len, feat_dim)))["params"]
                npar = sum(int(np.prod(v.shape))
                           for v in jax.tree_util.tree_leaves(hp))
                opt = optax.adam(3e-3); st = opt.init(hp)
                rng = np.random.default_rng(seed)

                @jax.jit
                def step(hp, st, batch):
                    feats = featurize(batch[:, :-1])
                    tgt = batch[:, 1:]
                    def loss(hp):
                        lg = head.apply({"params": hp}, feats)
                        return optax.softmax_cross_entropy_with_integer_labels(
                            lg, tgt).mean()
                    l, g = jax.value_and_grad(loss)(hp)
                    u, st = opt.update(g, st, hp)
                    return optax.apply_updates(hp, u), st, l

                bs = 16
                for _ in range(args.steps):
                    batch = sample_batch(
                        rng, train_ids, batch_size=bs, seq_len=args.seq_len
                    )
                    hp, st, _ = step(hp, st, jnp.asarray(batch))

                # eval CE on val
                vbatch = jnp.asarray(sample_batch(
                    rng, val_ids, batch_size=64, seq_len=args.seq_len
                ))
                feats = featurize(vbatch[:, :-1])
                lg = head.apply({"params": hp}, feats)
                ce = float(optax.softmax_cross_entropy_with_integer_labels(
                    lg, vbatch[:, 1:]).mean())
                ppl = float(np.exp(ce))
                db.record(suite=args.suite, variant=name, dataset=dataset,
                          seed=seed, steps=args.steps, n_params=npar,
                          val_loss=ce, val_ppl=ppl, val_bpc=ce / math.log(2),
                          wall_seconds=0.0)
                db.record_metrics(args.suite, name, dataset, seed,
                                  {"excess_ppl": ppl - floor_ppl,
                                   "floor_ppl": floor_ppl})
                print(f"d={dens:.2f} {name:16s} s{seed} params={npar:4d} "
                      f"ppl={ppl:.4f} floor={floor_ppl:.4f} "
                      f"excess={ppl - floor_ppl:+.4f}")
    # summary
    import statistics as st
    for dens in args.density:
        dataset = f"seqcancel-d{dens:.2f}"
        ex = {}
        for x in db.fetch_metrics(args.suite, dataset):
            if x["name"] == "excess_ppl":
                ex.setdefault(x["variant"], []).append(x["value"])
        if not ex:
            continue
        print(f"\n=== density {dens:.2f}: excess perplexity ===")
        for name in sorted(ex, key=lambda n: st.mean(ex[n])):
            sd = st.stdev(ex[name]) if len(ex[name]) > 1 else 0.0
            print(f"  {name:16s} excess_ppl={st.mean(ex[name]):+.4f} ± {sd:.4f}")


if __name__ == "__main__":
    main()
