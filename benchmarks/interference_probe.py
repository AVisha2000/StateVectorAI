#!/usr/bin/env python3
"""Does a coherent (interference) output head beat a param-matched positive
mixture head on a cancellation task? (Novel: quantum at the OUTPUT, not memory.)

Same transformer body; three heads:
  linear        single softmax (capacity floor)
  mixture-H     positive mixture of H softmaxes (classical, can only add)
  interference-K  coherent sum of K complex branches (can cancel)
Interference K and mixture 2K are parameter-matched (complex doubles params).
Task: interference XOR-cancellation set (data kind 'interference').

Usage: python benchmarks/interference_probe.py --steps 2000 --seeds 0 1 2
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qllm.config import (DataConfig, ExperimentConfig, ModelConfig,  # noqa: E402
                         TrackingConfig, TrainConfig)
from qllm.resultsdb import ResultsDB  # noqa: E402
from qllm.train.loop import fit  # noqa: E402

VOCAB = 16


def body(width):
    return dict(d_model=width, n_heads=2, n_blocks=1,
               d_ff=2 * width, max_seq_len=128)
VARIANTS = {
    "linear": dict(head_type="linear"),
    "mixture-2": dict(head_type="mixture", head_hypotheses=2),
    "mixture-4": dict(head_type="mixture", head_hypotheses=4),
    "mixture-8": dict(head_type="mixture", head_hypotheses=8),
    "interference-1": dict(head_type="interference", head_hypotheses=1),
    "interference-2": dict(head_type="interference", head_hypotheses=2),
    "interference-4": dict(head_type="interference", head_hypotheses=4),
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", default="interference-v1")
    p.add_argument("--dataset", default="interference",
                   choices=["interference", "text"])
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--width", type=int, default=64)
    args = p.parse_args()

    db = ResultsDB()
    if args.dataset == "interference":
        dcfg = DataConfig(kind="interference", ctx_observables=VOCAB,
                          gen_sequences=64, gen_len=2048, gen_seed=0)
    else:
        dcfg = DataConfig()

    for name in args.variants:
        for seed in args.seeds:
            ds_key = f"{args.dataset}-w{args.width}"
            if db.exists(args.suite, name, ds_key, seed, args.steps):
                print(f"skip {name} s{seed}")
                continue
            cfg = ExperimentConfig(
                model=ModelConfig(**body(args.width), **VARIANTS[name]),
                train=TrainConfig(seed=seed, steps=args.steps, batch_size=16,
                                  seq_len=64, lr=1e-3, weight_decay=0.01,
                                  grad_clip=1.0, eval_every=max(args.steps//2, 1),
                                  eval_batches=16),
                data=dcfg,
                tracking=TrackingConfig(
                    experiment="qllm-interference",
                    run_name=f"{args.suite}-{args.dataset}-{name}-s{seed}",
                    log_quantum_diagnostics=False, log_grad_norms=False))
            res = fit(cfg, verbose=False)
            s = res["summary"]
            db.record(suite=args.suite, variant=name, dataset=ds_key,
                      seed=seed, steps=args.steps, n_params=s["n_params"],
                      val_loss=s["val_loss"], val_ppl=s["val_ppl"],
                      val_bpc=s["val_bpc"], wall_seconds=s["wall_seconds"],
                      resources=s.get("resources"),
                      manifest=res["manifest"])
            print(f"done {name:16s} s{seed} params={s['n_params']:6,d} "
                  f"val_ppl={s['val_ppl']:.4f} ({s['wall_seconds']:.0f}s)")

    rows = db.fetch(args.suite, f"{args.dataset}-w{args.width}")
    if rows:
        import statistics as st
        by = {}
        for r in rows:
            by.setdefault(r["variant"], []).append(r["val_ppl"])
        print(f"\n=== {args.suite} / {args.dataset} ===")
        for name in sorted(by, key=lambda k: st.mean(by[k])):
            ppls = by[name]
            sd = st.stdev(ppls) if len(ppls) > 1 else 0.0
            params = next(r["n_params"] for r in rows if r["variant"] == name)
            print(f"{name:16s} {params:7,d}  {st.mean(ppls):.4f} ± {sd:.4f}")


if __name__ == "__main__":
    main()
