#!/usr/bin/env python3
"""Per-m resonance search for the separation flagship.

For each memory size m, scan generator settings (theta_x x steps_per_token)
and measure the SEPARATION TARGET: gap = markov3_ppl - planted_ppl, where
planted = the exact quantum Bayes filter (no training involved). The
flagship needs this gap large at every m; v0.8 screens showed it collapses
at fixed settings (0.169 at m=5 -> 0.025 at m=8). Uses reduced-size
corpora for speed; rank by gap, confirm winners at full size later.

Usage: python benchmarks/resonance_search.py --memory-qubits 5 6 --theta-x 0.3 0.45 0.6 --spt 1 2
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

import optax
from flax.training.train_state import TrainState

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qllm.config import DataConfig, TrainConfig  # noqa: E402
from qllm.data.datasets import load_dataset_bundle  # noqa: E402
from qllm.data.text import train_val_split  # noqa: E402
from qllm.evaluation import markov_baseline_ppl  # noqa: E402
from qllm.quantum.recurrent import QRNNLM  # noqa: E402
from qllm.resultsdb import ResultsDB  # noqa: E402
from qllm.train.loop import evaluate, make_eval_step  # noqa: E402
from benchmarks.planted_qrnn import planted_params  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", default="resonance-v1")
    p.add_argument("--memory-qubits", type=int, nargs="+", required=True)
    p.add_argument("--theta-x", type=float, nargs="+",
                   default=[0.3, 0.45, 0.6])
    p.add_argument("--spt", type=int, nargs="+", default=[1, 2])
    p.add_argument("--sequences", type=int, default=32)
    p.add_argument("--length", type=int, default=1024)
    args = p.parse_args()

    db = ResultsDB()
    for m in args.memory_qubits:
        results = []
        for spt in args.spt:          # spt-major: planted compile reused
            for tx in args.theta_x:
                dataset = f"ising-m{m}-tx{tx}-spt{spt}-screen"
                prior = [r for r in db.fetch(args.suite, dataset)
                         if r["variant"] == "planted"]
                if prior:
                    mrow = {x["name"]: x["value"] for x in
                            db.fetch_metrics(args.suite, dataset)}
                    results.append((mrow.get("gap", 0), tx, spt,
                                    mrow.get("markov3", 0),
                                    prior[0]["val_ppl"]))
                    print(f"skip m={m} tx={tx} spt={spt}")
                    continue
                dcfg = DataConfig(kind="monitored_ising", gen_qubits=m + 1,
                                  gen_measured=1, gen_sequences=args.sequences,
                                  gen_len=args.length, gen_theta_x=tx,
                                  gen_steps_per_token=spt, gen_seed=0)
                bundle = load_dataset_bundle(dcfg)
                tok = bundle.tokenizer
                train_ids, val_ids = train_val_split(
                    bundle.ids, dcfg.val_fraction
                )
                m3 = markov_baseline_ppl(train_ids, val_ids,
                                         tok.vocab_size, 3)
                model = QRNNLM(vocab_size=tok.vocab_size, n_qubits=m + 1,
                               n_layers=spt, entangler="none")
                params = planted_params(m + 1, spt, dcfg.gen_theta_zz, tx,
                                        tok.vocab_size)
                tcfg = TrainConfig(seed=0, steps=0, batch_size=16,
                                   seq_len=64, eval_batches=16)
                state = TrainState.create(apply_fn=model.apply,
                                          params=params, tx=optax.adamw(1e-3))
                ev = evaluate(make_eval_step(), state, val_ids, tcfg)
                gap = m3 - ev["val_ppl"]
                db.record(suite=args.suite, variant="planted",
                          dataset=dataset, seed=0, steps=0, n_params=0,
                          val_loss=ev["val_loss"], val_ppl=ev["val_ppl"],
                          val_bpc=ev["val_bpc"], wall_seconds=0.0)
                db.record_metrics(args.suite, "planted", dataset, 0,
                                  {"markov3": m3, "gap": gap})
                results.append((gap, tx, spt, m3, ev["val_ppl"]))
                print(f"m={m} tx={tx:.2f} spt={spt}: markov3={m3:.4f} "
                      f"planted={ev['val_ppl']:.4f} GAP={gap:+.4f}")
        results.sort(reverse=True)
        print(f"\n== m={m} ranked ==")
        for gap, tx, spt, m3, pl in results:
            print(f"  tx={tx:.2f} spt={spt}: gap={gap:+.4f} "
                  f"(markov3={m3:.3f}, planted={pl:.3f})")


if __name__ == "__main__":
    main()
