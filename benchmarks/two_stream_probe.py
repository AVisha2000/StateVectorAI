#!/usr/bin/env python3
"""Does a quantum sentence encoder guide a classical word transformer better
than a param-matched CLASSICAL sentence encoder?

Strict causal design: identical word transformer; sentence stream is quantum,
classical (param-matched, ~0.7% larger = conservative), or none (baseline).
Every position receives only its cumulative left-prefix summary. Three
conditioning modes (film/token/bias). Run on BOTH classical text and
quantum-structured Ising data for contrast. The quantum-vs-classical-
encoder gap at matched params is the result; quantum-vs-none only shows if
pooling helps at all.

Usage: python benchmarks/two_stream_probe.py --dataset text --seeds 0 1
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qllm.config import (DataConfig, ExperimentConfig, ModelConfig,  # noqa: E402
                         QuantumConfig, TrackingConfig, TrainConfig,
                         to_flat_dict)
from qllm.dashboard import with_dashboard  # noqa: E402
from qllm.research_protocol import (  # noqa: E402
    HISTORICAL_TWO_STREAM_SUITES,
    TWO_STREAM_CAUSAL_PROTOCOL,
    TWO_STREAM_CAUSAL_SUITE,
)
from qllm.resultsdb import ResultsDB  # noqa: E402
from qllm.train.loop import fit  # noqa: E402

Q = QuantumConfig(n_qubits=4, n_circuit_layers=2, readout="z")
BODY = dict(d_model=32, n_heads=2, n_blocks=2, d_ff=64, max_seq_len=128, quantum=Q)


def variants():
    v = {"none": dict(arch="two_stream", encoder_kind="none", **BODY)}
    for enc in ("quantum", "classical"):
        for cond in ("film", "token", "bias"):
            v[f"{enc}-{cond}"] = dict(
                arch="two_stream", encoder_kind=enc, condition=cond,
                d_sent=8, sent_hidden=8, **BODY)
    return v


def data_cfg(dataset):
    if dataset == "text":
        return DataConfig()
    return DataConfig(kind="monitored_ising", gen_qubits=6, gen_measured=1,
                      gen_sequences=64, gen_len=2048, gen_theta_x=0.75,
                      gen_steps_per_token=2, gen_seed=0)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", default=TWO_STREAM_CAUSAL_SUITE)
    p.add_argument("--dataset", choices=("text", "ising"), default="text")
    p.add_argument("--variants", nargs="+", default=list(variants()))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--dashboard", action="store_true",
                  help="stream per-step curves to the dashboard DB")
    return p


def validate_suite(suite: str) -> None:
    if suite in HISTORICAL_TWO_STREAM_SUITES:
        raise ValueError(
            f"{suite} is an immutable full-window side-information suite; "
            f"use {TWO_STREAM_CAUSAL_SUITE} (or another new suite name) for "
            "causal-prefix reruns."
        )


def main() -> None:
    p = build_parser()
    args = p.parse_args()
    try:
        validate_suite(args.suite)
    except ValueError as exc:
        p.error(str(exc))

    db = ResultsDB()
    V = variants()
    for name in args.variants:
        for seed in args.seeds:
            if db.exists(args.suite, name, args.dataset, seed, args.steps):
                print(f"skip {name} s{seed}"); continue
            cfg = ExperimentConfig(
                model=ModelConfig(**V[name]),
                train=TrainConfig(seed=seed, steps=args.steps, batch_size=16,
                                  seq_len=64, lr=1e-3, weight_decay=0.01,
                                  grad_clip=1.0, eval_every=max(args.steps//2, 1),
                                  eval_batches=16),
                data=data_cfg(args.dataset),
                tracking=TrackingConfig(
                    experiment="qllm-two-stream",
                    run_name=f"{args.suite}-{args.dataset}-{name}-s{seed}",
                    log_quantum_diagnostics=False, log_grad_norms=False))
            if args.dashboard:
                cfg = with_dashboard(cfg, args.suite, name, args.dataset, seed)
            res = fit(cfg, verbose=False)
            s = res["summary"]
            db.record(suite=args.suite, variant=name, dataset=args.dataset,
                      seed=seed, steps=args.steps, n_params=s["n_params"],
                      val_loss=s["val_loss"], val_ppl=s["val_ppl"],
                      val_bpc=s["val_bpc"], wall_seconds=s["wall_seconds"],
                      config={
                          **to_flat_dict(cfg),
                          "research.two_stream_protocol": TWO_STREAM_CAUSAL_PROTOCOL,
                      })
            print(f"done {name:16s} s{seed} params={s['n_params']:6,d} "
                  f"ppl={s['val_ppl']:.4f} ({s['wall_seconds']:.0f}s)")

    import statistics as st
    rows = db.fetch(args.suite, args.dataset)
    by = {}
    for r in rows:
        by.setdefault(r["variant"], []).append(r["val_ppl"])
    print(f"\n=== {args.suite} / {args.dataset} ===")
    for name in sorted(by, key=lambda k: st.mean(by[k])):
        sd = st.stdev(by[name]) if len(by[name]) > 1 else 0.0
        params = next(r["n_params"] for r in rows if r["variant"] == name)
        print(f"  {name:16s} {params:6,d}  ppl={st.mean(by[name]):.4f} ± {sd:.4f}")


if __name__ == "__main__":
    main()
