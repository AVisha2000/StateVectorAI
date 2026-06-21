#!/usr/bin/env python3
"""Contextuality memory-wall sweep.

Theory (arXiv:2209.14353): expressing interleaved parity contexts needs
classical latent space growing with the live-context depth, while a
contextual quantum recurrence needs O(n). We hold the QUANTUM model fixed
and grow the classical GRU, measuring constrained-position accuracy as we
increase task contextuality (n_live x context_size). The prediction: small
GRUs plateau below ceiling on constrained tokens and need ever more hidden
units as contextuality rises; the quantum recurrent cell tracks it cheaply.

Usage: python benchmarks/contextual_sweep.py --live 2 3 4 --models gru8 gru16 gru32 gru64 qrnn
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import qllm.data.datasets as D  # noqa: E402
from qllm.config import (DataConfig, ExperimentConfig, ModelConfig,  # noqa: E402
                         QuantumConfig, TrackingConfig, TrainConfig)
from qllm.data.datasets import load_dataset  # noqa: E402
from qllm.evaluation_contextual import constrained_accuracy  # noqa: E402
from qllm.resultsdb import ResultsDB  # noqa: E402
from qllm.train.loop import fit  # noqa: E402


def model_cfg(name: str, vocab: int) -> ModelConfig:
    if name.startswith("cqrnn"):
        nq = int(name[5:]) if len(name) > 5 else 4
        return ModelConfig(arch="contextual_qrnn",
                           quantum=QuantumConfig(n_qubits=nq, n_circuit_layers=2))
    if name.startswith("rqrnn"):
        nq = int(name[5:]) if len(name) > 5 else 5
        return ModelConfig(arch="routed_contextual",
                           quantum=QuantumConfig(n_qubits=nq, n_circuit_layers=1))
    return ModelConfig(arch="gru", rnn_hidden=int(name[3:]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", default="contextual-v1")
    p.add_argument("--live", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--context-size", type=int, default=4)
    p.add_argument("--observables", type=int, default=12)
    p.add_argument("--models", nargs="+",
                   default=["gru8", "gru16", "gru32", "gru64"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--steps", type=int, default=1500)
    args = p.parse_args()

    db = ResultsDB()
    for live in args.live:
        dcfg = DataConfig(kind="contextual", ctx_observables=args.observables,
                          ctx_context_size=args.context_size, ctx_n_live=live,
                          gen_sequences=64, gen_len=2048, gen_seed=0)
        ids, tok = load_dataset(dcfg)
        mask = D._LAST_CTX_MASK
        dataset = f"ctx-live{live}-cs{args.context_size}"
        for name in args.models:
            for seed in args.seeds:
                if db.exists(args.suite, name, dataset, seed, args.steps):
                    print(f"skip {name} live={live} s{seed}")
                    continue
                is_q = name == "qrnn"
                cfg = ExperimentConfig(
                    model=model_cfg(name, tok.vocab_size),
                    train=TrainConfig(seed=seed, steps=args.steps,
                                      batch_size=16, seq_len=64,
                                      lr=0.01 if is_q else 3e-3,
                                      weight_decay=0.0, grad_clip=1.0,
                                      eval_every=max(args.steps // 2, 1),
                                      eval_batches=8),
                    data=dcfg,
                    tracking=TrackingConfig(
                        experiment="qllm-contextual",
                        run_name=f"{args.suite}-{dataset}-{name}-s{seed}",
                        log_quantum_diagnostics=False, log_grad_norms=False))
                res = fit(cfg, verbose=False)
                s = res["summary"]
                acc = constrained_accuracy(res["model"], res["state"].params,
                                           ids, mask, seq_len=64)
                db.record(suite=args.suite, variant=name, dataset=dataset,
                          seed=seed, steps=args.steps, n_params=s["n_params"],
                          val_loss=s["val_loss"], val_ppl=s["val_ppl"],
                          val_bpc=s["val_bpc"], wall_seconds=s["wall_seconds"])
                db.record_metrics(args.suite, name, dataset, seed, acc)
                print(f"done {name:6s} live={live} s{seed} "
                      f"params={s['n_params']:6,d} "
                      f"con_acc={acc['constrained_acc']:.3f} "
                      f"unc_acc={acc['unconstrained_acc']:.3f} "
                      f"({s['wall_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
