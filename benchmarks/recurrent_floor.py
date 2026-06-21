#!/usr/bin/env python3
"""Parameters-to-floor benchmark: recurrent quantum vs classical models.

The v0.5 question: the monitored-Ising data has a known ideal floor
(~1.59 ppl) reachable only by tracking the quantum memory's belief
state. The QRNN's model class CONTAINS the generator (it can hit the
floor with ~tens of params in principle); classical recurrences must
approximate the belief filter in their hidden vector. We measure the
parameters-to-floor curve for both, plus the Markov-twin control.

Usage:
    python benchmarks/recurrent_floor.py --dataset ising --steps 1500
    python benchmarks/recurrent_floor.py --dataset markov --models qrnn gru16
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qllm.config import (  # noqa: E402
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    QuantumConfig,
    TrackingConfig,
    TrainConfig,
)
from qllm.train.loop import fit  # noqa: E402

FLOORS = {"ising": 1.59, "markov": 1.73}

MODELS = {
    "qrnn": ModelConfig(
        arch="qrnn",
        quantum=QuantumConfig(n_qubits=6, n_circuit_layers=3),
    ),
    "gru4": ModelConfig(arch="gru", rnn_hidden=4),
    "gru8": ModelConfig(arch="gru", rnn_hidden=8),
    "gru16": ModelConfig(arch="gru", rnn_hidden=16),
    "gru32": ModelConfig(arch="gru", rnn_hidden=32),
}


def data_cfg(dataset: str) -> DataConfig:
    base = DataConfig(
        kind="monitored_ising" if dataset == "ising" else "markov_control",
        gen_qubits=6,
        gen_measured=1,
        gen_sequences=64,
        gen_len=2048,
        gen_theta_x=0.6,
        gen_steps_per_token=2,
        gen_seed=0,
        markov_order=3,
    )
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ising", "markov"), default="ising")
    parser.add_argument(
        "--models", nargs="+", choices=list(MODELS), default=list(MODELS)
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    rows = []
    t_all = time.time()
    for name in args.models:
        for seed in args.seeds:
            cfg = ExperimentConfig(
                model=MODELS[name],
                train=TrainConfig(
                    seed=seed,
                    steps=args.steps,
                    batch_size=16,
                    seq_len=64,
                    lr=0.01 if name == "qrnn" else 0.003,
                    weight_decay=0.0,
                    grad_clip=1.0,
                    eval_every=max(args.steps // 3, 1),
                    eval_batches=16,
                ),
                data=data_cfg(args.dataset),
                tracking=TrackingConfig(
                    experiment="qllm-recurrent",
                    run_name=f"{args.dataset}-{name}-s{seed}",
                    log_quantum_diagnostics=False,
                ),
            )
            res = fit(cfg, verbose=False, out_dir=args.out)
            s = res["summary"]
            rows.append(
                {
                    "dataset": args.dataset,
                    "model": name,
                    "seed": seed,
                    "n_params": s["n_params"],
                    "val_ppl": s["val_ppl"],
                    "wall_seconds": s["wall_seconds"],
                }
            )
            print(
                f"{args.dataset}:{name:6s} seed={seed} "
                f"params={s['n_params']:6,d}  val_ppl={s['val_ppl']:.4f}  "
                f"({s['wall_seconds']:.0f}s)"
            )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"recurrent_floor_{args.dataset}.csv"
    exists = csv_path.exists()
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)

    floor = FLOORS[args.dataset]
    print(f"\nfloor for {args.dataset}: ~{floor} ppl")
    print(f"{'model':8s} {'params':>8s} {'ppl mean±std':>16s}  at-floor?")
    for name in args.models:
        ppls = [r["val_ppl"] for r in rows if r["model"] == name]
        n_params = next(r["n_params"] for r in rows if r["model"] == name)
        mean = statistics.mean(ppls)
        std = statistics.stdev(ppls) if len(ppls) > 1 else 0.0
        hit = "YES" if mean <= floor * 1.02 else "no"
        print(f"{name:8s} {n_params:8,d} {mean:10.4f}±{std:.4f}  {hit}")
    print(f"\ntotal wall: {time.time() - t_all:.0f}s -> {csv_path}")


if __name__ == "__main__":
    main()
