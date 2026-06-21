#!/usr/bin/env python3
"""QRNN optimization-landscape study (Ising ansatz, m=5 data).

The planted solution sits at ppl~1.527. Does gradient descent find it?
Variants: lr x depth grid from random init, plus planted-init fine-tune
(basin stability) and planted+noise inits (basin size).

Usage: python benchmarks/qrnn_landscape.py --variants L2-lr0.01 planted-ft
"""
from __future__ import annotations

import argparse, math, sys
from pathlib import Path

import jax.numpy as jnp, numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qllm.config import (DataConfig, ExperimentConfig, ModelConfig,  # noqa: E402
                         QuantumConfig, TrackingConfig, TrainConfig)
from qllm.resultsdb import ResultsDB  # noqa: E402
from qllm.train.loop import fit  # noqa: E402
from benchmarks.planted_qrnn import planted_params  # noqa: E402

DCFG = DataConfig(kind="monitored_ising", gen_qubits=6, gen_measured=1,
                  gen_sequences=64, gen_len=2048, gen_theta_x=0.6,
                  gen_steps_per_token=2, gen_seed=0)
GRID = {f"L{L}-lr{lr}": (L, lr)
        for L in (2, 3) for lr in (0.003, 0.01, 0.03)}
GRID["planted-ft"] = (2, 0.003)
GRID["planted-noise0.3-ft"] = (2, 0.003)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", default="qrnn-landscape-v1")
    p.add_argument("--variants", nargs="+", default=list(GRID))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--steps", type=int, default=1500)
    args = p.parse_args()

    db = ResultsDB()
    for name in args.variants:
        L, lr = GRID[name]
        for seed in args.seeds:
            if db.exists(args.suite, name, "ising", seed, args.steps):
                print(f"skip {name} s{seed}")
                continue
            cfg = ExperimentConfig(
                model=ModelConfig(arch="qrnn",
                                  quantum=QuantumConfig(n_qubits=6,
                                                        n_circuit_layers=L,
                                                        ansatz="ising")),
                train=TrainConfig(seed=seed, steps=args.steps, batch_size=16,
                                  seq_len=64, lr=lr, weight_decay=0.0,
                                  grad_clip=1.0,
                                  eval_every=max(args.steps // 3, 1),
                                  eval_batches=16),
                data=DCFG,
                tracking=TrackingConfig(
                    experiment="qllm-landscape",
                    run_name=f"{args.suite}-{name}-s{seed}",
                    log_quantum_diagnostics=False, log_grad_norms=False))
            init = None
            if name.startswith("planted"):
                init = planted_params(6, L, DCFG.gen_theta_zz,
                                      DCFG.gen_theta_x, 2)
                if "noise" in name:
                    scale = float(name.split("noise")[1].split("-")[0])
                    rng = np.random.default_rng(seed)
                    init = {k: jnp.asarray(np.asarray(v) + scale *
                            rng.normal(size=np.shape(v)).astype(np.float32))
                            for k, v in init.items()}
            res = fit(cfg, verbose=False, init_params=init)
            s = res["summary"]
            db.record(suite=args.suite, variant=name, dataset="ising",
                      seed=seed, steps=args.steps, n_params=s["n_params"],
                      val_loss=s["val_loss"], val_ppl=s["val_ppl"],
                      val_bpc=s["val_bpc"], wall_seconds=s["wall_seconds"])
            print(f"done {name:22s} s{seed} ppl={s['val_ppl']:.4f} "
                  f"({s['wall_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
