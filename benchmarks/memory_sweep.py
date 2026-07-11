#!/usr/bin/env python3
"""Separation flagship: memory-qubit sweep (designed for GPU, smoke-testable on CPU).

For m memory qubits, a classical model must track a belief filter of
dimension ~4^m while the QRNN carries m+1 qubits natively. This sweep
measures memory-gain (ppl below the order-3 Markov floor) per model as m
grows. The thesis predicts: fixed-size classical recurrences lose their
gain as 4^m outruns their hidden state; the QRNN keeps it. Exact
statevector simulation suffices to m~18 — no MPS needed.

GPU:   python benchmarks/memory_sweep.py --memory-qubits 6 8 10 12 --steps 2000 --seeds 0 1
Smoke: python benchmarks/memory_sweep.py --memory-qubits 8 --smoke
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qllm.config import (DataConfig, ExperimentConfig, ModelConfig,  # noqa: E402
                         QuantumConfig, TrackingConfig, TrainConfig)
from qllm.data.datasets import load_dataset  # noqa: E402
from qllm.data.text import train_val_split  # noqa: E402
from qllm.evaluation import markov_baseline_ppl  # noqa: E402
from qllm.resultsdb import ResultsDB  # noqa: E402
from qllm.train.loop import fit  # noqa: E402


def model_cfg(name: str, m: int) -> ModelConfig:
    if name == "qrnn":
        # Ising ansatz (exactly contains the generator); L = steps_per_token
        return ModelConfig(arch="qrnn",
                           quantum=QuantumConfig(n_qubits=m + 1,
                                                 n_circuit_layers=2,
                                                 ansatz="ising"))
    assert name.startswith("gru")
    return ModelConfig(arch="gru", rnn_hidden=int(name[3:]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", default="memory-sweep-v2")
    p.add_argument("--memory-qubits", type=int, nargs="+", default=[6, 8, 10, 12])
    p.add_argument("--models", nargs="+",
                   default=["planted", "qrnn", "gru8", "gru16", "gru32",
                            "gru64", "gru128"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--dashboard", action="store_true",
                  help="stream per-step curves to the dashboard DB")
    p.add_argument("--smoke", action="store_true",
                   help="tiny data/steps to validate the pipeline on CPU")
    args = p.parse_args()

    db = ResultsDB()
    seqs, length, steps = ((16, 512, 30) if args.smoke
                           else (64, 2048, args.steps))

    for m in args.memory_qubits:
        dataset = (f"ising-m{m}-tx{0.75}-spt{2}"
                   + ("-smoke" if args.smoke else ""))
        dcfg = DataConfig(kind="monitored_ising", gen_qubits=m + 1,
                          gen_measured=1, gen_sequences=seqs, gen_len=length,
                          gen_theta_x=0.75, gen_steps_per_token=2, gen_seed=0)
        ids, tok = load_dataset(dcfg)
        train_ids, val_ids = train_val_split(ids, dcfg.val_fraction)
        floor3 = markov_baseline_ppl(train_ids, val_ids, tok.vocab_size, 3)
        db.record_metrics(args.suite, "floor", dataset, 0,
                          {"markov3_ppl": floor3})
        print(f"[m={m}] markov-3 floor: {floor3:.4f}")

        if "planted" in args.models and not db.exists(
            args.suite, "planted", dataset, 0, 0
        ):
            # exact Bayes filter: the generator's own parameters, eval-only.
            # Sidesteps the QRNN's bad-basin landscape (see qrnn-landscape-v1)
            # and gives the true per-m achievable floor.
            import jax.numpy as jnp  # noqa: F401
            import optax
            from flax.training.train_state import TrainState

            from benchmarks.planted_qrnn import planted_params
            from qllm.quantum.recurrent import QRNNLM
            from qllm.train.loop import evaluate, make_eval_step

            model = QRNNLM(vocab_size=tok.vocab_size, n_qubits=m + 1,
                           n_layers=dcfg.gen_steps_per_token,
                           entangler="none")
            params = planted_params(m + 1, dcfg.gen_steps_per_token,
                                    dcfg.gen_theta_zz, dcfg.gen_theta_x,
                                    tok.vocab_size)
            tcfg = TrainConfig(seed=0, steps=0, batch_size=16, seq_len=64,
                               eval_batches=4 if args.smoke else 16)
            state = TrainState.create(apply_fn=model.apply, params=params,
                                      tx=optax.adamw(1e-3))
            ev = evaluate(make_eval_step(), state, val_ids, tcfg)
            db.record(suite=args.suite, variant="planted", dataset=dataset,
                      seed=0, steps=0, n_params=0, val_loss=ev["val_loss"],
                      val_ppl=ev["val_ppl"], val_bpc=ev["val_bpc"],
                      wall_seconds=0.0)
            db.record_metrics(args.suite, "planted", dataset, 0,
                              {"memory_gain": floor3 - ev["val_ppl"]})
            print(f"done planted m={m} ppl={ev['val_ppl']:.4f} "
                  f"gain={floor3 - ev['val_ppl']:+.4f}")

        for name in args.models:
            if name == "planted":
                continue
            for seed in args.seeds:
                if db.exists(args.suite, name, dataset, seed, steps):
                    print(f"skip {name} m={m} s{seed}")
                    continue
                is_q = name == "qrnn"
                cfg = ExperimentConfig(
                    model=model_cfg(name, m),
                    train=TrainConfig(seed=seed, steps=steps, batch_size=16,
                                      seq_len=64, lr=0.01 if is_q else 3e-3,
                                      weight_decay=0.0, grad_clip=1.0,
                                      eval_every=max(steps // 3, 1),
                                      eval_batches=4 if args.smoke else 16),
                    data=dcfg,
                    tracking=TrackingConfig(
                        experiment="qllm-memory-sweep",
                        run_name=f"{args.suite}-m{m}-{name}-s{seed}",
                        log_quantum_diagnostics=False, log_grad_norms=False))
                if args.dashboard:
                    from qllm.dashboard import with_dashboard
                    cfg = with_dashboard(cfg, args.suite, name, dataset, seed)
                res = fit(cfg, verbose=False)
                s = res["summary"]
                db.record(suite=args.suite, variant=name, dataset=dataset,
                          seed=seed, steps=steps, n_params=s["n_params"],
                          val_loss=s["val_loss"], val_ppl=s["val_ppl"],
                          val_bpc=s["val_bpc"], wall_seconds=s["wall_seconds"],
                          resources=s.get("resources"),
                          manifest=res["manifest"])
                db.record_metrics(args.suite, name, dataset, seed,
                                  {"memory_gain": floor3 - s["val_ppl"]})
                print(f"done {name:6s} m={m} s{seed} params={s['n_params']:6,d} "
                      f"ppl={s['val_ppl']:.4f} gain={floor3 - s['val_ppl']:+.4f} "
                      f"({s['wall_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
