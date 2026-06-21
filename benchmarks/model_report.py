#!/usr/bin/env python3
"""Full quality battery for one trained variant: ppl + memory gain vs
Markov floors + calibration + GENERATIVE fidelity (sample from the model,
compare statistics to held-out truth). Records to the metrics table.

Usage: python benchmarks/model_report.py --variant qrnn --dataset ising
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qllm.config import (DataConfig, ExperimentConfig, ModelConfig,  # noqa: E402
                         QuantumConfig, TrackingConfig, TrainConfig)
from qllm.data.datasets import load_dataset  # noqa: E402
from qllm.data.text import train_val_split  # noqa: E402
from qllm.evaluation import (calibration, generative_report,  # noqa: E402
                             markov_baseline_ppl)
from qllm.resultsdb import ResultsDB  # noqa: E402
from qllm.train.loop import fit  # noqa: E402

Q = QuantumConfig(n_qubits=4, n_circuit_layers=2, readout="zz")
BASE = dict(d_model=64, n_heads=4, n_blocks=2, d_ff=256, max_seq_len=128, quantum=Q)
VARIANTS = {
    "classical": ModelConfig(**BASE),
    "gru-64": ModelConfig(arch="gru", rnn_hidden=64),
    "q-ffn": ModelConfig(**BASE, ffn_type="quantum"),
    "qrnn": ModelConfig(arch="qrnn",
                        quantum=QuantumConfig(n_qubits=6, n_circuit_layers=3)),
}
DATA = {
    "text": DataConfig(),
    "ising": DataConfig(kind="monitored_ising", gen_qubits=6, gen_measured=1,
                        gen_sequences=64, gen_len=2048, gen_theta_x=0.6,
                        gen_steps_per_token=2, gen_seed=0, markov_order=3),
    "markov": DataConfig(kind="markov_control", gen_qubits=6, gen_measured=1,
                         gen_sequences=64, gen_len=2048, gen_theta_x=0.6,
                         gen_steps_per_token=2, gen_seed=0, markov_order=3),
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=list(VARIANTS), required=True)
    p.add_argument("--dataset", choices=list(DATA), default="ising")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gen-tokens", type=int, default=1200)
    args = p.parse_args()

    is_qrnn = args.variant == "qrnn"
    cfg = ExperimentConfig(
        model=VARIANTS[args.variant],
        train=TrainConfig(seed=args.seed, steps=args.steps, batch_size=16,
                          seq_len=64, lr=0.01 if is_qrnn else 1e-3,
                          weight_decay=0.0 if is_qrnn else 0.01,
                          grad_clip=1.0, eval_every=max(args.steps // 2, 1),
                          eval_batches=16),
        data=DATA[args.dataset],
        tracking=TrackingConfig(experiment="qllm-battery",
                                run_name=f"battery-{args.dataset}-"
                                         f"{args.variant}-s{args.seed}",
                                log_quantum_diagnostics=False))
    res = fit(cfg, verbose=False)
    s = res["summary"]
    model, params, tok = res["model"], res["state"].params, res["tokenizer"]
    ids, _ = load_dataset(cfg.data)
    train_ids, val_ids = train_val_split(ids, cfg.data.val_fraction)
    vocab = tok.vocab_size

    m = {"val_ppl": s["val_ppl"], "n_params": s["n_params"],
         "wall_seconds": s["wall_seconds"]}
    m["markov1_ppl"] = markov_baseline_ppl(train_ids, val_ids, vocab, 1)
    m["markov3_ppl"] = markov_baseline_ppl(train_ids, val_ids, vocab, 3)
    m["memory_gain"] = m["markov3_ppl"] - s["val_ppl"]
    m.update(calibration(model, params, val_ids, vocab,
                         seq_len=cfg.train.seq_len))
    k = 4 if vocab <= 4 else 2
    m.update(generative_report(model, params, val_ids, vocab,
                               n_tokens=args.gen_tokens,
                               context_len=cfg.train.seq_len, k=k,
                               seed=args.seed))

    db = ResultsDB()
    db.record(suite="battery-v1", variant=args.variant, dataset=args.dataset,
              seed=args.seed, steps=args.steps, n_params=s["n_params"],
              val_loss=s["val_loss"], val_ppl=s["val_ppl"],
              val_bpc=s["val_bpc"], wall_seconds=s["wall_seconds"])
    db.record_metrics("battery-v1", args.variant, args.dataset, args.seed,
                      {k_: v for k_, v in m.items() if isinstance(v, float)})

    print(f"\n=== battery: {args.variant} / {args.dataset} ===")
    for key, value in m.items():
        print(f"  {key:24s} {value:.4f}" if isinstance(value, float)
              else f"  {key:24s} {value}")


if __name__ == "__main__":
    main()
