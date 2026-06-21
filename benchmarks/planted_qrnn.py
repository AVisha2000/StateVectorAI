#!/usr/bin/env python3
"""Planted-solution diagnostic: set the QRNN's parameters to the data
generator's OWN values (Ising ansatz, no entangler) and evaluate.

If planted ppl ~= ideal floor: representability + Bayes filtering both
work, so any training gap is pure optimization. If not, the model class
or step semantics still mismatch the generator.

Usage: python benchmarks/planted_qrnn.py [--layers 2]
"""
from __future__ import annotations

import argparse, math, sys
from pathlib import Path

import jax, jax.numpy as jnp, numpy as np, optax
from flax.training.train_state import TrainState

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qllm.config import DataConfig, TrainConfig  # noqa: E402
from qllm.data.datasets import load_dataset  # noqa: E402
from qllm.data.text import train_val_split  # noqa: E402
from qllm.evaluation import markov_baseline_ppl  # noqa: E402
from qllm.quantum.recurrent import QRNNLM  # noqa: E402
from qllm.train.loop import evaluate, make_eval_step  # noqa: E402


def planted_params(n_qubits: int, n_layers: int, theta_zz: float,
                   theta_x: float, vocab: int) -> dict:
    rot = np.array([math.pi / 2, 2 * theta_x, -math.pi / 2], np.float32)
    return {
        "circuit_weights": jnp.asarray(
            np.tile(rot, (n_layers, n_qubits, 1))),
        "zz_phase": jnp.full((n_layers,), theta_zz, jnp.float32),
        "inject_angles": jnp.zeros((vocab, n_qubits), jnp.float32),
        "logit_scale": jnp.asarray(1.0, jnp.float32),
        "logit_bias": jnp.zeros((vocab,), jnp.float32),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layers", type=int, default=2)  # = steps_per_token
    args = p.parse_args()

    dcfg = DataConfig(kind="monitored_ising", gen_qubits=6, gen_measured=1,
                      gen_sequences=64, gen_len=2048, gen_theta_x=0.6,
                      gen_steps_per_token=2, gen_seed=0)
    ids, tok = load_dataset(dcfg)
    train_ids, val_ids = train_val_split(ids, dcfg.val_fraction)
    tcfg = TrainConfig(seed=0, steps=0, batch_size=16, seq_len=64,
                       eval_batches=32)

    model = QRNNLM(vocab_size=tok.vocab_size, n_qubits=dcfg.gen_qubits,
                   n_layers=args.layers, entangler="none")
    params = planted_params(dcfg.gen_qubits, args.layers, dcfg.gen_theta_zz,
                            dcfg.gen_theta_x, tok.vocab_size)
    state = TrainState.create(apply_fn=model.apply, params=params,
                              tx=optax.adamw(1e-3))
    ev = evaluate(make_eval_step(), state, val_ids, tcfg)
    m3 = markov_baseline_ppl(train_ids, val_ids, tok.vocab_size, 3)
    print(f"PLANTED QRNN (ising ansatz, L={args.layers}): "
          f"val_ppl={ev['val_ppl']:.4f}")
    print(f"references: markov-3 floor={m3:.4f}, "
          f"trained-model floor (v0.4) ~1.59, trained QRNN plateau ~1.637")


if __name__ == "__main__":
    main()
