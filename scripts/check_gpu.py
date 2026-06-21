#!/usr/bin/env python3
"""Sanity-check the GPU setup before running anything from GPU_QUEUE.md.

Prints the JAX device, then times one training step each of: a classical
transformer, the QRNN (hand-rolled JAX quantum recurrence), and a
PennyLane QuantumCore circuit (the one path that goes through PennyLane
rather than pure jnp). If everything reports a CUDA/GPU device and the
PennyLane step doesn't silently fall back to CPU, you're set.

Usage: python scripts/check_gpu.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np


def main() -> None:
    print(f"JAX version: {jax.__version__}")
    devices = jax.devices()
    print(f"JAX devices: {devices}")
    is_gpu = any(d.platform != "cpu" for d in devices)
    print(f"-> {'GPU detected' if is_gpu else 'CPU ONLY -- check CUDA install (see GPU_SETUP.md step 2)'}")
    print()
    if not is_gpu:
        raise SystemExit(1)

    from qllm.config import (DataConfig, ExperimentConfig, ModelConfig,
                             QuantumConfig, TrackingConfig, TrainConfig)
    from qllm.train.loop import fit

    def time_fit_data(name, model_cfg, data_cfg, steps=30):
        cfg = ExperimentConfig(
            model=model_cfg,
            train=TrainConfig(seed=0, steps=steps, batch_size=16, seq_len=64,
                              eval_every=steps, eval_batches=4),
            data=data_cfg,
            tracking=TrackingConfig(enabled=False, log_quantum_diagnostics=False,
                                    log_grad_norms=False))
        t0 = time.time()
        res = fit(cfg, verbose=False)
        dt = time.time() - t0
        print(f"{name:30s} {steps} steps in {dt:.1f}s "
              f"({dt/steps*1000:.0f} ms/step)  ppl={res['summary']['val_ppl']:.3f}")

    def time_fit(name, model_cfg, steps=30):
        time_fit_data(name, model_cfg, DataConfig(), steps=steps)

    print("Classical transformer (no quantum ops):")
    time_fit("transformer", ModelConfig(d_model=64, n_heads=4, n_blocks=2,
                                        d_ff=256, max_seq_len=128))

    print("\nQRNN (hand-rolled jnp quantum recurrence -- rides JAX device):")
    print("  (needs binary-vocab data: vocab=2**k < n_qubits, using monitored_ising)")
    ising_data = DataConfig(kind="monitored_ising", gen_qubits=6, gen_measured=1,
                            gen_sequences=16, gen_len=512, gen_seed=0)
    time_fit_data("qrnn", ModelConfig(arch="qrnn",
                  quantum=QuantumConfig(n_qubits=6, n_circuit_layers=3)),
                  ising_data)

    print("\nVQC FFN (goes through PennyLane's default.qubit + jax interface):")
    time_fit("quantum-ffn", ModelConfig(
        d_model=64, n_heads=4, n_blocks=2, d_ff=256, max_seq_len=128,
        ffn_type="quantum",
        quantum=QuantumConfig(n_qubits=4, n_circuit_layers=2)))

    print("\nIf the PennyLane step's ms/step is far higher than the QRNN's "
          "relative to step 1 (transformer), PennyLane's default.qubit may "
          "not be picking up the GPU device -- this is the one path worth "
          "double-checking; the other two are pure jnp and will always "
          "follow jax.devices().")


if __name__ == "__main__":
    main()
