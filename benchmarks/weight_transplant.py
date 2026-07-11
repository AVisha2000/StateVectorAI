#!/usr/bin/env python3
"""Transplant experiment v2 — with the controls that decide causality.

Variants (suite transplant-v2, all resumable via the DB):
  donor    : classical transformer (the weight source)
  q-warm   : quantum_linear FFN warm-started from contracted donor
             (SVD basis + compiled polar rotation + positive part)
  q-cold   : same architecture, random init
  lr-warm  : LOW-RANK CLASSICAL twin warm-started from the SAME SVD
             factors — if this matches q-warm, the win is low-rank +
             warm-start, not circuit structure
  lr-cold  : low-rank classical twin, random init
Zero-shot rows (steps=0) recorded for both warm variants.

Usage: python benchmarks/weight_transplant.py --seeds 0 1 --layers 20
"""
from __future__ import annotations

import argparse, copy, sys
from pathlib import Path

import jax, jax.numpy as jnp, numpy as np, optax
from flax.training.train_state import TrainState

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qllm.config import (DataConfig, ExperimentConfig, ModelConfig,  # noqa: E402
                         QuantumConfig, TrackingConfig, TrainConfig, to_flat_dict)
from qllm.data.datasets import load_dataset  # noqa: E402
from qllm.data.text import train_val_split  # noqa: E402
from qllm.models.model import build_model  # noqa: E402
from qllm.quantum.transplant import transplant_from_donor  # noqa: E402
from qllm.resultsdb import ResultsDB  # noqa: E402
from qllm.train.artifacts import RunOptions  # noqa: E402
from qllm.train.loop import evaluate, fit, make_eval_step  # noqa: E402

BASE = dict(d_model=64, n_heads=4, n_blocks=2, d_ff=256, max_seq_len=128)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", default="transplant-v2")
    p.add_argument("--variants", nargs="+",
                   default=["donor", "q-warm", "q-cold", "lr-warm", "lr-cold"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--qubits", type=int, default=4)
    p.add_argument("--layers", type=int, default=20)
    p.add_argument("--compile-steps", type=int, default=600)
    args = p.parse_args()

    db = ResultsDB()
    rank = 2**args.qubits
    qcfg = QuantumConfig(n_qubits=args.qubits, n_circuit_layers=args.layers)
    cfgs = {
        "donor": ModelConfig(**BASE),
        "q-warm": ModelConfig(**BASE, ffn_type="quantum_linear", quantum=qcfg),
        "q-cold": ModelConfig(**BASE, ffn_type="quantum_linear", quantum=qcfg),
        "lr-warm": ModelConfig(**BASE, ffn_type="lowrank", ffn_rank=rank),
        "lr-cold": ModelConfig(**BASE, ffn_type="lowrank", ffn_rank=rank),
    }

    for seed in args.seeds:
        train_cfg = TrainConfig(seed=seed, steps=args.steps, batch_size=16,
                                seq_len=64, lr=1e-3, weight_decay=0.01,
                                grad_clip=1.0, eval_every=100, eval_batches=8)

        def run(name, init_params=None, steps_key=None):
            steps_key = args.steps if steps_key is None else steps_key
            if db.exists(args.suite, name, "text", seed, steps_key):
                print(f"skip {name:10s} s{seed}")
                return None
            cfg = ExperimentConfig(
                model=cfgs[name.replace("-zs", "")], train=train_cfg,
                data=DataConfig(),
                tracking=TrackingConfig(
                    experiment="qllm-transplant",
                    run_name=f"{args.suite}-{name}-s{seed}",
                    log_quantum_diagnostics=False,
                    log_grad_norms=False))
            res = fit(
                cfg,
                verbose=False,
                init_params=init_params,
                run_options=RunOptions(
                    caller_metadata={
                        "warm_start_source": f"weight_transplant:{name}"
                    }
                ),
            )
            s = res["summary"]
            db.record(suite=args.suite, variant=name, dataset="text",
                      seed=seed, steps=steps_key, n_params=s["n_params"],
                      val_loss=s["val_loss"], val_ppl=s["val_ppl"],
                      val_bpc=s["val_bpc"], wall_seconds=s["wall_seconds"],
                      resources=s.get("resources"),
                      config=to_flat_dict(cfg), manifest=res["manifest"])
            print(f"done {name:10s} s{seed} params={s['n_params']:7,d} "
                  f"ppl={s['val_ppl']:.3f} ({s['wall_seconds']:.0f}s)")
            return res

        need_warm = any(v in args.variants for v in ("q-warm", "lr-warm"))
        donor_params = None
        if "donor" in args.variants or need_warm:
            donor_res = run("donor")
            if donor_res is None and need_warm:  # donor row exists; retrain for params
                cfg = ExperimentConfig(
                    model=cfgs["donor"], train=train_cfg, data=DataConfig(),
                    tracking=TrackingConfig(experiment="qllm-transplant",
                                            run_name=f"{args.suite}-donor-re-s{seed}",
                                            log_quantum_diagnostics=False,
                    log_grad_norms=False))
                donor_res = fit(cfg, verbose=False)
            donor_params = None if donor_res is None else donor_res["state"].params

        transplants = {}
        if need_warm and donor_params is not None:
            for b in range(BASE["n_blocks"]):
                t = transplant_from_donor(donor_params[f"block_{b}"]["ffn"],
                                          args.qubits, args.layers,
                                          compile_steps=args.compile_steps)
                transplants[b] = t
                print(f"  block_{b}: energy={t.retained_energy:.3f} "
                      f"fidelity={t.compile_fidelity:.3f}")
            db.record_metrics(args.suite, "transplant", "text", seed, {
                **{f"retained_energy_b{b}": t.retained_energy
                   for b, t in transplants.items()},
                **{f"compile_fidelity_b{b}": t.compile_fidelity
                   for b, t in transplants.items()},
            })

        def surgery(kind):
            model, _ = build_model(cfgs[kind], vocab_size=65)
            dummy = jnp.zeros((2, 8), dtype=jnp.int32)
            params = copy.deepcopy(jax.device_get(
                model.init(jax.random.PRNGKey(seed), dummy)["params"]))
            for key, value in donor_params.items():
                if not key.startswith("block_"):
                    params[key] = value
            for b, t in transplants.items():
                blk = f"block_{b}"
                for sub in ("ln_attn", "attn", "ln_ffn"):
                    params[blk][sub] = donor_params[blk][sub]
                B = np.asarray(t.basis, np.float32)
                C = (t.target_unitary @ t.positive_part).astype(np.float32)
                if kind == "q-warm":
                    params[blk]["ffn"] = {
                        "pre_proj": {"kernel": jnp.asarray(B)},
                        "circuit_weights": jnp.asarray(t.compiled_weights, jnp.float32),
                        "zz_phase": jnp.asarray(t.compiled_zz, jnp.float32),
                        "global_phase": jnp.asarray(t.compiled_phase, jnp.float32),
                        "post_proj": {"kernel": jnp.asarray(
                            (B @ np.asarray(t.positive_part, np.float32)).T)},
                    }
                else:
                    params[blk]["ffn"] = {
                        "pre_proj": {"kernel": jnp.asarray(B)},
                        "post_proj": {"kernel": jnp.asarray(C @ B.T)},
                    }
            return model, params

        for kind in ("q-warm", "lr-warm"):
            if kind not in args.variants or not transplants:
                continue
            model, warm = surgery(kind)
            if not db.exists(args.suite, f"{kind}-zs", "text", seed, 0):
                ids, _ = load_dataset(DataConfig())
                _, val_ids = train_val_split(ids, 0.1)
                state = TrainState.create(apply_fn=model.apply, params=warm,
                                          tx=optax.adamw(1e-3))
                zs = evaluate(make_eval_step(), state, val_ids, train_cfg)
                db.record(suite=args.suite, variant=f"{kind}-zs", dataset="text",
                          seed=seed, steps=0, n_params=0,
                          val_loss=zs["val_loss"], val_ppl=zs["val_ppl"],
                          val_bpc=zs["val_bpc"], wall_seconds=0.0)
                print(f"  {kind} ZERO-SHOT ppl={zs['val_ppl']:.3f}")
            run(kind, init_params=warm)

        for kind in ("q-cold", "lr-cold"):
            if kind in args.variants:
                run(kind)


if __name__ == "__main__":
    main()
