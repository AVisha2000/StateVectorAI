"""Training loop, evaluation, and generation.

One pipeline for every model variant: the loop never knows whether the
blocks inside the model are classical or quantum — that is the payoff of
the plugin architecture.
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization, traverse_util
from flax.training.train_state import TrainState

from ..config import (
    ExperimentConfig,
    TrainConfig,
    to_flat_dict,
    two_stream_position_count,
    validate_config,
)
from ..data.datasets import load_dataset_bundle
from ..data.text import CharTokenizer, sample_batch, train_val_split
from ..models.model import build_model, uses_quantum
from ..tracking import ExperimentTracker, log_quantum_diagnostics


def make_train_step():
    @jax.jit
    def train_step(state: TrainState, batch: jnp.ndarray):
        inputs, targets = batch[:, :-1], batch[:, 1:]

        def loss_fn(params):
            logits = state.apply_fn({"params": params}, inputs)
            return optax.softmax_cross_entropy_with_integer_labels(
                logits, targets
            ).mean()

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss

    return train_step


def make_eval_step():
    @jax.jit
    def eval_step(state: TrainState, batch: jnp.ndarray):
        inputs, targets = batch[:, :-1], batch[:, 1:]
        logits = state.apply_fn({"params": state.params}, inputs)
        return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

    return eval_step


def make_grad_norm_step():
    """Per-group gradient norms: is the circuit actually receiving signal?

    The v0.2 trained~=frozen result demanded this diagnostic: it splits
    the gradient L2 norm into circuit_weights vs everything else, logged
    at every eval point so under-training of the quantum block is visible
    DURING runs, not post-hoc.
    """

    @jax.jit
    def grad_norm_step(state: TrainState, batch: jnp.ndarray):
        inputs, targets = batch[:, :-1], batch[:, 1:]

        def loss_fn(params):
            logits = state.apply_fn({"params": params}, inputs)
            return optax.softmax_cross_entropy_with_integer_labels(
                logits, targets
            ).mean()

        grads = jax.grad(loss_fn)(state.params)
        flat = traverse_util.flatten_dict(grads)
        circuit_sq = sum(
            jnp.sum(v**2) for k, v in flat.items() if "circuit_weights" in k
        )
        other_sq = sum(
            jnp.sum(v**2) for k, v in flat.items() if "circuit_weights" not in k
        )
        return jnp.sqrt(circuit_sq), jnp.sqrt(other_sq)

    return grad_norm_step


def evaluate(
    eval_step,
    state: TrainState,
    val_ids: np.ndarray,
    cfg: TrainConfig,
    seed_offset: int = 9999,
) -> dict[str, float]:
    """Average loss over a fixed (seeded) set of validation batches."""
    rng = np.random.default_rng(cfg.seed + seed_offset)
    losses = []
    for _ in range(cfg.eval_batches):
        batch = jnp.asarray(sample_batch(rng, val_ids, cfg.batch_size, cfg.seq_len))
        losses.append(eval_step(state, batch))
    loss = float(jnp.mean(jnp.stack(losses)))
    return {
        "val_loss": loss,
        "val_ppl": float(np.exp(loss)),
        "val_bpc": loss / float(np.log(2)),
    }


def count_params(params) -> int:
    return sum(int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(params))


def make_optimizer(train_cfg: TrainConfig, params, freeze_circuit: bool = False):
    """AdamW + grad clipping; optionally freeze quantum circuit weights.

    Freezing uses ``optax.multi_transform`` keyed on parameter paths: any
    leaf whose path contains ``circuit_weights`` receives zero updates, so
    a frozen circuit stays EXACTLY at its random initialization — the
    random-feature control arm of the 2x2 ablation.
    """
    base = optax.chain(
        optax.clip_by_global_norm(train_cfg.grad_clip),
        optax.adamw(train_cfg.lr, weight_decay=train_cfg.weight_decay),
    )
    if not freeze_circuit:
        return base
    flat = traverse_util.flatten_dict(params)
    labels = traverse_util.unflatten_dict(
        {k: ("frozen" if "circuit_weights" in k else "trainable") for k in flat}
    )
    return optax.multi_transform(
        {"trainable": base, "frozen": optax.set_to_zero()}, labels
    )


def fit(
    cfg: ExperimentConfig,
    verbose: bool = True,
    out_dir: str | Path = "results",
    init_params=None,
    should_cancel=None,
) -> dict:
    """Train a model end to end from an ExperimentConfig.

    Returns dict with the final TrainState, model, tokenizer, and a JSON-able
    summary (also written to ``results/<run_name>/summary.json``).
    """
    validation_errors = validate_config(cfg)
    if validation_errors:
        details = "\n- ".join(validation_errors)
        raise ValueError(f"Invalid experiment config:\n- {details}")

    dataset = load_dataset_bundle(cfg.data)
    tokenizer = dataset.tokenizer
    train_ids, val_ids = train_val_split(dataset.ids, cfg.data.val_fraction)

    runtime_cfg = dataclasses.replace(
        cfg,
        model=dataclasses.replace(cfg.model, vocab_size=tokenizer.vocab_size),
    )
    runtime_errors = validate_config(runtime_cfg)
    if runtime_errors:
        details = "\n- ".join(runtime_errors)
        raise ValueError(f"Invalid runtime experiment config:\n- {details}")

    model, model_cfg = build_model(
        runtime_cfg.model, vocab_size=tokenizer.vocab_size
    )

    rng_np = np.random.default_rng(cfg.train.seed)
    init_key = jax.random.PRNGKey(cfg.train.seed)
    sample = jnp.asarray(
        sample_batch(rng_np, train_ids, cfg.train.batch_size, cfg.train.seq_len)
    )
    params = (
        init_params
        if init_params is not None
        else model.init(init_key, sample[:, :-1])["params"]
    )
    n_params = count_params(params)

    freeze_circuit = uses_quantum(model_cfg) and not model_cfg.quantum.trainable
    tx = make_optimizer(cfg.train, params, freeze_circuit=freeze_circuit)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    run_name = cfg.tracking.run_name or (
        f"{model_cfg.attn_type}-attn_{model_cfg.ffn_type}-ffn"
    )
    # own-dashboard per-step logger (replaces MLflow when configured)
    dash = None
    dash_key = None
    dash_config = to_flat_dict(cfg)
    if model_cfg.arch == "two_stream":
        from ..research_protocol import TWO_STREAM_CAUSAL_PROTOCOL

        dash_config["research.two_stream_protocol"] = TWO_STREAM_CAUSAL_PROTOCOL
    if cfg.tracking.dashboard_db:
        from ..resultsdb import ResultsDB

        dash = ResultsDB(cfg.tracking.dashboard_db)
        dash_key = (
            f"{cfg.tracking.dashboard_suite}/{cfg.tracking.dashboard_variant}/"
            f"{cfg.tracking.dashboard_dataset}/{cfg.tracking.dashboard_seed}/"
            f"{cfg.train.steps}"
        )
        dash.start_run(
            run_key=dash_key, run_name=run_name,
            suite=cfg.tracking.dashboard_suite or "adhoc",
            variant=cfg.tracking.dashboard_variant or run_name,
            dataset=cfg.tracking.dashboard_dataset or cfg.data.kind,
            seed=cfg.tracking.dashboard_seed
            if cfg.tracking.dashboard_seed is not None else cfg.train.seed,
            total_steps=cfg.train.steps, config=dash_config)

    tracker = ExperimentTracker(cfg.tracking)
    tracker.log_params(
        to_flat_dict(cfg)
        | {"n_params": n_params, "vocab_size": tokenizer.vocab_size}
    )
    tracker.set_tags(
        {
            "qubits": model_cfg.quantum.n_qubits if uses_quantum(model_cfg) else 0,
            "ansatz": model_cfg.quantum.ansatz if uses_quantum(model_cfg) else "none",
            "backend": model_cfg.quantum.backend if uses_quantum(model_cfg) else "none",
        }
    )

    if verbose:
        print(
            f"[{run_name}] vocab={tokenizer.vocab_size} params={n_params:,} "
            f"attn={model_cfg.attn_type} ffn={model_cfg.ffn_type}"
            + (" [circuit FROZEN]" if freeze_circuit else "")
        )

    diagnostics = None
    if uses_quantum(model_cfg) and cfg.tracking.log_quantum_diagnostics:
        diagnostics = log_quantum_diagnostics(tracker, model_cfg.quantum)
        if verbose:
            pretty = {k: f"{v:.3e}" for k, v in diagnostics.items()}
            print(f"[{run_name}] quantum diagnostics: {pretty}")

    train_step = make_train_step()
    eval_step = make_eval_step()
    grad_norm_step = (
        make_grad_norm_step()
        if uses_quantum(model_cfg) and cfg.tracking.log_grad_norms
        else None
    )

    history: list[dict] = []
    t0 = time.time()
    cancelled = False
    for step in range(1, cfg.train.steps + 1):
        if should_cancel is not None and should_cancel():
            cancelled = True
            break
        batch = jnp.asarray(
            sample_batch(rng_np, train_ids, cfg.train.batch_size, cfg.train.seq_len)
        )
        state, loss = train_step(state, batch)

        if step == 1 and verbose:
            print(f"[{run_name}] first step (incl. jit) {time.time() - t0:.1f}s")
        if step % 10 == 0 or step == 1:
            tracker.log_metrics({"train_loss": float(loss)}, step=step)
            if dash is not None:
                dash.log_step(dash_key, step, {"train_loss": float(loss)},
                              train_loss=float(loss))
            if verbose and step % 50 == 0:
                print(f"[{run_name}] step {step:5d}  train_loss {float(loss):.4f}")

        if step % cfg.train.eval_every == 0 or step == cfg.train.steps:
            ev = evaluate(eval_step, state, val_ids, cfg.train)
            if grad_norm_step is not None:
                g_circ, g_other = grad_norm_step(state, batch)
                ev["grad_norm_circuit"] = float(g_circ)
                ev["grad_norm_classical"] = float(g_other)
                ev["grad_norm_ratio"] = float(g_circ) / (float(g_other) + 1e-12)
            tracker.log_metrics(ev, step=step)
            if dash is not None:
                dash.log_step(dash_key, step,
                              {k: float(v) for k, v in ev.items()},
                              val_ppl=float(ev.get("val_ppl", 0)) or None)
            history.append({"step": step, **ev})
            if verbose:
                extra = (
                    f"  g_circ/g_cls {ev['grad_norm_ratio']:.3e}"
                    if "grad_norm_ratio" in ev
                    else ""
                )
                print(
                    f"[{run_name}] step {step:5d}  val_loss {ev['val_loss']:.4f}  "
                    f"val_ppl {ev['val_ppl']:.2f}  val_bpc {ev['val_bpc']:.3f}"
                    + extra
                )

    wall = time.time() - t0
    final = history[-1] if history else {}
    if dash is not None:
        if final:
            dash.record(
                suite=cfg.tracking.dashboard_suite or "adhoc",
                variant=cfg.tracking.dashboard_variant or run_name,
                dataset=cfg.tracking.dashboard_dataset or cfg.data.kind,
                seed=cfg.tracking.dashboard_seed
                if cfg.tracking.dashboard_seed is not None else cfg.train.seed,
                steps=cfg.train.steps,
                n_params=n_params,
                val_loss=float(final.get("val_loss", 0.0)),
                val_ppl=float(final.get("val_ppl", 0.0)),
                val_bpc=float(final.get("val_bpc", 0.0)),
                wall_seconds=wall,
                config=dash_config,
            )
        dash.finish_run(dash_key, status="cancelled" if cancelled else "done")

    out = Path(out_dir) / run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "params.msgpack").write_bytes(serialization.to_bytes(state.params))
    summary = {
        "run_name": run_name,
        "n_params": n_params,
        "vocab_size": tokenizer.vocab_size,
        "wall_seconds": round(wall, 2),
        "steps": cfg.train.steps,
        "cancelled": cancelled,
        "quantum_diagnostics": diagnostics,
        "history": history,
        **final,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    tracker.log_metrics({"wall_seconds": wall})
    tracker.log_artifact(out / "summary.json")
    tracker.end()

    return {
        "state": state,
        "model": model,
        "model_cfg": model_cfg,
        "dataset": dataset,
        "tokenizer": tokenizer,
        "summary": summary,
    }


def generate(
    model,
    params,
    tokenizer: CharTokenizer,
    prompt: str = "\n",
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    seed: int = 0,
) -> str:
    """Autoregressive sampling with a fixed-shape window (single jit trace)."""
    context_len = model.cfg.max_seq_len
    if model.cfg.arch == "two_stream":
        positions_per_token = two_stream_position_count(
            1, model.cfg.encoder_kind, model.cfg.condition
        )
        context_len //= positions_per_token
        if context_len < 1:
            raise ValueError(
                "two-stream conditioning has no usable real-token capacity"
            )
    ids = [tokenizer.stoi[c] for c in prompt if c in tokenizer.stoi] or [0]
    key = jax.random.PRNGKey(seed)

    @jax.jit
    def step_fn(window, t_index, key):
        logits = model.apply({"params": params}, window[None])[0]
        key, sub = jax.random.split(key)
        next_id = jax.random.categorical(sub, logits[t_index] / temperature)
        return next_id, key

    for _ in range(max_new_tokens):
        window = ids[-context_len:]
        padded = np.zeros(context_len, dtype=np.int32)
        padded[: len(window)] = window
        next_id, key = step_fn(
            jnp.asarray(padded), jnp.asarray(len(window) - 1), key
        )
        ids.append(int(next_id))

    return tokenizer.decode(ids)
