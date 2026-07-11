"""Ablation infrastructure tests: circuit freezing + parameter matching."""
from __future__ import annotations

import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
from flax import traverse_util

from benchmarks.ablation import build_ablation_report, paired_ablation_analysis
from qllm.config import ModelConfig, QuantumConfig, TrainConfig
from qllm.models.model import (
    build_model,
    count_model_params,
    matched_classical_d_ff,
)
from qllm.research_protocol import normalize_seed_axes
from qllm.train.loop import fit, make_optimizer

QMODEL = ModelConfig(
    d_model=16,
    n_heads=2,
    n_blocks=1,
    d_ff=32,
    max_seq_len=16,
    ffn_type="quantum",
    quantum=QuantumConfig(n_qubits=2, n_circuit_layers=1),
)


def _params():
    model, _ = build_model(QMODEL, vocab_size=7)
    tokens = jnp.zeros((1, 6), dtype=jnp.int32)
    return model.init(jax.random.PRNGKey(0), tokens)["params"]


def _circuit_update_norms(freeze: bool):
    params = _params()
    tx = make_optimizer(TrainConfig(), params, freeze_circuit=freeze)
    opt_state = tx.init(params)
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    updates, _ = tx.update(grads, opt_state, params)
    circuit, other = [], []
    for key, value in traverse_util.flatten_dict(updates).items():
        target = circuit if "circuit_weights" in key else other
        target.append(float(jnp.abs(value).max()))
    return circuit, other


def test_optimizer_freezes_circuit_weights():
    circuit, other = _circuit_update_norms(freeze=True)
    assert circuit and other  # both groups present in the tree
    assert all(c == 0.0 for c in circuit), "frozen circuit received updates"
    assert any(o > 0.0 for o in other), "trainable params received no updates"


def test_optimizer_updates_circuit_weights_when_trainable():
    circuit, _ = _circuit_update_norms(freeze=False)
    assert all(c > 0.0 for c in circuit)


def test_matched_d_ff_within_tolerance():
    target = count_model_params(QMODEL, vocab_size=7)
    d_ff = matched_classical_d_ff(QMODEL, vocab_size=7)
    twin = dataclasses.replace(QMODEL, ffn_type="classical", d_ff=d_ff)
    got = count_model_params(twin, vocab_size=7)
    assert abs(got - target) / target < 0.05


def test_frozen_flag_plumbed_through_fit(tiny_quantum_cfg, tmp_path):
    """Same seed => identical init; trained circuit must move, frozen not.

    Combined with test_optimizer_freezes_circuit_weights this verifies the
    frozen run stays exactly at its random initialization.
    """
    frozen_cfg = dataclasses.replace(
        tiny_quantum_cfg,
        model=dataclasses.replace(
            tiny_quantum_cfg.model,
            quantum=dataclasses.replace(
                tiny_quantum_cfg.model.quantum, trainable=False
            ),
        ),
    )
    trained = fit(tiny_quantum_cfg, verbose=False, out_dir=tmp_path / "t")
    frozen = fit(frozen_cfg, verbose=False, out_dir=tmp_path / "f")

    def circuit_weights(res):
        return res["state"].params["block_0"]["ffn"]["QuantumCore_0"][
            "circuit_weights"
        ]

    cw_trained = np.asarray(circuit_weights(trained))
    cw_frozen = np.asarray(circuit_weights(frozen))
    assert not np.allclose(cw_trained, cw_frozen)
    assert np.isfinite(frozen["summary"]["val_loss"])


def _ablation_row(variant: str, seed: int, val_ppl: float, n_params: int) -> dict:
    axes = normalize_seed_axes(
        seed, circuit_applicable=variant.startswith("quantum-")
    )
    return {
        "variant": variant,
        "seed": seed,
        "n_params": n_params,
        "val_loss": float(np.log(val_ppl)),
        "val_ppl": val_ppl,
        "wall_seconds": 1.0,
        "grad_norm_ratio": None,
        "protocol_version": 2,
        "protocol_hash": "shared-protocol",
        "data_config_hash": "a" * 64,
        "data_kind": "text",
        "data_gen_seed": 0,
        "steps": 300,
        "eval_every": 50,
        "device_target": "cpu",
        "batch_size": 4,
        "seq_len": 16,
        "lr": 0.001,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "eval_batches": 2,
        "seed_axes_json": json.dumps(axes, sort_keys=True),
    }


def test_paired_ablation_analysis_matches_scores_by_seed():
    rows = [
        _ablation_row("quantum-trained", 1, 20.0, 100),
        _ablation_row("quantum-frozen", 0, 7.0, 100),
        _ablation_row("quantum-trained", 0, 5.0, 100),
        _ablation_row("quantum-frozen", 1, 21.0, 100),
    ]

    analysis = paired_ablation_analysis(
        rows,
        "quantum-trained",
        "quantum-frozen",
        equivalence_margin=0.1,
        smallest_useful_effect=0.1,
    )

    assert analysis["paired_seeds"] == [0, 1]
    assert analysis["paired_stats"]["mean_improvement"] == 1.5
    assert analysis["fairness"]["same_seed"] is True
    assert analysis["fairness"]["valid"] is True


def test_three_seed_ablation_report_stays_pilot_only():
    values = {
        "quantum-trained": ([9.0, 9.1, 8.9], 100),
        "quantum-frozen": ([10.0, 10.2, 9.8], 100),
        "classical-matched": ([10.5, 10.7, 10.4], 101),
        "classical-full": ([8.8, 8.9, 8.7], 200),
    }
    rows = [
        _ablation_row(variant, seed, val_ppl, n_params)
        for variant, (per_seed, n_params) in values.items()
        for seed, val_ppl in reversed(list(enumerate(per_seed)))
    ]

    report = build_ablation_report(
        rows,
        tag="three-seed",
        steps=300,
        base_config="configs/test.yaml",
        matched_d_ff=32,
        equivalence_margin=0.1,
        smallest_useful_effect=0.1,
    )
    normalized = report.lower()

    assert "paired n=3 on seeds [0, 1, 2]" in report
    assert "pilot" in normalized
    assert "practical equivalence" in normalized
    assert "power planning" in normalized
    assert "separated" not in normalized
    assert "paired empirical edge" not in normalized


def test_ablation_fails_closed_for_legacy_or_mixed_protocol_rows():
    rows = [
        _ablation_row("quantum-trained", 0, 5.0, 100),
        _ablation_row("quantum-frozen", 0, 6.0, 100),
    ]
    rows[0]["protocol_hash"] = None
    legacy = paired_ablation_analysis(rows, "quantum-trained", "quantum-frozen")
    assert legacy["fairness"]["protocol_complete"] is False
    assert legacy["claim"]["label"] == "insufficient fairness"

    rows = [
        _ablation_row("quantum-trained", 0, 5.0, 100),
        _ablation_row("quantum-frozen", 0, 6.0, 100),
    ]
    rows[1]["protocol_hash"] = "different-protocol"
    mixed = paired_ablation_analysis(rows, "quantum-trained", "quantum-frozen")
    assert mixed["fairness"]["same_protocol_hash"] is False
    assert mixed["fairness"]["valid"] is False
