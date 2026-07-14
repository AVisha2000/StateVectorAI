"""End-to-end integration: full fit() pipeline for classical and hybrid models.

These tests are the contract that the plugin architecture works: the SAME
pipeline runs with quantum blocks selected purely by config.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from pathlib import Path

from qllm import registry
from qllm.config import ProblemConfig
from qllm.train import loop as train_loop
from qllm.train.loop import fit, generate


def test_fit_classical_end_to_end(tiny_classical_cfg, tmp_path):
    result = fit(tiny_classical_cfg, verbose=False, out_dir=tmp_path)
    summary = result["summary"]
    assert np.isfinite(summary["val_loss"])
    assert summary["val_ppl"] > 1.0
    assert summary["primary_metric_type"] == "strict_autoregressive_next_token"
    assert summary["primary_metric_name"] == "val_ppl"
    assert summary["primary_metric_value"] == pytest.approx(summary["val_ppl"])
    artifact_dir = Path(summary["artifact_dir"])
    assert (artifact_dir / "params.msgpack").exists()
    assert (artifact_dir / "summary.json").exists()


def test_fit_quantum_end_to_end(tiny_quantum_cfg, tmp_path):
    result = fit(tiny_quantum_cfg, verbose=False, out_dir=tmp_path)
    summary = result["summary"]
    assert np.isfinite(summary["val_loss"])
    # quantum FFN params present in the trained tree
    block = result["state"].params["block_0"]
    assert "ffn" in block
    assert "circuit_weights" in str(block["ffn"].keys()) or any(
        "circuit_weights" in str(k) for k in _walk_keys(block["ffn"])
    )


def test_sequence_runner_rejects_non_sequence_primary_metric_before_work(
    monkeypatch, tiny_classical_cfg, tmp_path
):
    monkeypatch.setattr(
        registry,
        "METRIC_TYPES",
        {
            **registry.METRIC_TYPES,
            "ground_state_energy_error": {
                "lower_is_better": True,
                "units": "hartree",
                "pairable": True,
                "extraction_key": "energy_error",
                "comparator_class": "classical_solver",
            },
        },
    )

    with pytest.raises(ValueError, match="metric-specific sibling runner"):
        fit(
            tiny_classical_cfg,
            verbose=False,
            out_dir=tmp_path,
            primary_metric_type="ground_state_energy_error",
        )

    assert list(tmp_path.iterdir()) == []


def test_sequence_runner_rejects_non_sequence_task_before_device_or_data_work(
    monkeypatch, tiny_classical_cfg, tmp_path
):
    cfg = replace(
        tiny_classical_cfg,
        problem=ProblemConfig(
            task_type="ground_state",
            instance_id="tfim-open-n4-j1-h1-v1",
        ),
    )

    def unexpected(*_args, **_kwargs):
        raise AssertionError("sequence-only work must not start")

    monkeypatch.setattr(train_loop, "resolve_execution_device", unexpected)
    with pytest.raises(ValueError, match="task-specific sibling runner"):
        fit(cfg, verbose=False, out_dir=tmp_path)

    monkeypatch.setattr(train_loop, "load_dataset_bundle", unexpected)
    with pytest.raises(ValueError, match="task-specific sibling runner"):
        train_loop._fit_on_device(cfg, verbose=False, out_dir=tmp_path)

    assert list(tmp_path.iterdir()) == []


def _walk_keys(tree):
    keys = []

    def rec(node, prefix=""):
        if hasattr(node, "items"):
            for k, v in node.items():
                keys.append(f"{prefix}{k}")
                rec(v, f"{prefix}{k}.")

    rec(tree)
    return keys


def test_generate_produces_text(tiny_classical_cfg, tmp_path):
    result = fit(tiny_classical_cfg, verbose=False, out_dir=tmp_path)
    text = generate(
        result["model"],
        result["state"].params,
        result["tokenizer"],
        prompt="hello",
        max_new_tokens=20,
        seed=0,
    )
    assert isinstance(text, str)
    assert len(text) >= 20
