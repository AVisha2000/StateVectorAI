"""End-to-end integration: full fit() pipeline for classical and hybrid models.

These tests are the contract that the plugin architecture works: the SAME
pipeline runs with quantum blocks selected purely by config.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

from qllm.train.loop import fit, generate


def test_fit_classical_end_to_end(tiny_classical_cfg, tmp_path):
    result = fit(tiny_classical_cfg, verbose=False, out_dir=tmp_path)
    summary = result["summary"]
    assert np.isfinite(summary["val_loss"])
    assert summary["val_ppl"] > 1.0
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
