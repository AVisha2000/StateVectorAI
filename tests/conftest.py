"""Shared fixtures. Tests use the synthetic corpus fallback (hermetic CI)."""
from __future__ import annotations

import pytest

from qllm.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    QuantumConfig,
    TrackingConfig,
    TrainConfig,
)


@pytest.fixture(scope="session")
def tiny_text() -> str:
    return "hello quantum world! " * 50


@pytest.fixture()
def tiny_classical_cfg() -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(
            d_model=16, n_heads=2, n_blocks=1, d_ff=32, max_seq_len=16
        ),
        train=TrainConfig(
            steps=3, batch_size=2, seq_len=8, eval_every=3, eval_batches=2
        ),
        data=DataConfig(corpus_path="__nonexistent__"),  # -> synthetic fallback
        tracking=TrackingConfig(enabled=False, log_quantum_diagnostics=False),
    )


@pytest.fixture()
def tiny_quantum_cfg(tiny_classical_cfg) -> ExperimentConfig:
    import dataclasses

    model = dataclasses.replace(
        tiny_classical_cfg.model,
        ffn_type="quantum",
        quantum=QuantumConfig(n_qubits=2, n_circuit_layers=1),
    )
    return dataclasses.replace(tiny_classical_cfg, model=model)
