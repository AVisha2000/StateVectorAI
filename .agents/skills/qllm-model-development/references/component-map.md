# Component Map

## Core Contracts

- `qllm/config.py`: `ModelConfig`, `QuantumConfig`, `DataConfig`, `TrainConfig`, `TrackingConfig`.
- `qllm/registry.py`: canonical architecture, component, dataset, circuit, backend, readout, and conditioning choices.
- `qllm/models/model.py`: `uses_quantum`, registry-backed component builders, `build_model`, parameter counting.
- `qllm/train/loop.py`: expects model output shape `(batch, time, vocab)` and integer token batches.
- `qllm/data/datasets.py`: canonical `DatasetBundle` dispatch with the compatible flat adapter.

## Model Extension Points

- Transformer attention: add one registry choice, wire the builder, and test validation/model/dashboard drift.
- Transformer FFN: add one registry choice, wire the builder, and update the claim-specific analogue ladder when the intervention changes.
- Whole architecture: add one registry choice, branch in `build_model`, ensure `uses_quantum` and generation capability reflect it.
- Per-block overrides: use `BlockConfig` and `model_block_config`.
- Output heads: inspect `head_type` handling in `QLLM.__call__`.

## Quantum Extension Points

- `qllm/quantum/layers.py`: Flax modules wrapping circuits.
- `qllm/quantum/circuits.py`: ansatz registry and weight shapes.
- `qllm/quantum/backends.py`: circuit backend/readout dispatch.
- `qllm/quantum/capabilities.py`: exactness and capability declarations; unsupported operations remain explicit.
- `qllm/quantum/recurrent.py`: QRNN memory model.
- `qllm/quantum/contextual_cell.py`: contextual phase-accumulator cells.
- `qllm/quantum/metrics.py`: diagnostics such as expressibility/entanglement.

## Data Extension Points

- `qllm/data/text.py`: char-level text loading and batching.
- `qllm/data/quantum_seq.py`: monitored Ising and Markov-control tasks.
- `qllm/data/contextual.py`: contextual parity tasks.
- `qllm/data/interference_task.py`: interference synthetic tasks.
- `qllm/data/seq_cancellation.py`: sequence cancellation tasks.

## Useful Tests

- Config/data: `tests/test_config_data.py`, `tests/test_quantum_data.py`.
- Quantum layers/circuits: `tests/test_quantum.py`, `tests/test_gradients.py`, `tests/test_metrics.py`.
- Model integration: `tests/test_integration.py`, `tests/test_classical_model.py`.
- Recurrent/contextual: `tests/test_recurrent.py`, `tests/test_contextual.py`, `tests/test_contextual_cell.py`.
- Research controls: `tests/test_ablation.py`, `tests/test_advantage.py`, `tests/test_research_protocol.py`.
- Dashboard model specs/tests: `tests/test_dashboard_lab.py`.

## Common Pitfalls

- Adding a supported value outside `qllm/registry.py` or a config key without adding it to the dataclass/validator.
- Forgetting `uses_quantum`, which disables diagnostics/freezing/resource tags.
- Breaking classical-only import paths by importing PennyLane at module import time.
- Returning logits with a shifted time dimension.
- Creating a benchmark script that does not skip completed cells.
- Changing a synthetic task without updating cache keys.
- Flattening independent trajectories or losing `DatasetBundle` provenance/masks.
- Computing claim or warning semantics in React instead of the research protocol/backend view model.
