# Component Map

## Core Contracts

- `qllm/config.py`: `ModelConfig`, `QuantumConfig`, `DataConfig`, `TrainConfig`, `TrackingConfig`.
- `qllm/models/model.py`: `uses_quantum`, `build_attention`, `build_ffn`, `build_model`, parameter counting.
- `qllm/train/loop.py`: expects model output shape `(batch, time, vocab)` and integer token batches.
- `qllm/data/datasets.py`: dispatches `DataConfig.kind` into token ids and tokenizer.

## Model Extension Points

- Transformer attention: add strings to `ATTN_TYPES`, branch in `build_attention`, tests in attention/model paths.
- Transformer FFN: add strings to `FFN_TYPES`, branch in `build_ffn`, update matched baseline logic if parameter matching changes.
- Whole architecture: add string to `ARCH_TYPES`, branch in `build_model`, ensure `uses_quantum` reflects it.
- Per-block overrides: use `BlockConfig` and `model_block_config`.
- Output heads: inspect `head_type` handling in `QLLM.__call__`.

## Quantum Extension Points

- `qllm/quantum/layers.py`: Flax modules wrapping circuits.
- `qllm/quantum/circuits.py`: ansatz registry and weight shapes.
- `qllm/quantum/backends.py`: circuit backend/readout dispatch.
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

- Adding a config key without adding it to the dataclass.
- Forgetting `uses_quantum`, which disables diagnostics/freezing/resource tags.
- Breaking classical-only import paths by importing PennyLane at module import time.
- Returning logits with a shifted time dimension.
- Creating a benchmark script that does not skip completed cells.
- Changing a synthetic task without updating cache keys.
