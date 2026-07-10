---
name: qllm-model-development
description: Add, modify, debug, or review QLLM architectures, quantum layers, classical controls, datasets, configs, and training behavior. Use for new circuits/backends/readouts, dataset kinds, model dispatch, fair baselines, and JAX/Flax/PennyLane behavior.
---

# QLLM Model Development

Use this skill when changing the scientific or model-building surface of QLLM. The repo is a plugin-style testbed: new components should fit the existing config, dispatch, train, benchmark, and evidence paths instead of becoming isolated scripts.

## First Steps

1. Locate the repo root by finding `pyproject.toml` with project name `qllm`.
2. Read `qllm/config.py`, `qllm/models/model.py`, and the touched model/data module.
3. Read `references/component-map.md` for extension points and focused tests.
4. Check existing benchmark scripts before adding a new one.

## Design Rules

- Add config fields in frozen dataclasses and preserve YAML loading through `from_dict`.
- Register model choices in `qllm/models/model.py` so `build_model()` remains the composition root.
- Keep `tokens -> logits` as the shared model contract for train/eval.
- Preserve lazy imports for PennyLane/quantum code where classical paths should not need quantum dependencies.
- Add classical controls alongside quantum variants when the change affects research claims.
- Name trainable circuit parameters `circuit_weights` when gradient logging or freezing should apply.

## Data Rules

- Route new task generators through `qllm/data/datasets.py`.
- Cache expensive synthetic datasets under `results/.data_cache`.
- Return `(ids, tokenizer)` where tokenizer exposes `vocab_size`, `encode`, and `decode` when possible.
- Keep generated task parameters represented in `DataConfig` so runs are reproducible.

## Validation

- Run the most focused tests for the touched path first.
- Add or update tests when introducing a new config key, dispatch string, shape contract, dataset kind, or fairness control.
- For JAX/Flax changes, check initialization, JIT compatibility, gradient flow, and causality where applicable.
- For quantum layers, test shape, finite outputs, gradient reachability, and trainable-vs-frozen behavior if relevant.
