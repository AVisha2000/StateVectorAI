---
name: qllm-model-development
description: Add, modify, debug, or review QLLM architectures, quantum layers, classical controls, datasets, configs, and training behavior. Use for new circuits/backends/readouts, dataset kinds, model dispatch, fair baselines, and JAX/Flax/PennyLane behavior.
---

# QLLM Model Development

Use this skill when changing the scientific or model-building surface of QLLM. The repo is a plugin-style testbed: new components should fit the existing config, dispatch, train, benchmark, and evidence paths instead of becoming isolated scripts.

## First Steps

1. Locate the repo root by finding `pyproject.toml` with project name `qllm`.
2. Read `qllm/registry.py`, `qllm/config.py`, `qllm/models/model.py`, and the
   touched model/data module.
3. Read `references/component-map.md` for extension points and focused tests.
4. Check existing benchmark scripts before adding a new one.

## Design Rules

- Add config fields in frozen dataclasses, preserve YAML loading through
  `from_dict`, and keep `validate_config` dependency-free so invalid runs fail
  before dataset, model, backend, or optional-library initialization.
- Register canonical choices in `qllm/registry.py`, then wire their dispatch in
  `qllm/models/model.py` so `build_model()` remains the composition root.
- Keep `tokens -> logits` as the shared model contract for train/eval.
- Preserve lazy imports for PennyLane/quantum code where classical paths should not need quantum dependencies.
- Declare exactness, approximation, supported diagnostics, and state access in
  backend capabilities. Never silently fall back to a different backend.
- Add classical controls alongside quantum variants when the change affects research claims.
- Name trainable circuit parameters `circuit_weights` when gradient logging or freezing should apply.

## Data Rules

- Route new task generators through `qllm/data/datasets.py` and return a
  `DatasetBundle`; keep `(ids, tokenizer)` flattening compatibility-only.
- Cache expensive synthetic datasets under `results/.data_cache`.
- Preserve trajectory rows, masks, provenance, sampler policy, and config/content
  hashes through split, sampling, evaluation, and benchmark callers.
- Sample every next-token crop (`seq_len + 1`) within one trajectory. Cover the
  exact-fit boundary and reject data that cannot provide one legal crop.
- Keep generated task parameters represented in `DataConfig` so runs are reproducible.

## Validation

- Run the most focused tests for the touched path first.
- Add or update tests when introducing a new config key, dispatch string, shape contract, dataset kind, or fairness control.
- For config changes, test `from_dict` plus `validate_config` and prove CLI,
  model-spec, and queue ingress use the shared validator where affected.
- For JAX/Flax changes, check initialization, JIT compatibility, gradient flow, and causality where applicable.
- For quantum layers, test shape, finite outputs, gradient reachability, and trainable-vs-frozen behavior if relevant.
