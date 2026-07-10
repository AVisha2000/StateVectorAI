# QLLM Package Instructions

The root `AGENTS.md` remains the baseline. This file adds rules for Python code
under `qllm/` except the dashboard, which has its own nested instructions.

## Route and read

- Use `$qllm-model-development` for models, layers, circuits, backends, data,
  configuration, evaluation, or training behavior.
- Also use `$qllm-research-protocol` when a code change alters the meaning of a
  metric, comparator, resource count, or scientific conclusion.
- Trace the narrow path from config/data through model dispatch and training;
  do not inventory the entire package unless the contract is genuinely broad.

## Package contracts

- Keep classical and quantum variants in the same data, training, evaluation,
  and logging path. New quantum behavior requires a fair classical control.
- Preserve explicit shapes, dtypes, seed flow, and JAX/Flax functional
  behavior. Make randomness injectable and reproducible.
- Register new architecture, backend, circuit, readout, or dataset choices in
  their canonical dispatch point; reject invalid combinations early.
- Keep dataset trajectory boundaries and train/validation separation intact.
  Never create synthetic transitions or leak future/validation information.
- Do not make full-statevector access a hidden requirement for an interface
  intended to reach finite-shot, approximate, or hardware backends.
- Record limitations instead of disguising unsupported behavior with a silent
  fallback. Preserve existing public config behavior unless migration is part
  of the task.
- Treat measurements such as entanglement, gradient variance, expressivity,
  geometry, or parameter count as diagnostics, not advantage claims.

## Tests

Run the smallest relevant set while iterating, then expand for shared contracts:

```powershell
pytest -q tests/test_config_data.py tests/test_integration.py
pytest -q tests/test_quantum.py tests/test_gradients.py tests/test_metrics.py
pytest -q tests/test_recurrent.py tests/test_contextual_cell.py
pytest -q tests/test_quantum_data.py tests/test_contextual.py tests/test_seq_cancellation.py
```

For changes to registries, shared config, training, evaluation, or public model
contracts, finish with:

```powershell
pytest -q
```

Add an invariant or regression test for every corrected scientific or data
contract. Do not validate a mechanism solely by checking that code executes.
