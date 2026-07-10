# Experiment Configuration Instructions

The root `AGENTS.md` remains the baseline. A YAML file here is a versioned
experiment contract, not merely a convenient example.

## Rules

- Use `$qllm-model-development`; add `$qllm-experiment-runner` when the config
  will be executed or compared.
- Keep keys aligned with frozen dataclasses and validation in `qllm/config.py`.
  Unknown or incompatible choices should fail before training.
- Make seeds, data identity, sequence length, precision-relevant settings,
  shots, backend/device, trainability, and tracking intent explicit when they
  affect interpretation.
- Quantum and classical comparison configs must share the same task, split,
  training budget, evaluation, and logging unless a documented study factor
  intentionally differs.
- Do not raise steps, qubits, batch size, sequence length, or memory demand and
  then run the config without checking the compute gate.
- A config name must describe the implemented mechanism. Do not encode an
  advantage conclusion in a filename.

## Validation

```powershell
pytest -q tests/test_config_data.py
python scripts/verify_changes.py --plan
```

Loading and validation are not evidence that training is scientifically fair.
Do not run `scripts/train.py` solely to validate YAML; use the approved
experiment workflow and explicit CPU/GPU target when execution is in scope.
