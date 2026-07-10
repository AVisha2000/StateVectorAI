# Benchmark Instructions

The root `AGENTS.md` remains the baseline. Files here are scientific
measurement programs, not ordinary demo scripts.

## Required workflow

- Use `$qllm-experiment-runner` for benchmark design or execution and
  `$qllm-research-protocol` for comparisons, statistics, or conclusions.
- Before changing a benchmark, state the hypothesis, task/instance
  distribution, size variable, quantum resource, classical challengers,
  stopping rule, and falsifying outcome. Follow `docs/RESEARCH_PROGRAM.md`.
- Keep generator, split, initialization, minibatch, circuit, and hardware seed
  axes distinct where they differ. A multi-seed single instance is not
  multi-instance replication.
- Use the same data access, evaluation split, optimization opportunity, and
  reporting path for compared arms. Parameter matching is one control, not a
  substitute for strong classical challengers.
- Record failures, null results, timeouts, OOMs, precision, shots/noise,
  backend, and real resource usage. Never overwrite an unfavorable result.
- Simulator wall time measures classical simulation cost, not QPU runtime.

## Execution and validation

Argument discovery is safe:

```powershell
python benchmarks/scaling_probe.py --help
```

Run the focused tests for the changed benchmark, then the benchmark-facing
suite when contracts overlap:

```powershell
pytest -q tests/test_ablation.py tests/test_advantage.py tests/test_qnlp_suite.py
pytest -q tests/test_contextual.py tests/test_interference.py tests/test_recurrent.py tests/test_two_stream.py
```

Do not launch a benchmark just because its code changed. CPU micro-smokes must
be explicitly bounded; GPU/QPU runs, sweeps, and high-memory simulations need
user approval. Benchmark output does not authorize a claim update.
