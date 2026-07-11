---
name: qllm-experiment-runner
description: Run, queue, resume, debug, and compare QLLM experiments and artifacts with durable identity, checkpoints, telemetry, and fair controls. Use for training, benchmarks, dashboard jobs, queue recovery, checkpoint/resume, GPU requests, quantum/classical comparisons, or experiment documentation; hardware and spend remain human-gated.
---

# QLLM Experiment Runner

Use this skill to run experiments without losing the project's fairness discipline. The training pipeline is intentionally shared across classical and quantum variants; preserve that by changing configs, presets, or benchmark harnesses rather than one-off training code unless the task truly requires it.

## First Steps

1. Locate the repo root by finding `pyproject.toml` with project name `qllm`.
2. Read `README.md` for current setup, then read `docs/RESEARCH_PROGRAM.md` and `GPU_QUEUE.md` if the task mentions the roadmap, GPU, scaling, memory, or "next experiment".
3. Read `references/experiment-map.md` for commands, output locations, and common run paths.
4. Check `git status --short` before edits and avoid touching unrelated generated outputs in `results/`, `mlruns/`, or `mlflow.db`.

## Running Work

- Use `python scripts/train.py --config <config>` for one YAML-backed run.
- Require the shared `validate_config` result to pass before initializing data,
  models, backends, or a dashboard job.
- Use `python benchmarks/<name>.py ...` for research sweeps; prefer existing benchmark scripts over hand-written loops.
- Use dashboard queue paths when the user wants live progress, dataset imports, paired comparison jobs, or model-spec jobs.
- For GPU/QPU-targeted work, stop for explicit human approval before spend or hardware execution. After approval, verify device readiness and keep the first run small unless the user explicitly asks for a full sweep.
- For comparison work, keep dataset, seed, steps, eval cadence, and device target matched unless the research question says otherwise.

## Durable Run Rules

- Keep scientific `ExperimentConfig` separate from operational `RunOptions`.
  Preserve immutable manifest identity, dataset hashes, environment identity,
  UUIDs, and resume lineage.
- Resume only through validated checkpoints. Restore train state, optimizer,
  RNG, history, and completed step; reject incompatible, corrupt, or older
  same-run checkpoints instead of silently starting fresh.
- Distinguish same-run recovery from a fork/warm start. A fork requires an
  explicit new run UUID and artifact directory while retaining parent lineage.
- Treat SQLite as authoritative for dashboard jobs. Use transactional claims,
  lease heartbeats, fenced completion, and identity-matched checkpoint
  recovery; never repair queue state with ad hoc row edits.
- Log UUID-backed step metrics idempotently. Identical retries are harmless;
  conflicting values for the same run/step/metric must fail, and displayed
  progress must remain monotonic.

## Resource Honesty

- Preserve configured, derived, measured, unsupported, and unmeasured fields as
  distinct categories in the resource ledger.
- Report requested and resolved device, environment packages, compile/first-step
  timing, steady-state timing, logical circuit calls, and allocator-memory
  status when available. Do not convert estimates into measured values.
- Label MPS/tensor-network simulation as approximate when its capability record
  says so. Do not present simulator wall time as QPU time or infer unmeasured
  truncation error, discarded weight, convergence, or peak memory.

## Interpreting Results

- Report `val_loss`, `val_ppl`, `val_bpc`, parameter count, wall time, seed count, and whether the run was cancelled.
- Treat single-seed results as smoke evidence. Use the `qllm-research-protocol` skill before presenting an empirical quantum edge.
- Change `RESULTS.md` only after explicit user approval for the claim-bearing
  edit, and use `$qllm-research-protocol` to preserve conservative wording and
  historical provenance.

## Validation

- For code changes to training/data/model behavior, run the focused tests that cover the touched path, then run broader `pytest -q` when the blast radius is high.
- For durability changes, run focused cases from `tests/test_durable_runs.py`;
  for telemetry/backends, use `tests/test_resource_accounting.py` and the
  backend capability suite.
- For benchmark-only changes, run the benchmark with tiny parameters or a smoke flag where available.
- For dashboard queue changes, after the local API is available run
  `python scripts/queue_smoke.py --steps 1 --eval-every 1 --device-target cpu`.
