---
name: qllm-experiment-runner
description: Run, queue, debug, and compare QLLM experiments and artifacts. Use for training, benchmarks, dashboard jobs, GPU requests, quantum/classical comparisons, or experiment run documentation; hardware and spend remain human-gated.
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
- Use `python benchmarks/<name>.py ...` for research sweeps; prefer existing benchmark scripts over hand-written loops.
- Use dashboard queue paths when the user wants live progress, dataset imports, paired comparison jobs, or model-spec jobs.
- For GPU/QPU-targeted work, stop for explicit human approval before spend or hardware execution. After approval, verify device readiness and keep the first run small unless the user explicitly asks for a full sweep.
- For comparison work, keep dataset, seed, steps, eval cadence, and device target matched unless the research question says otherwise.

## Interpreting Results

- Report `val_loss`, `val_ppl`, `val_bpc`, parameter count, wall time, seed count, and whether the run was cancelled.
- Treat single-seed results as smoke evidence. Use the `qllm-research-protocol` skill before presenting an empirical quantum edge.
- Update `RESULTS.md` only when the user asks for consolidated findings or the run materially changes project knowledge.

## Validation

- For code changes to training/data/model behavior, run the focused tests that cover the touched path, then run broader `pytest -q` when the blast radius is high.
- For benchmark-only changes, run the benchmark with tiny parameters or a smoke flag where available.
- For dashboard queue changes, run a queue smoke such as `python scripts/queue_smoke.py` after the dashboard/API path is available.
