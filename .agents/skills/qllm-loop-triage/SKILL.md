---
name: qllm-loop-triage
description: Run report-first QLLM daily or weekly triage across repo state, tests, dashboard jobs, results, GPU queue, budgets, and docs; update STATE.md and loop-run-log.md without source fixes unless explicitly authorized.
---

# QLLM Loop Triage

Use this skill to operate the first QLLM loop at L1 report level. The loop should create clarity and durable state before it starts making changes automatically.

## First Steps

1. Locate the repo root by finding `pyproject.toml` with project name `qllm`.
2. Read `LOOP.md`, `STATE.md`, `loop-budget.md`, `loop-constraints.md`, and recent `loop-run-log.md` entries.
3. Read `references/triage-flow.md`.
4. Inspect `git status --short`, recent docs/results changes, and high-priority items from `GPU_QUEUE.md`.
5. Stop immediately if `STATE.md` contains `loop-pause-all`. Switch to report-only at 80% of the daily cap and stop at 100%.

## Triage Scope

Check:

- Dirty worktree and uncommitted project files.
- Recent failures or stale artifacts if test output is available.
- `RESULTS.md` and `GPU_QUEUE.md` drift.
- Dashboard queue health if `results/qllm_results.db` exists.
- Obvious next experiment candidates.
- Skills or loop files that need maintenance.

## Output

Update `STATE.md` with:

- `Last run` timestamp.
- High-priority items that need action.
- Watch items that should be monitored.
- Noise/ignored items.
- Human handoff notes.

Append to `loop-run-log.md` with:

- Timestamp.
- Inputs inspected.
- Findings count.
- Actions taken.
- Escalations or blockers.
- Duration, estimated tokens, and outcome (`no-op`, `report-only`, `fix-proposed`, or `escalated`).

## Safety

- Stay report-only unless the user explicitly asks for fixes.
- L1 uses zero subagents; never exceed the current cap in `loop-budget.md` at higher levels.
- Do not start long experiments during triage.
- Do not auto-merge, auto-push, or delete generated artifacts.
- Escalate when a fix touches research interpretation, large GPU spend, secrets, environment setup, or dashboard job cancellation.
