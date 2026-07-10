# QLLM Loop Engineering

This file documents the first loop for the StateVectorAI QLLM project. The loop starts at L1 report-only: it creates durable state and clear next actions, but it does not make code changes, cancel jobs, or launch expensive runs by itself.

## Active Loops

| Pattern | Cadence | Status | Automation prompt |
|---------|---------|--------|-------------------|
| QLLM Daily Triage | Daily or manual | L1 report-only | `Run $loop-constraints, then $loop-budget, then $loop-triage. Read STATE.md, GPU_QUEUE.md, RESULTS.md, and git status. Update STATE.md and append loop-run-log.md. Do not modify source files.` |

## Scope

- Watch project health, local worktree state, dashboard queue state, result/documentation drift, and the GPU experiment queue.
- Surface the smallest useful next action for research progress.
- Keep quantum-evidence wording cautious by routing claims through `$qllm-research-protocol`.

## Human Gates

- Long GPU sweeps, CUDA/JAX environment changes, and high-memory quantum runs require explicit approval.
- Edits to `RESULTS.md` that strengthen claims require `$qllm-research-protocol`.
- Dashboard job cancellation requires explicit approval.
- No auto-fix until L2 checklist is intentionally completed.

## Worktrees

- Use the current Codex thread/worktree for L1 report-only work.
- For L2+ fixes, use one isolated branch or worktree per fix attempt.
- A verifier pass must approve before proposing a PR.

## Connectors

- MCP/connectors are not required for L1 manual triage.
- If GitHub is connected later, start with read-only issue/PR/status discovery plus comment-only write scope.
- Do not let a connector merge PRs, cancel dashboard jobs, or launch GPU work without explicit human approval.

## Skills

- Global personal skills: `$qllm-experiment-runner`, `$qllm-research-protocol`, `$qllm-model-development`, `$qllm-dashboard-development`, `$qllm-loop-triage`.
- Project loop skills: `$loop-triage`, `$loop-budget`, `$loop-constraints`.

## Budget

- Week one: L1 report-only, unlimited manual runs/day, no subagents.
- Switch to report-only immediately at 80% of daily token budget.
- Stop immediately if `loop-pause-all` appears in `STATE.md`.

## Promotion Path

- L1: report-only, update state/log.
- L2: single-file or narrowly scoped fixes with verifier approval.
- L3: unattended only after repeated clean L2 runs, explicit denylist, and budget controls.

Safety policy: see `docs/safety.md`.
