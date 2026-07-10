# QLLM Triage Flow

## Read

1. `LOOP.md`: active loop design, cadence, gates.
2. `STATE.md`: current high-priority and watch items.
3. `loop-budget.md`: budget and kill-switch limits.
4. `loop-constraints.md`: push, merge, path, test, spend, and claim boundaries.
5. Recent `loop-run-log.md` entries: prior usage and repeated blockers.
6. `GPU_QUEUE.md`: prioritized experiment backlog.
7. `RESULTS.md`: current evidence ledger.
8. `git status --short`: local change awareness.

Optional:

- `results/qllm_results.db`: dashboard job/run state.
- `.portal*.log`: portal launch/debug logs.
- `docs/UI_UPGRADE_PLAN.md`: dashboard design drift.

## Classify

- High priority: affects correctness, blocks active research, risks wasted GPU time, or needs human decision today.
- Watch: useful but not urgent, waiting for data, or lower confidence.
- Noise: checked and intentionally ignored.

## L1 Report Format

```markdown
## High Priority

- <item>: <why it matters>. Next: <action>.

## Watch List

- <item>: <condition to revisit>.

## Recent Noise

- <thing inspected and ignored>.

## Handoff

- <decision or approval needed>.
```

## Promotion Path

- L1: report only, update state/log.
- L2: propose small fixes in isolated branch/worktree, verifier required.
- L3: unattended only after repeated clean L2 runs, clear denylist, and budget controls.

## QLLM Denylist For Automation

- Secrets, environment files, credentials, CUDA/JAX installation changes.
- Long GPU sweeps or high-memory runs.
- Edits to `RESULTS.md` that strengthen claims without research-protocol review.
- Destructive cleanup of `results/`, `mlruns/`, or `mlflow.db`.
- Dashboard job cancellation unless user explicitly requests it.

## Budget Exit

- `loop-pause-all` in `STATE.md`: stop.
- At least 80% of daily budget: report only.
- At least 100%: append a capped outcome and stop.
- No actionable items: append one concise no-op record and exit; do not manufacture work.
