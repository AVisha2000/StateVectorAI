# QLLM Issue-Closeout Checklist

## Establish Current State

- Run `git status --short` and preserve unrelated changes.
- Read the current issue state, body, linked material, and complete comment
  history with `gh issue view`/`gh api` or the available GitHub connector.
- If a closeout names commits, verify that they are reachable from the delivery
  branch; do not treat an unmerged hash as shipped evidence.

## Requirement Matrix

| Acceptance criterion | Current source evidence | Test/UI evidence | Status | Gate or residual limit |
| --- | --- | --- | --- | --- |
| `<latest criterion, including comment amendments>` | `<file:line or commit>` | `<command/result or rendered check>` | complete / deferred / unmet | `<authority or limitation>` |

Inspect implementation, callers, tests, and domain documentation for each row.
Distinguish stale issue prose from current behavior and never hide an unmet
criterion behind an unrelated passing suite.

## Conditional Evidence

- Dashboard visuals: use an isolated loopback instance with temporary DB,
  results, and data paths; check both themes, desktop and narrow widths, console
  errors, interactions, and local versus page overflow. Do not queue jobs.
- Research: inspect claim identity, fairness, provenance, resource accounting,
  paired statistics, and causal/teacher-forcing limits with
  `$qllm-research-protocol`.
- Durability: exercise the focused checkpoint, queue recovery, idempotence, or
  security regressions that correspond to the criterion.

## Deterministic Closeout Evidence

- `python scripts/check_agent_setup.py`
- `python scripts/verify_changes.py --plan`
- `python scripts/verify_changes.py --run` when selected checks are safe
- Focused regression tests and the broader suite warranted by blast radius
- Fresh verifier output for multi-file, high-risk, or research-sensitive work

An empty or agent-only verifier plan on a clean worktree does not prove product
acceptance. Record exact commands, exit codes, and key results.

## Closeout Template

```markdown
Acceptance: <criterion-to-evidence summary, including any explicit deferral>

Delivery: <reachable commits and behavior-changing files>

Verification: <exact commands, results, and UI evidence>

Residual limits/gates: <known limits and approvals still required>

Issue state: <open/closed/reopened after the authorized action>
```

Never comment, close, or reopen remotely until the user explicitly authorizes
that GitHub action. Separate approvals still apply to commits, pushes, claims,
hardware, cloud/paid work, destructive operations, and publication.
