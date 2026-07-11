# QLLM Issue-Closeout Checklist

## Requirement Matrix

For every acceptance criterion, record its source, current evidence, command
or UI proof, and whether it is complete, deferred, or blocked by a human gate.

## Required Reconciliation

- Check current Git state and preserve unrelated user changes.
- Read current issue text and comments with `gh issue view`; distinguish stale
  issue prose from current repository behavior.
- Inspect the implementation, callers, tests, and domain documentation.
- Verify rendered dashboard requirements in an isolated local instance; check
  both themes, desktop and narrow widths, console errors, and local scrolling.
- For research-sensitive requirements, inspect fairness, provenance, resource
  accounting, and causal/teacher-forcing limits before describing a result.

## Closeout Evidence

- `python scripts/check_agent_setup.py`
- `python scripts/verify_changes.py --plan`
- `python scripts/verify_changes.py --run` when its selected checks are safe
  and authorized
- Focused regression tests and the broader suite warranted by blast radius
- Fresh verifier output for multi-file, high-risk, or research-sensitive work

## Human Gates

Never close remotely until the user explicitly authorizes the GitHub action.
Do not use issue closeout to bypass approvals for commits, pushes, claims,
hardware, cloud services, paid work, destructive operations, or publication.
