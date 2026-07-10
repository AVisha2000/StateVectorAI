---
name: qllm-code-change-verification
description: Select and run the smallest sufficient deterministic checks for QLLM code, dashboard, benchmark, configuration, documentation, or agent-workflow changes. Use after implementation and before reporting a change complete; do not use for read-only answers.
---

# QLLM Code Change Verification

Verification is evidence collection, not a final scientific judgment. Start focused, widen with blast radius, and preserve exact command output.

## Workflow

1. Read `references/check-matrix.md`.
2. Run `python scripts/verify_changes.py --plan` to inspect changed-path routing and human gates.
3. Run `python scripts/verify_changes.py --run` unless the selected command requires authority the user has not granted.
4. If a hook already ran checks for the same fingerprint, inspect and report that evidence instead of rerunning blindly.
5. Use the read-only `verifier` after high-risk or claim-sensitive work. Give it the original request, actual diff, and exact results.

## Rules

- A passing narrow test does not prove unrelated behavior.
- A failed check blocks completion; distinguish new failures from pre-existing ones with evidence.
- Do not weaken, skip, or delete a test to make verification pass.
- Do not let the Stop hook launch GPU/QPU work. It may only run safe local CPU checks and flag gated paths.
- Human-gated paths remain gated after tests pass. Record the needed approval instead of claiming the gate is cleared.
- Docs-only changes may use static checks, except evidence/claim docs, which require `$qllm-research-protocol` and human review.

## Report

List each command, exit status, key result, skipped check with reason, and remaining human gate. Never report simply "tests pass" without naming the tests.
