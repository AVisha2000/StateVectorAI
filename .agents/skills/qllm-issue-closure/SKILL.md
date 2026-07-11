---
name: qllm-issue-closure
description: Audit, verify, and close QLLM backlog or GitHub issues only after reconciling acceptance criteria with current code, documentation, tests, UI evidence, and human gates. Use when asked to resolve, validate, close, or publish a QLLM issue.
---

# QLLM Issue Closure

Use this skill to turn an issue into an evidence-backed closeout, not a stale
status update. A closed issue must say precisely which acceptance criteria are
met, what remains deferred, and which gate authorized the external action.

## First Steps

1. Read the root and relevant nested `AGENTS.md` files, linked issue text, and
   `references/issue-closure-checklist.md`.
2. Inspect `git status --short`, the current implementation, callers, tests,
   relevant docs, and existing issue comments before assuming the issue is open
   or incomplete.
3. Create a requirement-to-evidence matrix. Separate current repository facts,
   stale issue prose, known limits, deferred hardware work, and actions that
   still need explicit human authority.

## Verification and Reconciliation

- Use `gh issue view` and, where useful, `gh issue list` to check the current
  external issue state. Do not infer it from a local closeout note.
- Trace each requirement to code, test, documentation, rendered UI, or a
  recorded deferral. A commit message or passing unrelated suite is not enough.
- Route implementation changes to the appropriate QLLM domain skill. Use
  `$qllm-dashboard-development` for UI/API evidence and
  `$qllm-research-protocol` for claims, fairness, historical results, or
  causal interpretation.
- For visual dashboard requirements, use a temporary local database and check
  the changed route in both themes at desktop and narrow widths. Never queue a
  job or start hardware work merely to close an issue.
- Before closeout, run `python scripts/check_agent_setup.py`,
  `python scripts/verify_changes.py --plan`, and the selected safe checks from
  `python scripts/verify_changes.py --run`, plus focused regression tests.
  Use a fresh read-only verifier after multi-file, high-risk, or
  research-sensitive work.

## External-State Gate

- Posting a GitHub comment, closing/reopening an issue, committing, pushing,
  merging, publishing, and all remote side effects require explicit user
  authority even when local verification passes.
- GPU/QPU, cloud, paid, destructive, and claim-strengthening actions retain
  their separate human gates. Record a deferral instead of treating a gated
  result as completed evidence.

## Closeout Record

After authority is granted and the evidence is current, post or report:

1. The resolved acceptance criteria and the source locations that satisfy them.
2. Commits and files that changed behavior, if any.
3. Exact verification commands, exit status, and relevant UI evidence.
4. Remaining limits, deferred work, and active human gates.
5. The resulting issue state.

Do not close an issue with an unresolved requirement hidden behind optimistic
wording. Keep research language conservative: a simulator result, feature
completion, or UI display is not evidence of quantum advantage.
