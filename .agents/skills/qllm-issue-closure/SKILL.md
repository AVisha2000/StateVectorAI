---
name: qllm-issue-closure
description: Audit and close QLLM backlog or GitHub issues only after reconciling acceptance criteria with current code, documentation, tests, UI evidence, and human gates. Use when asked to assess issue readiness, resolve a backlog item, prepare closeout evidence, comment on, close, or reopen a QLLM issue.
---

# QLLM Issue Closure

Use this skill to turn an issue into an evidence-backed closeout, not a stale
status update. A closed issue must say precisely which acceptance criteria are
met, what remains deferred, and which gate authorized the external action.

## First Steps

1. Read the root and relevant nested `AGENTS.md` files and
   `references/issue-closure-checklist.md`.
2. Inspect `git status --short` and the current remote issue state, body, and
   full comment history before assuming that it is open, closed, or complete.
3. Build the checklist's requirement-to-evidence matrix from the latest
   acceptance criteria, including amendments made in comments.

## Verification and Reconciliation

- Trace each requirement to current code, tests, documentation, rendered UI,
  or an explicitly accepted deferral. A commit message, stale closeout comment,
  or passing unrelated suite is not enough.
- Route implementation changes to the appropriate QLLM domain skill. Use
  `$qllm-dashboard-development` for UI/API evidence and
  `$qllm-research-protocol` for claims, fairness, historical results, or
  causal interpretation.
- Use `$qllm-code-change-verification` for current deterministic evidence and a
  fresh read-only verifier after multi-file, high-risk, or research-sensitive
  work. On a clean worktree, an empty or agent-only change plan is not evidence
  for the issue's product behavior; run the issue-specific focused checks in
  the checklist.
- Never queue a job, start hardware work, or mutate an artifact merely to close
  an issue.

## External-State Gate

- Posting a GitHub comment, closing/reopening an issue, committing, pushing,
  merging, publishing, and all remote side effects require explicit user
  authority even when local verification passes.
- GPU/QPU, cloud, paid, destructive, and claim-strengthening actions retain
  their separate human gates. Record a deferral instead of treating a gated
  result as completed evidence.

## Closeout Record

After authority is granted and the evidence is current, use the checklist's
closeout template. Include the resolved criteria, source locations, commits,
exact checks and UI evidence, remaining limits, active gates, and resulting
issue state.

Do not close an issue with an unresolved requirement hidden behind optimistic
wording. Keep research language conservative: a simulator result, feature
completion, or UI display is not evidence of quantum advantage.
