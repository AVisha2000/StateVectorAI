---
name: qllm-agent-workflow
description: Orchestrate substantial QLLM repository work with scoped planning, bounded subagents, disjoint ownership, verification, and human gates. Use for ambiguous, cross-subsystem, multi-file, or high-risk tasks; skip for simple one-file fixes and direct questions.
---

# QLLM Agent Workflow

Use this workflow when coordination will improve correctness enough to justify extra agent tokens. The lead agent owns requirements, integration, final checks, and the user-facing answer.

## Start

1. Read the root and nearest path-scoped `AGENTS.md` files.
2. Classify the request as explain, diagnose, small change, substantial change, experiment, evidence claim, loop triage, or publish action.
3. Read `PLANS.md` and create or update an ExecPlan only for substantial work.
4. Read `references/delegation-contract.md` before spawning agents.
5. Invoke the narrow domain skill that matches the work. This skill coordinates those procedures; it does not replace them.

## Choose The Smallest Workflow

- Direct: use the lead alone when the desired diff and verification fit in one clear sentence.
- Scout then build: use a read-only explorer for an unfamiliar subsystem or noisy logs, then let one writer implement.
- Plan then build: use the read-only planner for cross-cutting design, unclear acceptance, or research-sensitive work.
- Parallel research: use two or three read-only agents only when source areas are independent.
- Maker/checker: after risky implementation, give a fresh read-only verifier the original request, actual diff, and exact test evidence.

## Delegate Safely

- Cap direct children at three and keep nesting depth at one.
- Give each child one concrete objective, explicit non-goals, mutation authority, acceptance checks, and a compact return format.
- Prefer read-heavy exploration, source verification, test interpretation, and independent review.
- Assign at most one writer to any file or shared contract. The lead integrates shared files.
- Stop or redirect duplicate work early. Ask for distilled evidence, not raw transcripts or full logs.

## Execute And Prove

- Record observable acceptance criteria before editing.
- Preserve the dirty worktree and keep changes within assigned scope.
- Run deterministic checks directly; do not spend an agent merely executing a known command.
- Use `$qllm-code-change-verification` after changes and a fresh verifier when the risk table requires one.
- Never let a local or lightweight drafter decide scientific claims, GPU/QPU work, security, publishing, or irreversible actions.
- Stop at the human gates for spend, hardware execution, claim strengthening, push/merge, destructive cleanup, secrets, or external side effects.

## Finish

Return the outcome, changed paths, exact checks and results, unresolved risks, and any human decision still required. Do not present an agent verdict as stronger evidence than the deterministic checks it reviewed.
