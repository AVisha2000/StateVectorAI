# Codex Playbook for QLLM

This is the practical operating rhythm for using Codex (and, when useful,
Claude Code) on this repository. The repository contract remains in
`AGENTS.md`; this page explains how to work with it day to day.

## The short version

1. Put rough thoughts in [`IDEAS.md`](../IDEAS.md).
2. Ask the agent to review and refine one idea with you. It should return a
   problem statement, proposed scope, acceptance evidence, risks, and a small
   implementation plan. It should not edit product code yet.
3. Decide explicitly: approve, revise, park, or reject. Only “approve” starts
   implementation.
4. Ask Codex to run the implementation loop. The parent agent owns the goal,
   assigns disjoint packets to bounded roles, integrates the work, and runs
   verification.
5. Review the diff and evidence. Ask for another correction loop if needed.
   Commit, push, merge, hardware execution, spend, and claim promotion remain
   your decisions.

## 1. Start from the repository root

Open Codex in the repo root so it can discover the complete instruction chain:

```powershell
codex -C .
```

At the beginning of a session, ask Codex to inspect `git status --short`, read
the nearest `AGENTS.md`, and summarize the relevant skills before changing
anything. For Claude Code, use `claude` from the same directory; the sibling
`CLAUDE.md` files import the shared guidance.

For a fresh setup or after changing agent configuration, run:

```powershell
python scripts/check_agent_setup.py
python scripts/verify_changes.py --plan
```

## 2. Capture ideas without over-specifying them

Use `IDEAS.md` for incomplete thoughts. Good raw entries say who has the
problem, what outcome would help, and any evidence or constraints. They do not
need a design, file list, or model choice.

Examples of useful prompts:

```text
Review IDEA-2026-07-11-dashboard-study-filter with me. Ask at most one
clarifying question at a time, inspect the relevant code and tests, and return
two scoped options with trade-offs. Do not edit product files or start jobs.
```

```text
Turn this idea into a candidate implementation plan. Include acceptance tests,
affected paths, non-goals, dependencies, risks, and any human gates. Keep the
idea in reviewed status until I approve it.
```

The review phase is where Codex is most useful as a thinking partner: it can
trace callers, find existing patterns, identify hidden coupling, and challenge
an underspecified request. Treat its proposal as a draft for your decision,
not as authorization.

## 3. Promote deliberately

Use a clear promotion message so the boundary is unambiguous:

```text
I approve IDEA-2026-07-11-dashboard-study-filter as written. Implement the
smallest complete change. Do not run GPU/QPU work, spend money, change claims,
delete artifacts, commit, or push.
```

For substantial work, require the agent to put the objective and acceptance
evidence in `PLANS.md` before implementation. For a small one-file fix, an
in-turn plan is enough. If the scope changes during implementation, stop and
return to review rather than silently expanding it.

## 4. Run an implementation loop

An implementation loop is a bounded triage cycle, not “let every agent work on
everything.” Ask the parent agent to:

- restate the approved objective and non-goals;
- inspect the relevant instructions, skills, code, callers, and tests;
- split only independent work into packets with exclusive write ownership;
- use a planner or explorer for uncertainty, a worker for one implementation
  packet, and a fresh verifier after integration;
- run focused checks while iterating and the change-aware verifier before
  reporting completion;
- stop at human gates and report the exact decision needed from you.

Recommended invocation:

```text
Run the implementation loop for the approved plan in PLANS.md. Work in triage
style: first inspect and identify blockers, then delegate only disjoint packets,
integrate, verify, and report. Keep me in the loop at each material decision.
Do not cross the repository's human gates.
```

The agent should not start parallel workers for a simple edit. Parallelism is
for independent or read-heavy work; one parent remains responsible for the
final diff and user handoff.

## 5. Verify and review the handoff

The normal completion check is:

```powershell
python scripts/verify_changes.py --run
```

For a broad product change, run `pytest -q`; for a dashboard frontend change,
also run the documented frontend build. Ask Codex to report commands and exit
statuses, changed files, skipped checks, residual risks, and any human gate.
“The agent said it passed” is not evidence; fresh command output is.

Before accepting the work, inspect the diff and ask:

```text
Review the complete diff as a fresh read-only verifier. Look for scope creep,
missing regression tests, unsafe permissions or validation, compatibility
breaks, generated-file edits, and claims stronger than the evidence. Return
only actionable findings with file paths and severity.
```

## 6. Keep the inbox and plans healthy

Use `IDEAS.md` for possibilities and unresolved requests. Use `PLANS.md` for
one active substantial objective and its evidence. Move completed detail into
the canonical docs, and keep only concise handoff state in `PLANS.md`.

Useful maintenance prompts include:

```text
Review the idea inbox. Group duplicates, identify stale entries, and propose
which one is worth refining next. Do not implement anything or change claims.
```

```text
Run a report-first triage of the active plan. Compare the stated acceptance
evidence with the current repo and tests, list blockers, and propose the next
smallest safe check. Do not fix source files or start expensive work.
```

## Human gates to repeat every time

Explicit approval is required before GPU/QPU/cluster runs, long or costly
sweeps, paid services, CUDA/JAX environment changes, destructive artifact or
database actions, changing claim-bearing research text, exposing the dashboard
remotely, or committing, pushing, merging, or publishing. Agents can prepare a
packet and explain the trade-offs; they do not grant the approval themselves.

For the architecture and role details behind this playbook, see
[`AGENT_OPERATING_MODEL.md`](AGENT_OPERATING_MODEL.md). For scientific scope,
use [`RESEARCH_PROGRAM.md`](RESEARCH_PROGRAM.md) and
[`RESEARCH_MAP.yaml`](RESEARCH_MAP.yaml).
