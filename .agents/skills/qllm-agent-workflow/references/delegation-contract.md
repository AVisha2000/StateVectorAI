# Delegation Contract

## Decision Rule

Delegate only when a bounded child context materially reduces exploration noise, adds independent scrutiny, or runs useful work in parallel. Do not delegate a trivial task, tightly coupled edits, or a command whose result the lead can read directly.

## Required Task Packet

```text
Objective:
Scope and owned files:
Required inputs:
Non-goals:
Mutation authority: read-only | owns <exact paths>
Required skill:
Acceptance checks:
Return format and limit:
- findings with file:line or primary-source URL
- risks and unresolved questions
- exact tests/evidence
```

The packet must be self-contained enough for a fresh context. Include decisions and constraints, not the lead's entire transcript.

## Role Routing

- `planner`: read-only, high-reasoning decomposition for ambiguous, multi-stage, or high-risk work. Produces acceptance, sequence, ownership, tests, and gates; never implements.
- `explorer`: read-only, fast targeted repository or documentation scan. Returns a short evidence map; never edits.
- built-in `worker`: one bounded implementation with disjoint ownership. Reports files and checks; never commits or publishes.
- `verifier`: fresh read-only checker. Reviews the original request, diff, and test output; never fixes its own findings.
- local git scribe: drafts structured commit text from an already-staged diff. It never stages, commits, pushes, or evaluates scientific meaning.

## Verification Return

```text
VERDICT: PASS | NEEDS_WORK | HUMAN_GATE
Scope reviewed:
Deterministic evidence:
Acceptance criteria not proven:
Claim or safety concerns:
Required next action:
```

Missing evidence is `NEEDS_WORK`. Medium-or-higher operational risk, GPU/QPU execution, or a strengthened research claim is `HUMAN_GATE` even when tests pass.

## Durable Handoff

For multi-session work only, preserve:

```text
Done
In progress
Next
Decisions
Evidence/tests
Blockers
```

Do not create a handoff artifact for ordinary one-turn tasks.
