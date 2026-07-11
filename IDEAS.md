# Feature and Research Idea Inbox

This is the low-friction scratchpad for ideas, feature requests, experiments,
workflow improvements, and questions. Write ideas in plain language. Do not
wait until you know the implementation path.

The inbox is intentionally not a backlog that agents may implement
automatically. An agent may review and refine entries, but implementation starts
only after the user explicitly promotes an entry.

## How to add an idea

Append a dated entry using this small template:

```text
## [IDEA-YYYY-MM-DD-short-name] Draft

Owner: [person or team]
Date: YYYY-MM-DD
Type: feature | bug | research | workflow | question

### Raw idea
[What should exist or change? Include the user problem and why it matters.]

### Notes
- [links, examples, constraints, or open questions]

### Agent review
Status: inbox
Decision: pending
Refined objective: pending
Acceptance evidence: pending
Risks/gates: pending
Plan link: pending
```

## States

`inbox` means the idea is unreviewed. `reviewed` means an agent has clarified
the problem, scope, acceptance evidence, dependencies, and risks with the user
still in the loop. `ready` means the user agrees with the refined objective
but has not yet authorized implementation. `approved` means the user has
explicitly asked to implement it. `implementing`, `verified`, `done`, and
`parked` describe the subsequent lifecycle.

When an idea becomes `approved` and is substantial, create or update a focused
entry in `PLANS.md`; keep the inbox entry as the traceability link. Small fixes
can go directly from `approved` to implementation with an in-turn plan.

Archive completed or parked entries in place rather than deleting their
history. Keep the raw idea and the final plan link together so later research
decisions remain auditable.
