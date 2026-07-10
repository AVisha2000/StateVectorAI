---
name: qllm-research-protocol
description: Audit QLLM evidence and quantum/classical claims for fairness, statistics, controls, resources, and cautious wording. Use for advantage or edge questions and updates to RESULTS.md, study reports, comparisons, or evidence summaries.
---

# QLLM Research Protocol

Use this skill to keep QLLM research claims honest. The project is explicitly designed to avoid over-reading quantum results: a good answer should say what the evidence supports, what it does not support, and what run would settle the next question.

## First Steps

1. Locate the repo root by finding `pyproject.toml` with project name `qllm`.
2. Read `qllm/research_protocol.py` before classifying a claim.
3. Read `RESULTS.md`, the relevant area in `docs/RESEARCH_MAP.yaml`, and `references/evidence-checklist.md`. Treat `docs/RESEARCH_PROGRAM.md` as a plan, not evidence.
4. If the result comes from the dashboard, inspect linked candidate/baseline jobs and final run records before summarizing.

## Evidence Rules

- Prefer paired candidate-vs-baseline comparisons with matched dataset, seed, steps, eval interval, and device target.
- Treat one fair pair as smoke evidence only.
- Require parameter matching or an explicitly justified resource mismatch before discussing model quality.
- Separate trainability observations from architecture observations: frozen/random-circuit controls matter.
- Include wall time and state dimension when quantum components are more expensive.
- Check whether an architecture-aware classical surrogate, matched GRU, low-rank FFN, or Markov control explains the result.

## Claim Language

- Use "no evidence" when the baseline wins or paired mean improvement is not positive.
- Use "smoke evidence" for a positive single pair or too few paired runs.
- Use "positive but not significant" when paired runs improve but confidence/p-value is weak.
- Use "paired empirical edge" only when fairness checks pass, enough pairs exist, and paired stats support it.
- Avoid "quantum advantage" unless the evidence includes fair baselines, statistical support, resource accounting, dequantization/hardware caveats, and the user explicitly wants that framing.

## Output Shape

When reviewing a result, provide:

1. Verdict in one sentence.
2. Fairness status.
3. Metrics and paired statistics.
4. Most plausible non-quantum explanation.
5. Next experiment or control.

Keep the wording suitable for `RESULTS.md`: concrete, dated when useful, and cautious.
