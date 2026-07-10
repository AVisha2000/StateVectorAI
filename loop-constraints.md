# QLLM Loop Constraints

The `loop-constraints` skill reads this file at the start of every loop run. These rules are binding for automated or recurring loop work.

## Push And Merge

- Do not push before telling the user.
- Never auto-merge to main.
- Open draft PRs only after the user asks for publishing.

## QLLM Research Safety

- Do not claim quantum advantage from a single run or unmatched baseline.
- Use `$qllm-research-protocol` before strengthening conclusions in `RESULTS.md`.
- Do not start long GPU sweeps, high-memory jobs, CUDA/JAX installs, or dashboard job cancellations without explicit user approval.
- Do not delete or rewrite `results/`, `mlruns/`, `mlflow.db`, or `results/qllm_results.db` during triage.

## Paths

- Never edit `.env`, `.env.*`, secrets, credentials, auth, or payment files.
- Treat generated experiment outputs as artifacts to inspect, not cleanup targets.
- Avoid unrelated refactors in `qllm/`, `benchmarks/`, `scripts/`, and dashboard files.

## Code

- L1 loop runs are report-only.
- For L2 fixes, keep one fix per run and run focused tests before proposing completion.
- Never disable tests or loosen assertions to make a run pass.
- Escalate after 3 failed fix attempts on the same item.

## Communication

- State what will be inspected before a loop run.
- Put ambiguous decisions in `STATE.md` under Handoff.
- Keep `loop-run-log.md` append-only except for pruning old entries.

## Budget

- Switch to report-only at 80% of the daily cap.
- Exit immediately if `loop-pause-all` appears in `STATE.md`.
