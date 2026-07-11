# QLLM Loop Safety

This project is a research codebase with local experiment artifacts, GPU-cost risk, and claim-sensitive result summaries. Loop automation should make the work easier to inspect before it makes the work faster to change.

## Denylist

- Secrets, credentials, `.env`, `.env.*`, auth, payment, or infrastructure files.
- Destructive cleanup of `results/`, `mlruns/`, `mlflow.db`, or `results/qllm_results.db`.
- Long GPU sweeps, high-memory quantum runs, CUDA/JAX installation changes, or dashboard job cancellation.
- Claim-strengthening edits to `RESULTS.md` without `$qllm-research-protocol`.
- Auto-merge, force-push, or branch deletion.

## Human Gates

- Approve before spending GPU time beyond smoke runs.
- Approve before publishing or updating empirical claims.
- Approve before any connector writes to GitHub beyond comments.
- Approve before changing loop level from L1 report-only to L2 assisted fixes.

## Verification

- Use focused tests for the touched path before proposing a fix.
- Use `pytest -q` for broad Python changes.
- Use `npm run build` in `qllm/dashboard/frontend` for dashboard UI changes.
- After the local dashboard path is available, use
  `python scripts/queue_smoke.py --steps 1 --eval-every 1 --device-target cpu`
  for queue/API changes.

## Stop Conditions

- `loop-pause-all` appears in `STATE.md`.
- The same item has failed 3 fix attempts.
- The loop reaches 80% of daily token budget and needs to switch to report-only.
- The loop cannot distinguish generated artifacts from source changes.
