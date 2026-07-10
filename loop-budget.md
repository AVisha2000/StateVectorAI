# QLLM Loop Budget

Primary loop: QLLM Daily Triage, scaffolded from `cobusgreyling/loop-engineering` and adapted for this repo.

## Daily Limits

| Loop | Max runs/day | Max tokens/day | Max sub-agent spawns/run |
|------|--------------|----------------|--------------------------|
| QLLM Daily Triage | Unlimited | 100k | L1: 0; L2: 2 sub-agent spawns only |

Manual QLLM triage runs are unlimited per day; the token budget remains capped. The L2 value of 2 applies only to sub-agent spawns per run, not to the daily run count.

## On Token Budget Exceed

1. Switch to report-only at 80% of the daily token budget.
2. Exit immediately at 100% of the daily token budget.
3. Append the event to `loop-run-log.md`.
4. Add a one-line note to `STATE.md` under High Priority.

## Kill Switch

- Add `loop-pause-all` to `STATE.md` to stop loop work.
- Resume only after the human removes the flag.

## Spend Estimate

```powershell
npx.cmd @cobusgreyling/loop-cost --pattern daily-triage --level L1
```

## Alerts This Period

- None.
