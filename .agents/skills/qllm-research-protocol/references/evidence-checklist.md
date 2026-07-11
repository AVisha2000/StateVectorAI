# Evidence Checklist

## Files To Check

- `qllm/research_protocol.py`: paired stats, fairness checks, claim classification, resource ledger.
- `qllm/dashboard/evidence.py`: dashboard evidence payloads.
- `qllm/dashboard/analogues.py`: classical analogue definitions.
- `qllm/models/model.py`: parameter counting and matched classical FFN logic.
- `RESULTS.md`: prior project conclusions.
- `docs/RESEARCH_MAP.yaml`: hypotheses, status, dependencies, and coverage; not proof by itself.
- `docs/RESEARCH_PROGRAM.md`: planned methods and acceptance criteria; not completed evidence.
- `GPU_QUEUE.md`: deferred decisive experiments.

## Minimal Fairness Fields

- Same dataset or explicit dataset role.
- Same seed for paired runs unless doing aggregate unpaired analysis.
- Same training steps.
- Same eval interval and eval budget.
- Same device target.
- Parameter count difference recorded.
- Same preprocessing/data generator settings.

## Controls By Family

- Quantum FFN on text: frozen-circuit control and parameter-matched low-rank/classical FFN.
- Quantum attention: classical attention with matched parameter budget and same training schedule.
- QRNN on monitored Ising: planted floor, GRU ladder, Markov-control twin.
- Contextual QRNN: GRU ladder across hidden sizes and routed-vs-plain contextual cells.
- Two-stream: quantum sentence encoder vs parameter-matched classical sentence encoder, same conditioning mode.
- Kernel/advantage screens: classical kernel family, random Fourier surrogate, finite-shot/noise ladder when relevant.

## Reporting Template

```markdown
Verdict: <claim level and one-sentence reason>

Fairness: <pass/fail fields, parameter delta, resource caveats>

Metrics: <candidate vs baseline, seed count, paired mean, CI/p-value when available>

Interpretation: <what this supports and what it does not support>

Next check: <smallest decisive run/control>
```

## Historical Correction Checklist

- Read `qllm/research_protocol.py`, `research/claims.yaml`, and the relevant
  `docs/RESEARCH_MAP.yaml` entry before deciding whether a correction is needed.
- Obtain explicit approval before changing claim-bearing research prose.
- Preserve historical metrics, plots, run IDs, and artifact provenance.
- Record `teacher_forced_side_information` when side information enters the
  evaluation path, and `rerun_required` when a causal replacement is needed.
- State that the historical result does not establish a strict autoregressive
  conclusion; identify the replacement causal protocol.
- Add a static regression for the corrected wording and rerun relevant
  protocol/dashboard tests.

## Red Flags

- "Quantum wins" based on one seed.
- Baseline has fewer steps, different data, or weaker parameter budget.
- Frozen circuit performs the same as trained circuit.
- Full classical model wins but only a tiny classical control is discussed.
- GPU/CPU wall time is omitted for expensive quantum simulations.
- Result is from teacher-forced side-information models but described as strict autoregressive performance.
