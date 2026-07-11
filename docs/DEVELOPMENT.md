# QLLM Development Guide

QLLM changes should preserve the project rule that evidence comes before
claims. This guide points contributors to the smallest extension path and the
checks that usually prove it.

## Before changing code

1. Read the nearest `AGENTS.md` files for the paths you will touch.
2. Pick the matching project skill from the root instructions.
3. Inspect `docs/RESEARCH_MAP.yaml` when a change affects an explored research
   area, and treat `docs/RESEARCH_PROGRAM.md` as the roadmap rather than
   evidence.
4. Define the baseline, control, and validation command before implementing.

## Extension paths

| Change type | Primary files | Required evidence |
| --- | --- | --- |
| New model or layer | `qllm/models/`, `qllm/quantum/`, `qllm/classical/`, `configs/` | Shape tests, parameter-count or analogue rationale, focused model tests, and CPU verification. |
| New synthetic task | `qllm/data/`, `configs/`, `DATA.md` | Boundary-aware sampling tests, deterministic seeds, cache/provenance metadata, and Markov or classical controls where relevant. |
| New benchmark | `benchmarks/`, `scripts/`, `configs/` | Resume/idempotence behavior, recorded config/data/code identity, and a dry CPU-scale fixture before any long run. |
| Dashboard API/UI card | `qllm/dashboard/`, `qllm/dashboard/frontend/`, `tests/test_dashboard_lab.py` | Additive payload fields, warning-first rendering for missing evidence, backend tests, frontend tests/build. |
| Research summary | `RESULTS.md`, `docs/RESEARCH_MAP.yaml`, study reports | Research-protocol review, paired statistics when enough seeds exist, and explicit limitations. |

## Example: adding a new FFN variant

1. Add the layer or dispatch option in the smallest model module that owns the
   behavior.
2. Add a config that differs from the nearest baseline only in the intended
   fields.
3. Record parameter counts or the analogue policy; do not call a comparison
   parameter-matched unless the final counts support it.
4. Add tests for initialization, forward shape, config validation, and any
   unsupported backend path.
5. Run the focused model tests and `python scripts/verify_changes.py --run`.

## Example: adding a synthetic task

1. Preserve trajectory identity through generation, splitting, caching, and
   batching; never flatten away sequence boundaries before sampling.
2. Store generator parameters, seed, cache identity, and split metadata.
3. Add a classical control such as a Markov twin when the research question is
   long-memory structure.
4. Test train/eval sampling boundaries and reproducibility with small fixtures.
5. Update `DATA.md` only with cautious wording supported by those fixtures or
   completed runs.

## Example: adding a dashboard comparison card

1. Build the interpretation in Python first, preferably from
   `qllm/research_protocol.py` or existing dashboard evidence helpers.
2. Return additive JSON fields so older jobs and saved databases remain
   readable.
3. Include structured warnings for one-seed comparisons, unmatched parameters,
   missing controls, resource-heavy negligible gains, side-information metrics,
   and synthetic-data mismatches when applicable.
4. Render warnings before optimistic summaries in the React component.
5. Cover loading, empty, error, filtered-empty, desktop, and narrow-width states
   where the route exposes new behavior.

## Verification quick map

```bash
python scripts/check_agent_setup.py
python scripts/verify_changes.py --plan
python scripts/verify_changes.py --run
pytest -q
```

Use focused tests while iterating. For example, model changes usually start with
`pytest -q tests/test_quantum.py`, dashboard changes with
`pytest -q tests/test_dashboard_lab.py`, and research-protocol changes with
`pytest -q tests/test_research_protocol.py`.

## Human gates

Stop for explicit approval before GPU/QPU work, paid services, destructive
artifact changes, claim strengthening, public dashboard exposure, or remote Git
operations not already authorized for the current milestone.
