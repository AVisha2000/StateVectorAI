# Documentation Instructions

The root `AGENTS.md` remains the baseline. Documentation is part of the
research evidence system, not post-hoc marketing.

## Research records

- Use `$qllm-research-protocol` for claim-bearing text, comparisons, result
  summaries, evidence levels, or changes to `RESULTS.md`.
- `RESEARCH_PROGRAM.md` is the program roadmap; `RESEARCH_MAP.yaml` is the
  machine-readable exploration index; `ENHANCEMENT_PLAN.md` is the engineering
  backlog. Keep their roles distinct and cross-links relative.
- Preserve negative, null, corrected, failed, OOM, and superseded findings.
  Explain a correction; never silently rewrite history.
- Distinguish observation, inference, hypothesis, mechanism evidence, scaling
  evidence, and hardware evidence. State the studied regime and comparator.
- Do not describe a heuristic task as theorem-faithful, a simulator resource
  as QPU cost, or a diagnostic as advantage.
- Strengthening a conclusion or evidence level is human-gated even when tests
  pass. Cite primary literature for external scientific facts.
- Update `RESEARCH_MAP.yaml` when an area's status, blocker, next decisive test,
  or evidence level materially changes; keep schema values internally
  consistent.

## Validation

Parse the research map after YAML edits:

```powershell
python -c "from pathlib import Path; import yaml; yaml.safe_load(Path('docs/RESEARCH_MAP.yaml').read_text(encoding='utf-8')); print('RESEARCH_MAP.yaml: OK')"
```

For research-protocol code or claim classification changes:

```powershell
pytest -q tests/test_research_protocol.py
```

For agent instruction or operating-model changes:

```powershell
python scripts/check_agent_setup.py
pytest -q tests/test_agent_configuration.py
```

Check links, commands, dates, and file names against the repository. Do not
copy transient run counts or status into multiple documents without a clear
canonical source.
