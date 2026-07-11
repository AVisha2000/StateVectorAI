# Test Instructions

The root `AGENTS.md` remains the baseline. Tests are executable research and
software contracts; do not weaken them to accommodate an implementation.

## Test design

- Reproduce the bug or failed invariant first when practical, then implement
  the fix.
- Prefer deterministic seeds, small arrays/circuits, explicit tolerances, and
  direct behavioral assertions. Test shapes and execution only when those are
  the actual contract.
- For quantum/classical comparisons, test both arms through the shared public
  path. Separate mathematical invariants from stochastic performance claims.
- Add boundary/leakage checks for generated sequences and split behavior;
  fixed-seed equality alone is not enough.
- Avoid network, GPU, QPU, live dashboard, global database, and user artifact
  dependencies in the pytest suite. Use temporary paths and CPU-sized cases.
- Do not add broad skips, loosen tolerances without numerical evidence, mock
  away the behavior under test, or assert a research claim from a single
  stochastic run.

## Running tests

One file while iterating:

```powershell
pytest -q tests/test_quantum.py
```

Agent configuration contract:

```powershell
pytest -q tests/test_agent_configuration.py tests/test_verify_changes.py
python scripts/check_agent_setup.py
```

Finish broad or shared-contract changes with:

```powershell
pytest -q
```

If Windows temp ACLs cause `WinError 5`, use a known writable `--basetemp` and
report that rerun explicitly. Distinguish collection/setup failures from test
assertion failures in the handoff.
