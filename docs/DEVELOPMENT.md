# Engineer Guide

This guide maps the extension paths that are safe to use without reading the
whole repository. Start Codex or Claude Code and run commands from the
repository root so the scoped `AGENTS.md` contracts and project skills are
discovered. Claude Code reaches the same canonical files through import-only
`CLAUDE.md` and `.claude/skills/` adapters.

## Install and verify

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements-cpu.txt
python -m pip install --no-deps -e .
python scripts/check_agent_setup.py
pytest -q --basetemp .tmp/pytest-dev
```

`requirements-cpu.txt` is the authoritative CPU profile. GPU/WSL setup is
separate and human-gated; follow [`GPU_SETUP.md`](../GPU_SETUP.md) rather than
altering the CPU environment in place.

## Canonical contracts

| Concern | Canonical location |
| --- | --- |
| Supported architectures/components/datasets/backends/readouts/conditioning | `qllm/registry.py` |
| Dataclasses, flat/YAML adapters, semantic validation | `qllm/config.py` |
| Dataset metadata, boundaries, masks, hashes, sampler policy | `qllm/data/datasets.py:DatasetBundle` |
| Model composition and `tokens -> logits` contract | `qllm/models/model.py:build_model` |
| Circuits and backend dispatch/capabilities | `qllm/quantum/circuits.py`, `qllm/quantum/backends.py` |
| Training, generation, checkpoints, resource summaries | `qllm/train/loop.py`, `qllm/train/artifacts.py` |
| Claim registry/schema and loading | `research/claims.yaml`, `qllm/claims.py` |
| Statistics, fairness, analogue evaluation, classification | `qllm/research_protocol.py` |
| Durable SQLite run/job state | `qllm/resultsdb.py`, `qllm/dashboard/runner.py` |
| Dashboard API/view models | `qllm/dashboard/server.py`, `lab.py`, `workspace.py`, `studies.py`, `evidence.py` |
| React evidence presentation | `qllm/dashboard/frontend/src/components/` and `pages/` |

Supported-value lists must come from `qllm/registry.py`. Do not introduce a
second dashboard or CLI list. `validate_config()` is the shared validation
entry point and must reject invalid combinations before expensive work.

## Change workflow

1. Inspect `git status --short` and preserve existing artifacts/user changes.
2. Read the root and nearest nested `AGENTS.md` plus the matching canonical
   project skill. Claude Code's adapters must lead to the same files.
3. Trace the narrow data/config/model/API path and state acceptance evidence.
4. Change the smallest coherent contract and add behavioral tests.
5. Run focused checks, `python scripts/verify_changes.py --plan`, then
   `python scripts/verify_changes.py --run` and the broader CPU suite warranted
   by the blast radius.
6. Inspect `git diff --check`, the final diff, and research meaning.

GPU/QPU runs, paid compute, destructive artifact/database operations, remote
exposure, claim promotion, and `RESULTS.md` strengthening remain human gates.

## Add a model component

For a new FFN (the attention/head/architecture paths are analogous):

1. Implement the Flax module under `qllm/classical/` or `qllm/quantum/`.
2. Add the supported name once in the appropriate registry in
   `qllm/registry.py`.
3. Wire construction through `qllm/models/model.py`; preserve output shape
   `(batch, time, vocab)` and functional JAX/Flax state/randomness.
4. Update `uses_quantum` and resource/capability reporting when applicable.
5. Define an architecture-aware classical control and claim-specific allowed
   differences; parameter matching alone is not a universal fairness proof.
6. Ensure model graph/spec serialization uses the same config/registry path.
7. Add shape, JIT, gradient, config rejection, integration, and control tests.

Useful focused checks:

```powershell
pytest -q tests/test_classical_model.py tests/test_integration.py
pytest -q tests/test_quantum.py tests/test_gradients.py tests/test_metrics.py
pytest -q tests/test_config_data.py tests/test_qnlp_suite.py
```

## Add a dataset or synthetic task

1. Add the kind to the dataset registry and `DataConfig` validation path.
2. Return a `DatasetBundle`; preserve trajectory boundaries, masks,
   provenance, content/config hashes, metadata, and sampler policy.
3. Split independent trajectories before sampling. Never flatten first or draw
   windows across a generated boundary.
4. Give cache/import identity every generator parameter and revision needed to
   reproduce content. Keep the flat `load_dataset()` adapter compatible.
5. Add a mandatory classical/control generator when the scientific question
   requires one.
6. Test boundary-adjacent windows, split isolation, cache identity, masks,
   provenance, truncation warnings, and invalid config.

Use `tests/test_config_data.py`, `tests/test_quantum_data.py`,
`tests/test_contextual.py`, and `tests/test_seq_cancellation.py` as patterns.

## Add or change a backend

Declare capabilities for state access, expectations, probabilities, sampling,
gradients, noise, reset, and dynamic circuits. Record exact/sampled/noisy/
approximate semantics explicitly and fail unsupported operations rather than
silently falling back. Add small CPU overlap tests only where two backends have
the same semantics. A dense TensorCircuit path is not an MPS implementation.

## Extend the dashboard

1. Keep SQLite migrations additive, repeatable, and compatible with historical
   databases.
2. Compute scientific interpretation in Python (`qllm/research_protocol.py`
   and `qllm/dashboard/evidence.py`), not React.
3. Preserve existing response keys; make new payload fields additive or ship an
   adapter and regression fixture.
4. Render structured warnings before verdicts/metrics. Missing evidence must
   remain unavailable.
5. Reuse `EvidenceWarnings`, `EvidenceSummary`, and `RunLedger` for a new
   comparison card or evidence-bearing route.
6. Cover loading, empty, filtered-empty, error/retry, rerun, and narrow-width
   behavior.

Validation:

```powershell
pytest -q tests/test_dashboard_lab.py --basetemp .tmp/pytest-dashboard
npm.cmd test --prefix qllm/dashboard/frontend
npm.cmd run build --prefix qllm/dashboard/frontend
python scripts/queue_smoke.py --steps 1 --eval-every 1 --device-target cpu
```

Use an isolated database/server for browser QA when the normal local dashboard
has active work. Never restart, cancel, or mutate a live queue just to test UI.

## Durable-run rules

- Use immutable experiment/run UUIDs and manifest hashes.
- Checkpoints contain parameters, optimizer state, completed step, RNG lineage,
  and resume metadata; write atomically.
- Step logging is unique/idempotent. DB workers claim transactionally, heartbeat,
  recover stale leases, and release terminal reservations.
- Never delete or silently rewrite historical databases, caches, runs, or
  artifacts during a migration.

The more detailed component map is
[`qllm-model-development/references/component-map.md`](../.agents/skills/qllm-model-development/references/component-map.md).
