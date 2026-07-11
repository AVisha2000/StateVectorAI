# Living Plans

This file carries substantial work across agent turns. The parent owns the
plan, integration, deterministic verification, human gates, and final handoff.
Completed implementation details move into canonical documentation; this file
keeps only concise progress, decisions, and current evidence.

## Active plan: local platform completion

Owner: parent agent
Started: 2026-07-10
Objective: complete the local CPU-capable QLLM engineering, evidence,
dashboard, UI, and documentation platform through nine verified milestones.

Scope: repository agent workflow, data/config correctness, causal metrics,
claim protocol, durable runs, localhost safety, local scaling, dashboard
evidence surfaces, and onboarding. GPU/cluster/QPU execution, paid services,
destructive artifact migration, and stronger scientific claims are excluded
without separate approval.

Acceptance evidence:

- Each milestone is implemented on its named `codex/mNN-*` branch.
- Focused checks and change-aware full verification pass before review.
- A fresh read-only verifier reviews material changes.
- The parent inspects the final diff and requests approval before commit,
  merge, or push.
- Existing databases, caches, runs, and research artifacts remain readable and
  are never silently discarded.

Progress:

- [x] M01 Agent system and workflow cleanup (`codex/m01-agent-workflow`),
  delivered to `main` in commit `98b91f7`.
- [x] M02 Trust inputs and configuration (`codex/m02-inputs-config`).
- [ ] M03 Causal two-stream replacement (`codex/m03-causal-two-stream`).
- [ ] M04 Trust claims (`codex/m04-claim-integrity`).
- [ ] M05 Trust runs (`codex/m05-durable-runs`).
- [ ] M06 Local safety and resource reproducibility (`codex/m06-safety-resources`).
- [ ] M07 Local scaling architecture (`codex/m07-local-scaling`).
- [ ] M08 Dashboard and UI evidence completion (`codex/m08-dashboard-evidence`).
- [ ] M09 Documentation and completion audit (`codex/m09-docs-audit`).

Current milestone: M03 causal two-stream replacement.

M02 acceptance evidence:

- `DatasetBundle` carries tokens, tokenizer, boundary/sampler policy, optional
  masks, provenance metadata, and a deterministic config/content identity;
  contextual evaluation no longer depends on module globals.
- Canonical registries are the only source for supported model, component,
  dataset, circuit, backend, readout, and conditioning choices.
- CLI, model-spec, and queue paths reject the same invalid numeric and semantic
  configurations before model initialization.
- Kernel regularization is selected without reading final test labels.
- Imported text datasets record revision, limits, hash, and truncation status.
- Focused data/config/kernel/dashboard tests, full CPU tests, and a fresh
  verifier pass.

M02 progress:

- [x] Deliver M01 and create `codex/m02-inputs-config` from updated `main`.
- [x] Map registry, validation, dataset-bundle, kernel, and import gaps.
- [x] Implement the canonical data/config contracts and compatibility paths.
- [x] Add split-hygiene and import-provenance behavior with regression tests.
- [x] Run focused, change-aware, and full verification.
- [x] Obtain a fresh verifier pass and deliver under standing Git approval.

M02 implementation notes:

- Full transactional paired-job creation remains assigned to M05. M02
  preflights candidate and analogue configuration errors before the first job
  insert; no durability claim is made for later database or process failures.
- Imported corpus character/byte limits bound materialized UTF-8 output.
  Streaming avoids eager full-dataset materialization, but remote shard/chunk
  transfer may exceed the stored-output limit and is labeled accordingly.

M02 focused validation so far:

```text
.venv/Scripts/python.exe -m pytest -q tests/test_config_data.py tests/test_dashboard_lab.py tests/test_advantage.py tests/test_contextual.py tests/test_quantum_data.py --basetemp .tmp/pytest-m02-registry-final
PASS: 115 passed; 15 existing JAX complex128-to-complex64 warnings.
npm.cmd run build  (qllm/dashboard/frontend)
PASS: 856 modules transformed; existing bundle-size advisory only.
desktop + 390px in-app browser inspection
PASS: dataset provenance UI and registry-driven model controls render; narrow
page width equals viewport, wide table is locally scrollable, console clean.
python scripts/verify_changes.py --plan
PASS: selected agent, dashboard, benchmark, CLI, and full Python checks.
python scripts/verify_changes.py --run
PASS: agent-setup, agent-tests, dashboard-build, dashboard-tests,
benchmark-tests, script-syntax, train-entrypoint-tests, and python-tests.
.venv/Scripts/python.exe -m pytest -q --basetemp .tmp/pytest-m02-full
PASS: 206 passed, 1 skipped; 37 existing JAX precision warnings.
git diff --check
PASS (Git emitted only the repository's CRLF checkout notices).
first fresh verifier
NEEDS_WORK: registry-exposed contextual recurrent architectures were rendered
as transformers and architecture switching retained invalid block configs.
verifier fix
PASS: contextual/routed recurrent graphs are quantum/family-honest; recurrent
transitions clear transformer-only fields; global quantum editing is supported;
7 Node frontend transition tests and backend regressions pass.
python scripts/verify_changes.py --run  (final code fingerprint)
PASS: agent-setup, agent-tests, dashboard-frontend-tests, dashboard-build,
dashboard-tests, benchmark-tests, script-syntax, train-entrypoint-tests, and
python-tests.
.venv/Scripts/python.exe -m pytest -q --basetemp .tmp/pytest-m02-full-final
PASS: 215 passed, 1 skipped; 37 existing JAX precision warnings.
final in-app browser pass on code fingerprint
dd08bd54d0d7afaff9661b9a860ebeb9bf15b48aa3ebf8735a4655407924215c
PASS: a delayed loopback proxy exposed the Models loading state, the empty
saved-spec state, the ready transformer state, the contextual-QRNN transition,
the architecture-level quantum inspector, and the shared invalid-qubit error.
PASS: the Datasets view exposed its primary/default-dataset state, empty task
cards, disabled `Importing...` state, and the expected missing-source error.
Direct backend validation returned HTTP 400; the proxy was corrected to relay
upstream HTTP errors instead of misreporting them as HTTP 500.
PASS: Models and Datasets were inspected at desktop and 390x844. Both narrow
pages had 375px client and document widths; the dataset table remained locally
scrollable (341px client / 792px content) with no page-level overflow. Browser
warning/error consoles were empty.
No model spec or job was saved. Two ignored text files created by accidental
QA imports while diagnosing the proxy were removed by exact path; the temporary
QA database was isolated under `.tmp` and no pre-existing artifact was changed.
third fresh verifier review
PASS: current fingerprint, full M02 diff, browser-state matrix, registry and
validation contracts, dataset identity/import/migration behavior, kernel split
hygiene, compatibility, and disclosed limitations were independently reviewed;
M02 is acceptance-ready for its authorized Git delivery.
```

M01 acceptance evidence:

- The local Ollama scribe, tests, verification routing, and workflow references
  are removed without changing the user's Ollama installation.
- GPT-5.6 is the configured project root; Sol planner/verifier, Terra worker,
  Luna explorer, Mini mechanical worker, and Spark text helper profiles are
  syntactically valid and discoverable.
- Exact requested model identifiers are never silently substituted. Runtime
  availability is proven by fresh-session profile smoke or recorded as
  unresolved.
- The engineering/UI roadmaps distinguish completed boundary/dashboard work
  from remaining milestones.

M01 progress:

- [x] Confirm the clean `main` checkpoint and create the milestone branch.
- [x] Verify the official custom-agent file schema.
- [x] Remove the scribe and install deterministic agent-profile validation.
- [x] Reconcile roadmap status and operating documentation.
- [x] Run focused, change-aware, and full verification.
- [x] Obtain the post-restart fresh verifier verdict and request the Git
  delivery gate.

Decisions:

- The local scribe is retired from StateVectorAI; Ollama itself remains
  untouched.
- The requested Luna/Mini/Spark model slugs are configuration inputs but are
  not claimed available until a fresh Codex session successfully starts them.
- Full-window two-stream conditioning will be replaced by a causal model in M03.
- Milestones merge directly to `main` only after explicit approval.

Human gates: on 2026-07-10 the user granted standing approval to stage, commit,
merge to `main`, and push each M01-M09 milestone after its required checks and
fresh verifier pass. This does not authorize claim-bearing `RESULTS.md`
changes, GPU/QPU work, paid services, destructive artifact/database cleanup,
or experiment cancellation.

Latest validation:

```text
python scripts/check_agent_setup.py
PASS: Agent setup validation passed.
.venv/Scripts/python.exe -m pytest -q tests/test_agent_configuration.py tests/test_verify_changes.py --basetemp .tmp/pytest-m01-focused-3
PASS: 16 passed.
python scripts/verify_changes.py --plan
PASS: selected agent-setup and agent-tests only; deleted helpers are not executed.
python scripts/verify_changes.py --run
PASS: agent-setup and agent-tests.
.venv/Scripts/python.exe -m pytest -q --basetemp .tmp/pytest-m01-full
PASS: 168 passed, 1 skipped, 36 existing JAX dtype warnings (187.38s).
git diff --check
PASS.
codex --version
UNVERIFIED: local codex.exe returned Windows Access is denied, so Luna/Mini/Spark
runtime availability requires a fresh Codex session after this config is loaded.
fresh verifier review
NEEDS_WORK: exact model/profile runtime discovery is not proven; restart Codex,
smoke-start every profile, and remove or disable any unsupported profile before
delivery.
post-restart profile smoke
PASS: planner, terra_worker, luna_explorer, mini_worker, and spark_helper all
started under their configured profile names and completed bounded read-only
packets. Child contexts do not expose the underlying model identifier; static
validation pins the requested strings and prevents silent repository-level
substitution.
post-restart fresh verifier
PASS: M01 acceptance, scope, safety, profile startup, and deterministic
evidence reviewed; no material acceptance criterion remains unproven.
```

## Completed foundation

### Agent operating system rollout

Completed 2026-07-10 in commit `d4a02e6`. Installed scoped instructions,
project skills, planner/explorer/verifier roles, deterministic verification,
and human gates. Historical verification: 169 passed, 1 skipped.

### Boundary-safe synthetic datasets

Completed 2026-07-10 in commit `83c5fa1`. Synthetic datasets retain trajectory
identity through loading, splitting, Markov-control generation, and batch
sampling while the legacy flat adapter remains available. Historical focused
verification: 21 passed; change-aware full Python verification passed.

## Entry template

```markdown
## Active plan: <objective>

Owner: <owner>
Started: YYYY-MM-DD
Objective: <one outcome>

Scope: <systems and explicit non-goals>

Acceptance evidence:

- <observable result>

Progress:

- [ ] <ordered milestone>

Decisions: <material decisions and unknowns>

Human gates: <approval state>

Latest validation:

    <exact command and result>
```
