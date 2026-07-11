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
- [x] M02 Trust inputs and configuration (`codex/m02-inputs-config`),
  delivered to `main` in commit `bb4a11e`.
- [x] M03 Causal two-stream replacement (`codex/m03-causal-two-stream`),
  delivered to `main` in commit `e5e5230`.
- [ ] M04 Trust claims (`codex/m04-claim-integrity`).
- [ ] M05 Trust runs (`codex/m05-durable-runs`).
- [ ] M06 Local safety and resource reproducibility (`codex/m06-safety-resources`).
- [ ] M07 Local scaling architecture (`codex/m07-local-scaling`).
- [ ] M08 Dashboard and UI evidence completion (`codex/m08-dashboard-evidence`).
- [ ] M09 Documentation and completion audit (`codex/m09-docs-audit`).

Current milestone: M04 trust claims.

M04 acceptance evidence:

- `research/claims.yaml` is the canonical, schema-validated claim registry and
  records claim ID, level, status, evidence, contradictions, limitations,
  metric type, and next decisive test without strengthening `RESULTS.md`.
- Paired effects use deterministic bootstrap intervals for the mean, exact
  sign-flip inference where enumerable (seeded Monte Carlo otherwise), explicit
  practical-equivalence margins, and pilot-variance power planning.
- Three-pair pilots cannot reach the `paired empirical edge` level regardless
  of apparent win rate or nominal sign-flip result.
- Generator, split, initialization, minibatch, circuit, and
  hardware-calibration seed axes are explicit in study/run contracts and remain
  backward-compatible with the legacy scalar seed.
- Claim-specific fairness schemas support documented intentional differences,
  report every mismatch, and require parameter/resource-matched analogue
  ladders before stronger evidence labels.
- Positive, negative, equivalent, underpowered, unfair, and mismatched fixtures
  pass focused tests; change-aware and full CPU validation pass; a fresh
  verifier returns PASS.

M04 progress:

- [x] Deliver M03 and create `codex/m04-claim-integrity` from updated `main`.
- [x] Inventory claim-bearing paths, statistical primitives, seed contracts,
  fairness checks, and analogue evidence surfaces.
- [x] Implement the canonical claim/statistics/seed/fairness contracts with
  additive payload compatibility.
- [x] Wire claims and complete mismatch reporting through studies, comparisons,
  and evidence ladders without editing `RESULTS.md`.
- [x] Run focused, change-aware, and full verification.
- [x] Obtain a fresh verifier pass; Git delivery follows under standing
  approval.

M04 design constraints:

- Scientific status changes remain conservative: missing controls, inadequate
  replication, or unfair comparisons can only lower or invalidate a claim.
- Deterministic routines must accept an explicit seed and produce stable output
  independent of process hash ordering.
- New payload fields are additive; existing dashboard keys and legacy scalar
  seeds remain supported.
- No experiment, GPU/QPU workload, paid service, artifact rewrite, or
  `RESULTS.md` edit is part of this milestone.
- The 19 stable `RESEARCH_MAP.yaml` area IDs are the initial canonical claim
  IDs. Canonical level, ledger status, replication status, assessment status,
  and display verdict remain separate dimensions.
- `RESULTS.md` is an immutable historical evidence reference, not the current
  verdict authority. Corrected, blocked, relabeled, and rerun-required states
  come from the research program/map and the new claim ledger.
- M04 records legacy seed coupling truthfully instead of inventing independent
  execution: initialization, minibatch, and circuit axes alias the scalar seed;
  generator seed is separate where applicable; split is deterministic; hardware
  calibration is not applicable to current local runs.
- Paired inference operates within a fixed claim/metric/dataset/sweep cell and
  pairs across independent seeds. Dataset or sweep cells never count as extra
  replications.

M03 acceptance evidence:

- Every encoder summary at position `t` is computed only from real-token
  embeddings at positions `<= t`; classical and quantum encoders receive the
  identical cumulative-prefix feature tensor.
- FiLM and bias conditioning use per-position summaries. Token conditioning
  interleaves each summary immediately before its corresponding real token and
  still returns one logit row per real input token.
- Shared validation and direct model guards reject an expanded token stream
  that exceeds the configured positional capacity.
- Deterministic leakage tests cover quantum and classical encoders under all
  three conditioning modes, as well as the unconditioned control.
- Historical `two-stream-v1` full-window results remain immutable, are marked
  teacher-forced side-information and rerun-required, and cannot be selected as
  a strict-autoregressive dashboard champion. New benchmark runs use a distinct
  causal suite identifier. `RESULTS.md` remains human-gated and unchanged.
- Focused two-stream/config/dashboard tests, change-aware and full CPU checks,
  and a fresh verifier pass.

M03 progress:

- [x] Deliver M02 and create `codex/m03-causal-two-stream` from updated `main`.
- [x] Define the causal prefix, token ordering, and sequence-capacity contract.
- [x] Map all two-stream dispatch, evidence, and historical-result surfaces.
- [x] Implement the causal model/config contract with leakage regressions.
- [x] Add historical result protocol labeling without rewriting artifacts.
- [x] Run focused, change-aware, and full verification.
- [x] Obtain a fresh verifier pass; Git delivery follows under standing
  approval.

M03 design decisions:

- Prefix summaries include the current input token, matching next-token LM
  semantics: logit `t` may use tokens `<= t` but never a future token.
- `model.max_seq_len` is the internal positional capacity. Active token
  conditioning requires `2 * train.seq_len` positions; FiLM, bias, and the
  unconditioned control require `train.seq_len`.
- Quantum prefix evaluation stays vectorized through `QuantumCore` over the
  same `(batch, time, d_model)` prefix features used by the classical control.
- No historical experiment is rerun in this milestone and no GPU/QPU workload
  is launched.

M03 browser evidence:

- The production dashboard was inspected against an isolated copy of the
  authoritative SQLite database. The legacy suite card labels
  `two-stream-v1` rerun-required without a best-perplexity promotion; suite and
  run detail pages display the full protocol limitation; the suite chart and
  best badge are absent; and the dataset evidence page selects current causal
  candidates instead of historical two-stream rows.
- Desktop and 390x844 checks covered the suite index, suite detail, run detail,
  dataset evidence, overview highlights, and the dataset-filter interaction.
  Both narrow pages stayed at 390px document width and browser console,
  page-error, and failed-request collections were empty.
- The in-app browser attempt first used the obsolete `/suites` route, which
  correctly rendered no React route. After identifying the valid
  `/results/legacy` entry route, the approved local Playwright fallback used
  system Edge and completed the rendered QA matrix; no browser dependency was
  installed.

M03 validation results:

- Focused causal/config/protocol/dashboard suite: `121 passed` with six known
  JAX complex-precision warnings.
- `python scripts/verify_changes.py --run`: PASS for agent setup/tests,
  frontend tests/build, dashboard tests, benchmark tests, and the complete
  Python suite. The corresponding plan fingerprint before this validation-note
  update was
  `3296dec0f26790cdec6a26cce118516c5080a87c9900336fcbd3a7de42646f5c`.
- Independent full CPU suite with repository-local temporary storage:
  `244 passed, 1 skipped`; the 41 warnings are the existing JAX complex128 to
  complex64 precision notices.
- `npm run build` PASS (857 modules; existing large-chunk advisory only) and
  `git diff --check` PASS (Windows line-ending notices only).
- Fresh read-only verifier: PASS. Independent checks covered 24 causal/model
  cases, nine historical protocol/dashboard cases, a direct metric-contract
  probe, research-map YAML parsing, and the complete M03 diff. The verifier
  confirmed that `RESULTS.md` is unchanged and that the remaining uncertainty
  is the intentionally rerun-required historical evidence itself.

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
