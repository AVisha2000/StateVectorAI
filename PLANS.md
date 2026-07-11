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
- The parent inspects the final diff before commit, merge, and push; the
  user's standing approval covers safe in-scope Git delivery.
- Existing databases, caches, runs, and research artifacts remain readable and
  are never silently discarded.

Progress:

- [x] M01 Agent system and workflow cleanup (`codex/m01-agent-workflow`),
  delivered to `main` in commit `98b91f7`.
- [x] M02 Trust inputs and configuration (`codex/m02-inputs-config`),
  delivered to `main` in commit `bb4a11e`.
- [x] M03 Causal two-stream replacement (`codex/m03-causal-two-stream`),
  delivered to `main` in commit `e5e5230`.
- [x] M04 Trust claims (`codex/m04-claim-integrity`),
  delivered to `main` in commit `2418d7b`.
- [x] M05 Trust runs (`codex/m05-durable-runs`), delivered to `main` in commit
  `e901139`.
- [x] M06 Local safety and resource reproducibility
  (`codex/m06-safety-resources`), delivered to `main` in commit `182192c`.
- [x] M07 Local scaling architecture (`codex/m07-local-scaling`), delivered
  to `main` in commit `dcec571`.
- [x] M08 Dashboard and UI evidence completion (`codex/m08-dashboard-evidence`), delivered on cloud branch in commit `f3daa85`.
- [ ] M09 Documentation and completion audit (`codex/m09-docs-audit`): in progress on cloud handoff.

Current milestone: M09 documentation and completion audit.

M08 acceptance evidence:

- Existing cockpit, diagrams, comparisons, studies, per-layer specifications,
  and visual builder remain backward compatible while evidence views expose
  metric type, claim identity/status, paired and equivalence statistics,
  power, complete fairness mismatches, analogue limitations, immutable
  manifests, checkpoints/recovery state, and resource/capability ledgers.
- Backend payloads make single-seed, unmatched, missing-control,
  negligible-gain/high-cost, and invalid-protocol conditions explicit and
  unavoidable; the UI renders them as prominent interpretation warnings rather
  than hiding them behind optional detail panels.
- Primary routes have deterministic loading, empty, error, filtering,
  comparison, rerun, and study-report behavior at desktop and narrow widths.
  Existing response keys remain compatible and new fields are additive.
- Focused dashboard/backend tests, frontend tests/build, CPU queue/API smoke,
  desktop+narrow browser inspection, change-aware/full CPU verification, and a
  fresh verifier PASS complete without GPU/QPU execution or claim promotion.

M08 progress:

- [x] Deliver M07 and create `codex/m08-dashboard-evidence` from updated main.
- [x] Audit route, payload, component, and warning coverage against the M04-M07
  evidence contracts and existing UI behavior.
- [x] Define additive backend view models and unavoidable warning semantics.
- [x] Implement backend/API integration with regression tests.
- [x] Implement frontend evidence presentation and resilient route states.
- [x] Run backend/frontend/queue/browser and full CPU verification.
- [x] Obtain a fresh verifier PASS and deliver M08 under standing Git approval.


M09 acceptance evidence:

- Researcher onboarding explains the claim ladder, required controls, resource
  accounting, dashboard vocabulary, and benchmark entry points without
  strengthening existing scientific conclusions.
- Engineer onboarding documents safe extension paths for models, synthetic
  tasks, benchmarks, dashboard evidence cards, and research summaries.
- Documentation links are current, roles of README/DATA/RESULTS/research-map/
  dashboard docs remain distinct, and claim-bearing wording stays cautious.
- Static documentation checks, agent setup, change-aware verification, and any
  focused docs/research tests pass before delivery.

M09 progress:

- [x] Resume plan from cloud while desktop is offline; worktree was clean at
  handoff.
- [x] Add researcher onboarding claim ladder and research-question map.
- [x] Add engineer development guide for common extension paths and checks.
- [x] Align dashboard evidence vocabulary with backend-authoritative warnings.
- [x] Run documentation-focused and change-aware verification.
- [x] Review final diff before commit and PR record.

M09 design constraints:

- Do not edit `RESULTS.md` or promote any claim level in this milestone unless
  separately routed through research-protocol review and human approval.
- Documentation must preserve negative/null results and describe dashboards as
  evidence surfaces, not proof of advantage.
- No GPU/QPU run, paid service, destructive artifact migration, remote exposure,
  or experiment cancellation is part of M09.

M08 design constraints:

- Research protocol code remains the authority for statistical/fairness
  interpretation; the dashboard presents those results without recomputing or
  strengthening them in JavaScript.
- Warning severity and visibility derive from explicit backend codes, with
  human-readable text as an additive presentation field. Missing evidence is
  never converted into a neutral or successful state.
- Existing dashboard response keys and saved jobs remain readable. Contract
  changes are additive unless an adapter and regression fixture are included.
- Browser and queue validation stay local and CPU-only. No `RESULTS.md` edit,
  claim promotion, artifact deletion, remote exposure, GPU/QPU work, or paid
  service is part of M08.

M08 validation results:

```text
.venv/Scripts/python.exe -m pytest -q tests/test_dashboard_lab.py --basetemp .tmp/pytest-m08-dashboard-final
PASS: 74 passed after verifier corrections.
npm.cmd test; node --test src/api.test.js src/evidenceView.test.js; npm.cmd run build
PASS: 7 model-config tests, 6 API/evidence tests, and 861-module production build; existing bundle-size advisory only.
python scripts/queue_smoke.py --url http://127.0.0.1:8173 --steps 1 --eval-every 1 --device-target cpu
PASS: isolated copied database job #77 completed one step and wrote its latest checkpoint; live port 8000 queue untouched.
desktop + 390x844 in-app browser inspection
PASS: comparison warnings and ledgers, run workspace, studies empty state, experiment filtering and filtered-empty state, and recoverable API error render correctly; isolated server stopped after QA.
.venv/Scripts/python.exe scripts/verify_changes.py --run
PASS: agent setup/tests, frontend tests/build, and dashboard tests.
.venv/Scripts/python.exe -m pytest -q --basetemp .tmp/pytest-m08-full-final
PASS: 351 passed, 2 skipped; 44 existing dependency/JAX precision warnings.
git diff --check
PASS: no whitespace errors; line-ending notices only.
fresh read-only verifier
PASS: warning-first legacy route coverage, truthful standalone single-seed semantics, polling recovery, additive compatibility, and server-authoritative interpretation verified after corrections.
```

M07 acceptance evidence:

- Recurrent, contextual, routed, two-stream, and transplant transformations use
  static-shape `lax.scan`/vectorized execution only where numerical semantics
  and public model contracts remain intact.
- Backend capability metadata is canonical and additive across state access,
  expectations, probabilities, sampling, gradients, noise, reset, and dynamic
  circuits. Unsupported capabilities fail explicitly rather than being inferred.
- Exact state-vector and approximate/sampled/noisy outcomes are labeled and
  stored distinctly; small CPU overlap fixtures compare implementations only
  where their semantics genuinely coincide.
- Focused numerical parity, gradient, shape/JIT, backend-capability, and
  unsupported-path tests pass together with change-aware/full CPU verification
  and a fresh verifier PASS. No GPU/QPU workload is launched.

M07 progress:

- [x] Deliver M06 and create `codex/m07-local-scaling` from updated `main`.
- [x] Inventory hot-loop transformation candidates and backend/API drift.
- [x] Implement one coupled static-shape transformation path at a time with
  numerical/gradient parity tests.
- [x] Add canonical backend capability metadata and exact/approximate overlap
  fixtures without weakening existing dispatch.
- [x] Run focused, change-aware, and full CPU verification.
- [x] Obtain a fresh verifier PASS and deliver M07 under standing Git approval.

M07 design constraints:

- Sequence recurrence in QRNN/contextual/routed models, transplant circuit
  depth, and two-stream prefix evaluation are already scanned/vectorized. M07
  therefore targets the transplant compiler's Python-per-optimizer-step host
  dispatch; static gate-construction and named transformer-block loops remain
  unchanged because scanning them would add dynamic-index or parameter-tree
  risk without demonstrated benefit.
- Restart selection remains host-controlled and deterministic. Only the
  optimizer step axis is scanned, preserving the established loss convention
  and bounded memory behavior.
- The current TensorCircuit adapter is an exact dense state-vector path, not an
  MPS or approximate implementation. Capability metadata must say so, and M07
  will report that no approximate backend exists rather than fabricate one.
- Transformation boundaries are selected for measurable compile/runtime or
  memory benefit, not merely to replace readable Python with JAX primitives.
- Static shapes, carry state, masking, reset behavior, RNG use, and gradient
  semantics are explicit and covered by small deterministic CPU fixtures.
- Capability declarations describe actual adapter behavior. Configured backend
  names, native-JAX implementations, and simulator approximations remain
  distinct in manifests and result metadata.
- No GPU/QPU run, dependency installation, paid service, destructive artifact
  migration, claim promotion, or `RESULTS.md` edit is part of M07.

M06 acceptance evidence:

- Dashboard CORS and bind defaults permit loopback origins/hosts only. Remote
  access requires one explicit opt-in contract and emits an unavoidable
  startup/API warning without weakening path or origin checks silently.
- Every dashboard-controlled file read/write validates canonical containment
  within its configured data, result, or artifact root and rejects traversal,
  symlink escape, and untrusted absolute paths with actionable errors.
- Run manifests and result/job payloads record additive resource evidence for
  compile time, steady-state execution time, wall time, model parameters,
  state dimension, circuit calls, device/backend, precision, and available
  peak-memory information, with measured/estimated/unavailable provenance.
- The authoritative CPU development profile and optional GPU/WSL profile are
  explicit, non-conflicting, and documented; resolved environment versions
  remain attached to immutable run manifests.
- Focused loopback/CORS/path/symlink/remote-warning/resource fixtures,
  dashboard tests/build, change-aware/full CPU verification, and a fresh
  security/resource verifier return PASS. No GPU workload is launched.

M06 progress:

- [x] Deliver M05 and create `codex/m06-safety-resources` from updated `main`.
- [x] Inventory bind/CORS/path boundaries and resource/dependency evidence.
- [x] Implement local-access and filesystem-containment contracts with tests.
- [x] Implement additive resource ledgers and environment profiles with tests.
- [x] Run focused, isolated queue-smoke, frontend-build, change-aware, and full
  CPU verification. The verifier's GPU-file human gate is satisfied only for
  the approved profile/document edits; no GPU setup or workload was executed.
- [x] Obtain a fresh verifier PASS; Git delivery follows under standing
  approval.

M06 design constraints:

- Remote access is opt-in, visibly labeled, and never inferred from a nonlocal
  bind address alone. Loopback remains the zero-configuration default.
- Filesystem authorization is based on resolved containment in explicit roots,
  not string prefixes. Existing artifacts are read-only unless the requested
  operation already owns their exact run directory.
- Resource fields distinguish measured, estimated, available, and unsupported
  values. Simulator state size or memory estimates are never presented as QPU
  cost or measured accelerator consumption.
- Optional GPU/WSL dependencies cannot alter the authoritative CPU profile and
  are not installed or exercised in this milestone.
- Dashboard jobs execute on their requested CPU/GPU target instead of merely
  labeling it; requested and resolved execution devices remain distinct from
  the quantum simulator backend/device. M06 validation stays CPU-only.
- Circuit-call accounting records exact zero for classical runs and a labeled
  logical-forward derivation for quantum runs. Backend/gradient execution calls
  remain explicitly unsupported until instrumented rather than being inferred
  from JIT traces.
- Legacy `wall_seconds` remains compatible. New timing scopes use synchronized
  completed work and distinguish first-step compile-plus-execution,
  post-warmup steady-state steps, the active training loop, and total fit time.
- No GPU/QPU run, paid service, destructive artifact/database migration,
  claim promotion, or `RESULTS.md` edit is part of M06.

M05 acceptance evidence:

- Every new experiment and execution has immutable UUID identity plus canonical
  config, code, data, and environment hashes; legacy rows remain readable and
  explicitly report unavailable identity fields.
- Atomic latest/best checkpoints contain parameters, optimizer state,
  completed step, RNG lineage, metric state, and resume metadata. A resumed
  CPU run continues from the next step without replaying completed work.
- Additive, repeatable SQLite migrations make step logging unique/idempotent
  and preserve all existing databases, runs, jobs, studies, and artifacts.
- Dashboard workers claim jobs transactionally in SQLite with worker IDs,
  leases/heartbeats, deterministic restart recovery, stale-job handling, and
  terminal GPU-reservation release; the in-memory queue is only a wake-up hint.
- Generation is architecture-neutral at its public boundary and returns an
  explicit supported or unsupported outcome rather than assuming one model
  shape or silently falling back.
- Focused checkpoint/resume, idempotent-step, transactional-claim,
  restart/recovery, stale-job, terminal-release, manifest/hash, and generation
  fixtures pass; one-step CPU queue smoke, change-aware/full verification, and
  a fresh verifier return PASS.

M05 progress:

- [x] Deliver M04 and create `codex/m05-durable-runs` from updated `main`.
- [x] Inventory persistence, schema, queue, artifact, RNG, resume, and
  generation paths.
- [x] Implement additive identity/manifest and atomic checkpoint/resume
  contracts through CLI and dashboard jobs.
- [x] Replace in-memory queue authority with transactional DB claims,
  heartbeats, recovery, and idempotent logging.
- [x] Make generation support explicit across architecture families.
- [x] Run focused, queue-smoke, change-aware, frontend-compatibility, and full
  CPU verification.
- [x] Obtain a fresh verifier PASS; Git delivery follows under standing
  approval.

M05 design constraints:

- Existing SQLite files and artifact directories are never deleted, renamed,
  rewritten in place, or assigned fabricated hashes/UUIDs. Migrations are
  additive and safe to run repeatedly.
- Database transactions, not process-local memory, determine which worker owns
  a queued job. A lease transition must be atomic and terminal transitions must
  be idempotent.
- Resume reproducibility includes optimizer state and the exact next RNG/step;
  loading parameters alone is warm start, not resume, and must not be labeled
  otherwise.
- Latest and best checkpoint writes use a temporary sibling plus atomic
  replacement so interruption cannot expose a partial checkpoint.
- Code/data/environment hashes record what is knowable locally and use explicit
  unavailable/dirty states rather than silently claiming a clean repository.
- `experiment_uuid` groups a scientific submission or comparison while one
  immutable `run_uuid` identifies each logical job/run. Process recovery keeps
  that run identity; a deliberate resume fork records its parent and source
  checkpoint instead of rewriting the original identity.
- Historical `steps` rows remain untouched. New UUID-backed logging uses an
  additive canonical table with a `(run_uuid, step, metric)` key, same-value
  retries are idempotent, and conflicting retries fail loudly.
- `RunOptions` carries operational identity, checkpoint, resume, and artifact
  settings outside the scientific `ExperimentConfig`; CLI and dashboard jobs
  translate to the same contract.
- No GPU/QPU run, paid service, destructive artifact migration, claim
  promotion, or `RESULTS.md` edit is part of M05.

M05 validation evidence:

```text
.venv/Scripts/python.exe -m compileall -q qllm benchmarks scripts tests
PASS.
.venv/Scripts/python.exe -m pytest -q tests/test_durable_runs.py
PASS: 38 passed, including exact resume, fork crash boundaries,
generated-to-cache retry, legacy migration, result/manifest immutability,
SQLite claims/recovery, one-step CPU queue execution, and generation families.
.venv/Scripts/python.exe -m pytest -q tests/test_dashboard_lab.py
tests/test_integration.py tests/test_research_protocol.py
PASS: 95 passed; one existing JAX complex128-to-complex64 warning.
npm.cmd run build  (qllm/dashboard/frontend)
PASS: 857 modules transformed; existing bundle-size advisory only.
python scripts/verify_changes.py --run
PASS: agent-setup, 17 agent tests, frontend tests/build, 69 dashboard tests,
67 benchmark tests, script syntax, 34 train/config tests, and the full CPU
suite (306 passed, 1 skipped; 41 existing JAX precision warnings).
desktop + 390px in-app browser inspection
PASS: completed, loading, empty-curve, and error workspace states render;
artifact paths use backend-provided persisted locations; narrow document width
equals viewport width; warning/error console is clean. No job was created,
cancelled, or modified during browser QA.
compatibility/data-integrity auditor
PASS: additive migrations, historical projections, snapshot authority,
checkpoint integrity, queue fencing/recovery, and benchmark identity reviewed;
38 durable tests and diff check independently passed.
fresh verifier
PASS: manifest-only fork recovery and generated-to-cache access-provenance
transition preserve stable identity and lineage; 38 durable tests and diff
check independently passed. No RESULTS.md change or artifact leak found.
```

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
