# Living Plans

This file carries substantial work across agent turns. The parent owns the
plan, integration, deterministic verification, human gates, and final handoff.
Completed implementation details move into canonical documentation; this file
keeps only concise progress, decisions, and current evidence.

## Completed plan: backend hardening and research contracts

Owner: Codex backend track
Started: 2026-07-12
Objective: finish the remaining dashboard hardening and coverage work, then
ship strict Designer validation and a claim-safe Atlas ontology API without
changing frontend-owned files or promoting research claims.

Scope: concurrent additive SQLite migrations, bounded mutation request bodies,
hermetic environment-status tests, API route-contract coverage, Designer
circuit validation, Atlas ontology serving, OpenAPI snapshots, and backend
coordination evidence. `qllm/dashboard/frontend/**`, GPU/QPU work, paid
services, dependency/environment changes, claim promotion, artifact rewrites,
commits, pushes, and merges are excluded.

Acceptance evidence:

- Concurrent fresh and legacy `ResultsDB` construction converges on one schema
  and index set without errors or row rewrites.
- Mutating `/api/*` bodies are bounded while streaming; declared and chunked
  oversized requests return HTTP 413 before route side effects, and existing
  access, origin, and JSON media-type policy remains intact.
- Environment-status branches and every FastAPI API method/path pair have
  hermetic tests that fail when an uncovered route is added.
- Designer choices come from the canonical registry and validation rejects
  unsupported backend/ansatz/readout combinations without constructing or
  queueing a model.
- Atlas grouping and relation metadata is schema-validated while claim level,
  claim status, and replication status remain separate values loaded from the
  canonical research map.
- Derived result classification uses `assessment_level`; the legacy dashboard
  `claim_level` alias remains boundary-only and cannot collide with the
  canonical claim ledger in verdict storage.
- Focused tests, OpenAPI snapshot checks, change-aware verification, the full
  CPU suite, CPU-only queue smoke, and a fresh verifier review pass.

Progress:

- [x] Rebase `backend-enhancements` onto current `origin/main` and refresh the
  scoped dashboard/workflow/verification instructions.
- [x] Make additive `ResultsDB` migrations concurrency-safe and test fresh and
  legacy first-open races.
- [x] Enforce bounded mutating API bodies without weakening local-only access.
- [x] Add status and API route-contract coverage.
- [x] Add strict Designer validation and regenerate OpenAPI.
- [x] Rename derived classifier output to `assessment_level` while retaining a
  compatibility-only dashboard alias.
- [x] Add the validated Atlas ontology service; the six display groupings were
  approved while map-owned evidence fields stay non-overridable.
- [x] Run focused, change-aware, full CPU, OpenAPI, and queue-smoke checks; update
  the backend log.
- [x] Obtain fresh independent pre-delivery and post-publication Sol Ultra reviews.

Decisions:

- The mutation-body limit is fixed at 1 MiB and applies only to state-changing
  `/api/*` requests; it is intentionally enforced before JSON parsing.
- Designer validation is advisory and side-effect-free. It never reports
  advantage and does not trust client-computed parameter or gate estimates.
- Atlas status and evidence dimensions remain sourced from
  `docs/RESEARCH_MAP.yaml`; ontology metadata cannot promote them.
- Loopback dataset imports retain trusted-local file and URL support. The user
  accepted this L5 policy behind the strengthened local access boundary.
- Verification child processes use a short repository-keyed system temp root,
  avoiding synced-worktree atomic-write failures without changing dependencies.

Human gates: Atlas research grouping and Git publication were approved by the
user and completed on 2026-07-13. GPU/QPU execution, paid services, environment
changes, claim promotion, artifact rewrites, and frontend edits remain closed.
D4 remains stopped pending provider and daily budget.

Latest validation:

```text
pytest -q tests/test_durable_runs.py -k concurrent_resultsdb_initialization_converges_atomically
PASS: 2 passed.
pytest -q tests/test_dashboard_security.py
PASS: 15 passed, 1 skipped; one dependency deprecation warning.
pytest -q tests/test_dashboard_status.py tests/test_dashboard_routes.py
PASS: 78 tests across hermetic status branches and all 60 API route pairs.
pytest -q tests/test_dashboard_designer.py
PASS: 19 passed, including QRNN/MPS applicability and derivation checks.
pytest -q tests/test_dashboard_atlas.py
PASS: 11 passed across exact map coverage, non-overridable evidence fields,
dimension validation, live relations, and the typed HTTP contract.
pytest -q tests/test_research_protocol.py tests/test_dashboard_verdicts.py
PASS: 38 passed; assessment and canonical claim levels remain distinct.
python scripts/dump_openapi.py --check
PASS: committed snapshot is current.
pytest -q tests/test_verify_changes.py
PASS: 39 passed; normal checks retain 600 seconds, the full suite receives 900,
explicit overrides win, and Atlas ontology changes trigger human review.
pytest -q tests/test_durable_runs.py::test_warm_start_manifest_is_honest_and_its_checkpoint_resumes
  --basetemp %LOCALAPPDATA%/Temp/qllm-warm-start-regression
PASS: 1 passed after isolating a transient OneDrive `WinError 5`; verifier child
temp roots now use the same short OS-local strategy.
python scripts/verify_changes.py --run
CHECKS PASS: agent setup; 56 agent tests; dashboard bundle 306 passed, 1
skipped; full CPU suite 663 passed, 1 skipped with 48 known dependency/JAX
warnings. Overall status is `human_review_required` for the Atlas ontology gate
approved by the user on 2026-07-13.
pytest -q --basetemp .tmp/pytest-full-final
PASS: 663 passed, 1 skipped; 48 known dependency/JAX warnings.
python scripts/queue_smoke.py --url http://127.0.0.1:8877 --steps 1
  --eval-every 1 --device-target cpu
PASS: isolated loopback job 1 completed step 1 with a durable checkpoint; the
temporary server was stopped afterward.
post-rebase focused protocol/verifier/OpenAPI tests
PASS: 67 passed.
post-rebase pytest -q with OS-local basetemp
PASS: 665 passed, 1 skipped; 48 known dependency/JAX warnings.
post-rebase CPU queue smoke at http://127.0.0.1:8878
PASS: isolated job 1 completed step 1 with a durable checkpoint; server stopped.
post-publication GPT-5.6 Sol Ultra review
PASS: local and remote main resolve to `44b35fd`; all backend requirements are
complete, scoped correctly, and published without unrelated files.
```

## Completed plan: backend audit hardening

Owner: Codex backend track
Started: 2026-07-12
Objective: implement the P0 and requested P1 hardening findings in
`docs/BACKEND_AUDIT.md` without changing research conclusions, frontend code,
or the local-only trust boundary.

Scope: mutation-request CSRF defenses, lease-heartbeat resilience, shared
dashboard helpers, GPU/high-memory claim exclusivity, idempotent submission,
400-vs-500 error classification, DB-layer forbidden-score enforcement, and
dashboard verification routing. `qllm/dashboard/frontend/**`, GPU/QPU runs,
claim promotion, dependency/environment changes, commits, pushes, and merges
are excluded.

Acceptance evidence:

- Cross-site-shaped or non-JSON mutation requests are rejected before route
  side effects; valid local JSON mutations retain their current contract.
- Transient SQLite heartbeat failures retry without losing ownership; unexpected
  heartbeat failures set `ownership_lost` instead of silently killing renewal.
- Config decoding, curve projection, job variant selection, and artifact-root
  resolution use shared helpers with identical behavior across dashboard views.
- A second worker cannot claim a GPU/high-memory job while another such job is
  running; CPU work remains claimable.
- Concurrent duplicate `run_uuid` submission returns the existing job, expected
  client errors remain HTTP 400, and unexpected server failures become HTTP 500.
- Forbidden composite/advantage score keys are rejected by `ResultsDB` itself.
- Dashboard source changes select security, OpenAPI, verdict, diagnostics,
  stream, and lab tests in `scripts/verify_changes.py`.
- Focused tests, `scripts/dump_openapi.py --check`, change-aware verification,
  and the full CPU suite pass with fresh output.

Progress:

- [x] Rebase `backend-enhancements` onto latest `origin/main` and read the audit,
  scoped instructions, and dashboard/runner/verification skills.
- [x] Implement and verify P0 M1, M3, and T1.
- [x] Implement and verify P1 M2, L1, L2, L6, and TG3.
- [x] Run broad CPU-only verification and obtain a clean GPT-5.6 Sol Ultra
  read-only verifier verdict.

Decisions and residuals:

- Submission identity ignores only runtime lease, dashboard-tracking, artifact
  mirror, and comparison-topology metadata. New jobs persist immutable
  `lab.submission.comparison_mode`; legacy rows without it use a deliberately
  permissive compatibility fallback.
- Explicitly configured remote CORS origins remain usable for JSON mutations;
  default loopback mode still rejects hostile Origin/Sec-Fetch-Site requests.
- The verifier temp root is short enough for legacy Windows path limits. A
  transient OneDrive `WinError 5` was isolated by rerunning the identical
  affected-module command under a writable `%LOCALAPPDATA%` temp root.
- Exact focused, full-suite, OpenAPI, and CPU queue-smoke results are reported
  in the final handoff from the frozen worktree diff.

Human gates: no GPU/QPU execution, paid services, environment changes, claim
promotion, artifact rewrites, commits, pushes, or merges without explicit user
approval. No frontend contract edit is planned; any discovered UI contract need
will be logged in `docs/BUILD_COORDINATION.md` instead.

## Active plan: UI redesign Phase 2 — Bench, Run detail, Verdicts, diagnostics viz

Owner: apex (Claude Code, Opus 4.8) · branch `ui-redesign`
Started: 2026-07-11
Objective: turn the scaffolded Bench / Verdicts surfaces and the placeholder
`/runs/:id` route into real, data-bound surfaces, plus reusable diagnostics
charts, honoring verification-first integrity (no composite advantage score,
diagnostics labeled as diagnostics, claim classification stays backend-owned).

Scope: `qllm/dashboard/frontend/**` only. Non-goals (backend-blocked, tracked in
docs/BUILD_COORDINATION.md): SSE swap of the poll interval (`/stream/jobs`),
OpenAPI type codegen, and the persistent `/verdicts` + `/jobs/{id}/diagnostics`
stores — the UI is built to light up when these ship and degrade gracefully now.

Key facts (from backend-shape inventory):
- Claim ladder is backend-owned. Runtime `classify_claim` vocabulary
  (`empirical/quantum-inspired/smoke/weak/fragile/...`) differs from
  RESEARCH_MAP.yaml `claim_levels` (`untested→formal`); no mapping exists.
  Render `evidence_ladder.steps[]` + `verdict`/`claim` verbatim; never hardcode.
- `/jobs/{id}/comparison` is single-seed (`paired_stats/equivalence/power` null);
  no real seed-band here. `seedBand` helper reserved for study/verdict-store data.
- Diagnostics available today via `/jobs/{id}/model-tests` →
  `summary.quantum_diagnostics` (grad_var_*, meyer_wallach_q, expressibility_kl +
  per-metric `availability`); per-step `grad_norm_ratio` is in the `curve`.
  SNR + gradient-variance scaling-fit are NOT computed for dashboard jobs yet.

Acceptance evidence:

- `npm test` + `npm run build` green in qllm/dashboard/frontend.
- Bench queues a real CPU job via POST /api/jobs; `/runs/:id` renders run detail
  with diagnostics labeled as diagnostics; Verdicts renders a backend-driven
  scorecard + ladder with no composite score; all surfaces show calm
  loading / empty / not-yet-built states; both themes at desktop + narrow width.

Progress:

- [x] API status surfacing + proposed-endpoint hooks (graceful 404)
- [x] Pure curve helpers (mergeCurve/mergeComparison/seedBand/logLinearFit) + tests
- [x] Reusable chart components (ComparisonCurve, TrainabilityChart, ScalingChart, SeedBandChart)
- [x] Run detail surface (RunDetail.jsx) + clickable Runs rows + `/runs/:id` route
- [x] Verdicts surface (list + `/verdicts/:id` detail) + Scaling view (`/runs/scaling/:groupId`)
- [x] Bench surface with real POST /api/jobs queueing (CPU-default, GPU-gated)
- [x] Verify: npm test, npm run build clean, browser QA (routing + graceful states + both themes)
- [x] Post UI log note to docs/BUILD_COORDINATION.md on main (`3a6678d`)
- [x] Wire `/api/stream/jobs` SSE (change-token dedupe → invalidate; polling fallback)
- [x] Align to shipped contracts: diagnostics per-dimension `{status,value,reason}`;
      persistent `/verdicts` + `/verdicts/{id}` store (canonical claim vs derived
      assessment kept distinct); `/status` five-field shape (System running KPI)
- [ ] OpenAPI type codegen from `qllm/dashboard/openapi.json` (no longer backend-blocked — the snapshot is on `main`; the codegen work itself remains)

Decisions: render backend claim data verbatim (evidence_ladder/verdict/claim);
no composite advantage score in the scorecard; wall-clock labeled simulator cost;
default Bench device_target to CPU (GPU stays human-gated); Verdicts detail
derives from a job's single-seed comparison until the persistent verdict store
ships; diagnostics read summary.quantum_diagnostics today, degrade to
"awaiting backend" for SNR/scaling-fit.

Human gates: no GPU runs, no merge to main, no claim-bearing RESULTS.md edits.
Backend real-data QA not run — fastapi/jax/pennylane not installed in this
environment (no venv); reported as an environment limitation, not a product gap.

Latest validation:

    npm test                → 54 pass / 0 fail (node --test)
    npm run build           → built in ~3.1s, dist emitted, no errors
    browser (dev server, no backend): shell + all new routes render, legacy
      /launch→/bench redirect works, Verdicts store→comparison fallback,
      graceful empty/error states everywhere, no console errors, both themes
    Branch `ui-redesign` pushed to origin (`ad2a7c1`, force-with-lease over the
      pre-rebase Phase 1 tip). Backend endpoints lived on `backend-enhancements`
      at capture time; since fully merged to `main` (`b54ad30`) — the frontend
      now consumes the shipped `/designer/circuit` and `/atlas/ontology`
      contracts directly (2026-07-13).

### Phase 3 — Atlas (in progress, `a858ab0`)

- Slice A (model + seed ontology + list) and Slice B (Cytoscape graph) shipped.
- Data: `src/lib/atlasOntology.seed.js` transcribes RESEARCH_MAP's 19 areas
  verbatim (status/claim/replication), grouped into 6 domains — originally a
  labeled placeholder; `/api/atlas/ontology` has since shipped and the seed is
  now only the offline/older-backend fallback (2026-07-13).
  `atlasModel` joins to the live `/verdicts` store; `outcome_class` is a color
  bucket from RESEARCH_MAP `status` only (never a verdict, no composite score);
  `claim_level` (map ladder) and `replication_status` stay separate; null /
  "no advantage found" cells are first-class (own `--null` token, counted).
- Verified: `npm test` 71/71, build clean; the List view + summary + filters +
  node detail render correctly in-browser (nulls first-class confirmed).
- **Graph caveat:** the Cytoscape graph's element/style transforms are
  unit-tested and code-split, but its **canvas rendering could not be visually
  verified in this sandbox browser** — cytoscape's renderer emits nothing (even
  under `forceRender`) while a hand-drawn canvas paints/reads back fine, and the
  screenshot tool times out. Environment limitation, not a logic defect. The
  List is the default, verified render path; the graph needs a real-browser check.
- Not done (deferred/gated): Slice C polish (URL deep-link/expand-collapse UI),
  Slice D static-export helper, and the **human-gated public Atlas export**
  (internet exposure — out of scope until scoped with the user).
- **Graph resolution:** the cytoscape graph was rebuilt as hand-authored SVG
  (`atlasSvgLayout.js` + `AtlasGraphSvg.jsx`); cytoscape dependency removed
  entirely. Verified in-browser (19 clickable cells, correct shapes/colors,
  click→detail with claim/replication distinct). Graph now the equal of the List.

### Phase 4 — research loop (Library shipped; copilot human-gated)

- `Library` surface built (`7d52c5e`): consumes the backend's shipped, ungated
  bounded arXiv metadata scan (`POST /discover/arxiv/scan`) + the D4 capability
  gate (`GET /research/capabilities`), rendered as an honest boundary panel.
  `npm test` 68/68, build clean, verified in-browser (graceful when the service
  isn't reachable). Papers table + quota display.
- **Blocked on a human gate:** the Discover copilot / vault synthesis needs a
  paid LLM + embedding provider and a per-day cost budget (AGENTS.md human gate),
  and the integration is backend-owned. User approved *pursuing* it; still needs
  the specific provider + budget named and keys configured before the backend
  enables it. Frontend Discover UI will consume it once exposed.

### Phase 4 Discover + Phase 5 Designer — shipped

- **Discover** (`e202fcd`): capability-aware surface; copilot + idea queue stay
  disabled (no spend) until the paid provider is approved. Points to the working
  Library scan and to Bench/Designer.
- **Designer** (`905ed3d`): parameterized-ansatz circuit editor over registry.py
  families (hardware_efficient/reuploading/ising) with a hand-authored SVG circuit
  view; live properties (gates/params/entangling/depth); classical↔quantum toggle;
  Send-to-Bench carries `quantum_overrides`; Validate consumes the proposed
  `/designer/circuit` round-trip gracefully. Verified in-browser (ansatz switch
  re-renders, params update). npm test 73/73.
- **All ten surfaces are now real.** Remaining backend/human-gated items
  (updated 2026-07-13): ~~`/designer/circuit` round-trip~~ shipped and consumed
  live; the paid research copilot (D4, still user-gated); OpenAPI codegen (the
  snapshot is on `main`; the codegen work remains); the public Atlas export
  (exposure-gated).

## Completed workstream: GitHub Actions reliability and supply-chain hardening

Owner: parent agent
Started: 2026-07-11
Objective: make every repository change receive an appropriate deterministic
CI signal while reducing duplicate runs and hardening third-party Actions and
dependency updates.

Scope: add default Python, dashboard frontend, and dependency-review workflows;
pin every third-party Action to a reviewed immutable commit; add concurrency and
main-only push routing to existing workflows; configure weekly Dependabot for
GitHub Actions, Python, and npm; align change-aware verification, regression
tests, and dashboard validation docs. Runtime behavior, experiments, research
claims, remote repository settings, branch protection, secrets policy, and
artifact/database state are excluded. The user later authorized Git delivery
of this completed worktree after verification and final review.

Acceptance evidence:

- A stable Python CI check runs the exact CPU profile and complete pytest suite
  on Linux and Windows for every pull request and every push to `main`.
- Frontend changes run deterministic `npm ci`, all checked-in Node tests, and a
  production build on Linux and Windows.
- Pull requests receive dependency review, while weekly grouped Dependabot
  updates cover Actions, Python, and npm manifests.
- Existing and new workflows use least-privilege permissions, cancel superseded
  runs, avoid duplicate feature-branch push runs, and pin Actions by full SHA.
- The local verifier routes workflow edits to the checks they define, and tests
  reject mutable action references or missing CI contracts.
- Dashboard documentation names the same complete frontend validation commands
  used locally and in CI.

Progress:

- [x] Audit repository manifests, test surfaces, verifier routing, workflow
  history, branch protection, security settings, and current Action releases.
- [x] Implement workflows, Dependabot, immutable pins, and run controls.
- [x] Update verifier routing, regression tests, and validation documentation.
- [x] Run focused, frontend, YAML, change-aware, and full CPU verification.
- [x] Obtain a fresh read-only verifier review and inspect the complete diff.

Decisions:

- Keep the default Python check unfiltered so it can become a reliable required
  status check; specialized workflows remain path-scoped.
- Preserve the existing `node --test` package script because it already
  discovers all six checked-in files (22 tests); only stale docs need correction.
- Use current official releases verified through GitHub on 2026-07-11:
  checkout v7.0.0, setup-python v6.3.0, setup-node v6.4.0, and dependency-review
  v5.0.0, each pinned to its exact commit.
- Do not mutate remote settings in this workstream. `main` protection,
  Dependabot security updates, secret scanning, and push protection remain a
  separately approved GitHub administration step.

Human gates: no commit, push, branch-rule/security-setting mutation, GPU/QPU
work, experiment execution, claim edit, or artifact/database action was
authorized by this plan alone. On 2026-07-11 the user explicitly authorized
committing and pushing all completed in-scope changes after verification. No
remote branch rule, security feature, secret setting, PR, or merge is authorized.

Latest validation:

```text
.venv/Scripts/python.exe scripts/verify_changes.py --plan
PASS: selected agent setup/tests, frontend tests/build, dependency profile
validation/tests, and the complete Python suite; no human gate was selected.
.venv/Scripts/python.exe scripts/verify_changes.py --run --timeout 900
FOCUSED PASS: agent setup; 44 agent/verifier tests; all 22 frontend tests and
the 864-module production build; dependency profile static checks; and 37
dependency/verifier tests. The first complete Python pass reported 453 passed,
1 skipped, and one transient dashboard static-directory import failure.
isolated dashboard-security retry with a repository-local basetemp
PASS: the previously failing remote-mode case passed immediately.
.venv/Scripts/python.exe -m pytest -q --basetemp .tmp/pytest-ci-retry
PASS: 454 passed, 1 skipped; 48 existing dependency/JAX warnings in 378.47s.
npm.cmd ci; npm.cmd test; npm.cmd run build
PASS: clean lockfile install, 22 frontend tests, and 864-module production
build. npm audit reports one existing moderate esbuild advisory and one existing
high Vite advisory; the available fix is a separately reviewed Vite 8 major
upgrade that the new Dependabot configuration will propose.
workflow/dependabot YAML parse; pip check; CPU runtime-profile validation;
compileall; git diff --check
PASS: six YAML files parse, all 20 installed CPU pins match, Python sources
compile, and no whitespace error is present (checkout line-ending notices only).
fresh read-only verifier
PASS: all 12 in-scope paths, workflow semantics, immutable Action references,
verifier routing, regression contracts, documentation, evidence, exclusions,
and the complete diff were reviewed; no blocking criterion remains.
```

## Completed workstream: QLLM skill catalog refresh and forward-test

Owner: parent agent
Started: 2026-07-11
Objective: refresh reusable QLLM skill guidance after the engineering backlog
closeout, and add a narrowly scoped issue-closeout skill where the catalog has
a durable workflow gap. Re-audit the published catalog against real routes,
completed durability/resource contracts, and a fresh issue-closeout exercise.

Scope: strengthen the dashboard, research-protocol, and code-verification
skills; add `qllm-issue-closure` with its required Claude discovery bridge;
update skill routing and the operating-model catalog; and add policy-regression
coverage. Product behavior, claims, experiments, and remote issue operations
are excluded.

Acceptance evidence:

- Dashboard guidance requires deterministic frontend tests and rendered,
  theme-aware local QA for visual changes.
- Research guidance preserves historical evidence while correcting causal
  interpretation only after the existing human gate.
- Verification routing covers frontend tests and evidence-sensitive review.
- A discoverable issue-closeout workflow reconciles acceptance criteria,
  current repository evidence, and external-state gates before closure.
- Model and experiment workflows cover the now-canonical dataset/config,
  checkpoint/resume, queue, idempotence, telemetry, and backend contracts.
- A forward-test can apply issue closeout without mutating local or remote
  state, and its default prompt does not imply automatic closure authority.

Risks and gates: skill prose must not silently grant GitHub closure or
claim-edit authority; no issue, commit, push, or research claim is changed by
this work without separate approval.

Validation: `python scripts/check_agent_setup.py`, focused agent-configuration
tests, `python scripts/verify_changes.py --plan`, and its selected safe checks.

Progress:

- [x] Update canonical skills, references, catalog routing, and bridge.
- [x] Add regression coverage and run deterministic validation.
- [x] Integrate the post-publish catalog audit and issue forward-test findings.
- [x] Revalidate every changed skill and the repository agent contract.

Post-publish decisions:

- Do not add another skill: the existing model, experiment, dashboard,
  research, verification, issue-closeout, orchestration, and loop domains cover
  the reusable workflows without a new ownership boundary.
- Keep the issue-closeout skill, but move its detailed evidence matrix into the
  bundled checklist and make the default prompt explicitly non-mutating.
- The read-only issue-#1 forward-test found its final run/comparison/task report
  acceptance criterion unmet even though the issue is closed. Treat that as a
  separate product/remote-state follow-up; this skill refresh did not reopen,
  comment on, or modify the issue.
- Preserve concurrent Claude role model-pin edits as separately owned work;
  local agent validation passes, but this skill audit does not claim runtime
  availability of those Claude model identifiers.
- Concurrent Claude role model-pin edits and their setup-validator/test support
  remain separate uncommitted work. Do not stage, rewrite, or publish that
  mixed-ownership packet with this skill refresh.

Latest validation:

- `quick_validate.py .agents/skills/qllm-issue-closure`: passed.
- `python scripts/check_agent_setup.py`: passed.
- `python -m pytest -q tests/test_agent_configuration.py
  tests/test_verify_changes.py --basetemp .tmp\\pytest-skill-refresh`: 35 passed.
- `python scripts/verify_changes.py --plan`: selected agent setup and agent
  configuration tests.
- `python scripts/verify_changes.py --run` with `TMP`/`TEMP` set to the
  writable workspace temp root: passed (agent setup and agent tests).
- `git diff --check`: passed.
- Post-publish review: all eight skills passed `quick_validate.py`;
  `python scripts/check_agent_setup.py` passed; focused agent tests reported
  38 passed; `verify_changes.py --plan` selected agent setup/tests and
  `--run` passed; the fresh read-only verifier returned `PASS`.

## Completed workstream: Codex idea-to-implementation playbook

Owner: parent agent
Started: 2026-07-11
Objective: give the repository a low-friction idea inbox and an explicit,
human-in-the-loop path from rough feature idea to triaged implementation.

Scope: add `IDEAS.md`, document the Codex/Claude operating rhythm, explain when
to use the inbox versus `PLANS.md`, and link the playbook from onboarding.
Product behavior, research claims, experiment execution, and Git delivery are
excluded.

Acceptance evidence:

- A user can paste an unstructured idea into one obvious file without needing
  to know the repository architecture first.
- The playbook defines review, refinement, approval, implementation, triage,
  verification, and completion states.
- Agent delegation remains bounded and the user retains gates for spend,
  hardware, claims, destructive actions, and Git publication.
- The documentation names exact startup and verification commands.

Progress:

- [x] Add the idea inbox template.
- [x] Add the Codex playbook and link it from README onboarding.
- [x] Verify documentation and agent-configuration checks.

Decisions:

- Raw ideas live in `IDEAS.md`; accepted substantial work moves to `PLANS.md`.
- The user explicitly promotes an idea before implementation agents write code.
- An implementation loop is a triage run with one parent owner, disjoint file
  ownership, and a fresh verification pass.

Human gates: no GPU/QPU/cluster run, paid service, claim promotion,
destructive artifact/database action, commit, push, merge, or publication is
authorized by this documentation work.

Latest validation: see the final response for the exact commands and results.

## Completed workstream: Codex and Claude Code interoperability

Owner: parent agent
Started: 2026-07-11
Objective: expose the existing verification-first QLLM agent operating system
to both Codex and Claude Code without duplicating its substantive guidance.

Scope: preserve `AGENTS.md`, `.agents/skills/`, the delegation contract, and
`PLANS.md` as the portable policy core; add thin Claude Code discovery and role
adapters; share deterministic Stop-hook verification; validate both client
configurations and drift in CI; and update onboarding. Product code, research
claims, experiment artifacts, GPU/QPU execution, paid services, and Git
delivery are excluded.

Acceptance evidence:

- Every root or nested `AGENTS.md` has an import-only `CLAUDE.md` peer, so one
  instruction hierarchy serves both clients.
- Every canonical QLLM skill is discoverable from Claude Code through a thin,
  drift-checked adapter that points back to `.agents/skills/`.
- Claude Code has bounded planner, explorer, worker, helper, and verifier roles
  with the same delegation contract and human gates as Codex.
- Codex and Claude Code both invoke `scripts/verify_changes.py` at Stop with a
  protocol-correct response and an infinite-loop fuse.
- `scripts/check_agent_setup.py`, focused agent tests, change-aware
  verification, CI path routing, documentation, and the final diff all agree
  on the portable-core/client-adapter architecture.

Progress:

- [x] Inventory the current instruction hierarchy, skills, roles, hooks,
  validators, tests, CI, and onboarding while preserving the dirty research
  plan below.
- [x] Verify current Codex and Claude Code configuration mechanisms against
  official product documentation.
- [x] Add Claude Code instruction, skill, role, and hook adapters.
- [x] Reconcile delegation/CPU-smoke drift and generalize deterministic
  validation, tests, CI, and onboarding for both clients.
- [x] Run focused and change-aware checks, obtain a fresh read-only review, and
  inspect the complete diff.

Decisions:

- The pasted product-level Claude prompt is design input only and is not copied
  into the repository.
- `AGENTS.md` and `.agents/skills/` remain canonical. Claude files contain only
  the minimum product-specific discovery or execution adapter.
- Import adapters are used instead of committed symlinks because this checkout
  is native Windows/OneDrive and symlink support is not portable there.
- Claude role definitions inherit the session model rather than pinning
  transient Anthropic model identifiers.
- No blanket permission grants or unverified command-interception policy is
  added. Existing human gates remain authoritative, and deterministic checks
  are enforced at Stop.

Human gates: no GPU/QPU/cluster run, paid service, claim promotion,
destructive artifact/database action, remote dashboard exposure, commit,
push, merge, or publication is authorized by this workstream.

Latest validation:

```text
git status --short --branch
PASS: main matches origin/main; the pre-existing PLANS.md research-continuation
work remains the only user change before this workstream's additive entry.
official Codex and Claude Code configuration audit
PASS: thin imports, native client adapters, bounded roles, and shared hooks are
supported; Claude runtime smoke remains unavailable because `claude` is not
installed in this environment.
python scripts/check_agent_setup.py
PASS: shared instructions, both client adapters, roles, skills, hooks, and
local-ignore contracts are valid.
.venv/Scripts/python.exe -m pytest -q tests/test_agent_configuration.py tests/test_verify_changes.py --basetemp .tmp/pytest-agent-interop-final
PASS: 25 passed.
.venv/Scripts/python.exe scripts/verify_changes.py --run --timeout 120
PASS: agent-setup and agent-tests; no training, dashboard job, GPU, QPU, or Git
mutation selected.
skill-creator quick_validate.py
PASS: all seven Claude skill bridges and all three changed canonical skill
folders are structurally valid.
workflow/settings syntax audit and git diff --check
PASS: CI YAML and Claude settings JSON parse; no whitespace errors (Git emitted
only checkout line-ending notices).
fresh read-only verifier
NEEDS_WORK on first review: a bridge could append duplicated workflow prose
while preserving metadata and its canonical link. The checker now requires one
exact generated bridge template and the negative fixture proves appended policy
is rejected. Follow-up verdict: PASS; no material finding remains. Real client
discovery and Stop-hook execution remain unproven until Codex and Claude Code
can be smoke-tested outside this restricted environment.
```

## Proposed plan: engineering backlog verification and gap closure

Owner: parent agent
Started: 2026-07-11
Status: complete and delivered; GitHub issue #1 was closed with fresh evidence
on 2026-07-11
Objective: close the remaining acceptance gaps in the requested engineering
backlog, in priority order, without reimplementing behavior that is already
present and freshly verified.

Scope: the twelve user-requested backlog themes: boundary-safe sampling,
configuration validation, two-stream honesty, fairness comparison, paired
statistics, dashboard safety, durable queue recovery, checkpoint/resume,
idempotent logging, runtime/resource telemetry, the claim ledger and warnings,
and dependency/backend scaling. Existing research artifacts and the active
claim-guided continuation plan remain untouched. No experiment execution,
remote dashboard exposure, claim promotion, artifact or database rewrite was
part of this work. Sol approved the exact CPU-only optional dependency after a
dry run proved that it would not replace JAX/JAXLIB or alter CUDA.

Current phase: no local implementation phase remains. Engineering, approved
historical labeling, UI verification, Git delivery, and issue closeout are done.

Phase 1 progress:

- [x] Re-inspect the five synthetic-capable benchmark callers, dashboard model
  helper, shared evaluation functions, and focused tests.
- [x] Reproduce the hidden shared gap: 2-D `conditional_entropy` crashes and
  2-D `markov_baseline_ppl` can silently return a meaningless value.
- [x] Obtain the requested Sol technical gate. Verdict: PASS with raw
  within-trajectory observation weighting and strict 1-D compatibility.
- [x] Make evaluation metrics and generative reference sampling
  trajectory-aware with boundary sentinels.
- [x] Migrate callers to `load_dataset_bundle`/`sample_batch` without running
  research workloads.
- [x] Run focused, change-aware, full CPU, diff, and fresh-verifier checks.
- [x] Correct the completion audit only after deterministic evidence proves the
  expanded acceptance criteria.

Phase 12 implementation packet (Sol-approved and completed):

- **Objective:** close the dependency/backend acceptance gap with one real,
  CPU-testable approximate execution path and clean-install evidence, while
  keeping approximate rows distinguishable from dense exact simulation.
- **Backend scope:** add a distinct optional `tensorcircuit_mps` backend with
  an explicit positive fixed bond limit, analytic expectation readout, and no
  dense state access. Recognize but fail closed on TensorCircuit-NG 1.7's
  JIT-incompatible threshold/relative modes. Thread the supported setting
  through validation, cached circuit factories, quantum layers, manifests,
  resource estimates, and the local model builder.
- **Dependency scope:** validate the existing CPU/WSL top-level pinned profiles,
  add a pinned optional MPS profile, make change-aware verification notice all
  profile files, and add clean Windows/Linux CPU installation CI. Describe
  these honestly as top-level pinned profiles rather than transitive locks.
- **Acceptance evidence:** deterministic four-qubit overlap and gradient parity
  at a sufficient bond dimension; a low-bond fixture that demonstrates the
  approximation; explicit nonlocal ring-edge coverage; JIT/nested-vmap
  coverage; state-access rejection before an optional import; capability-aware
  default diagnostics with statevector-only metrics explicitly unsupported;
  exact approximation fields in capability/run metadata; no dense-storage or
  observed-convergence claim; profile drift regressions; clean isolated
  CPU/MPS installs whose MPS job runs (rather than skips) the optional behavior
  suite; focused, change-aware, full CPU, frontend, and fresh-verifier checks.
- **Risks:** ring-CNOT routing introduces additional MPS truncation points;
  TensorCircuit uses process-global backend selection; SVD differentiation can
  be numerically sensitive; the configured SVD error threshold is not a
  guaranteed realized error when the bond cap also binds; the bound
  `2*n_qubits*chi**2` covers stored post-truncation MPS state-tensor elements
  only, not AD/gate/SWAP intermediates, peak memory, or runtime; CI proves
  current clean resolution but not a hash-locked transitive environment.
- **Observed implementation constraint:** TensorCircuit-NG 1.7 chooses retained
  rank from traced singular values when `max_truncation_err` is active, which
  raises under QLLM's JIT/`vmap` training path. Sol approved a fail-closed
  fixed-bond contract: non-null error thresholds and relative truncation are
  recognized but rejected before optional import; there is no eager-only or
  silent fallback mode.
- **Gates:** no GPU/WSL/CUDA execution, JAX/JAXLIB replacement, long training,
  artifact/database mutation, or claim edit. Sol PASS was received before the
  optional install and backend implementation.

Phase 12 progress:

- [x] Obtain Sol PASS for the architecture, dependency, diagnostics, resource,
  and fixed-bond fail-closed contracts.
- [x] Add the distinct approximate MPS backend, config/capability validation,
  JIT/`vmap`/gradient support, and explicit statevector-diagnostic unavailability.
- [x] Add logical-vs-storage resource evidence and conservative dashboard
  estimates covering global and per-block quantum configs.
- [x] Add CPU/WSL/MPS profile validation, clean-install CI, expanded manifest
  versions, and the native-Windows Orbax compatibility pin discovered by clean
  installation.
- [x] Prove clean native-Windows CPU and MPS installs in isolated environments;
  run the required MPS behavior suite with zero skips in both the active and
  clean MPS environments.
- [x] Browser-verify the model-builder MPS selection, bounded small-qubit bond
  default, fixed-bond warning, resource band, and console/overlay health on an
  isolated loopback dashboard.
- [x] Make tracking tags and one-time diagnostics resolve active per-block
  configs rather than the unused global default; fail closed for mixed/native
  execution, retain full component evidence in run resources, and use bounded
  count/status/SHA-256 MLflow tags instead of truncation-prone JSON.

Audit baseline:

| Requested item | Current classification | Acceptance gap or evidence |
| --- | --- | --- |
| 1. Boundary-safe synthetic sampling | Complete | All identified benchmark/dashboard callers preserve bundles and shared evaluation is trajectory-aware; inventory and behavior regressions prevent flattening recurrence. |
| 2. Comprehensive config validation | Complete; revalidate only | Registry-backed validation is shared by CLI, model-spec, queue, model, data, circuit, and backend paths. |
| 3. Two-stream metric and causality honesty | Complete; causal rerun is research-dependent | The current model is causal and the approved `RESULTS.md` correction now matches dashboard/claim-ledger labeling: historical v1 rows are side-information, rerun-required, and support no strict autoregressive conclusion. |
| 4. Full fairness-field comparison | Complete; revalidate only | Claim-specific schemas compare the full normalized protocol and expose every allowed/disallowed mismatch through studies and UI payloads. |
| 5. Paired study statistics | Complete; revalidate only | Paired bootstrap/sign-flip/equivalence/power logic is wired into study cells and the conservative verdict ladder. |
| 6. Dashboard local safety | Complete; revalidate only | Loopback/CORS gates and canonical path containment are implemented and tested. |
| 7. Durable queue recovery | Complete for the documented local worker contract | SQLite claims, leases, heartbeats, stale recovery, fencing, and terminal reservation release are implemented. |
| 8. Checkpoint/resume | Complete on CPU; hardware resume remains unproven | Atomic latest/best checkpoints restore optimizer, step, RNG, identity, and lineage through CLI/dashboard paths. |
| 9. Idempotent per-step logging | Complete for UUID-backed runs | Canonical step rows use `(run_uuid, step, name)` uniqueness; same-value retry is idempotent and conflicting retry fails loudly. |
| 10. Quantum runtime/resource telemetry | Complete for local execution | Compile/first-step, steady-state, state dimension, logical circuit-forward estimates, device, precision, and available memory are labeled; physical backend-call and GPU/QPU telemetry remain unavailable. |
| 11. Claim ledger and dashboard warnings | Complete; revalidate only | `research/claims.yaml` is canonical and structured warnings reach API/UI with conservative status semantics. |
| 12. Dependency matrix and backend scaling | Complete for local CPU scope | Validated CPU/WSL/MPS profiles, clean-install CI and native-Windows evidence, exact environment manifests, and a real fixed-bond approximate MPS backend are implemented; GPU/QPU/hardware validation remains separately gated. |

Acceptance evidence:

- Every synthetic caller that can load trajectory data preserves the 2-D
  bundle through splitting, sampling, and evaluation; a regression inventory
  fails if an executable benchmark or dashboard helper reintroduces the flat
  compatibility loader.
- Config, causality, fairness, paired-statistics, safety, durability, resume,
  logging, telemetry, ledger, and warning contracts remain green under focused
  tests after each relevant phase.
- Historical two-stream wording is changed only after separate approval for
  claim-bearing text and only to add the already-recorded limitation; no claim
  level or conclusion is strengthened.
- Dependency/backend work separates reproducible install validation from a
  concrete scalable-backend design. No CUDA/JAX environment change, GPU/QPU
  run, cluster work, or new backend dependency occurs without a named approved
  packet.
- Each implementation phase ends with focused checks, change-aware
  verification, diff review, and a fresh read-only verifier when risk or claim
  sensitivity warrants it.

Phased execution:

1. **Boundary-safe caller closure (recommended first phase).** Migrate the five
   identified benchmarks and dashboard model-test helper to
   `load_dataset_bundle`, preserve trajectory shape through their split/sample
   paths, add caller-inventory and behavior regressions, and update the stale
   completion classification. Validate data/config/benchmark/dashboard-model
   tests, then `verify_changes.py --plan/--run`. Risk: helper-specific shape or
   mask assumptions; no research result is rerun or rewritten.
2. **Configuration gate confirmation.** Trace checked-in configs and all launch
   callers once more after phase 1; add code only if a concrete registry or
   semantic-validation hole is reproduced. Validate all YAML configs and the
   focused config/dashboard preflight tests.
3. **Two-stream historical labeling (approved and implemented).** With explicit
   claim-text approval, add a correction note to the historical section of
   `RESULTS.md` identifying full-window v1 as teacher-forced side-information
   and rerun-required. Do not alter the recorded numbers or strengthen the
   verdict. Validate research protocol, dashboard historical filtering, links,
   and receive fresh research-sensitive verification.
4. **Fairness and paired-statistics confirmation.** Re-run focused protocol and
   study tests, add only any reproduced missing field/cell regression, and
   perform fresh API/UI evidence inspection if code changes. No study or claim
   promotion is part of this phase.
5. **Dashboard local-safety confirmation.** Re-run containment/CORS tests and,
   if implementation changes are needed, inspect loopback behavior locally.
   Remote mode is not exercised without approval.
6. **Durable queue recovery confirmation.** Re-run deterministic SQLite
   claim/lease/recovery/cancellation fixtures. Any live smoke uses an isolated
   database and explicit CPU target; no existing job is cancelled or mutated.
7. **Checkpoint/resume confirmation.** Re-run exact CPU interruption/resume,
   optimizer/RNG, best/latest, corruption, and lineage fixtures. Long or
   accelerator resume validation remains separately gated.
8. **Idempotent logging confirmation.** Re-run migration, concurrent retry,
   conflicting retry, and monotonic-curve fixtures; preserve legacy rows and
   use additive migrations only if a reproduced gap exists.
9. **Runtime/resource telemetry confirmation.** Re-run CPU timing, state-size,
   capability, precision, logical-call, and unavailable-memory cases. Do not
   relabel logical simulator work as physical QPU calls.
10. **Claim ledger and dashboard warning confirmation.** Re-run schema,
    conservative-classification, structured-warning, and frontend evidence
    tests; browser-inspect changed evidence surfaces if needed. Claim status
    changes remain out of scope.
11. **Dependency matrix verification (completed).** Validate the profiles,
    prove clean CPU/MPS installs, add clean-install CI, and document the
    unverified WSL/CUDA boundary without altering CUDA/JAX.
12. **Scalable backend implementation (completed for local CPU scope).** Add
    fixed-bond TensorCircuit MPS with exact-overlap, gradient/JIT parity,
    approximation/resource evidence, fail-closed unsupported modes, and clear
    hardware gates.

Risks and decisions:

- The initial completion audit overstated items 1 and 3. Items 1 and 3 are now
  closed, and item 12 is closed for local CPU execution. The causal-v2 study
  remains research-dependent and is not evidence already obtained.
- Passing existing tests did not expose the flat-loader callers, so phase 1
  requires an inventory-style regression in addition to behavior tests.
- The working tree already contains user-owned agent/documentation changes,
  including earlier `PLANS.md` edits. The parent must preserve them and keep
  every phase diff narrowly scoped.
- If approved, this engineering workstream takes execution priority without
  deleting the existing claim-guided continuation plan.

Human gates: the user explicitly approved the conservative phase-3
`RESULTS.md` correction on 2026-07-11; that approval does not authorize any
stronger wording. GPU/QPU/cluster runs, CUDA/JAX changes, remote dashboard
exposure, destructive database/artifact action, claim promotion, and research
experiments remain outside this local CPU implementation.

Latest planning validation:

```text
python scripts/check_agent_setup.py
PASS: Agent setup validation passed.
.venv/Scripts/python.exe scripts/verify_changes.py --plan
PASS: selected only agent-setup and agent tests for the pre-existing dirty
documentation/agent worktree.
.venv/Scripts/python.exe -m pytest -q tests/test_config_data.py
tests/test_quantum_data.py tests/test_contextual.py
tests/test_seq_cancellation.py tests/test_two_stream.py
tests/test_research_protocol.py tests/test_dashboard_security.py
tests/test_durable_runs.py tests/test_resource_accounting.py
tests/test_backend_capabilities.py tests/test_dashboard_lab.py
--basetemp .tmp/pytest-backlog-audit
PASS: 243 passed, 1 skipped, 11 warnings in 97.67s; warnings are the existing
Starlette/httpx deprecation and JAX complex128-to-complex64 notices.
read-only bounded audits
PASS with gaps: items 1 and 3 are overstated by the completion audit; item 12
is partial/deferred. All other requested items have implementation and focused
regression evidence. No experiment, service, job, dependency, database, or
artifact was changed.
```

Phase 1 validation:

```text
.venv/Scripts/python.exe -m pytest -q tests/test_v07.py
tests/test_config_data.py --basetemp .tmp/pytest-phase1-integration
PASS: 55 passed in 14.39s.
.venv/Scripts/python.exe -m pytest -q tests/test_config_data.py
tests/test_quantum_data.py tests/test_contextual.py
tests/test_seq_cancellation.py tests/test_v07.py tests/test_durable_runs.py
tests/test_dashboard_lab.py tests/test_integration.py
--basetemp .tmp/pytest-phase1-focused
PASS: 196 passed with 3 existing JAX complex128-to-complex64 warnings in
68.14s.
.venv/Scripts/python.exe scripts/verify_changes.py --run --timeout 600
PASS in 363.3s: agent setup/tests, dashboard tests, benchmark tests, and the
complete Python suite.
isolated CPU micro-smoke: benchmarks/memory_sweep.py --suite
phase1-boundary-smoke --memory-qubits 2 --models planted --smoke
PASS in 1.8s: boundary-aware Markov and planted evaluation completed under
`.tmp/phase1-smoke`; the live results database and existing artifacts were not
touched.
git diff --check
PASS: no whitespace errors; Windows line-ending notices only.
fresh Sol verifier
PASS: trajectory semantics, caller migration, regression coverage, and
completion-audit correction are acceptance-ready; no claim was strengthened.
```

Phase 12 validation:

```text
.venv/Scripts/python.exe -m pip install --dry-run tensorcircuit-ng==1.7.0
PASS: only tensorcircuit-ng 1.7.0, tensornetwork-ng 0.5.1, graphviz 0.21,
and h5py 3.16.0 were proposed; JAX/JAXLIB/CUDA were unchanged.
.venv/Scripts/python.exe -m pip check
PASS: no broken requirements; JAX/JAXLIB remained 0.10.1.
.venv/Scripts/python.exe scripts/check_dependency_profiles.py
--runtime-profile mps
PASS: CPU 20, WSL 18, and MPS 21 top-level pins are consistent; all 21 MPS
runtime versions match.
clean native-Windows installs at C:\q2 and C:\m2
PASS: requirements-cpu.txt and requirements-mps.txt installed from scratch,
editable QLLM installed without re-resolution, pip check passed, runtime
profiles matched, and JAX reported CpuDevice(id=0). The clean-install audit
discovered and fixed unbounded Orbax 0.12.1's native-Windows path failure by
pinning the already-tested orbax-checkpoint 0.10.3 in CPU/WSL profiles.
QLLM_REQUIRE_TENSORCIRCUIT_MPS=1 pytest -q -rs
tests/test_tensorcircuit_mps.py
PASS twice with zero skips: 8 passed in the active environment and 8 passed in
the independently clean MPS environment; three existing JAX x64 warnings per
run.
focused backend/config/resource/dashboard/durability integration
PASS: 225 passed with 12 existing JAX x64 warnings.
complete Python suite
PASS: 438 passed, 1 Windows symlink-environment skip, 48 existing
dependency/JAX warnings in 344.53s.
npm run test / npm run build --prefix qllm/dashboard/frontend
PASS: 9 frontend behavior tests and production build; existing bundle-size
advisory only.
isolated loopback browser QA on 127.0.0.1:8011/models
PASS: MPS selection exposed bond 4 for the 4-qubit preset, retained a low
resource band, rendered fixed-bond/unmeasured-evidence warnings, and produced
no console errors or framework overlay; the isolated server was stopped.
.venv/Scripts/python.exe scripts/verify_changes.py --run --timeout 600
PASS: agent setup/tests and frontend tests/build.
active-config tracking/resource/config regression suite
PASS: 102 passed with two existing JAX x64 warnings; per-block MPS, mixed
configured backends, native-JAX execution, and a valid 32-block/64-component
model are covered. Canonical component evidence is represented in MLflow by a
bounded SHA-256 tag and remains complete in the resource/diagnostic payload.
training-loop integration and durable-run suite
PASS: 41 passed with one existing JAX x64 warning.
agent-workflow CI regression suite and setup check
PASS: 34 passed; agent setup validation passed. The workflow now installs the
exact PyYAML dependency required during pytest collection.
remote dependency-matrix workflow 29160547587
PASS: all six clean CPU/MPS Windows/Linux Python 3.11/3.12 jobs succeeded.
fresh Sol final engineering verifier
PASS: active configs drive tags/diagnostics, mixed/native cases fail closed,
the canonical digest matches, all tag values are bounded, and no phase-12
engineering blocker remains. At that checkpoint `RESULTS.md` was still a
separate semantic gate; approval was granted in the later phase-3 packet.
.venv/Scripts/python.exe scripts/verify_changes.py --run --timeout 600
PASS in 328.8s: agent setup, agent tests, dashboard frontend tests, production
frontend build, and the complete Python suite all passed on the final
engineering code fingerprint.
```

Phase 3 claim-text gate:

- Sol verdict: `HUMAN_GATE`. The proposed `RESULTS.md` section-20 correction is
  scientifically conservative and technically approved, but the repository
  requires explicit semantic approval for claim-bearing text.
- The user explicitly granted that semantic approval on 2026-07-11 for the
  exact constraints below. No experiment rerun, number change, claim promotion,
  or artifact rewrite is authorized or needed.
- Approved wording constraints are ready: preserve numbers/provenance, remove
  strict-perplexity/lead interpretation, state that no strict autoregressive
  conclusion is supported, and name `two-stream-causal-v2` as the required
  paired rerun.
- **Objective:** correct only the historical interpretation in `RESULTS.md`
  section 20 so it matches the canonical metric contract already enforced by
  code, dashboard payloads, the claim ledger, and the research map.
- **Acceptance evidence:** every historical number/table/plot reference remains;
  the section says `teacher_forced_side_information`, `rerun_required`, and no
  strict autoregressive conclusion; lead/exception wording is removed; the
  required powered paired `two-stream-causal-v2` rerun is named explicitly.
- **Risks:** accidental number/provenance drift, implying that historical
  within-v1 differences are causal evidence, or strengthening a claim while
  correcting it. The edit must be wording-only and conservative.
- **Validation:** focused research-protocol/dashboard contract tests, claim and
  research-map validation, documentation/link checks selected by
  `verify_changes.py`, diff review proving number preservation, and a fresh Sol
  research-sensitive verifier.
- **Progress:** complete. Approval was received, the conservative correction
  and regression guard were implemented, focused/change-aware checks passed,
  and a fresh Sol research-sensitive verifier returned PASS.

Phase 3 validation:

```text
.venv/Scripts/python.exe -m pytest -q tests/test_research_protocol.py
tests/test_two_stream.py tests/test_dashboard_lab.py
--basetemp .tmp/pytest-two-stream-results-correction
PASS: 117 passed with six existing JAX complex128-to-complex64 warnings.
historical section numeric/provenance comparison against HEAD
PASS: every pre-correction numeric token remains with at least its original
count; all six table rows and `results/two_stream.png` remain unchanged.
.venv/Scripts/python.exe scripts/verify_changes.py --run --timeout 600
PASS for every executable check: agent setup, agent tests, and focused research
tests. The command correctly returned `human_review_required` because
`RESULTS.md` is non-automatable; the user had explicitly approved the exact
correction and Sol performed the required semantic review.
strengthened static historical-contract regression
PASS: all table values, remaining reported values, both canonical status
labels, causal-v2 rerun, and removal of superseded lead/exception language are
locked.
fresh Sol research-sensitive verifier
PASS: wording is strictly weaker and conservative, numeric/table/plot
provenance is preserved, the metric contract matches the claim ledger and
research map, and no claim is promoted.
```

UI issue #1 closure packet:

- **Objective:** close the remaining acceptance gaps in the open Quantum
  Advantage Research Cockpit issue without redesigning unrelated routes or
  weakening research evidence contracts.
- **Scope:** render the existing backend study/group slice on Explore; replace
  dark-only chart/quantum-band colors with light/dark tokens; make `npm test`
  execute every checked-in frontend test; add small pure contract tests for
  theme selection, navigation, study filtering, and bounded theme-safe chart
  styles. Existing routes, API schemas, queue behavior, and model specs remain
  unchanged.
- **Acceptance evidence:** a selected domain visibly includes its related
  study/groups and canonical dataset/task links; saved theme and first-load
  system preference remain deterministic; tooltips, ticks, bands, tables, and
  cards are readable in both themes; all frontend tests are discovered; build
  and focused dashboard tests pass; desktop and narrow browser QA covers both
  themes with no console errors or page-level overflow.
- **Risks:** linking an inferred group to a nonexistent first-class study,
  using CSS variables unsupported by chart SVG attributes, reducing dark-theme
  contrast while fixing light mode, or creating a test command that behaves
  differently on Windows/Linux. Group cards therefore link only to canonical
  dataset/task routes, and the test command names all checked-in files through
  Node's native discovery.
- **Validation:** focused Node tests, production frontend build, Explore/backend
  payload regressions, change-aware verification, dual-theme responsive
  browser inspection, diff review, and fresh independent verification before
  posting evidence and closing GitHub issue #1.
- **Progress:** complete and delivered in `c1ac72e`. Disjoint theme/chart and
  Explore/test packets were integrated; focused, change-aware, backend, build,
  responsive browser, fresh-verifier, and post-push CI checks passed. The
  evidence was posted to GitHub and issue #1 was closed as completed.

UI issue #1 validation:

```text
npm.cmd test  (qllm/dashboard/frontend)
PASS: Node native discovery ran all 22 checked-in API, evidence, model-config,
app-shell, Explore, and chart/theme tests.
npm.cmd run build  (qllm/dashboard/frontend)
PASS: 864 modules transformed; only the existing >500 kB chunk advisory.
.venv/Scripts/python.exe -m pytest -q tests/test_dashboard_lab.py
--basetemp .tmp/pytest-ui-issue-dashboard-full
PASS: 74 passed.
.venv/Scripts/python.exe scripts/verify_changes.py --run --timeout 600
PASS: agent setup, agent tests, dashboard frontend tests, and production build.
isolated loopback browser QA on 127.0.0.1:8012
PASS: selected-domain study/group cards and canonical dataset/task links render;
dark/light body, SVG tick, tooltip, series, and quantum-band tokens switch;
desktop document width equals client width; at 390x844 the document/client are
375px and the 1266px result table remains in its 341px local scroller; both
theme chart screenshots are readable; console logs are empty. The temporary
viewport override was reset, tab closed, and server stopped.
fresh gap-finding verifier
PASS: all three previously reported failures are closed, no unrelated backend,
route, queue, model, database, or research-contract file changed, and GitHub
issue #1 is safe to close after this exact patch is delivered with evidence.
post-push Agent Configuration workflow 29162959791
PASS: Ubuntu and Windows jobs completed successfully on `c1ac72e`.
GitHub issue #1 closeout
PASS: exact verification evidence was posted and the issue was closed as
completed on 2026-07-11.
```

## Active plan: claim-guided research continuation

Owner: parent agent
Started: 2026-07-11
Objective: choose and prepare the next decisive QLLM research study after the
local CPU-capable platform completion.

Scope: refresh the stale loop state, reconcile current claim contracts and
research-map priorities, select one next study track, define controls and
resource gates, and prepare a concrete execution packet. GPU/cluster/QPU
execution, paid services, destructive artifact/database migration, remote
dashboard exposure, and stronger scientific claims remain excluded without
separate approval.

Acceptance evidence:

- `STATE.md` no longer presents pre-M01-M09 blockers as current engineering
  status.
- The selected next study has a claim ID from `research/claims.yaml`, a metric
  type, required controls, fairness fields, seed/pair count rationale, and a
  resource gate.
- The plan names the smallest useful CPU smoke or read-only audit before any
  GPU or long run.
- Any GPU queue command is treated as a proposed packet, not authorization to
  execute.
- No `RESULTS.md` claim level/status is strengthened during planning.

Progress:

- [x] Complete and push the local platform M09 audit to `main` in commit
  `c2a7562`.
- [x] Reload `PLANS.md`, `docs/COMPLETION_AUDIT.md`, `GPU_QUEUE.md`,
  `research/claims.yaml`, and `docs/RESEARCH_MAP.yaml`.
- [x] Refresh current state with a report-only triage against post-M09 `main`.
- [x] Rank candidate continuation tracks by claim value, blockers, cost, and
  falsification power.
- [x] Select one track and write a narrow execution packet with controls,
  validation, and human gates.
- [x] Run the smallest safe CPU/read-only check that de-risks the selected
  packet.

Current candidate tracks:

- **Monitored quantum memory:** highest project-value track, but currently
  gated by predictive-memory baselines, replicated generator instances, and
  careful resource accounting. GPU queue items 1-2 belong here but require
  explicit hardware approval.
- **Two-stream causal confirmation:** lower setup risk and a clean rerun path
  because the causal replacement exists. GPU queue item 6 can test the
  historical lean without mixing in side-information rows.
- **Kernel geometry controls:** useful methodology hardening with finite-shot
  and stronger classical challenger work before application claims.
- **Contextual/parity memory:** promising mechanism track, but needs relabeling
  discipline and theorem-faithful task design before scaling.
- **Backend scaling:** valuable infrastructure follow-on for MPS/Lightning,
  but this is a capability program rather than the next scientific result.

Recommended first continuation:

1. Run a post-M09 report-only triage to rewrite stale `STATE.md` around the
   completed platform and current claim contracts.
2. Choose between monitored quantum memory and causal two-stream as the first
   continuation track.
3. If choosing memory, start with a CPU/read-only design packet for classical
   predictive-memory baselines and the m=12/14/16 GPU gate.
4. If choosing two-stream, start with a CPU smoke and study packet for
   `two-stream-causal-v2` before considering the 12-seed GPU run.

### 2026-07-13 selection: R1 causal two-stream preflight

Selected track: `two_stream_conditioning`, because its causal replacement is
implemented, the historical protocol error is isolated, and the first
verification can remain local, CPU-only, and non-destructive. This is not a
claim promotion: the canonical ledger remains `diagnostic` / `rerun_required`.

Execution packet:

- **Question:** does the causal-prefix quantum sentence conditioner improve
  strict next-token prediction over the matched classical conditioner in the
  declared small text regime?
- **Protocol:** `two-stream-causal-v2` only, with metric type
  `strict_autoregressive_next_token`. Never pool or append v1
  teacher-forced-side-information observations.
- **Arms:** `quantum-bias` candidate, `classical-bias` parameter-matched
  conditioner, and `none` no-conditioning ablation. Hold dataset, seed,
  training/evaluation budget, CPU device, and claim-schema fields fixed; record
  allowed architecture differences, parameter count, simulator wall time,
  manifest identity, seed axes, and resource ledger.
- **Evidence stages:** first verify the benchmark's causal suite guard and
  focused protocol/model tests. The completed one-step, seed-0 CPU execution
  check establishes only the fully isolated harness and manifest route, not a
  comparison outcome. A later single fair pair at a declared training/evaluation
  budget is descriptive smoke evidence only; three fair pairs are a variance
  pilot; confirmatory assessment needs at least six pairs and the larger
  pilot-power count, practical equivalence, and all required analogue/control
  rungs.
- **Artifact rule:** do not use the current benchmark's default shared
  `results/qllm_results.db`, `results/`, MLflow store, or dashboard database
  for a throwaway preflight. Use an isolated, UUID-backed run path with
  `--no-mlflow`, route optional dashboard logging to the selected scratch
  database, and preserve every result including nulls and failures.
- **Gates:** no GPU/cluster/QPU execution, paid provider, credential use,
  `RESULTS.md` edit, claim-level/status change, destructive artifact action, or
  push is implied by this packet.

Decisions:

- The local platform is complete for the M01-M09 scope; continuation is a
  research-campaign plan, not another platform-hardening milestone.
- Serious runs should belong to a Study and claim contract.
- GPU queue entries remain proposals until the user explicitly approves the
  hardware gate for a named packet.

Human gates: no GPU/QPU/cluster run, paid service, claim promotion,
destructive artifact/database action, remote dashboard exposure, or push is
authorized by this planning entry alone.

Latest validation:

```text
git status --short --branch
PASS: main was aligned with origin/main before the current local continuation work.
2026-07-13 causal preflight
PASS: `.venv\Scripts\python.exe benchmarks\two_stream_probe.py --help`
confirmed the causal-v2 CLI surface; it was not run as a stateful benchmark.
PASS: `.venv\Scripts\python.exe scripts\check_agent_setup.py --repo .`.
PASS: `.venv\Scripts\python.exe -m pytest -q tests\test_research_protocol.py
  tests\test_two_stream.py -p no:cacheprovider --basetemp <temp>`:
44 passed; six existing JAX complex128-to-complex64 warnings.
PASS: `git diff --check`; only the repository's LF-to-CRLF notice for the
append-only loop log was emitted.
ENVIRONMENT: `verify_changes.py --run --timeout 900` could not persist its
ignored `.tmp/verify-changes/state.tmp` record because the relocated worktree
was not writable to that wrapper. Its result is not treated as a pass; focused
preflight evidence above remains authoritative for its planning precursor.
2026-07-13 fully isolated CPU execution check
PASS: `two_stream_probe.py --suite two-stream-causal-v2 --dataset text
  --variants quantum-bias classical-bias none --seeds 0 --steps 1
  --results-db <LOCALAPPDATA temp>/results.sqlite --out-dir <LOCALAPPDATA
  temp>/artifacts --device-target cpu --no-mlflow --dashboard` completed three
  scratch-only run records.
PASS: every manifest records disabled MLflow, the selected scratch dashboard DB,
  `resource_plan.execution_device.requested=cpu`, and a resolved JAX CPU device;
  the scratch database contains three `runs` and three `live_runs` records.
PASS: the harness now accepts explicit `--results-db`, `--out-dir`,
  `--device-target`, and `--no-mlflow` controls while retaining prior defaults;
  focused protocol and causal-model tests passed 46 tests with six existing JAX
  dtype warnings.
PASS: `verify_changes.py --run --timeout 900 --state-file <temp>` completed
  after 869 seconds: agent setup, agent tests, benchmark tests, and Python tests
  all passed.
FIXED: an earlier output-only check exposed the default MLflow path as shared;
  the explicit no-MLflow control and dashboard-DB routing close that gap.
INTERPRETATION: one seed and one optimization step support no performance
  conclusion, positive or negative.
```

### 2026-07-14 execution: R1 fair CPU pair + three-pair variance pilot (pilot-only)

Executed per the packet above, fully isolated
(`%LOCALAPPDATA%\Temp\qllm-r1-causal-pair-isolated\{results.sqlite,artifacts}`,
`--no-mlflow`, `--device-target cpu`, suite `two-stream-causal-v2`, dataset
text, 1500 steps, eval_every 750, arms quantum-bias / classical-bias / none).
Commands:

    .venv\Scripts\python.exe benchmarks\two_stream_probe.py --suite two-stream-causal-v2 ^
      --dataset text --variants quantum-bias classical-bias none --seeds 0 --steps 1500 ^
      --results-db <iso>\results.sqlite --out-dir <iso>\artifacts ^
      --device-target cpu --no-mlflow --dashboard
    # then --seeds 0 1 2 (seed-0 cells skipped via completed-cell dedup)

Recorded val_ppl (candidate 25,701 params; control 25,841 = ~0.5% larger,
conservative; ablation 25,217):

| seed | quantum-bias | classical-bias | none | delta (q - c) |
| --- | --- | --- | --- | --- |
| 0 | 8.9716 | 9.8266 | 10.1706 | -0.8550 |
| 1 | 8.8751 | 9.6609 | 9.7371 | -0.7858 |
| 2 | 9.6814 | 9.9992 | 10.4499 | -0.3178 |

Descriptive pilot statistics (NOT a claim, NOT confirmatory): paired
delta mean -0.6529, sd 0.2922, candidate lower in 3/3 pairs; both
conditioned arms beat no-conditioning in all seeds. Per protocol
(RESEARCH_PROGRAM.md; evidence-checklist), three pairs remain pilot-only
regardless of nominal statistics — no p-value is claimed, the claim ledger
and RESULTS.md are unchanged, and claim `two_stream_conditioning` stays
`diagnostic` / `rerun_required`.

Pilot-derived power plan (the step STATE.md requires before any
confirmation): to resolve the pilot mean effect beyond the predeclared
practical-equivalence margin (0.1 ppl), effective effect 0.5529, pilot sd
0.2922 -> n ~= ((1.96 + 0.8416) * 0.2922 / 0.5529)^2 ~= 2.2, ~4 with
small-sample t inflation; the protocol floor `minimum_confirmatory_pairs: 6`
therefore GOVERNS. If pilot variance holds, a 6-pair confirmation is
adequately powered; 6/6 sign consistency would reach two-sided exact
sign-flip p = 0.03125. CPU cost of a 6-pair confirmation at 1500 steps is
~4-5 minutes (quantum arm ~20-24 s/run); the queued 12-seed 3000-step GPU
proposal (GPU_QUEUE.md item 6) remains NOT authorized.

Gates honored: CPU only, isolated artifacts (preserved, never appended to
two-stream-v1 or shared results), no MLflow, no claim-level/status change,
no RESULTS.md edit. Next decisive step (user decision): promote the
confirmation to a first-class Study + claim-contract run with >= 6 paired
seeds, the ablation arm, and complete resource accounting — CPU-feasible,
or the GPU proposal if separately approved by name.

## Active plan: StateVector backend enhancements

Owner: Codex apex orchestrator
Started: 2026-07-11
Objective: ship the backend contracts required by the parallel dashboard redesign
without crossing frontend ownership or weakening research and localhost safety.

Scope: deterministic OpenAPI contract generation, frontend-build-independent
backend tests, the pinned status payload, loopback-only job streaming, per-job
quantum diagnostics, durable verdict adjudication, and a bounded metadata-only
research scanner. `qllm/dashboard/frontend/**`, GPU/QPU execution, dependency or
environment changes, paid services, remote exposure, claim promotion, and merging
`backend-enhancements` to `main` are excluded.

Acceptance evidence:

- `scripts/dump_openapi.py` deterministically regenerates the committed
  `qllm/dashboard/openapi.json`; tests fail when the snapshot is stale.
- Backend security tests pass when the frontend `dist` directory is absent or
  incomplete, while a complete build remains confined and serveable.
- `GET /api/status` has the pinned top-level shape `{worker, gpu_available,
  queued, running, runs}` with compatibility details nested under `worker`.
- `GET /api/stream/jobs` is an SSE route restricted to loopback clients and emits
  bounded snapshots when authoritative `lab_jobs` or `live_runs` rows change.
- `GET /api/jobs/{id}/diagnostics` exposes gradient variance,
  parameter-shift gradient SNR, expressibility, Meyer-Wallach entanglement, and
  scaling-fit fields with explicit measured/unavailable provenance and a warning
  that diagnostics are not advantage evidence.
- An additive verdict table and `GET /api/verdicts[/{id}]` preserve canonical
  claim level, derived assessment status, and replication status as separate
  fields, retain immutable source provenance, and never produce a composite
  advantage score.
- The research service has a provider-neutral interface and a standard-library,
  bounded arXiv metadata scanner with deterministic fixture tests. No LLM,
  embedding, vector/graph store, paid call, or new dependency is selected before
  D4 user approval and a per-day budget.
- The generated OpenAPI snapshot and backend-owned coordination summary match the
  shipped endpoints; no frontend file changes appear in the diff.
- Focused dashboard/security/durability/metrics/research tests, agent setup,
  change-aware verification, the full CPU suite, and the one-step CPU queue smoke
  complete with fresh output and a read-only verifier review.

Progress:

- [x] Rebase the clean backend worktree onto `origin/main` and read the contract,
  redesign sections, feature intake, scoped instructions, and domain maps.
- [x] Run bounded planner/scout discovery for API, queue, diagnostics, claim, and
  research-service boundaries.
- [x] Ship D5: OpenAPI dump script, deterministic snapshot, and stale-snapshot
  regression.
- [x] Ship D6: tolerate missing/incomplete frontend build in backend imports and
  security tests.
- [x] Pin `/api/status` and ship localhost-only SQLite-authoritative SSE updates;
  record D1 and D2.
- [x] Ship capability-aware diagnostics and focused regressions.
- [x] Ship the additive verdict store/API; record D3.
- [x] Ship the bounded research-service scaffold and stop at D4.
- [x] Regenerate the final OpenAPI snapshot and update only the backend-owned API
  contract and Backend log in `docs/BUILD_COORDINATION.md`.
- [x] Remediate the Ultra verifier's artifact-identity and protocol-cohort
  diagnostics findings.
- [x] Reconcile terminal claim-bound verdicts before listing so persistence no
  longer depends on comparison-read order.
- [x] Close the second Ultra pass: make cohorts schedule/config strict and make
  verdict reconciliation cursor-driven and metadata-only.
- [x] Close the third Ultra pass: reject non-standard and non-finite cohort JSON
  plus bounded decoder failures for both target and peer rows.
- [x] Run every requested focused/full CPU check, obtain a fresh verifier verdict,
  and commit authorized deliverables without merging the feature branch.

Decisions: SQLite remains authoritative for stream events; SSE uses bounded
change detection rather than process-local notification state. Diagnostic values
remain capability-aware observations and cannot raise a claim. Canonical
`claim_level`, derived `assessment_status`, and `replication_status` are distinct
verdict dimensions. The parent owns `server.py`, `PLANS.md`, OpenAPI regeneration,
and the shared coordination file so delegated writers have disjoint ownership.

Human gates: D4 remains open for the provider, dependency, vector/graph store,
and daily cost budget. No GPU/QPU/cluster workload, paid service, environment
change, remote dashboard exposure, claim promotion, artifact rewrite, feature
branch merge, or unrelated Git action is authorized.

Latest validation:

```text
git fetch origin
PASS: origin refreshed.
git rebase origin/main
PASS: backend-enhancements was already up to date.
git status --short --branch
PASS: clean backend-enhancements worktree before plan creation.
shared-venv pytest baseline
ENVIRONMENT: the default Windows pytest temp root raised WinError 5; all
subsequent pytest runs will use a repository-local --basetemp as required by
AGENTS.md. The first local path also showed that its parent `.tmp` must exist.
python scripts/dump_openapi.py --check
PASS: committed qllm/dashboard/openapi.json is current.
python -m pytest -q tests/test_openapi_contract.py --basetemp .tmp/pytest-openapi-d5b
PASS: 1 passed.
python -m pytest -q tests/test_dashboard_security.py --basetemp .tmp/pytest-security-d6b
PASS: 13 passed, 1 skipped; one existing Starlette/httpx deprecation warning.
python -m pytest -q tests/test_dashboard_stream.py --basetemp .tmp/pytest-stream-integration
PASS: 6 passed.
python -m pytest -q tests/test_dashboard_security.py --basetemp .tmp/pytest-security-stream
PASS: 14 passed, 1 skipped; one existing Starlette/httpx deprecation warning.
python -m pytest -q tests/test_openapi_contract.py --basetemp .tmp/pytest-openapi-contract-d1d2
PASS: 1 passed; the status schema has five typed fields and SSE advertises text/event-stream.
python -m pytest -q tests/test_dashboard_lab.py --basetemp .tmp/pytest-dashboard-d1d2
PASS: 74 passed.
python -m pytest -q tests/test_dashboard_diagnostics.py --basetemp .tmp/pytest-diagnostics-parent
PASS: 10 passed; repeated same-qubit rows cannot become a scaling axis.
python -m pytest -q tests/test_metrics.py tests/test_backend_capabilities.py --basetemp .tmp/pytest-diagnostics-metrics
PASS: 58 passed; six existing JAX complex128-to-complex64 warnings.
python -m pytest -q tests/test_openapi_contract.py --basetemp .tmp/pytest-openapi-diagnostics
PASS: 1 passed; the five diagnostics dimensions are explicit in OpenAPI.
python -m pytest -q tests/test_dashboard_verdicts.py tests/test_advantage.py tests/test_research_protocol.py --basetemp .tmp/pytest-verdicts-parent
PASS: 47 passed; 13 existing JAX complex128-to-complex64 warnings.
python -m pytest -q tests/test_dashboard_lab.py --basetemp .tmp/pytest-dashboard-verdicts
PASS: 74 passed with idempotent comparison snapshot materialization.
python -m pytest -q tests/test_durable_runs.py --basetemp .tmp/pytest-durable-verdicts
PASS: 38 passed; additive verdict migrations preserve durable-run behavior.
python -m pytest -q tests/test_openapi_contract.py --basetemp .tmp/pytest-openapi-verdicts-final
PASS: 1 passed; two subprocess checks prove worker-free OpenAPI cleanup on Windows.
python -m pytest -q tests/test_research_service.py --basetemp .tmp/pytest-research-parent
PASS: 14 passed with no live network call; fixed query, byte/result/day bounds,
version-aware deduplication, persistent quota, courtesy pacing, and D4 boundary covered.
python -m pytest -q tests/test_dashboard_lab.py --basetemp .tmp/pytest-dashboard-research
PASS: 74 passed.
python -m pytest -q tests/test_dashboard_security.py --basetemp .tmp/pytest-security-research
PASS: 14 passed, 1 skipped; one existing Starlette/httpx deprecation warning.
python -m pytest -q tests/test_openapi_contract.py --basetemp .tmp/pytest-openapi-research
PASS: 1 passed; capabilities and bounded scan routes are typed in OpenAPI.
python scripts/check_agent_setup.py
PASS: agent setup validation passed.
python scripts/verify_changes.py --plan
PASS: selected agent-setup for the clean committed worktree; the script does not
compare committed branch history to origin/main, so the full suite below supplies
the branch-wide behavioral verification.
python scripts/verify_changes.py --run
PASS: agent-setup passed for verification fingerprint
91f8dea9b78c842f01de8272747f3cc704d625fc35178ef1bcd22edc0a89e327.
python -m pytest -q --basetemp .tmp/pytest-backend-full-final
PASS: 492 passed, 1 skipped, 48 warnings in 312.66 seconds; the warnings are one
existing Starlette/httpx deprecation and 47 existing JAX complex128-to-complex64
warnings.
python scripts/queue_smoke.py --url http://127.0.0.1:8179 --steps 1
  --eval-every 1 --device-target cpu --timeout 180
PASS: isolated job 1 completed on CPU at step 1 and wrote latest/best checkpoints;
the temporary loopback dashboard was stopped and port 8179 was confirmed free.
python scripts/dump_openapi.py --check
PASS: committed qllm/dashboard/openapi.json is current after all endpoint changes.
git diff --name-only origin/main...HEAD
PASS: 20 backend/plan/test paths changed and no qllm/dashboard/frontend path changed.
git diff --check
PASS: no whitespace errors.
gpt-5.6-sol Ultra immutable-range verification
REMEDIATION REQUIRED: the verifier reproduced cross-run summary leakage,
protocol-incompatible scaling cohorts, and verdict persistence coupled to a
prior comparison read. D5, D6, status, SSE, bounded research, and human gates
passed independently.
python -m pytest -q tests/test_dashboard_diagnostics.py tests/test_dashboard_verdicts.py
  tests/test_dashboard_lab.py tests/test_durable_runs.py tests/test_dashboard_security.py
  tests/test_openapi_contract.py --basetemp .tmp/pytest-ultra-remediation-integration
PASS: 149 passed, 1 skipped; one existing Starlette/httpx deprecation warning.
python -m pytest -q tests/test_metrics.py tests/test_advantage.py
  tests/test_research_protocol.py tests/test_research_service.py
  --basetemp .tmp/pytest-ultra-remediation-research
PASS: 64 passed; 17 existing JAX complex128-to-complex64 warnings.
python scripts/dump_openapi.py --check
PASS: committed OpenAPI snapshot remains current after verifier remediation.
git diff --check
PASS: no whitespace errors; only LF-to-CRLF working-copy notices.
python scripts/check_agent_setup.py
PASS: agent setup validation passed after remediation.
python scripts/verify_changes.py --run
PASS: agent-setup passed; the clean-tree selector again had no uncommitted paths,
so the complete branch-wide suite below remains the behavioral authority.
python -m pytest -q --basetemp .tmp/pytest-backend-full-remediated
PASS: 499 passed, 1 skipped, 48 warnings in 316.68 seconds; warnings remain
the existing Starlette/httpx deprecation and JAX complex128-to-complex64 notices.
python scripts/queue_smoke.py --url http://127.0.0.1:8180 --steps 1
  --eval-every 1 --device-target cpu --timeout 180
PASS: isolated job 1 completed on CPU at step 1 with latest/best checkpoints and
no error; the temporary server was stopped and port 8180 confirmed free.
gpt-5.6-sol Ultra remediation verification
REMEDIATION REQUIRED: the verifier confirmed artifact identity was fixed, then
reproduced schedule/config under-grouping plus invalid-row starvation and
unbounded curve loading in list-time verdict reconciliation.
python -m pytest -q tests/test_dashboard_diagnostics.py tests/test_dashboard_verdicts.py
  tests/test_dashboard_lab.py tests/test_durable_runs.py tests/test_dashboard_security.py
  tests/test_openapi_contract.py tests/test_metrics.py tests/test_advantage.py
  tests/test_research_protocol.py --basetemp .tmp/pytest-third-pass-integration
PASS: 206 passed, 1 skipped; one existing Starlette/httpx warning and 17 existing
JAX complex128-to-complex64 warnings.
python scripts/dump_openapi.py --check
PASS: metadata-only reconciliation and additive cursor storage do not change the
public API; the committed OpenAPI snapshot remains current.
python scripts/check_agent_setup.py
PASS: agent setup validation passed after the third remediation pass.
python scripts/verify_changes.py --run
PASS: clean-tree agent-setup selection passed.
python -m pytest -q --basetemp .tmp/pytest-backend-full-third-pass
PASS: 506 passed, 1 skipped, 48 warnings in 308.72 seconds; warnings remain
the existing Starlette/httpx deprecation and JAX dtype notices.
python scripts/queue_smoke.py --url http://127.0.0.1:8181 --steps 1
  --eval-every 1 --device-target cpu --timeout 180
PASS: isolated job 1 completed on CPU at step 1 with both checkpoints and no
error; the temporary server was stopped and port 8181 confirmed free.
gpt-5.6-sol Ultra strict-cohort verification
REMEDIATION REQUIRED: all cursor/metadata/original-contract items passed; the
verifier reproduced Python JSON acceptance of NaN, Infinity, and overflowed
exponents in cohort config.
python -m pytest -q tests/test_dashboard_diagnostics.py
  --basetemp .tmp/pytest-diagnostics-strict-json
PASS: 26 passed, covering non-standard constants, overflowed non-finite values,
decoder recursion, target rejection, and peer exclusion.
python scripts/check_agent_setup.py
PASS: agent setup validation passed after strict-JSON remediation.
python scripts/verify_changes.py --run
PASS: clean-tree agent-setup selection passed.
python -m pytest -q --basetemp .tmp/pytest-backend-full-strict-json
PASS: 513 passed, 1 skipped, 48 warnings in 322.86 seconds; warnings remain
the existing Starlette/httpx deprecation and JAX dtype notices.
python scripts/queue_smoke.py --url http://127.0.0.1:8182 --steps 1
  --eval-every 1 --device-target cpu --timeout 180
PASS: isolated CPU job 1 completed at step 1 with both checkpoints and no error;
the temporary server was stopped and port 8182 confirmed free.
gpt-5.6-sol Ultra final immutable-range verification
PASS: no required fixes remain for
c3c2bdf26845bc61320feb338b7efb47f540106a..de4de904d8724ddd965a72cb366c5a5ce4c53b06.
Strict JSON, artifact identity, schedule cohorts, cursor/wrap behavior,
metadata-only reconciliation, the original API contract, and all human gates
were reconfirmed. Residual risks are resource-exhaustion failures outside the
strict-JSON contract and D4/hardware/paid/merge gates that intentionally remain closed.
```

## Completed plan: local platform completion

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
- [x] M08 Dashboard and UI evidence completion (`codex/m08-dashboard-evidence`),
  delivered to `main` in commit `f3daa85`.
- [x] M09 Documentation and completion audit (`codex/m09-docs-audit`),
  delivered to `main` in commit `c2a7562`.

Current milestone: M09 documentation and completion audit.

M09 acceptance evidence:

- Researcher onboarding explains claim levels, matched controls, multi-seed
  studies, resource ledgers, warnings, and safe interpretation without
  strengthening any scientific conclusion.
- Engineer onboarding maps registries, canonical datasets/config validation,
  model/backend extension points, durable runs, dashboard evidence contracts,
  tests, and safe local workflows.
- Every enhancement-plan item is classified completed, superseded, deferred,
  or human-gated with concrete repository evidence; unresolved hardware,
  cluster, paid-service, destructive-migration, and claim-promotion work stays
  explicitly outside local-platform completion.
- Documentation links and commands are valid, agent setup and change-aware
  verification pass, the complete CPU suite remains green, and a fresh
  read-only verifier confirms the repository audit before delivery.

M09 progress:

- [x] Deliver M08 and create `codex/m09-docs-audit` from updated `main`.
- [x] Build the documentation and enhancement-status coverage map.
- [x] Add researcher and engineer onboarding plus evidence-backed status audit.
- [x] Validate links, commands, documentation checks, and the complete CPU suite.
- [x] Obtain a fresh verifier PASS and deliver M09 under standing Git approval.

M09 design constraints:

- Documentation describes the implemented repository; it does not invent
  unverified behavior or silently close deferred/human-gated programs.
- `RESULTS.md` claim-bearing text and claim status/level remain unchanged.
- Existing historical plans and validation evidence remain intact. Mechanical
  status/link edits receive parent review before acceptance.
- No GPU/QPU/cluster run, paid service, destructive artifact/database action,
  remote dashboard exposure, or stronger claim is part of M09.

M09 validation results:

```text
relative Markdown link audit
PASS: every local target in repository Markdown resolves.
research YAML/claim registry audit
PASS: RESEARCH_MAP.yaml parses and 19 canonical claim contracts load.
documented CLI help audit
PASS: seven researcher benchmark entry points plus queue/GPU helpers resolve in the repository virtual environment.
python scripts/check_agent_setup.py --repo .
PASS: project agent/profile/skill setup is valid.
pytest -q tests/test_agent_configuration.py tests/test_verify_changes.py --basetemp .tmp/pytest-m09-agent
PASS: 17 passed.
python scripts/verify_changes.py --run
SAFE CHECKS PASS: agent setup/tests. The verifier reports its conservative `gpu` HUMAN GATE because GPU_QUEUE.md warning prose changed; no GPU workload is selected or launched, and the user authorized safe in-scope documentation delivery.
pytest -vv --tb=short --basetemp .tmp/pytest-m09-full-retry
PASS: 351 passed, 2 skipped; 44 existing dependency/JAX precision warnings. An earlier buffered run was interrupted by the external command timeout without an assertion result; the clean retry used a fresh temp root.
git diff --check
PASS: no whitespace errors; line-ending notices only.
independent research-documentation review
PASS after correcting seed-axis coupling language and claim-loader module attribution; no claim promotion or unsupported completion classification remains.
```

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

## Active plan: quantum-native expansion — from QML-only to general quantum-algorithm research

Owner: user (Arlind); apex Claude Code (Opus 4.8); backend implementation TBD
Started: 2026-07-14
Objective: open the existing verification-first pipeline (evidence ladder, claim
ledger, fair-control protocol, multi-seed statistics, scaling harness) to
non-language-model quantum tasks — ground-state chemistry (VQE), combinatorial
optimization (QAOA), and later finance/biology instances — without weakening a
single integrity guarantee. QML remains the active first vertical; nothing in
this plan pauses the two-stream program.

Scope: qllm/registry.py, qllm/config.py, qllm/resultsdb.py,
qllm/dashboard/{lab,studies}.py, research/claims.yaml schema, a NEW sibling
task runner (loop.py is not generalized), and eventually dashboard surfaces.
Non-goals: renaming the qllm package (cosmetic, high churn); restructuring
RESEARCH_MAP.yaml domains (new areas enter as flat entries first); any hardware
(QPU) execution; any change to existing QML claims or RESULTS.md.

Why this is a schema evolution, not a rewrite (2026-07-14 survey evidence):
the verdict/study machinery already routes through a `metric_type` string and
FAILS CLOSED on non-perplexity types (lab.py PAIRABLE_VAL_PPL_METRIC_TYPES;
studies.py rejects unknown metric types at creation) — a deliberate integrity
guard. The evidence engine already parameterizes `lower_is_better` and
`metric_name`; the seed-axis statistics map directly onto problem instances;
resultsdb already has a generic per-step metric name/value table; the claim
contract's `metric_type` is an extensible string. What is hard-wired: val_ppl
as a schema column and progress key, sequence-only Data/Model/Train configs,
no task-type dimension anywhere, and ~30 dashboard surfaces labeling val_ppl.

Acceptance evidence:

- A VQE toy instance (small molecule / transverse-field Ising ground state,
  exactly diagonalizable) runs end-to-end through registry -> config -> new
  runner -> resultsdb -> Study -> dashboard verdict with
  metric_type=ground_state_energy_error, and the verdict renders with the same
  ladder/fairness framing as a QML pair — with zero changes to the QML path's
  behavior (full pytest suite stays green).
- Relabeling is impossible by construction: metric admission and metric
  extraction live in ONE registry entry, so a perplexity number can never be
  presented as an energy (regression test).
- A classical-solver analogue ladder (e.g. exact diag < DMRG for VQE) is
  declared in the claim contract before the first comparative run.

Progress:

- [x] Step 1 — metric registry in qllm/registry.py: METRIC_TYPES mapping
      metric_type -> {lower_is_better, units, pairable, extraction_key,
      comparator_class}; replace the frozenset checks in lab.py and studies.py
      with registry lookups; serve the list via /config/choices so the frontend
      stops hard-coding val_ppl labels. (Small diff; unblocks everything.)
- [x] Step 2 — primary-metric indirection in resultsdb.py:
      primary_metric_name/value on the run row (or a view over the existing
      metrics table); progress/dedup keyed on it; val_ppl columns become the
      sequence-modeling specialization, not the universal schema.
- [x] Step 3 — task dimension: TASK_TYPES = (sequence_modeling, ground_state,
      combinatorial_optimization) in registry.py; a task-conditional
      ProblemConfig section in config.py validated in validate_config (copy the
      existing tensorcircuit_mps conditional-validation pattern); task_type
      recorded in claims.yaml entries and study specs.
- [x] Step 4 — one vertical slice, VQE first (exact diagonalization gives
      ground truth; QAOA lacks certified optima at toy scale): a SIBLING
      runner minimizing a problem-Hamiltonian expectation over the existing
      circuit/backend layer, metric_type=ground_state_energy_error vs exact
      solution, writing through the same resultsdb/manifest/Study path, with a
      classical-solver analogue ladder. No changes to train/loop.py.
      Completed (2026-07-14): the first bounded slice uses a registered
      two-qubit transverse-field Ising instance, analytic CPU statevector
      execution, and exact diagonalization as a correctness reference. It must
      label shots=None as simulator-diagnostic evidence, preserve immutable
      problem identity and checkpoint recovery, and block solver-edge
      inference until solver_competition_v1 and registered comparison runners
      exist; Step 5 declares only the fail-closed admission contract.
      Acceptance evidence: all seven post-implementation Sol Ultra findings
      were fixed with regressions; the fresh second Sol Ultra review returned
      PASS; the focused integration bundle passed 275 tests; the standalone
      full suite passed 751 tests with 1 skip; and the OpenAPI snapshot check
      passed. The user approved the VQE Atlas grouping on 2026-07-14; this
      records display placement only and does not promote a claim.
- [x] Step 5 — solver_competition_v1 fairness schema + comparator_kind in
      claims.yaml: equal-budget best-in-class competition semantics (declared
      solver versions, certified or best-known optima), distinct from
      controlled_component_ablation_v1 which stays QML-only.
      Completed (2026-07-14): admission now resolves every solver identity
      through the server-owned registry, pairs only unique immutable problem
      instances, binds per-instance optima and unique executed-run identities,
      enforces complete observed search ledgers and all prespecified ceilings,
      and cryptographically binds each evaluation run to the selected search
      configuration. The claim loader and evaluator fail closed on malformed
      persisted shapes. Production still has no comparison-eligible finite-shot
      quantum/classical runner pair, so inference and claim eligibility remain
      disabled and no composite or advantage score is produced.
      Acceptance evidence: focused solver/claim/metric tests passed 83 tests;
      the widened Step 4-5 integration bundle passed 375 tests with 9 existing
      warnings; the final fresh Sol Ultra review returned PASS after independent
      bypass probes; OpenAPI remained current; and the one-step CPU queue smoke
      completed with no recovery or error. The user approved the VQE Atlas
      grouping on 2026-07-14; Git delivery remains separately human-gated.
- [x] Backend Atlas verdict references: emit stable, persisted
      `claim:<claim_id>` projections without changing curated Atlas status,
      claim level, or replication fields. Source snapshots and projection
      revisions remain append-only; normal writes materialize projections
      immediately, bounded list-time reconciliation repairs legacy rows, and
      malformed or forged projection rows fail closed across list, detail, and
      history APIs. Concurrent source/key ownership and projection ordering are
      serialized in SQLite, and Atlas emits no reference until a fully
      source-validated projection exists.
      Acceptance evidence: focused Atlas/verdict/OpenAPI tests passed 33 tests;
      the widened dashboard contract passed 202 tests with 1 expected skip;
      the full CPU suite passed 835 tests with 1 expected skip and 55 existing
      warnings; the OpenAPI snapshot remained current; and the final fresh
      gpt-5.6-sol Ultra review returned PASS after temporal-window, delayed-
      writer, namespace-forgery, stable-key poisoning, and HTTP history probes.
      No frontend files, claim promotions, composite scores, or ontology
      approvals were introduced.
- [ ] Dashboard follow-through: metric labels from the registry (frontend
      verdictView.js), task-type facet on Runs/Studies, Atlas gains chemistry/
      optimization areas as new level-0/unexplored entries with full audit
      trail.

Decisions: VQE before QAOA (ground truth available); sibling runner rather
than generalizing loop.py (protects the QML path); expansion areas enter the
research map at level 0/unexplored — never pre-credited. Integrity guards that
MUST hold (from the 2026-07-14 review): (1) metric admission and extraction
coupled in one registry entry — widening the pairable set without coupling
would silently read val_ppl and present it as energy; (2) fairness schemas are
per-task — transplanting controlled_component_ablation_v1 to circuit-vs-solver
comparisons would fabricate fair verdicts; (3) pairing moves to the
problem-instance axis with model seeds nested; optimization tasks need
time-to-target/success-probability with censored-data handling before any
claim; (4) shot/measurement budgets are part of the claim for VQE/QAOA —
analytic-gradient (shots=None) convergence is a simulator diagnostic, never
hardware-relevant evidence; (5) per-domain practical-effect thresholds
(e.g. chemical accuracy 1.6 mHa) predeclared in the claim contract; (6) new
domains carry the same claim-ledger discipline from day one.

Human gates: this entry authorizes design and CPU-bounded engineering only —
no GPU/QPU runs, no paid solver/provider, no claim promotion, no RESULTS.md
edits, no new dependency without user sign-off (a chemistry Hamiltonian
library, if wanted later, is a named user decision).

Latest validation:

    2026-07-14 read-only survey (5-agent review): expansion points and
    integrity guards verified against qllm/registry.py, config.py,
    train/loop.py, resultsdb.py, research_protocol.py, dashboard/lab.py,
    dashboard/studies.py, research/claims.yaml, docs/RESEARCH_PROGRAM.md.
    No code changed under this entry yet.

    2026-07-14 local Steps 1-3 implementation: metric admission/extraction,
    primary metric persistence, and task identity are implemented in the
    backend worktree. Sol Ultra review passed after three fixes; pytest reports
    709 passed and 1 skipped, verify_changes.py --run passed, OpenAPI is
    current, and the isolated one-step CPU queue smoke passed. Changes remain
    uncommitted pending the normal Git gate.

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
