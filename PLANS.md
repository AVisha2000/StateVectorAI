# Living Plans

This file carries substantial work across agent turns. The parent owns the
plan, integration, deterministic verification, human gates, and final handoff.
Completed implementation details move into canonical documentation; this file
keeps only concise progress, decisions, and current evidence.

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
Status: awaiting explicit implementation approval
Objective: close the remaining acceptance gaps in the requested engineering
backlog, in priority order, without reimplementing behavior that is already
present and freshly verified.

Scope: the twelve user-requested backlog themes: boundary-safe sampling,
configuration validation, two-stream honesty, fairness comparison, paired
statistics, dashboard safety, durable queue recovery, checkpoint/resume,
idempotent logging, runtime/resource telemetry, the claim ledger and warnings,
and dependency/backend scaling. Existing research artifacts and the active
claim-guided continuation plan remain untouched. No experiment execution,
dependency installation, remote dashboard exposure, claim promotion, artifact
or database rewrite, or Git delivery is authorized by this proposal.

Audit baseline:

| Requested item | Current classification | Acceptance gap or evidence |
| --- | --- | --- |
| 1. Boundary-safe synthetic sampling | Partial; audit status is stale | Canonical training/evaluation use 2-D trajectory-aware bundles, but `benchmarks/memory_sweep.py`, `planted_qrnn.py`, `resonance_search.py`, `seq_interference_probe.py`, `model_report.py`, and `qllm/dashboard/model_tests.py` still call the flattening compatibility loader. |
| 2. Comprehensive config validation | Complete; revalidate only | Registry-backed validation is shared by CLI, model-spec, queue, model, data, circuit, and backend paths. |
| 3. Two-stream metric and causality honesty | Code/dashboard complete; claim-text gap is human-gated | The current model is causal and historical dashboard rows are side-information/rerun-required, but `RESULTS.md` section 20 does not carry that historical limitation. |
| 4. Full fairness-field comparison | Complete; revalidate only | Claim-specific schemas compare the full normalized protocol and expose every allowed/disallowed mismatch through studies and UI payloads. |
| 5. Paired study statistics | Complete; revalidate only | Paired bootstrap/sign-flip/equivalence/power logic is wired into study cells and the conservative verdict ladder. |
| 6. Dashboard local safety | Complete; revalidate only | Loopback/CORS gates and canonical path containment are implemented and tested. |
| 7. Durable queue recovery | Complete for the documented local worker contract | SQLite claims, leases, heartbeats, stale recovery, fencing, and terminal reservation release are implemented. |
| 8. Checkpoint/resume | Complete on CPU; hardware resume remains unproven | Atomic latest/best checkpoints restore optimizer, step, RNG, identity, and lineage through CLI/dashboard paths. |
| 9. Idempotent per-step logging | Complete for UUID-backed runs | Canonical step rows use `(run_uuid, step, name)` uniqueness; same-value retry is idempotent and conflicting retry fails loudly. |
| 10. Quantum runtime/resource telemetry | Complete for local execution | Compile/first-step, steady-state, state dimension, logical circuit-forward estimates, device, precision, and available memory are labeled; physical backend-call and GPU/QPU telemetry remain unavailable. |
| 11. Claim ledger and dashboard warnings | Complete; revalidate only | `research/claims.yaml` is canonical and structured warnings reach API/UI with conservative status semantics. |
| 12. Dependency matrix and backend scaling | Partial | CPU/WSL profiles and exact/sampled/unverified capability metadata exist. Clean-install/GPU compatibility is not freshly proven, and no real MPS/Lightning GPU/MPI/QPU scalable backend exists; those choices are a separate dependency/hardware gate. |

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
3. **Two-stream historical labeling (separate human gate).** With explicit
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
11. **Dependency matrix verification.** Add a clean CPU install/compatibility
    check in an isolated environment if approved and document the unverified
    WSL/CUDA boundary. Do not alter the active CUDA/JAX environment.
12. **Scalable backend decision packet.** Before implementation, choose one
    concrete target (for example MPS/tensor-network or Lightning), define exact
    overlap/parity, approximation metadata, resource evidence, dependency and
    hardware gates, and stopping criteria. This is deferred until the user
    approves that architecture/dependency packet; it is not implied by the
    local exact-backend capability work.

Risks and decisions:

- The completion audit overstates items 1 and 3 and must be corrected when the
  corresponding approved phase lands; item 12 must stay explicitly split
  between local metadata and real scaling.
- Passing existing tests did not expose the flat-loader callers, so phase 1
  requires an inventory-style regression in addition to behavior tests.
- The working tree already contains user-owned agent/documentation changes,
  including earlier `PLANS.md` edits. The parent must preserve them and keep
  every phase diff narrowly scoped.
- If approved, this engineering workstream takes execution priority without
  deleting the existing claim-guided continuation plan.

Human gates: implementation has not started. Separate explicit approval is
required for the phase-3 `RESULTS.md` correction, any dependency/CUDA/JAX
environment change, scalable-backend dependency or hardware work, remote
dashboard exposure, destructive database/artifact action, experiment run,
commit, push, merge, or publication.

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
- [ ] Refresh current state with a report-only triage against post-M09 `main`.
- [ ] Rank candidate continuation tracks by claim value, blockers, cost, and
  falsification power.
- [ ] Select one track and write a narrow execution packet with controls,
  validation, and human gates.
- [ ] Run the smallest safe CPU/read-only check that de-risks the selected
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
PASS: main is clean and aligned with origin/main before continuation planning.
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
