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
  implementation complete; awaiting Git delivery approval.
- [ ] M02 Trust inputs and configuration (`codex/m02-inputs-config`).
- [ ] M03 Causal two-stream replacement (`codex/m03-causal-two-stream`).
- [ ] M04 Trust claims (`codex/m04-claim-integrity`).
- [ ] M05 Trust runs (`codex/m05-durable-runs`).
- [ ] M06 Local safety and resource reproducibility (`codex/m06-safety-resources`).
- [ ] M07 Local scaling architecture (`codex/m07-local-scaling`).
- [ ] M08 Dashboard and UI evidence completion (`codex/m08-dashboard-evidence`).
- [ ] M09 Documentation and completion audit (`codex/m09-docs-audit`).

Current milestone: M01 Git delivery gate, then M02 inputs and configuration.

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
