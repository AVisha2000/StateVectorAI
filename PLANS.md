# Living Plans

This file carries state across agent turns and context windows for substantial
work. It is not a backlog and should not be created or expanded for a small,
single-concern task.

Use a living plan when work spans multiple subsystems or sessions, has an
uncertain design, requires delegation, reaches a human gate, or could change a
research conclusion. The parent agent owns the plan and updates it after each
material decision or completed validation step.

## Rules

- One active entry per substantial objective. State an owner and date.
- Bound the scope with owned paths or subsystems and explicit non-goals.
- Write acceptance evidence before implementation.
- Record facts and decisions, not conversation transcripts.
- Mark progress as work happens; do not declare a step complete from an agent's
  summary without inspecting the resulting state.
- Include human gates and explicitly say whether approval has been obtained.
- Keep exact validation commands and their latest results.
- At completion, leave a concise outcome and follow-up, then remove stale
  investigative detail. Durable architecture or research decisions belong in
  their canonical docs, not only here.

## Active plan: agent operating system rollout

Owner: parent agent
Started: 2026-07-10
Objective: make efficient, evidence-first Codex orchestration the repository
default without giving agents authority over science, compute spend, or Git
publication.

Scope: repository agent instructions, project skills and roles, deterministic
verification, and the optional local git scribe. No model, experiment, result,
or dashboard behavior changes.

Acceptance evidence:

- Root and nested instructions route tasks, skills, tests, and human gates.
- Project `planner`, `explorer`, and `verifier` roles load successfully; the
  built-in `worker` receives bounded, disjoint task packets.
- Agent-workflow and verification skills validate and pass forward tests.
- Deterministic setup/change checks cover the policy that prose cannot enforce.
- The optional local git scribe reads staged diffs only and has no Git or
  scientific side effects.
- Focused tests and the full repository suite pass, or environmental failures
  are recorded precisely.

Progress:

- [x] Audit the repository's current instructions, skills, loop constraints,
  tests, and research roadmap.
- [x] Review primary OpenAI, Anthropic, AGENTS.md, and Ollama patterns.
- [x] Install the instruction hierarchy and human operating model.
- [x] Install and validate skills, custom agents, deterministic checks, and the
  local git scribe.
- [x] Run forward tests, focused tests, and full verification.
- [x] Review the integrated diff and record final evidence and follow-up.

Human gates: no GPU/QPU work, result-claim strengthening, destructive cleanup,
commit, push, PR publication, or merge is approved by this plan.

Latest validation:

```text
python scripts/check_agent_setup.py
PASS: Agent setup validation passed (2026-07-10).
python scripts/verify_changes.py --plan
PASS: change-aware verification plan generated.
python scripts/verify_changes.py --run
PASS: agent-setup and agent-tests.
python scripts/local_git_scribe.py --check
PASS: local Ollama endpoint and configured model are ready.
.venv/Scripts/python.exe -m pytest -q tests/test_agent_configuration.py tests/test_local_git_scribe.py tests/test_verify_changes.py --basetemp .tmp/pytest-agent-focused
PASS: 20 passed.
python scripts/verify_changes.py --hook --state-file .tmp/verify-changes/manual-hook.json --timeout 120
PASS: Stop hook JSON allowed after safe CPU verification passed.
.venv/Scripts/python.exe -m pytest -q --basetemp .tmp/pytest-full-agent
PASS: 169 passed, 1 skipped, 36 warnings.
.venv/Scripts/python.exe %USERPROFILE%/.codex/skills/.system/skill-creator/scripts/quick_validate.py <each project skill>
PASS: all seven project skills are valid.
Forward test: fresh-context agent selected the new workflow/model/dashboard/experiment/research/verification skills and caught the dense TensorCircuit backend caveat.
```

## Template

## Active plan: boundary-safe synthetic datasets

Owner: parent agent
Started: 2026-07-10
Objective: prevent generated language-model samples and validation splits from
crossing or sharing synthetic trajectory boundaries.

Scope/non-goals: dataset dispatch, synthetic trajectory representation,
training split/batching integration, and focused invariant tests. No experiment
reruns, result reinterpretation, dashboard changes, GPU/QPU work, or claim
changes.

Acceptance evidence:

- Synthetic datasets retain explicit trajectory boundaries through loading.
- Train and validation partitions contain disjoint whole trajectories.
- Sampled next-token windows stay within one trajectory.
- Existing text-data behavior and the public `load_dataset` compatibility path
  remain intact.
- Focused data/integration tests and the full suite pass.

Progress:

- [x] Confirm the current flatten-then-split path and roadmap priority.
- [x] Add the boundary-aware dataset bundle and compatibility wrapper.
- [x] Route training through trajectory-safe splitting and batching.
- [x] Add invariant/regression tests and run deterministic verification.
- [x] Review the final diff and record outcome/evidence.

Decisions/unknowns:

- Preserve existing generators and `load_dataset()` callers; the new bundle
  contract will reshape generated arrays from recorded config dimensions.
- Text remains a single contiguous stream because it has no synthetic
  trajectory identity.

Human gates: no GPU/QPU work, experiment execution, conclusion strengthening,
destructive cleanup, commit, push, or PR is approved.

Outcome: synthetic datasets now preserve trajectory identity through loading,
splitting, control generation, and training batch sampling. The legacy flat
loader remains available for analysis callers. Follow-up research runs remain
human-scoped and were not started.

Latest validation:

```text
.venv/Scripts/python.exe -m pytest -q tests/test_config_data.py tests/test_quantum_data.py --basetemp .tmp/pytest-boundary-focused-2
PASS: 21 passed, 2 JAX dtype warnings.
python scripts/verify_changes.py --run
PASS: agent-setup, agent-tests, and full python-tests (197.8 seconds).
git diff --check
PASS after removing plan trailing whitespace.
```

````markdown
## Active plan: <objective>

Owner: <parent agent or human>  
Started: YYYY-MM-DD  
Objective: <one outcome>

Scope/non-goals: <owned systems and explicit exclusions>

Acceptance evidence:

- <observable result>

Progress:

- [ ] <ordered milestone>

Decisions/unknowns:

- <decision with reason, or open question>

Human gates: <actions requiring approval and current approval state>

Latest validation:

```text
<exact command and result>
```
````
