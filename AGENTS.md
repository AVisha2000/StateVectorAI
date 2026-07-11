# QLLM Agent Instructions

## Mission

QLLM is a verification-first research system for finding, falsifying, scaling,
and eventually hardware-testing quantum mechanisms in machine learning. Do not
optimize for a quantum-advantage headline. Null and negative results are useful.
Use `docs/RESEARCH_PROGRAM.md` for the scientific roadmap and
`docs/RESEARCH_MAP.yaml` for the machine-readable map of explored areas.

## Instruction scope

This file is the repository baseline. A nested `AGENTS.md` adds path-specific
guidance and takes precedence only where it is more specific; all unrelated
root rules remain in force. Sibling `CLAUDE.md` files are import-only adapters
for Claude Code; never duplicate substantive guidance in them. Start Codex or
Claude Code from the repository root so the full instruction chain and project
skills are discoverable.

## Route the task before reading broadly

Load only the skills that match the task, and read each selected `SKILL.md`
before acting.

| Work | Required skill |
| --- | --- |
| Multi-file, delegated, or high-ambiguity repository change | `$qllm-agent-workflow` |
| Model, quantum layer/backend, data, config, or training change | `$qllm-model-development` |
| Dashboard API, queue, database view, or React UI | `$qllm-dashboard-development` |
| Run, queue, debug, or compare an experiment | `$qllm-experiment-runner` |
| Interpret evidence or change claim-bearing text | `$qllm-research-protocol` |
| Verify a repository change | `$qllm-code-change-verification` |
| Daily/weekly loop triage | `$qllm-loop-triage` (it reads the loop budget and constraints) |

Prefer targeted `rg`/`rg --files`, the relevant nested instructions, and the
maps referenced by a skill over scanning the whole repository.

## Working loop

1. Inspect `git status --short`; preserve user changes and generated research
   artifacts.
2. State the objective, acceptance evidence, scope, gates, and validation.
3. For substantial work, create or update the living entry in `PLANS.md`
   before implementation. A small, single-concern change needs only a short
   in-turn plan.
4. Make the smallest coherent change. Add or update tests with behavior.
5. Run focused checks first, then the broader checks warranted by the blast
   radius. Do not claim success without fresh command output.
6. Review the final diff for scope, scientific meaning, and unintended
   artifacts. Report files changed, exact checks and results, and remaining
   uncertainty.

Substantial means any of: multiple subsystems, uncertain architecture, more
than one session, expensive or human-gated validation, or research conclusions.

## Delegation

The parent agent owns the plan, integration, final verification, and user
handoff. Delegate only when separation saves context or wall-clock time.

- Use `planner` for substantial ambiguous work and `explorer` for bounded,
  read-only discovery. Use `terra_worker` (Claude Code: `terra-worker`) for a
  disjoint implementation packet. Use `verifier` after implementation with
  fresh, read-only context.
- Spawn at most two or three children, only for independent or read-heavy
  work. Keep dependent steps in one context.
- This is a shared workspace: give writing agents disjoint file ownership.
  Never let two agents edit the same file. The parent reviews and integrates
  every return.
- Do not delegate trivial lookups, one-file edits, or work whose result must be
  reread in full by the parent.
- Use
  `.agents/skills/qllm-agent-workflow/references/delegation-contract.md` as the
  sole task-packet schema and role-routing reference; do not maintain a second
  template here.
- One parent owns an objective. If Codex and Claude Code work concurrently,
  name the parent explicitly and give every writing agent disjoint files.

## Token and model economy

- Keep the main context for decisions and integration; send noisy searches,
  inventories, and logs to bounded children.
- Ask children for evidence-bearing summaries, not transcript dumps. Resume a
  child when continuity matters instead of repeating discovery.
- Use the strongest available reasoning for planning, research interpretation,
  and verification. Use smaller agents for narrow mechanical work with explicit
  tests.
- Route bounded implementation to `terra_worker`, read-only inventories to
  `luna_explorer`, mechanical edits to `mini_worker`, and isolated text-only
  suggestions to `spark_helper` when those profiles are available. Mini output
  requires Terra or parent review. The parent retains architecture, security,
  research, integration, and Git authority.

## Validation commands

From the repository root:

```powershell
python scripts/check_agent_setup.py
python scripts/verify_changes.py --plan
python scripts/verify_changes.py --run
pytest -q
```

Use a focused test file while iterating, for example:

```powershell
pytest -q tests/test_quantum.py
```

Dashboard frontend:

```powershell
Push-Location qllm/dashboard/frontend
npm run build
Pop-Location
```

With the local dashboard already running, queue/API smoke tests must stay on
CPU unless the user approves GPU work:

```powershell
python scripts/queue_smoke.py --steps 1 --eval-every 1 --device-target cpu
```

If Windows raises a temp-directory `WinError 5`, rerun pytest with a known
writable `--basetemp` and report the environment issue separately from product
failures. Never weaken or skip a test to make a check pass.

## Human gates

Get explicit user approval before:

- starting GPU/QPU runs, long sweeps, high-memory simulation, cluster/cloud
  spend, CUDA/JAX environment changes, or paid external services;
- strengthening a scientific conclusion, promoting an advantage level, or
  changing claim-bearing `RESULTS.md` text;
- cancelling jobs or deleting, rewriting, or moving experiment artifacts,
  databases, user work, or tracked files;
- committing, pushing, publishing a PR, merging, force-updating, or other
  consequential Git/remote side effects.

Never expose secrets, make a public service from the local dashboard, silently
discard negative results, or present simulator cost as QPU cost. A lower loss,
larger Hilbert space, entanglement, expressivity, or parameter-count win is not
by itself quantum advantage.

## Loop work is a separate mode

`LOOP.md`, `STATE.md`, `loop-budget.md`, and `loop-constraints.md` govern only
explicit loop/triage runs. L1 report-only and loop subagent limits do not apply
to ordinary user-directed implementation. Do not turn a loop request into an
auto-fix unless its documented promotion gate has been met.
