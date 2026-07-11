# QLLM Agent Operating Model

Status: adopted dual-client repository workflow
Date: 2026-07-11

QLLM uses a small agent operating system so that AI assistance improves
research throughput without manufacturing confidence. The objective is not to
maximize agent count. It is to put the right context, capability, and check at
the right boundary while humans retain control of scientific claims, expensive
compute, destructive actions, and publication.

The research mission and evidence rules remain canonical in
[RESEARCH_PROGRAM.md](RESEARCH_PROGRAM.md) and
[RESEARCH_MAP.yaml](RESEARCH_MAP.yaml). This document explains the operating
model; it does not duplicate those scientific policies.

| Concern | Portable core | Codex adapter | Claude Code adapter |
| --- | --- | --- | --- |
| Instructions | Root and nested `AGENTS.md` | Native discovery | Import-only sibling `CLAUDE.md` |
| Workflows | `.agents/skills/` | Native project skills | `.claude/skills/` metadata bridges |
| Delegation | Canonical delegation contract | `.codex/agents/*.toml` | `.claude/agents/*.md` |
| Verification | `scripts/verify_changes.py` | `.codex/hooks.json` | `.claude/settings.json` |

The portable core owns repository meaning. Client adapters contain only
discovery, tool, model, permission, or hook details that cannot be shared.

## Four layers

### 1. Scoped instructions

The root `AGENTS.md` contains the invariant repository contract. Nested files
under `qllm/`, `qllm/dashboard/`, `benchmarks/`, `configs/`, `tests/`, and
`docs/` add only local detail. A nearer file can specialize a rule, while the
rest of the parent baseline remains active. This progressive disclosure keeps
unrelated implementation detail out of each task's context. Each guide has a
sibling `CLAUDE.md` containing only `@AGENTS.md`; the validator rejects copied
or divergent Claude instructions.

### 2. On-demand skills

Repository skills under `.agents/skills/` hold reusable domain workflows:
model development, dashboard development, experiment execution, research
review, orchestration, and verification. Their short metadata enables routing;
their full instructions are loaded only when the task triggers them. This
separates always-on constraints from capability-specific procedure. Claude
Code discovers thin `.claude/skills/` bridges whose metadata must match the
canonical skill and whose body directs the agent back to that canonical file.
This avoids Windows-fragile symlinks and duplicated workflow bodies.

### 3. Bounded roles

The main agent is the orchestrator and final integrator. Project roles are:

- `planner`: converts a substantial, ambiguous objective into acceptance
  evidence, risk gates, dependencies, and disjoint work packets;
- `explorer`: performs read-only repository or literature discovery and returns
  an evidence-bearing summary rather than raw search context;
- `terra_worker`: implements one bounded packet with exclusive file ownership
  and named validation;
- `luna_explorer`: handles lower-cost read-only inventories, usage tracing,
  test-gap analysis, and documentation maps;
- `mini_worker`: performs only low-risk mechanical edits and requires Terra or
  parent review;
- `spark_helper`: provides isolated text-only suggestions and never edits;
- `verifier`: starts fresh and read-only, inspects the final state and test
  evidence, and defaults to rejection when evidence is absent.

The parent uses no more than two or three children at once. Parallelism is for
independent or read-heavy work; sequential dependencies stay together. Because
agents share a workspace, write ownership never overlaps, and only the parent
integrates results. One parent owns an objective even when Codex and Claude Code
are both active; cross-client work uses the same disjoint-ownership contract.

The main agent uses GPT-5.6 as the root orchestrator. Planner and verifier work
remain on GPT-5.6, Terra is the default implementation tier, Luna handles
read-only discovery, Mini handles reviewed mechanical work, and Spark is
restricted to text-only assistance. If a configured model is unavailable, the
runtime must report that failure rather than silently substitute another tier.
Claude role files use the hyphenated equivalents (`terra-worker`,
`luna-explorer`, `mini-worker`, and `spark-helper`) and inherit the active
Claude model instead of pinning transient product identifiers.

### 4. Deterministic verification

Prose guides decisions; executable checks enforce what can be checked. The
repository validates agent configuration, plans relevant checks from changed
paths, runs those checks on request, and exercises the orchestration setup in
pytest on Windows and Linux. Codex and Claude Code both call the same verifier
at Stop. The hook has a one-block loop fuse, returns only the documented
`decision: block` shape when more work is required, and otherwise omits a
decision. A fresh verifier checks scope, commands, failures, gates, and the
final diff. Hooks may invoke deterministic checks, but a hook result never
overrides the human gates in `AGENTS.md`.

`PLANS.md` is the cross-session handoff for substantial work only. It records
the objective, acceptance evidence, milestones, decisions, gates, and exact
validation. Small tasks avoid that overhead.

## Task packets and integration

Every delegation packet follows the sole schema in
`.agents/skills/qllm-agent-workflow/references/delegation-contract.md`. It names
an objective, mutation authority or exclusive write ownership, required
context, non-goals, gates, validation, and a bounded return. The result is a
lead for the parent, not proof; the parent inspects repository state and reruns
the checks that matter.

For research work, the packet also states the hypothesis, comparator, data
access, resource measure, falsifier, and claim ceiling. An implementation agent
does not promote its own result.

## Token efficiency

- Route first, then read the nearest instructions and matching skills.
- Keep repository maps and living plans as indexes; open source files only
  along the affected execution path.
- Isolate noisy inventories, web research, and logs in `explorer` context and
  return summaries with file paths or primary links.
- Reuse or resume a child when continuity is valuable; do not repeat the same
  scan in several contexts.
- Use strong reasoning for planning, scientific interpretation, and fresh
  verification. Use smaller contexts/models only for bounded mechanical work.
- Prefer focused tests during iteration and a blast-radius-based final suite.
  Command output, not confident prose, is the completion signal.

## Adoption and checks

Start either client from the repository root so root-to-leaf instructions,
project configuration, agents, and skills are discoverable:

```powershell
codex -C .
claude
```

Validate the installed operating layer and preview the checks selected for the
current diff:

```powershell
python scripts/check_agent_setup.py
python scripts/verify_changes.py --plan
```

Run the selected deterministic checks after a change:

```powershell
python scripts/verify_changes.py --run
```

After a configuration change, start a fresh client session. In Codex, inspect
the active configuration, agents, skills, and hooks. In Claude Code, use
`/memory`, `/agents`, `/skills`, and `/hooks`; accept workspace trust only after
reviewing the committed hook. Static validation runs without either client,
but these discovery checks require installed runtimes and cannot be proven by
CI alone.

## Source basis

This design intentionally combines vendor-neutral repository guidance with
tool-specific capabilities:

- OpenAI's [AGENTS.md discovery guide](https://learn.chatgpt.com/docs/agent-configuration/agents-md),
  [subagent guide](https://learn.chatgpt.com/docs/agent-configuration/subagents),
  [skills guide](https://learn.chatgpt.com/docs/build-skills), and the
  [open-source Codex repository](https://github.com/openai/codex) establish
  scoped instructions, focused agent roles, and on-demand capabilities. The
  OpenAI Agents Python repository is a direct combined example through its
  [AGENTS.md](https://github.com/openai/openai-agents-python/blob/main/AGENTS.md),
  [PLANS.md](https://github.com/openai/openai-agents-python/blob/main/PLANS.md),
  and [.agents skills](https://github.com/openai/openai-agents-python/tree/main/.agents/skills).
- Anthropic's [`.claude` directory map](https://code.claude.com/docs/en/claude-directory),
  [project-memory and `AGENTS.md` import guidance](https://code.claude.com/docs/en/memory),
  [skills documentation](https://code.claude.com/docs/en/skills),
  [subagent documentation](https://code.claude.com/docs/en/sub-agents),
  [hooks reference](https://code.claude.com/docs/en/hooks),
  [context-engineering guidance](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents),
  and [Claude Code repository](https://github.com/anthropics/claude-code)
  support thin imports, on-demand workflows, isolated roles, lifecycle checks,
  and concise handoffs.
- The open [AGENTS.md format](https://agents.md/) and
  [reference repository](https://github.com/agentsmd/agents.md) support a
  portable root-plus-nested instruction hierarchy.
These sources are design inputs, not dependencies. Deterministic repository
tests and the QLLM human gates remain authoritative for this project.
