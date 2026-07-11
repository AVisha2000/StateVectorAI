# StateVector Dashboard — UI Redesign Plan

Status: design locked, implementation not started
Date: 2026-07-11
Owner: user (Arlind); apex builder Claude Code (Opus 4.8)
Companion files: [FEATURE_UPGRADES.md](FEATURE_UPGRADES.md) (feature intake),
[AGENT_OPERATING_MODEL.md](AGENT_OPERATING_MODEL.md) (how we build),
[RESEARCH_PROGRAM.md](RESEARCH_PROGRAM.md) (why we build).

**Relationship to existing docs.** This file is the *current* UI roadmap and
supersedes [UI_UPGRADE_PLAN.md](UI_UPGRADE_PLAN.md), which is retained only as
the historical record of the shipped dashboard. Where this plan and
UI_UPGRADE_PLAN.md disagree, this plan wins.

This is the living plan for a full redesign of the QLLM/StateVector dashboard.
The old dashboard is functionally rich but disorganized. The redesign
reorganizes the product around a single research spine and adds the missing
research and design surfaces. Implementation happens after this plan; the plan
is refined continuously as the backend is built.

## 1. Mission for the UI

Be **the** interface for quantum-ML experimentation — a personal research
cockpit that is also a presentable, hostable product. Verification-first, not
advantage-seeking: per [RESEARCH_PROGRAM.md](RESEARCH_PROGRAM.md), "do not
optimize for a quantum-advantage headline," and mapping where advantage is
**absent** is as valuable as where it is present. The UI's job is to make honest
claims easy and overclaiming hard: surface diagnostics as diagnostics, keep null
results first-class, and require matched controls and human sign-off before any
verdict is promoted.

## 2. Users and the two faces (security-reconciled)

- **Public (logged-out): the Atlas, read-only.** Root `AGENTS.md` forbids making
  the local dashboard a public service, and `qllm/dashboard/AGENTS.md` mandates
  localhost-only serving with no broadened CORS/file/path routes. Therefore the
  public Atlas must **not** be co-hosted with the cockpit's FastAPI app (which
  exposes filesystem/dataset/queue routes). The public face is a **separate
  static export** (or a dedicated read-only API with no filesystem/path/queue
  surface). Public hosting is its own **human-gated** project, reconciled with
  the localhost-only rule before anything is exposed.
- **Researcher (logged-in): the full cockpit** on localhost — Discover, Library,
  Designer, Bench, Runs, Verdicts, Datasets, System. Simple auth is acceptable
  for the local cockpit; it never justifies broadening the dashboard's network
  exposure.

## 3. Product spine

```
Literature loop:  Scan → Library → Synthesize → Dialogue → Ideate → Queue
Experiment loop:  Bench → Runs → Verdict → Atlas
                        (Verdict + Atlas feed back into what we Scan for)
```

The unifying interaction is **bilateral**: the agent scans and presents to the
researcher; the researcher explores and presents to the agent. Both feed the
same knowledge vault and idea queue.

## 4. Information architecture

Sidebar, three groups:

- **Research** — Overview · Discover · Library · Atlas
- **Experiments** — Designer · Bench · Runs · Verdicts
- **System** — Datasets · Queue & Backends

### Old route → new surface migration

The current app (see `qllm/dashboard/frontend/src/main.jsx`) is consolidated,
not deleted. Nothing is removed until its replacement carries its inbound links.

| Old page/route | New surface | Kind |
| --- | --- | --- |
| `LabOverview` (`/`) | Overview | rename + trim |
| `Jobs` (`/experiments`), `Run`, `RunWorkspace` | Runs (+ run detail) | consolidate |
| `Launch` (`/launch`), `Comparison` | Bench + Verdicts | consolidate |
| `ResultsHub` (`/results`), `Suites`/`Suite` (`/results/legacy`), `ResearchResults`, `StudyReport` | Verdicts + Atlas | **retire the 3 overlapping results systems** |
| `Studies`/`Study` | Bench rigor levels + Verdicts | fold in |
| `ScalingTest`/`ScalingTests` | Runs detail → scaling view | fold in |
| `Explore` | Atlas | replace |
| `Models` | Designer | replace |
| `Datasets`, `GPU`, `Live`, `Docs` | Datasets, Queue & Backends | rename/absorb |
| net-new | Discover, Library | new |

`/launch` is **not orphaned/deletable** — it is referenced by `main.jsx`,
`Explore.jsx`, `Jobs.jsx`, `LabOverview.jsx`, `Launch.jsx`, and
`ScalingTests.jsx`. Its problem is that it is absent from the sidebar; the
migration moves its New-Experiment / Queue / Rerun entry points onto Bench and
re-points those inbound links, rather than removing the route.

### Surfaces

| Surface | Purpose | Key elements & integrity notes |
| --- | --- | --- |
| **Overview** | live lab state | running/queued tiles, latest verdicts, open hypotheses, new-papers strip. Any "advantage candidate" tile must show its current claim level and read "candidate, not established" — never a headline result. |
| **Discover** | research copilot | dialogue grounded in the vault, auto-prioritized idea queue. Ranking includes a **decisiveness/falsification** term (value of resolving an open Atlas cell, closing a track, or strengthening the classical challenger) alongside novelty/feasibility/expected-advantage, so null-producing and baseline-strengthening ideas rank on par with advantage-seeking ones. |
| **Library** | paper archive + knowledge vault | filterable table, feature-potential score, bidirectional experiment links, knowledge graph, scanner config, synthesis status. Greenfield — see §5. |
| **Designer** | semantic-zoom QML builder | zoom breadcrumb (illustrative target), circuit render + editor, classical↔quantum toggle, matched-control + properties. Renderer split: read-only preview vs interactive editor (§6). |
| **Atlas** | classical-vs-quantum map (public face) | domains → components → head-to-head; quantum-only branches; literature-suggested nodes. Data is a **new curated ontology** joined to **derived verdict comparisons** (§ Atlas data). Each node shows its **claim level** (per `RESEARCH_MAP.yaml` `claim_levels`) kept **visually distinct from replication status**; **null / "no advantage found" cells render with equal prominence** to positive ones — no single green "quantum wins" gloss. |
| **Bench** | hypothesis → fair test | hypothesis (may cite a paper/Discover idea), candidate + auto-matched control (a **proposal only**, not a passed fairness gate), protocol, rigor selector. The fairness gate references the multiple Pareto frontiers (param-, memory-, sample-, training-call-, wall-time-, cost-matched) and the baseline/dequantization ladder from RESEARCH_PROGRAM.md; the researcher confirms the challenger set before promotion. |
| **Runs** | every run in one table | live/queued/done/failed; type, backend, dataset, val_ppl. Wall-time is labeled **simulator cost**, never placed where a reader would take it as QPU cost. Failed runs keep logs. |
| **Verdicts** | advantage adjudication | seed-band curve, **per-dimension scorecard bound to the claim ladder** (each diagnostic dimension — expressibility, entanglement, param count, Hilbert dimension — labeled explicitly as *diagnostic / mechanism candidate*, never *advantage*). **No composite/averaged advantage score is produced, and a strong diagnostic dimension can never raise the overall claim level.** Auto caveats; promotion up the ladder is human-gated. |
| **Datasets** | synthetic quantum-native + HF import | provenance, "which experiments used this" |
| **Queue & Backends** | worker, backends, reservations | pennylane / tensorcircuit / tensorcircuit_mps, GPU gated, queue depth. "QPU-readiness" is a **hardware-feasibility** indicator (noisy-sim/emulator stage), distinct from hardware reproduction. |

### Atlas data (accurate)

`RESEARCH_MAP.yaml` is a **flat list of ~19 mechanism areas** tagged by
`pipeline_stages` / `quantum_resources` / `advantage_targets` — **not** an
`ML → LLMs → Attention → Encoder` tree. The domains→components→head-to-head
ontology the Atlas/Designer zoom implies is a **new curated schema** to be
hand-populated (a new `RESEARCH_MAP.yaml` section or a sibling file) with
budgeted maintenance. There is also **no persistent verdict store**:
`qllm/resultsdb.py` has runs/metrics/steps/studies tables but no verdict table;
verdicts are **derived on the fly** from paired comparisons in
`qllm/dashboard/evidence.py` / `explore.py`, backed by the real claim-ladder and
advantage logic in `qllm/quantum/advantage.py`. So the Atlas is "new ontology +
derived verdicts"; a persistent verdict/adjudication store is a Phase 2/3
dependency to build, not an existing asset.

## 5. Research discovery loop — greenfield subsystem

See [FEATURE_UPGRADES.md](FEATURE_UPGRADES.md) §1. **This is greenfield**: there
is no runtime LLM / embedding / vector / graph code or dependency in the repo
today (`requirements-cpu.txt` carries `pennylane` and no
openai/anthropic/langchain/chromadb/faiss/sentence-transformers/feedparser/
arxiv). The only "agents" are the dev-time `.claude/agents` and `.codex/agents`
configs, **not** a paper-reading service. Scoping this subsystem requires
naming, up front and human-gated: the LLM/embedding provider, the vector store,
the graph store, the new dependencies, and a recurring per-day cost budget. Any
paid external LLM/embedding service is human-gated per `AGENTS.md`.

## 6. QML Designer & semantic zoom

See [FEATURE_UPGRADES.md](FEATURE_UPGRADES.md) §2. Two separate renderer needs:

- **Read-only preview** — genuinely reusable: server-render PennyLane
  `qml.draw_mpl` to SVG.
- **Interactive editor** (gate palette, live editing, drag-drop) — **no drawer
  provides this.** `qml.draw_mpl` and the Qiskit drawer emit static images (and
  the repo has no `qiskit` dependency). The editor must adopt a named OSS JS
  circuit composer (candidate: the `quantum-circuit` npm package — confirm
  license) or be an accepted from-scratch SVG build. Do not assume drawer reuse
  eliminates the editor.

The editor must round-trip to the existing model/config schema in
`qllm/registry.py` (`BACKEND_TYPES` / `CIRCUIT_ANSATZ_TYPES`) so a built circuit
becomes a runnable Bench experiment with a matched classical control.

## 7. Design system

- **Palette (validated CVD-safe, both themes):** Quantum magenta `#a233a8`
  (light) / `#c05fc0` (dark); Classical blue `#2a78d6` / `#3987e5`; Accent
  indigo `#4f46c8` / `#8f8af0`. Validated with the dataviz skill validator
  against surfaces light `#fdfdfc` / dark `#17171c`. Supersedes the old
  GitHub-style chart tokens.
- **Type:** system UI sans; tabular numerals; balanced headings.
- **Tokens:** CSS custom properties; light + dark both first-class.
- **Aesthetic anchors:** Weights & Biases, Datadog, MLflow — clean, dense but
  not crowded, no dead widgets.

## 8. Tech stack (decided)

- Keep **React 18 + Vite + react-router 6**.
- Add **TanStack Query** for server state/caching.
- **Live run updates:** new backend work, not just a client change — there is no
  streaming endpoint today (`LabOverview` polls 3000ms, `Jobs`/`Live` 2000ms).
  Requires new FastAPI streaming endpoints (SSE/WebSocket) + event emission from
  `runner.py` and the `lab_jobs`/`live_runs` tables.
- Keep **CSS-variable design tokens** (no Tailwind).
- **Charts:** recharts for standard charts; hand-authored SVG for the circuit
  renderer and seed-band/scaling viz.
- **Graph surfaces (Atlas, knowledge graph):** RESEARCH_PROGRAM.md (line 499)
  makes a canonical decision: **Cytoscape.js**, built locally. The redesign
  **adopts Cytoscape.js** for the Atlas and knowledge graph rather than
  hand-authored SVG, to honor that decision and get robust graph layout; SVG is
  used only for the smaller bespoke viz.
- Backend keeps **FastAPI**; gains the greenfield **research service** (§5) and
  Designer schema round-tripping.
- Rationale: lowest migration risk, biggest wins, solo-maintainable.

## 9. Build phases

Each phase is shippable, has explicit acceptance evidence, and runs the standard
checks. Frontend checks (from `AGENTS.md`):
`Push-Location qllm/dashboard/frontend; npm test; npm run build; Pop-Location`.
Queue/API smoke (CPU): `python scripts/queue_smoke.py --steps 1 --eval-every 1
--device-target cpu`. Plus `python scripts/verify_changes.py --run` and `pytest -q`.

1. **Foundation.** New IA + shell + design tokens + TanStack Query; port
   Overview, Runs, Bench, Verdicts onto it; consolidate the three results
   systems; migrate `/launch` entry points (do not break inbound links).
   *Acceptance:* new nav navigable, no dead routes, old results routes redirect,
   `npm test`/`npm run build` green, queue smoke green.
2. **Diagnostics + Verdicts depth.** Surface `qllm/quantum/metrics.py`
   (barren-plateau variance + scaling fit, `parameter_shift_gradient_snr`,
   expressibility, Meyer-Wallach entanglement); claim-ladder-bound scorecard;
   scaling view; build the persistent verdict store. *Acceptance:* diagnostics
   render with correct claim-level labels; no composite advantage score; tests
   green.
3. **Atlas.** Curated ontology schema + Cytoscape.js graph + derived-verdict
   join; null cells first-class; **static read-only public export** separated
   from the cockpit app. *Acceptance:* graph renders from the schema; null and
   positive cells equally visible; public export has no filesystem/queue routes.
4. **Research loop (greenfield, de-risk first).** Spike the LLM+vault end-to-end
   on ONE paper (provider, cost, embedding, retrieval) before building the full
   loop; then Library + bounded scanners + synthesis + Discover copilot + idea
   queue. Human-gate the provider/cost decision. *Acceptance:* one paper flows
   scan→vault→cited-in-dialogue; per-day cost model documented.
5. **Designer (highest-risk, de-risk first).** Prototype exactly one zoom
   LOD transition and one interactive-editor round-trip to `registry.py` before
   committing the full canvas; then semantic zoom + editor + classical↔quantum
   toggle. *Acceptance:* one built circuit runs on the Bench with a matched control.

Phases 4 and 5 are each plausibly larger than 1–3 combined; the spikes exist so
they are not treated as incremental.

## 10. Execution model — parallel agents under triage

See [AGENT_OPERATING_MODEL.md](AGENT_OPERATING_MODEL.md) for the canonical model.
**Today vs target, stated honestly:**

- **Today:** the apex Claude Code session (Opus 4.8) dispatches into the **Claude
  triage tree** via model-pinned subagents in `.claude/agents/`: planner/verifier
  on Opus 4.8; explorer/terra-worker on Sonnet 5; luna-explorer/mini-worker/
  spark-helper on Haiku 4.5. The **Codex triage tree** (`.codex/agents/`:
  planner/verifier GPT-5.6, explorer + terra_worker on gpt-5.6-terra, Luna/Mini/
  Spark cheaper tiers) triages within itself. There is **no harness bridge** for
  Claude to invoke Codex directly; the two clients coordinate through the shared
  workspace with one named parent owning the objective.
- **Target:** Claude Opus 4.8 at the apex over **both** trees, dispatching into
  either. All delegation uses the single task-packet schema in
  `.agents/skills/qllm-agent-workflow/references/delegation-contract.md` with
  **disjoint file ownership** (never two writers on one file), and only the
  parent integrates. Human gates always hold.

## 11. Current status / how to pick up in a new chat

- **Where the plan lives:** these docs are committed to `main` under `docs/`.
  A fresh session should read **this file** plus `FEATURE_UPGRADES.md`. The
  canonical **visual reference** is
  [design/statevector-cockpit-v3.html](design/statevector-cockpit-v3.html)
  (committed). The interactive `claude.ai/code/artifact/...` mock URLs are
  **session-private and not portable** — do not rely on them across sessions.
- **Branch ownership & coordination:** `main` carries the backend/experiment
  work and these docs, worked by the Codex client (branch `codex/*`) and Claude
  Code. UI implementation happens on the **`ui-redesign`** branch/worktree.
  Coordination protocol: **disjoint file ownership, one integrator, verify a
  subset commit in an isolated worktree before pushing** (see §12). Do not have
  two agents edit the same file concurrently.
- **Before working on `ui-redesign`:** that worktree branched at `44ff6a6` and
  is behind `main`; **rebase/merge it onto current `main` first** so this plan,
  `FEATURE_UPGRADES.md`, the reconciled `AGENT_OPERATING_MODEL.md`, and
  `design/statevector-cockpit-v3.html` are present there.
- **Done so far:** design direction locked through mock v3; validated palette;
  stack decided; feature intake captured; Claude model triage pinned (commit
  `03b7ea4`); this plan reviewed by a 5-dimension adversarial pass and corrected.
- **Next actions (in order):** (1) rebase `ui-redesign` onto `main`; (2) start
  Phase 1 there; (3) scope the greenfield research service (§5) incl. the
  human-gated provider/cost decision. The community-circuit source ("ECDA", §
  FEATURE_UPGRADES.md §3) is a **Phase 5 dependency, off the critical path** —
  a fresh session proceeds without it, using the recorded candidate leads.
- **Not yet started:** any frontend rebuild code, the research service, the
  Designer backend, auth, the public Atlas export.

## 12. Constraints and lessons

- **Human gates unchanged:** GPU/QPU runs, claim promotion, `RESULTS.md` edits,
  spend, paid external services, and consequential Git actions stay human-gated.
  Auto-builder and auto-queue *propose*; the researcher approves.
- **Negative results are first-class**; the UI makes nulls visible, never hides
  them; "no advantage found" is a successful outcome.
- **No overclaiming:** lower loss / larger Hilbert space / entanglement /
  expressivity are **diagnostics, not advantage**. Verdicts enforce matched
  controls, keep claim-level and replication distinct, and never emit a composite
  advantage score.
- **Simulator ≠ QPU:** simulator wall-time is labeled as such and never presented
  as QPU cost.
- **Coordination lesson:** during the model-triage work two agents edited the
  same files concurrently and nearly pushed a broken `main`. Rule: disjoint file
  ownership, one integrator, verify a subset commit in an isolated worktree
  before pushing.
- **Reuse over rebuild** with a license check and copy-then-refine — but verify
  the package actually does what is needed (several first-pass reuse claims here
  were wrong; see §6 and FEATURE_UPGRADES.md §4).
