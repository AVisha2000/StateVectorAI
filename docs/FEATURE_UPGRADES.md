# StateVector — Feature Upgrades (redesign-era intake)

Forward-looking feature backlog captured during the UI redesign. These are
**not** yet implemented. The first deliverable is the UI redesign
([UI_REDESIGN_PLAN.md](UI_REDESIGN_PLAN.md)); the capabilities below are backend/
product work taken on afterwards and refined as we build. This file is the
*intake* for new features — it feeds `ENHANCEMENT_PLAN.md` (engineering backlog)
and `RESEARCH_PROGRAM.md` (scientific roadmap) once a feature is scheduled. Keep
it honest: nulls and dead-ends are first-class, and no capability here is a
quantum-advantage claim.

Guiding principle from the user: **do not rebuild what already exists** — but
verify a package actually does what is needed before claiming reuse (several
first-pass reuse assumptions in earlier drafts were wrong; corrected below).
Check the license and copy-then-refine. Record source repo + license per feature.

Status legend: `proposed` · `scoped` · `parking-lot` · `in-progress` · `done` · `dropped`.

---

## 1. Research discovery loop  (status: proposed — GREENFIELD)

Turn the cockpit from an experiment *tracker* into a research *engine*. Closed
loop:

> Scan → Library → Synthesize → Dialogue → Ideate → Auto-prioritized queue →
> Bench → Verdict → Atlas → (feeds back into what we scan for)

The core interaction is **bilateral**: the agent scans and presents to the
researcher; the researcher explores and presents to the agent.

**Reality check:** this whole subsystem is **greenfield**. The repo has **no**
runtime LLM/embedding/vector/graph code or dependency — its only quantum-ML
dependency is `pennylane`, with no openai/anthropic/langchain/chromadb/faiss/
sentence-transformers/feedparser/arxiv anywhere in the requirements. The
`.claude/agents` and `.codex/agents`
files are dev-time configs, not a paper-reading service. Plausibly larger than
the UI redesign itself; scope it as new infrastructure, not reuse.

**1a. Library / paper vault.** Archive of everything the lab reads.
- Scanners: arXiv `quant-ph` + a **QML-filtered** `cs.LG` slice (cs.LG alone is
  hundreds/day — do not ingest unfiltered), with a **per-day cap**. Add
  `stat.ML`/PennyLane demos later. Manual dumps always allowed.
- Per-paper: status (inbox → reviewing → linked → feature-candidate → archived),
  field tags, feature-potential score, **bidirectional** experiment links.
- Prior art to evaluate: arXiv API, `feedparser`; metadata via Semantic Scholar /
  OpenAlex.

**1b. Synthesis agent + knowledge vault (new subsystem).** An agent reads papers
and builds a knowledge graph (papers ↔ concepts ↔ our experiments ↔ Atlas nodes).
Decisions that **cannot be deferred** and are **human-gated** (spend):
- LLM/embedding **provider** (paid external service → human gate per `AGENTS.md`);
- **vector store** (e.g. chromadb/faiss) and **graph store**;
- new dependencies + a **recurring per-day cost budget**.
Scoring must be **tiered/cheap**: metadata + keyword heuristics first, an LLM
call only on shortlisted papers, so the "new papers today" strip does not imply
an LLM call per paper.

**1c. Discover copilot.** Dialogue grounded in the vault with inline citations
and "draft this in the Bench" actions, beside an **auto-prioritized idea queue**.
Ranking = `novelty × feasibility × expected-advantage × Atlas-gap` **plus an
explicit decisiveness/falsification term** (value of resolving an open Atlas
cell, closing a track, or strengthening the classical challenger). Null-producing
and baseline-strengthening ideas rank **on par** with advantage-seeking ones —
the mission is verification-first. Falsified ideas update the vault so they stop
being re-proposed.

**1d. Automatic experiment builder.** Turn an idea into a runnable spec
(model + circuit + fair control). The hard part is **schema-constrained
idea→spec generation** (an LLM emitting a constrained output that round-trips
into `qllm/registry.py` `BACKEND_TYPES`/`CIRCUIT_ANSATZ_TYPES` with a matched
control) — **not** something `sQUlearn` supplies. Treat sQUlearn only as
inspiration for the target runnable-spec **API shape**, not as the assembly step
(it is also a heavy Qiskit/PennyLane-stack dependency the repo does not carry).

---

## 2. Integrated QML Designer — semantic zoom  (status: proposed)

A **continuous zoom** from the whole field down to individual gates, for
non-experts.

**2a. Semantic zoom (one continuous surface with the Atlas).**
Zoomed out = the Atlas; zoom in through `ML → LLMs → Attention → Encoder →
model`. **The domains→components hierarchy does not exist in repo data today** —
`RESEARCH_MAP.yaml` is a flat list of ~19 mechanism areas, so this ontology is a
**new curated schema** to hand-populate with budgeted maintenance (see
UI_REDESIGN_PLAN.md § Atlas data). Classical slots render the neural network;
quantum slots render the circuit with visible gates.

**2b. Circuit & gate builder — split the renderer honestly.**
- **Read-only preview:** reusable — server-render PennyLane `qml.draw_mpl` to SVG.
- **Interactive editor** (drag-drop gates, live editing): **no drawer provides
  this.** `qml.draw_mpl` and the Qiskit drawer emit static images, and the repo
  has no `qiskit` dependency. Adopt a named OSS JS circuit composer (candidate:
  the `quantum-circuit` npm package — confirm license) or accept a from-scratch
  SVG editor. Must round-trip to `qllm/registry.py`.

**2c. Classical ↔ quantum toggle at every level** — same slot, NN vs circuit.

---

## 3. Community circuit stream  (status: PARKING-LOT — no confirmed source)

Ingest promising community quantum circuits into the Library/Designer.
- The referenced source (heard as **"ECDA"**) is **unidentified** and possibly
  mis-heard; **gated on identifying a real API** before any scoping.
- The named fallbacks — **MQT Bench, QASMBench** — are **static benchmark
  corpora, not live community-upload APIs**. The realistic near-term version is a
  **periodic batch import from a static corpus**, not a live stream.
- **Not** a committed Designer surface or a build phase until a real source is
  confirmed. This item stays in the parking lot; it is a Phase 5 dependency at
  most, off the critical path.

---

## 4. Package leverage map (don't rebuild — but verify reuse holds)

Backend/experiment-engine capabilities the UI exposes. Confirm license and
copy-then-refine. **Reuse-reality notes** flag where a first-pass reuse claim
does not hold.

| Package | Role | Reuse reality |
| --- | --- | --- |
| TensorCircuit-NG | GPU/JAX/tensor-network execution | already partially in as `tensorcircuit_mps`; genuine |
| PennyLane (present) | differentiable hybrid models; `qml.draw_mpl` for read-only circuit preview | genuine for preview only, not the editor |
| PennyLane Demos | algorithm reference | seeds Library |
| sQUlearn | quantum-kernel/QSVM baselines; API-shape inspiration | **not** the auto-builder assembly step; heavy new dep |
| Qrisp | higher-level quantum algorithm design | for quantum-only paradigms |
| Mitiq | noise / error mitigation | **gated on first adding a noisy/hardware backend** — repo runs noiseless `default.qubit` with `diff_method=backprop`, so there is nothing to mitigate today. Near-term QPU-readiness uses the existing `parameter_shift_gradient_snr` shot-noise gate in `qllm/quantum/metrics.py`, not Mitiq. |
| Qiskit drawer | static circuit drawing | static image only; adds a large dep — prefer the JS composer for the editor |
| Qibo, OpenQAOA, Classiq lib | high-perf sim, QAOA, algorithm library | evaluate as needed |

**Avoid** Qiskit Algorithms as a foundation — no longer officially supported by IBM.

---

## 5. Cross-cutting notes

- **Provenance everywhere:** experiment ↔ papers ↔ circuits ↔ Atlas node ↔
  verdict, both directions, so "feature potential" is graded by evidence.
- **Human gates unchanged:** GPU/QPU runs, claim promotion, spend, and paid
  external services stay human-gated. Auto-builder and auto-queue *propose*.
- **Negative results are first-class** across the whole loop.
- **Hosting:** the public Atlas is a **separate static export / read-only
  surface**, never co-hosted with the localhost cockpit's filesystem/queue
  routes (root `AGENTS.md` localhost rule); public hosting is its own
  human-gated project. See UI_REDESIGN_PLAN.md §2.
