# QLLM Lab UI Upgrade Plan

> **Superseded as the forward roadmap.** This document is the historical record
> of the *shipped* dashboard. The current UI direction is a full redesign —
> see [UI_REDESIGN_PLAN.md](UI_REDESIGN_PLAN.md). Retain this file for history;
> do not treat it as the active plan.

## Implementation status

The original phases are retained as a design record. All structural phases are
implemented: Lab Overview, filtered jobs, model diagrams, dedicated
comparisons, studies/reports, persisted per-layer model specs, and the visual
builder. M08 completed canonical metric/claim IDs, paired/equivalence/power
statistics, complete fairness mismatches, manifests, checkpoints/recovery,
resource ledgers, unavoidable warnings, and responsive/browser QA.

The original Phase 7 name, “Quantum Advantage Reporting,” is superseded by
cautious evidence reporting: the UI presents claim contracts and limitations
but never promotes a scientific claim. See
[`COMPLETION_AUDIT.md`](COMPLETION_AUDIT.md) for phase-by-phase evidence.

## Purpose

QLLM Lab should become a local research cockpit for quantum machine learning
experiments, not just a collection of run pages. The upgraded UI should help a
researcher decide what to run, understand the model being tested, control queued
and active experiments, compare quantum and classical variants, and judge whether
there is credible evidence of quantum advantage.

The current app already has the right foundation:

- React dashboard routes for launch, jobs, datasets, suites, live telemetry,
  GPU status, run details, and job workspaces.
- FastAPI endpoints for presets, datasets, jobs, workspace payloads, live runs,
  run detail, and linked comparison payloads.
- A local SQLite-backed results store and single-worker experiment queue.
- Presets with classical twins for several quantum candidates.
- Model config fields for transformer, recurrent, two-stream, classical,
  quantum-attention, quantum-FFN, quantum recurrent, and quantum diagnostics.

The main missing layer is workflow coherence: the UI should be organized around
research decisions instead of separate technical pages.

## External Product Patterns To Borrow

- W&B-style experiment tracking: searchable runs, config/metric tables,
  comparison dashboards, grouped studies, and sweep-like workflows.
- TensorBoard-style model/metric inspection: curves, graph-level model
  structure, distributions, embeddings, and profiling views.
- Netron-style architecture browsing: clear model nodes, edges, shapes, and
  layer metadata.
- IBM Quantum Composer-style circuit editing: visual quantum operations,
  register/wire layout, execution target awareness, and resource feedback.
- PennyLane-style QML framing: demos organized by algorithms, resource
  estimation, quantum machine learning tasks, and hardware/simulator details.

This project should not copy any one of those tools. It should combine their
best ideas into a local, QML-specific research interface.

## North Star Workflow

The UI should make these questions easy to answer:

1. What research question am I testing?
2. What model architecture is being run?
3. Which layers/components are classical or quantum?
4. What jobs are queued, running, failed, cancelled, or finished?
5. What is the right classical baseline for this candidate?
6. Are the runs matched by dataset, seed, steps, parameter count, and budget?
7. Does the result indicate quantum advantage, or only a tradeoff?
8. What should I run next?

## Proposed Information Architecture

Replace the current nav labels with workflow-oriented sections:

- **Lab Overview**
  - Active/running jobs.
  - Queue health.
  - GPU/JAX backend status.
  - Recent comparisons.
  - Best current candidate per task.
  - Failed runs requiring attention.

- **Experiments**
  - Queue new experiment.
  - Inspect queued/running/done/failed runs.
  - Cancel, duplicate, rerun, and compare selected jobs.
  - Filter by status, dataset, preset, model family, seed, device, and group.

- **Model Builder**
  - Visual model breakdown.
  - Preset browser.
  - Layer/component editor.
  - Quantum circuit/resource controls.
  - Save model spec as preset.
  - Run directly from the edited spec.

- **Comparisons**
  - Quantum vs classical paired comparisons.
  - Parameter-matched baselines.
  - Frozen/random quantum controls.
  - Multi-seed evidence summaries.
  - Delta tables and comparison curves.

- **Datasets & Tasks**
  - Text corpora.
  - Imported Hugging Face datasets.
  - Synthetic quantum sequence tasks.
  - Contextuality/parity tasks.
  - Interference/memory tasks.
  - Task cards explaining what kind of advantage each task can test.

- **Results**
  - Leaderboards.
  - Study dashboards.
  - Statistical summaries across seeds.
  - Exportable research reports.
  - Artifact links.

- **System**
  - GPU/JAX status.
  - Backend availability.
  - Dependency checks.
  - Local storage paths.

## UX Principles

- The first screen should be operational: show what is running and what needs
  attention.
- A run should always show its matched baseline status if one exists.
- A quantum claim should always be shown beside cost: parameter count, wall time,
  simulator backend, qubits, depth, and shots.
- Model controls should start constrained and safe. Freeform editing comes only
  after the underlying config model supports it.
- Every comparison should show whether the protocol is fair: same dataset, seed,
  steps, eval cadence, and comparable parameter budget.
- Results should separate "better on this run" from "evidence across a study."

## Phase 1: Research Cockpit Refresh

Goal: make the existing dashboard navigable without changing model internals.

Frontend changes:

- Rename and reorganize routes:
  - `/` -> Lab Overview.
  - `/experiments` -> improved Jobs/Live hybrid.
  - `/launch` remains available as New Experiment.
  - `/jobs/:id` remains Run Workspace.
  - `/comparisons/:id` becomes a dedicated comparison page.
  - `/models` becomes a preset/model browser.
  - `/results` becomes leaderboards/studies.
- Add status tabs in Experiments: all, queued, running, done, failed,
  cancelled.
- Add run filters: dataset, preset, kind, device target, comparison role,
  group id.
- Add row actions: open, cancel, duplicate, rerun, compare.
- Add a compact "currently running" strip to Lab Overview.
- Improve empty states so a new user knows the next action.

Backend changes:

- Add aggregate endpoint:
  - `GET /api/lab/overview`
  - returns job counts, active jobs, recent failed jobs, recent comparisons,
    GPU status, and latest leaderboard highlights.
- Extend job payloads with derived fields:
  - `kind`
  - `uses_quantum`
  - `model_family`
  - `comparison_state`
  - `elapsed_or_wall_seconds`

Acceptance criteria:

- A user can tell what is running from the first screen.
- A user can filter jobs without scanning every run manually.
- A user can open a run, cancel it, rerun it, or start a comparison from the
  Experiments page.

## Phase 2: Read-Only Model Diagram

Goal: every preset/run gets a visual architecture breakdown.

Backend changes:

- Add model graph builder:
  - `qllm/dashboard/model_graph.py`
  - `model_graph_from_config(cfg) -> dict`
- Add endpoints:
  - `GET /api/presets/{preset_id}/model-graph`
  - `GET /api/jobs/{job_id}/model-graph`

Suggested graph payload:

```json
{
  "nodes": [
    {"id": "tokens", "label": "Tokens", "kind": "input"},
    {"id": "embed", "label": "Classical Embedding", "kind": "classical"},
    {"id": "block_0_attn", "label": "Block 1 Attention", "kind": "classical"},
    {"id": "block_0_ffn", "label": "Block 1 Quantum FFN", "kind": "quantum"},
    {"id": "head", "label": "LM Head", "kind": "classical"}
  ],
  "edges": [
    ["tokens", "embed"],
    ["embed", "block_0_attn"],
    ["block_0_attn", "block_0_ffn"],
    ["block_0_ffn", "head"]
  ],
  "quantum": {
    "n_qubits": 4,
    "n_circuit_layers": 2,
    "ansatz": "reuploading",
    "backend": "pennylane",
    "device": "default.qubit",
    "shots": null,
    "readout": "z"
  }
}
```

Frontend changes:

- Add `ModelDiagram.jsx`.
- Show node colors by component kind:
  - classical
  - quantum
  - hybrid
  - frozen/control
  - input/output
- Add hover/side-panel metadata:
  - dimensions
  - heads
  - qubits
  - circuit depth
  - backend
  - trainable/frozen
  - expected cost band
- Embed the diagram in:
  - Model Builder.
  - Run Workspace.
  - Comparison View.

Acceptance criteria:

- A user can visually see which parts of a run are quantum.
- A user can compare a candidate and baseline architecture side by side.
- The diagram is generated from config, not handwritten preset text.

## Phase 3: Dedicated Quantum vs Classical Comparison

Goal: comparison becomes a research object, not a section buried in a run page.

Comparison page sections:

- Protocol card:
  - candidate run
  - baseline run
  - dataset/task
  - seed
  - steps
  - eval interval
  - device/backend
  - matched or unmatched flags
- Architecture side-by-side:
  - candidate model diagram
  - baseline model diagram
- Metric deltas:
  - validation loss
  - validation perplexity
  - bpc
  - wall time
  - parameter count
  - gradient norm ratio
- Curves:
  - train loss
  - validation loss
  - validation perplexity
  - gradient diagnostics
- Verdict panel:
  - "incomplete"
  - "candidate better on this run"
  - "baseline better on this run"
  - "insufficient fairness"
  - "needs multi-seed study"

Backend changes:

- Extend `comparison_payload` with fairness flags:
  - same dataset
  - same seed
  - same steps
  - same eval interval
  - same device target
  - parameter delta ratio
  - quantum/classical role validation
- Add suggested verdict generation from available data.

Acceptance criteria:

- A user can pick or open quantum vs classical comparisons directly.
- The UI clearly separates metric deltas from scientific evidence.
- The page shows when a comparison is unfair or incomplete.

## Phase 4: Studies, Sweeps, And Multi-Seed Evidence

Goal: support actual research claims, not one-off run anecdotes.

Add a study concept:

- A study groups jobs by research question.
- A study defines:
  - task/dataset
  - candidate model spec
  - baseline model spec
  - seeds
  - training budget
  - metrics of interest
  - ablations

Study examples:

- `quantum-ffn-vs-classical-text-smoke`
- `qrnn-vs-gru-sequence-memory`
- `quantum-attention-contextual-parity`
- `two-stream-quantum-bias-ablation`

Add sweep support:

- grid over qubits
- grid over circuit depth
- random/Bayesian-ready hyperparameter schema later
- seed repeats
- batch queue creation

Study result views:

- leaderboard by mean metric
- confidence intervals or mean/std across seeds
- per-seed scatter plot
- performance vs wall-time plot
- performance vs parameter count plot
- scaling over qubits/depth
- ablation matrix

Acceptance criteria:

- A user can create a multi-run study from the UI.
- A result can be summarized across seeds.
- The UI can say whether a result is promising, inconclusive, or negative under
  the study protocol.

## Phase 5: Editable Per-Layer Model Specs

Goal: support user edits such as "replace classical attention with quantum
attention in layers 1-5 only."

Current limitation:

- `ModelConfig` has global `attn_type` and `ffn_type` fields.
- `TransformerBlock` receives the same config for every block.
- This cannot represent layer-specific component swaps.

Proposed config extension:

```python
@dataclass(frozen=True)
class BlockConfig:
    attn_type: str = "classical"
    ffn_type: str = "classical"
    quantum: QuantumConfig | None = None

@dataclass(frozen=True)
class ModelConfig:
    ...
    blocks: tuple[BlockConfig, ...] | None = None
```

Rules:

- If `blocks` is `None`, preserve current behavior using global `attn_type`,
  `ffn_type`, and `n_blocks`.
- If `blocks` is present, `n_blocks == len(blocks)`.
- A block-level quantum config overrides the model-level quantum config.
- Presets can be migrated gradually.

Model assembly changes:

- `QLLM.__call__` should pass each block its own component config.
- `uses_quantum` should inspect block specs.
- `to_flat_dict` must serialize block configs predictably.
- Preset metadata should expose layer summaries.

Validation:

- Reject unknown component names.
- Reject quantum components without quantum config.
- Warn when a quantum-heavy layer selection is likely too slow.

Acceptance criteria:

- Config can represent quantum attention in layers 1-5 and classical attention
  elsewhere.
- Existing presets still work.
- Tests cover global and per-layer configs.

## Phase 6: Visual Model Builder

Goal: turn the read-only diagram into a controlled editor.

Editing modes:

- Select layer range.
- Swap attention type.
- Swap FFN type.
- Toggle quantum trainable/frozen.
- Edit qubits/depth/ansatz/readout/backend/shots.
- Save as model spec.
- Queue experiment.
- Queue matched comparison.

UI flow:

1. Start from preset.
2. Inspect architecture.
3. Select components/layer range.
4. Apply swap.
5. Review resource/fairness warnings.
6. Name the model.
7. Save and run.

Backend changes:

- Add persisted custom model specs.
- Add endpoints:
  - `GET /api/model-specs`
  - `POST /api/model-specs`
  - `GET /api/model-specs/{id}`
  - `POST /api/model-specs/{id}/jobs`
- Add validation endpoint:
  - `POST /api/model-specs/validate`

Acceptance criteria:

- A user can create a custom hybrid model from the UI.
- The generated config is inspectable and reproducible.
- The model can be run without hand-editing YAML.

## Phase 7: Quantum Advantage Reporting

Goal: help the user make careful claims.

The UI should avoid declaring "quantum advantage" from a single run. It should
present an evidence ladder:

- **Run-level improvement**: candidate beats baseline in one matched run.
- **Repeated improvement**: candidate beats baseline across multiple seeds.
- **Parameter-matched improvement**: candidate beats a classical model with
  comparable trainable parameter count.
- **Ablation-supported improvement**: trainable quantum beats frozen/random
  quantum and classical controls.
- **Task-specific advantage**: improvement appears on a task where quantum
  structure is expected to matter.
- **Cost-aware advantage**: improvement justifies wall-time/simulation/resource
  overhead for the research question.

Suggested verdict labels:

- `Incomplete`: not enough runs or comparison missing.
- `Unfair comparison`: mismatched dataset, seed, steps, or budget.
- `Negative`: baseline wins or deltas are not meaningful.
- `Promising run`: candidate wins one fair comparison.
- `Promising study`: candidate wins across seeds with acceptable variance.
- `Task-specific evidence`: candidate wins on a quantum-structured task with
  controls.
- `Quantum advantage candidate`: strong multi-seed, parameter-matched,
  ablation-supported evidence. This should be rare.

Report sections:

- Research question.
- Protocol.
- Candidate architecture.
- Baselines and controls.
- Dataset/task.
- Metrics.
- Curves and tables.
- Statistical summary.
- Cost/resource summary.
- Verdict and limitations.

## Results By Dataset vs Task

The UI should distinguish datasets from tasks:

- A dataset is the data source.
- A task is the scientific probe.

For normal text, the task might be "language modeling on corpus X." For QML
research, better tasks include:

- sequence memory
- contextual parity
- monitored quantum sequence prediction
- interference/cancellation
- two-stream semantic conditioning

Quantum advantage should be shown primarily at the task/study level, then linked
to the dataset used to instantiate the task.

Recommended result hierarchy:

```text
Task
  -> Dataset or generator settings
  -> Study
  -> Candidate/baseline groups
  -> Runs
  -> Curves/artifacts
```

This keeps the UI honest: a model may show advantage on a contextual task but not
on general text, and both findings are useful.

## Data Model Additions

Recommended new tables or equivalent JSON-backed records:

- `model_specs`
  - id
  - name
  - source preset
  - config json
  - graph json
  - created at

- `studies`
  - id
  - name
  - research question
  - task
  - dataset
  - protocol json
  - created at

- `study_jobs`
  - study id
  - job id
  - role: candidate, baseline, parameter_matched, frozen_quantum, ablation

- `comparisons`
  - id
  - candidate job id
  - baseline job id
  - fairness json
  - verdict

The existing lab job group id can remain useful for pairwise queueing, but
studies need a more explicit research-level grouping.

## Implementation Order

Recommended first implementation sequence:

1. Add `docs/UI_UPGRADE_PLAN.md`.
2. Add Lab Overview endpoint and page.
3. Replace sidebar labels and improve route organization.
4. Add job filters/actions in Experiments.
5. Add read-only model graph backend and `ModelDiagram`.
6. Add dedicated comparison page with fairness flags.
7. Add study grouping and multi-seed result summaries.
8. Add model spec persistence.
9. Add per-layer block config support.
10. Add visual model editor.
11. Add quantum advantage reports.

## Near-Term Cutline

The best first production-quality milestone is:

- Lab Overview.
- Experiments page with filters and actions.
- Read-only model diagram for presets and jobs.
- Dedicated comparison page with fairness/protocol cards.

This milestone makes the app much easier to navigate and directly supports
quantum-vs-classical research without forcing the riskier per-layer model config
migration immediately.
