# QLLM Enhancement Plan

This plan turns the technical review into staged engineering and research
work. The guiding principle is simple: keep QLLM a fast research testbed, but
make every result reproducible, comparable, and difficult to overclaim.

## Objectives

- Preserve the shared `tokens -> logits` training contract across classical,
  quantum, recurrent, contextual, and two-stream models.
- Make every dataset/model/dashboard comparison fair by construction.
- Prevent experiment artifacts from introducing synthetic transitions,
  hidden side information, or incomplete controls.
- Improve local dashboard reliability without turning the project into a
  public SaaS application.
- Raise the repo from promising research code to production-quality research
  infrastructure.

## Priority Overview

| Priority | Theme | Why it matters |
| --- | --- | --- |
| Critical | Boundary-safe synthetic data | Prevents fake cross-trajectory transitions from biasing memory experiments. |
| Critical | Strict config validation | Invalid model/data/quantum combinations should fail before expensive runs. |
| Critical | Two-stream metric honesty | Full-window side information must not be ranked as strict autoregressive LM evidence. |
| High | Study-level paired statistics | Multi-seed dashboard claims need confidence intervals and p-values. |
| High | Durable dashboard queue | Queued jobs should survive dashboard restarts. |
| High | Dashboard local safety | Local file serving and open CORS need localhost-only guardrails. |
| Medium | Checkpoint/resume and DB idempotence | Long GPU runs need recovery and clean curves. |
| Medium | Dependency matrix | JAX/PennyLane/Flax/GPU install paths need tested, explicit locks. |
| Long term | Registries and scalable backends | Cleaner extension points and larger-qubit simulation paths. |

## Phase 0: Planning Baseline

Goal: make the current state explicit before code changes.

Tasks:

- Capture this file as the canonical engineering roadmap.
- Add links from `README.md`, `GPU_QUEUE.md`, or `LOOP.md` once execution starts.
- Keep the current cautious claim language from `RESULTS.md`; do not upgrade
  any research claims until the fixes below land.

Acceptance criteria:

- The plan is checked in.
- Each major work item has priority, files, tests, and acceptance criteria.
- No implementation behavior changes are included in this phase.

## Phase 1: Correctness Gates

These are the fixes to do before trusting additional experiment results.

### 1. Boundary-Safe Synthetic Dataset Sampling

Priority: Critical

Problem:

Synthetic generators flatten independent trajectories:

- `qllm/data/quantum_seq.py`
- `qllm/data/contextual.py`
- `qllm/data/interference_task.py`
- `qllm/data/seq_cancellation.py`

The generic sampler in `qllm/data/text.py` can draw windows that cross
trajectory boundaries. That creates transitions the generator never emitted.
For memory and contextuality tasks, this can corrupt the exact phenomenon
being measured.

Plan:

- Introduce a dataset return shape or metadata that preserves sequence
  boundaries.
- Add a boundary-aware batch sampler for generated datasets.
- Keep the text path unchanged unless the generic API needs a small adapter.
- Update `fit`, `evaluate`, `calibration`, and benchmark helpers to use the
  sampler associated with the loaded dataset.
- Ensure train/validation split happens along sequence boundaries for generated
  tasks.

Suggested files:

- `qllm/data/datasets.py`
- `qllm/data/text.py`
- `qllm/data/quantum_seq.py`
- `qllm/data/contextual.py`
- `qllm/data/interference_task.py`
- `qllm/data/seq_cancellation.py`
- `qllm/train/loop.py`
- `qllm/evaluation.py`
- `tests/test_config_data.py`
- `tests/test_quantum_data.py`
- `tests/test_contextual.py`
- `tests/test_seq_cancellation.py`

Acceptance criteria:

- No sampled batch crosses a generated sequence boundary.
- Train/validation splits do not cut a generated trajectory in half.
- Cached generated datasets include enough metadata to reconstruct boundaries.
- Tests explicitly construct boundary-adjacent sequences and prove sampler
  windows stay inside one trajectory.

### 2. Comprehensive Config Validation

Priority: Critical

Problem:

`qllm/config.py::validate_config` validates only a narrow slice of the actual
supported surface. `qllm/data/datasets.py::DATASET_KINDS` is also stale:
the dispatcher supports more kinds than the exported tuple reports.

Plan:

- Centralize model, data, backend, ansatz, readout, embedding, head, condition,
  and architecture registries.
- Validate numeric constraints: positive steps, batch size, sequence length,
  qubit count, circuit depth, context sizes, Markov order, measurement count,
  and `seq_len <= max_seq_len` where applicable.
- Validate semantic constraints:
  - QRNN vocab must be `2**k` and `k < n_qubits`.
  - Interference/mixture heads should only be used with supported arch paths.
  - Two-stream condition must be one of `film`, `token`, `bias`.
  - `ising` ansatz is currently QRNN-specific unless added to circuit backend
    registries.
  - Synthetic task parameters must be reproducible from `DataConfig`.
- Make CLI and dashboard queue call validation before starting jobs.

Suggested files:

- `qllm/config.py`
- `qllm/models/model.py`
- `qllm/data/datasets.py`
- `qllm/quantum/backends.py`
- `qllm/quantum/circuits.py`
- `qllm/dashboard/model_specs.py`
- `qllm/dashboard/runner.py`
- `scripts/train.py`
- `tests/test_config_data.py`
- `tests/test_dashboard_lab.py`

Acceptance criteria:

- Every config under `configs/` passes validation.
- Unknown or unsupported `data.kind` values report the full supported set.
- Invalid model/data/quantum combinations fail before model initialization.
- Dashboard model spec validation and CLI training share the same validator.

### 3. Two-Stream Causality and Metric Labeling

Priority: Critical

Problem:

`qllm/models/two_stream.py` mean-pools the full input window before predicting
each position. This is useful as a side-information probe, but it is not strict
autoregressive evaluation.

Plan:

- Decide between two valid paths:
  - Build causal prefix pooling so the sentence stream only sees left context.
  - Keep full-window pooling but mark the model family and metrics as
    teacher-forced side-information, not strict LM perplexity.
- Update dashboard comparison cards and `RESULTS.md` wording so two-stream
  results are never mixed with strict autoregressive leaderboard claims.
- Add tests that detect future-token leakage for all conditioning modes.

Suggested files:

- `qllm/models/two_stream.py`
- `qllm/dashboard/lab.py`
- `qllm/dashboard/explore.py`
- `qllm/dashboard/frontend/src/pages/Comparison.jsx`
- `qllm/dashboard/frontend/src/pages/ResearchResults.jsx`
- `benchmarks/two_stream_probe.py`
- `tests/test_two_stream.py`

Acceptance criteria:

- Either two-stream is causal, or dashboard/results label it as side-info.
- Two-stream comparisons only compare quantum vs classical encoders under the
  same conditioning and side-information policy.
- Tests fail if a future token can affect an earlier strict-LM logit.

### 4. Kernel Advantage Evaluation Split Hygiene

Priority: High

Problem:

`qllm/quantum/advantage.py::kernel_ridge_r2` chooses regularization on the
test split. This leaks evaluation information into model selection.

Plan:

- Replace train/test-only selection with train/validation/test or nested
  cross-validation.
- Preserve deterministic seeds for reproducibility.
- Update reports to distinguish validation-selected and final test scores.

Suggested files:

- `qllm/quantum/advantage.py`
- `benchmarks/advantage_probe.py`
- `benchmarks/data_screen.py`
- `tests/test_advantage.py`

Acceptance criteria:

- Test labels are used only once for final scoring.
- Existing advantage positive/negative controls still pass.
- Reports include selected regularization and split sizes.

## Phase 2: Research Evidence Hardening

These items make dashboard and benchmark conclusions statistically safer.

### 5. Study-Level Paired Statistics

Priority: High

Problem:

Dashboard study evidence currently summarizes wins and mean deltas, but does
not fully use `qllm/research_protocol.py::paired_stats`.

Plan:

- Pair candidate and baseline runs by dataset, seed, steps, eval cadence,
  device target, and sweep point.
- Compute paired improvements, mean/median, confidence interval, p-value,
  win rate, and effect size.
- Feed paired stats into `classify_claim`.
- Surface paired stats in study detail and report markdown.

Suggested files:

- `qllm/dashboard/studies.py`
- `qllm/dashboard/evidence.py`
- `qllm/dashboard/lab.py`
- `qllm/dashboard/frontend/src/pages/StudyReport.jsx`
- `tests/test_dashboard_lab.py`
- `tests/test_research_protocol.py`

Acceptance criteria:

- A study with three or more fair pairs reports paired stats.
- A positive mean with weak p-value/CI is labeled "positive but not
  significant."
- Single fair pairs remain "smoke evidence" or "anecdote."

### 6. Full Fairness Field Comparison

Priority: High

Problem:

Dashboard fairness checks compare high-level fields but miss many synthetic
data generator parameters and some training-budget details.

Plan:

- Define an intentional-difference allowlist per comparison family.
- Compare all relevant flattened config fields outside the allowlist.
- Include synthetic generator fields:
  - `data.gen_qubits`
  - `data.gen_measured`
  - `data.gen_theta_x`
  - `data.gen_theta_zz`
  - `data.gen_steps_per_token`
  - `data.gen_seed`
  - `data.markov_order`
  - contextual and cancellation fields
- Display mismatched fields in the dashboard comparison payload.

Suggested files:

- `qllm/dashboard/lab.py`
- `qllm/dashboard/analogues.py`
- `qllm/research_protocol.py`
- `qllm/dashboard/frontend/src/pages/Comparison.jsx`
- `tests/test_dashboard_lab.py`

Acceptance criteria:

- Candidate/baseline pairs with mismatched synthetic generator fields fail
  fairness.
- Intended model-only differences pass when all protocol fields match.
- Dashboard shows the exact mismatched config keys.

### 7. Parameter-Matched Analogue Generation

Priority: High

Problem:

Automatic analogues are component swaps, but not always parameter matched.
Research claims need parameter deltas recorded and bounded.

Plan:

- Add analogue builders for:
  - Quantum FFN -> matched classical FFN/low-rank FFN.
  - Quantum attention -> matched classical projection/attention width where
    possible.
  - QRNN/contextual -> GRU ladder rather than one arbitrary hidden size.
  - Two-stream quantum encoder -> classical encoder matched to output dim and
    parameter budget.
- Add dashboard warnings when a linked analogue is not parameter matched.

Suggested files:

- `qllm/dashboard/analogues.py`
- `qllm/models/model.py`
- `qllm/dashboard/presets.py`
- `benchmarks/*`
- `tests/test_ablation.py`
- `tests/test_dashboard_lab.py`

Acceptance criteria:

- Each quantum family reports analogue type, parameter delta ratio, and known
  limitations.
- Dashboard evidence cannot mark "parameter-matched" unless the delta is
  within the configured tolerance.

### 8. Result Claim Ledger

Priority: Medium

Problem:

`RESULTS.md` is careful, but claim status is embedded in prose. It should be
machine-checkable and dashboard-reusable.

Plan:

- Add a small claim ledger file such as `results/claims.json` or
  `docs/CLAIMS.md`.
- Track claim level, supporting runs, limitations, next decisive check, and
  current status.
- Let dashboard result pages read or mirror the same ladder.

Suggested files:

- `RESULTS.md`
- `qllm/dashboard/evidence.py`
- `qllm/dashboard/explore.py`
- `qllm/dashboard/frontend/src/pages/ResearchResults.jsx`

Acceptance criteria:

- Every highlighted result has a claim level from the research protocol.
- No dashboard card can imply "advantage" without the protocol fields.

## Phase 3: Training and Experiment Reliability

These items make longer GPU runs less fragile.

### 9. Durable Dashboard Queue

Priority: High

Problem:

`qllm/dashboard/runner.py::ExperimentQueue` stores pending job IDs in an
in-memory queue. `lab_jobs` persists rows, but a dashboard restart can leave
queued work stranded.

Plan:

- Add DB-backed job claiming:
  - `queued -> running` with worker ID and heartbeat.
  - recover stale `running` jobs after timeout.
  - resume queued jobs at process startup.
- Keep single-worker default, but make the state durable.
- Add safe cancellation semantics for queued and running jobs.

Suggested files:

- `qllm/dashboard/runner.py`
- `qllm/resultsdb.py`
- `qllm/dashboard/gpu_reservation.py`
- `tests/test_dashboard_lab.py`

Acceptance criteria:

- Restarting the dashboard resumes queued jobs.
- Stale running jobs are marked recoverable or failed with a clear error.
- GPU reservation state is released after done/error/cancelled.

### 10. Checkpointing and Resume

Priority: Medium

Problem:

`fit` writes final params and summary, but long runs need periodic checkpoints
and resumability.

Plan:

- Save latest and best-validation checkpoints.
- Persist optimizer state and step.
- Add `resume_from` to CLI, dashboard jobs, and benchmark scripts.
- Record checkpoint path and resumed step in summaries and DB.

Suggested files:

- `qllm/train/loop.py`
- `scripts/train.py`
- `qllm/dashboard/runner.py`
- `qllm/resultsdb.py`
- `tests/test_integration.py`
- `tests/test_dashboard_lab.py`

Acceptance criteria:

- Interrupted runs can resume without resetting optimizer state.
- Best-val checkpoint is distinguishable from final checkpoint.
- Summary JSON records whether the run was resumed.

### 11. Idempotent Per-Step Logging

Priority: Medium

Problem:

`steps` can accumulate duplicate rows if a run restarts with the same
`run_key`, step, and metric name.

Plan:

- Add a uniqueness constraint or delete/replace policy for
  `(run_key, step, name)`.
- Make reruns explicit: either same run replaces curves or a new run key is
  created.

Suggested files:

- `qllm/resultsdb.py`
- `qllm/train/loop.py`
- `qllm/dashboard/queries.py`
- `tests/test_dashboard_lab.py`

Acceptance criteria:

- Re-running a job cannot silently double-count curve points.
- Dashboard curves remain monotonic by step.

### 12. Generation Contract Across Architectures

Priority: Medium

Problem:

`qllm/train/loop.py::generate` assumes `model.cfg.max_seq_len`, which is not
true for every model architecture.

Plan:

- Make generation take `context_len` explicitly.
- Add architecture-neutral model metadata or helper.
- Add generation smoke tests for transformer, GRU, QRNN, and two-stream where
  supported.

Suggested files:

- `qllm/train/loop.py`
- `qllm/evaluation.py`
- `qllm/dashboard/model_tests.py`
- `tests/test_integration.py`
- `tests/test_recurrent.py`
- `tests/test_two_stream.py`

Acceptance criteria:

- Generation either works or fails with a clear unsupported-architecture
  message.
- Dashboard model tests do not crash on non-transformer jobs.

## Phase 4: Dashboard Safety and UX

These are local-tool hardening improvements.

### 13. Localhost Safety Guardrails

Priority: High

Problem:

The dashboard defaults to local host, but the API has open CORS and file
routes that should not be exposed casually.

Plan:

- Restrict CORS to localhost origins by default.
- Add a documented `--allow-remote` or env flag for wider access.
- Resolve file paths and reject traversal before serving plot/static files.
- Add warnings in `/api/status` when bound to non-loopback hosts.

Suggested files:

- `qllm/dashboard/server.py`
- `qllm/dashboard/run.py`
- `qllm/dashboard/status.py`
- `tests/test_dashboard_lab.py`

Acceptance criteria:

- Path traversal attempts return 404/400.
- CORS is not wildcard unless explicitly enabled.
- Remote binding is deliberate and visible.

### 14. Dashboard Comparison Warnings

Priority: Medium

Problem:

The dashboard has an evidence ladder, but the UI should make weak evidence
hard to miss at job-launch and result-reading time.

Plan:

- Show warnings for:
  - one seed only
  - unmatched parameter counts
  - missing frozen/random control
  - two-stream side-info metrics
  - candidate slower with negligible improvement
  - synthetic config mismatch
- Add launch-time warnings when a job has no planned analogue.

Suggested files:

- `qllm/dashboard/evidence.py`
- `qllm/dashboard/lab.py`
- `qllm/dashboard/frontend/src/pages/Launch.jsx`
- `qllm/dashboard/frontend/src/pages/Comparison.jsx`
- `qllm/dashboard/frontend/src/pages/Studies.jsx`

Acceptance criteria:

- A weak comparison cannot appear as a clean win without visible caveats.
- Evidence warnings are present in API payloads, not only UI text.

### 15. Dataset Import Boundaries

Priority: Medium

Problem:

Hugging Face import is useful, but imported corpora need provenance and safer
limits.

Plan:

- Store source dataset ID, split, revision if available, row limit, text
  column, and import timestamp.
- Hash imported corpora.
- Add row/size caps and clear warnings when truncating.

Suggested files:

- `qllm/dashboard/datasets.py`
- `qllm/resultsdb.py`
- `qllm/dashboard/frontend/src/pages/Datasets.jsx`
- `tests/test_dashboard_lab.py`

Acceptance criteria:

- Imported datasets are reproducible enough to identify.
- Dashboard displays corpus size, row count, and provenance.

## Phase 5: Performance and Scaling

These improvements reduce GPU pain and compile overhead.

### 16. Quantum Compile and Runtime Telemetry

Priority: Medium

Problem:

Quantum runs can fail slowly through compile time, state dimension, or nested
`vmap` cost.

Plan:

- Record first-step compile time, steady-state step time, state dimension,
  token calls per step, quantum component multiplier, and peak memory if
  available.
- Surface this in summaries, DB, and dashboard resource panels.

Suggested files:

- `qllm/train/loop.py`
- `qllm/research_protocol.py`
- `qllm/dashboard/resources.py`
- `qllm/dashboard/lab.py`
- `qllm/dashboard/frontend/src/pages/GPU.jsx`

Acceptance criteria:

- Every quantum run records resource ledger fields.
- Dashboard can rank runs by wall time and resource cost.

### 17. Scan/Vectorization Cleanup

Priority: Medium

Problem:

Some quantum paths still use Python-level loops over layers or qubits. Static
loops are acceptable for small configs, but deeper GPU experiments need better
compile behavior.

Plan:

- Identify loops in recurrent/contextual/transplant paths that can become
  `jax.lax.scan`.
- Keep shapes static and avoid tracing Python work inside hot paths.
- Add compile-time regression smoke tests for deeper circuits.

Suggested files:

- `qllm/quantum/recurrent.py`
- `qllm/quantum/contextual_cell.py`
- `qllm/quantum/transplant.py`
- `qllm/quantum/layers.py`
- `tests/test_recurrent.py`
- `tests/test_contextual_cell.py`

Acceptance criteria:

- Increasing depth does not cause avoidable Python-unrolled compile blowups.
- Existing shape/gradient tests still pass.

### 18. Scalable Backend Roadmap

Priority: Long term

Problem:

Statevector simulation is honest and exact, but memory scales exponentially.
The repo needs a clear path for larger memory experiments.

Plan:

- Define exact-vs-approximate backend roles.
- Add overlap-region parity tests between PennyLane statevector and
  TensorCircuit or MPS paths.
- Record backend and approximation caveats in result metadata.

Suggested files:

- `qllm/quantum/backends.py`
- `qllm/quantum/metrics.py`
- `benchmarks/memory_sweep.py`
- `tests/test_quantum.py`
- `tests/test_metrics.py`

Acceptance criteria:

- Approximate backend runs are never mixed with exact statevector runs without
  metadata.
- Small-qubit overlap tests pass within tolerance.

## Phase 6: Architecture Cleanup

These changes reduce future extension friction.

### 19. Component Registries

Priority: Medium

Problem:

`qllm/models/model.py` is a clean composition root, but it is becoming a large
switchboard.

Plan:

- Keep `build_model` as the composition root.
- Extract registries for attention, FFN, embedding, head, and architecture
  builders.
- Use registries for validation, dashboard model graphs, and config docs.

Suggested files:

- `qllm/models/model.py`
- `qllm/config.py`
- `qllm/dashboard/model_graph.py`
- `tests/test_classical_model.py`
- `tests/test_qnlp_suite.py`

Acceptance criteria:

- Adding a new model component requires one registry entry plus tests.
- Validation and dashboard supported-options lists cannot drift apart.

### 20. Dataset Object Contract

Priority: Medium

Problem:

`load_dataset` returns only `(ids, tokenizer)`, and contextual masks are exposed
through a module global. That is fragile.

Plan:

- Introduce a lightweight dataset bundle with:
  - token IDs
  - tokenizer
  - optional sequence shape/boundaries
  - optional masks
  - metadata/config hash
  - sampler function or sampler policy
- Preserve backwards compatibility through helper adapters where needed.

Suggested files:

- `qllm/data/datasets.py`
- `qllm/data/text.py`
- `qllm/evaluation_contextual.py`
- `benchmarks/contextual_sweep.py`
- `tests/test_contextual.py`

Acceptance criteria:

- No module global is needed to retrieve contextual masks.
- Batch sampling and evaluation can consume dataset metadata directly.

## Phase 7: Dependencies and Packaging

### 21. Tested Dependency Matrix

Priority: Medium

Problem:

`pyproject.toml` uses broad dependency ranges while `requirements.txt` pins
strict versions. GPU install instructions correctly warn about JAX wheels, but
the project should make the supported matrix explicit.

Plan:

- Define:
  - CPU development lock.
  - GPU WSL lock or install recipe.
  - optional dashboard/HF dependencies.
- Add a small environment sanity script that reports versions and rejects known
  bad combinations.
- Keep native Windows GPU limitations visible.

Suggested files:

- `pyproject.toml`
- `requirements.txt`
- `GPU_SETUP.md`
- `scripts/check_gpu.py`
- `README.md`

Acceptance criteria:

- A new contributor can install CPU dev dependencies from one command.
- A GPU user can verify JAX sees CUDA before queueing GPU jobs.
- Version metadata is included in run summaries.

## Phase 8: Documentation

### 22. Researcher Onboarding Guide

Priority: Medium

Problem:

The repo has strong docs, but a new researcher still has to infer the evidence
ladder from several files.

Plan:

- Add a "How to make a claim" section:
  - single pair = smoke
  - three or more pairs = paired analysis
  - controls required by model family
  - resource accounting required for quantum components
- Add diagrams for data/model/training/dashboard flow.
- Add a quick map from research question to benchmark script.

Suggested files:

- `README.md`
- `RESULTS.md`
- `DATA.md`
- `GPU_QUEUE.md`
- `qllm/dashboard/README.md`

Acceptance criteria:

- A new researcher can identify the right baseline/control for each quantum
  family.
- Dashboard study reports use the same vocabulary as the docs.

### 23. Engineer Onboarding Guide

Priority: Nice to have

Problem:

Extension points exist, but contributors need a concise guide to adding a new
component safely.

Plan:

- Document model, data, benchmark, dashboard, and test extension paths.
- Include examples:
  - new FFN variant
  - new synthetic task
  - new dashboard comparison card

Suggested files:

- `docs/DEVELOPMENT.md`
- `README.md`
- `.codex/skills/qllm-model-development/references/component-map.md`

Acceptance criteria:

- A contributor can add a new component without reading the whole repo first.
- Required tests are listed by change type.

## Recommended Execution Order

1. Boundary-safe synthetic sampling.
2. Comprehensive config validation.
3. Two-stream causality/metric labeling.
4. Dashboard fairness field expansion.
5. Study paired statistics.
6. Dashboard local safety guardrails.
7. Durable queue recovery.
8. Checkpoint/resume.
9. Dependency matrix and environment metadata.
10. Component registries and dataset bundle refactor.

## Proposed Milestones

### Milestone A: Trust the Data

Scope:

- Boundary-safe sampling.
- Dataset bundle contract.
- Config validation.
- Synthetic fairness fields.

Exit criteria:

- All synthetic benchmarks use boundary-aware windows.
- Invalid configs fail before training.
- Dashboard comparisons detect generator mismatches.

### Milestone B: Trust the Claims

Scope:

- Paired study statistics.
- Claim ledger.
- Parameter-matched analogues.
- Two-stream labeling or causal rewrite.

Exit criteria:

- Study reports include paired stats and protocol labels.
- No side-info run is mixed with strict autoregressive LM rankings.
- Dashboard evidence matches `RESULTS.md` wording.

### Milestone C: Trust the Runs

Scope:

- Durable queue.
- Checkpoint/resume.
- Idempotent step logging.
- GPU/resource telemetry.

Exit criteria:

- Dashboard restarts do not strand queued jobs.
- Long runs can resume.
- Resource cost is visible for every quantum result.

### Milestone D: Scale the Testbed

Scope:

- Component registries.
- Scan/vectorization cleanup.
- Scalable backend overlap tests.
- Dependency matrix.

Exit criteria:

- New components are registry-driven.
- Larger memory sweeps have explicit backend metadata.
- Install paths are reproducible on CPU and GPU setups.

## GitHub Issue Backlog

### Issue 1: Make synthetic dataset sampling boundary-safe

Priority: Critical

Description:

Generated datasets flatten independent trajectories, allowing random crops to
cross sequence boundaries. This can introduce fake transitions into memory
tasks.

Acceptance criteria:

- Generated datasets preserve boundary metadata.
- Samplers never cross boundaries.
- Train/validation splitting respects trajectory boundaries.
- Tests cover boundary-adjacent samples.

Suggested files:

- `qllm/data/datasets.py`
- `qllm/data/text.py`
- `qllm/train/loop.py`
- `tests/test_quantum_data.py`

### Issue 2: Expand config validation across all registries

Priority: Critical

Description:

Config validation should cover all model/data/quantum/dashboard-supported
options and fail before model initialization or job launch.

Acceptance criteria:

- Supported option registries are centralized.
- All checked-in YAML configs pass.
- Invalid configs return actionable errors.
- CLI and dashboard use the same validator.

Suggested files:

- `qllm/config.py`
- `qllm/models/model.py`
- `qllm/data/datasets.py`
- `qllm/dashboard/model_specs.py`

### Issue 3: Fix or label two-stream side-information metrics

Priority: Critical

Description:

Two-stream sentence encoders currently pool the full window. Either make them
causal or label them as side-information probes.

Acceptance criteria:

- Future-token leakage is removed or surfaced in all result views.
- Dashboard comparisons do not imply strict autoregressive wins.
- Causality tests cover all conditioning modes.

Suggested files:

- `qllm/models/two_stream.py`
- `benchmarks/two_stream_probe.py`
- `tests/test_two_stream.py`

### Issue 4: Use paired statistics in dashboard studies

Priority: High

Description:

Study verdicts should use paired improvements, confidence intervals, p-values,
and protocol claim levels.

Acceptance criteria:

- Study report includes paired stats for fair completed pairs.
- `classify_claim` decides study claim level.
- Weak positive effects are labeled cautiously.

Suggested files:

- `qllm/dashboard/studies.py`
- `qllm/dashboard/evidence.py`
- `tests/test_dashboard_lab.py`

### Issue 5: Compare full fairness fields for synthetic runs

Priority: High

Description:

Candidate/baseline fairness checks must include generator and preprocessing
fields, not just dataset name and training budget.

Acceptance criteria:

- Synthetic generator mismatch fails fairness.
- Mismatched fields are listed in the API.
- Intentional model differences are allowlisted.

Suggested files:

- `qllm/dashboard/lab.py`
- `qllm/research_protocol.py`
- `tests/test_dashboard_lab.py`

### Issue 6: Make the dashboard queue durable

Priority: High

Description:

Queued jobs are currently in process memory. Persist queue claiming and recover
jobs after dashboard restart.

Acceptance criteria:

- Queued jobs resume after restart.
- Stale running jobs are handled explicitly.
- GPU reservations are released on terminal states.

Suggested files:

- `qllm/dashboard/runner.py`
- `qllm/resultsdb.py`
- `tests/test_dashboard_lab.py`

### Issue 7: Harden dashboard local file serving and CORS

Priority: High

Description:

Restrict dashboard access assumptions to localhost by default and protect file
routes.

Acceptance criteria:

- No wildcard CORS unless explicitly enabled.
- Plot/static serving rejects path traversal.
- Non-loopback binding surfaces a warning.

Suggested files:

- `qllm/dashboard/server.py`
- `qllm/dashboard/run.py`
- `tests/test_dashboard_lab.py`

### Issue 8: Remove test-set model selection leakage in kernel probes

Priority: High

Description:

Kernel ridge regularization is selected on the test split. Add validation
selection.

Acceptance criteria:

- Test labels are used only for final score.
- Existing positive/negative controls still pass.
- Reports include selected regularization.

Suggested files:

- `qllm/quantum/advantage.py`
- `tests/test_advantage.py`

### Issue 9: Add checkpoint/resume support

Priority: Medium

Description:

Long GPU experiments should survive interruption and resume optimizer state.

Acceptance criteria:

- Latest and best checkpoints are saved.
- Resume restores step, params, optimizer state, and RNG path.
- Dashboard and CLI expose resume metadata.

Suggested files:

- `qllm/train/loop.py`
- `scripts/train.py`
- `qllm/dashboard/runner.py`

### Issue 10: Add tested dependency matrix and environment metadata

Priority: Medium

Description:

CPU/GPU dependency paths need one authoritative matrix and run-level version
metadata.

Acceptance criteria:

- CPU dev install is documented and tested.
- GPU setup records JAX/JAXLIB/CUDA visibility.
- Run summaries include key package versions and device backend.

Suggested files:

- `pyproject.toml`
- `requirements.txt`
- `GPU_SETUP.md`
- `scripts/check_gpu.py`
- `qllm/train/loop.py`

## Definition of Done

The enhancement plan is complete when:

- New experimental results cannot be produced with boundary-crossing synthetic
  batches.
- Dashboard comparisons fail closed when fairness fields mismatch.
- Multi-seed claims use paired statistics.
- Two-stream side-information is either removed or labeled everywhere.
- Long dashboard jobs can survive restart or resume from checkpoint.
- Dependency and device metadata are recorded with every serious run.

