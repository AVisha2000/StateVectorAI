# QLLM Loop Run Log

Append one entry per run. Prune entries older than 30 days when the log becomes noisy.

## Format

```json
{
  "run_id": "2026-07-04T12:54:25+01:00",
  "pattern": "qllm-daily-triage",
  "duration_s": 0,
  "items_found": 0,
  "actions_taken": 0,
  "escalations": 0,
  "tokens_estimate": 0,
  "outcome": "no-op | report-only | fix-proposed | escalated"
}
```

## Recent Runs

```json
{
  "run_id": "2026-07-04T12:54:25+01:00",
  "pattern": "qllm-daily-triage",
  "duration_s": 0,
  "items_found": 4,
  "actions_taken": 1,
  "escalations": 0,
  "tokens_estimate": 0,
  "outcome": "report-only",
  "notes": "Initialized loop engineering scaffold and project-specific QLLM skills."
}
```

```json
{
  "run_id": "2026-07-04T13:41:32+01:00",
  "pattern": "enhancement-plan-boundary-sampling-triage",
  "duration_s": 0,
  "items_found": 5,
  "actions_taken": 1,
  "escalations": 0,
  "tokens_estimate": 9000,
  "outcome": "report-only",
  "notes": "Inspected the first critical enhancement item. Generated datasets, splitting, and sampling currently operate on flattened streams without sequence boundary metadata; the next L2 fix should add a boundary-aware dataset bundle while preserving flat compatibility."
}
```

```json
{
  "run_id": "2026-07-04T13:43:19+01:00",
  "pattern": "qllm-daily-triage-budget-guard",
  "duration_s": 0,
  "items_found": 1,
  "actions_taken": 1,
  "escalations": 0,
  "tokens_estimate": 1500,
  "outcome": "no-op",
  "notes": "Daily loop run cap already reached for 2026-07-04; no further triage or source changes were performed."
}
```

```json
{
  "run_id": "2026-07-04T13:48:53+01:00",
  "pattern": "qllm-daily-triage",
  "duration_s": 0,
  "items_found": 5,
  "actions_taken": 2,
  "escalations": 0,
  "tokens_estimate": 6500,
  "outcome": "report-only",
  "notes": "After removing the run-count cap, inspected loop gates, worktree, GPU_QUEUE.md, RESULTS.md, docs/ENHANCEMENT_PLAN.md, boundary-sampling call sites, and dashboard DB state. Next recommended L2 fix is boundary-safe synthetic dataset metadata, splitting, sampling, and tests before long GPU runs."
}
```

```json
{
  "run_id": "2026-07-04T13:51:46+01:00",
  "pattern": "enhancement-plan-config-validation-triage",
  "duration_s": 0,
  "items_found": 6,
  "actions_taken": 2,
  "escalations": 0,
  "tokens_estimate": 5500,
  "outcome": "report-only",
  "notes": "Inspected the comprehensive config-validation item. Existing validate_config covers block attention/FFN names and block count only; .venv probes showed checked-in configs pass but invalid data/model/train/quantum settings return no validator errors. Dashboard model specs call validate_config, while CLI training and queue execution need shared pre-run validation."
}
```

```json
{
  "run_id": "2026-07-04T13:55:41+01:00",
  "pattern": "enhancement-plan-two-stream-metric-triage",
  "duration_s": 0,
  "items_found": 5,
  "actions_taken": 2,
  "escalations": 0,
  "tokens_estimate": 6500,
  "outcome": "report-only",
  "notes": "Inspected two-stream causality and metric labeling. The sentence encoder pools the full token window for quantum/classical encoders; tests only prove causality when encoder_kind='none'. Dashboard comparison/results/suite views and docs still present generic validation perplexity without a side-information label, while RESULTS v0.16 is statistically cautious but not side-info explicit."
}
```

```json
{
  "run_id": "2026-07-04T13:57:45+01:00",
  "pattern": "enhancement-plan-kernel-split-hygiene-triage",
  "duration_s": 0,
  "items_found": 4,
  "actions_taken": 2,
  "escalations": 0,
  "tokens_estimate": 5200,
  "outcome": "report-only",
  "notes": "Inspected kernel advantage split hygiene under the research protocol. kernel_ridge_r2 selects ridge regularization by maximizing test R2; advantage_experiment uses a train/test split only; advantage_probe reports test R2 without selected regularization or split sizes. Recommended later L2 fix: deterministic train/validation/test or nested-CV selection plus metadata while preserving positive/negative controls."
}
```

```json
{
  "run_id": "2026-07-04T14:01:40+01:00",
  "pattern": "enhancement-plan-study-paired-stats-triage",
  "duration_s": 185,
  "items_found": 5,
  "actions_taken": 2,
  "escalations": 0,
  "tokens_estimate": 5400,
  "outcome": "report-only",
  "notes": "Inspected study-level evidence hardening under the research protocol. paired_stats/classify_claim already provide CI, p-value, effect size, win rate, and cautious claim levels, but dashboard study evidence currently uses fair pair counts, wins, mean delta, and std delta only; Study and Study Report UI do not surface paired statistical significance. Recommended later L2 fix: compute paired stats over fair completed study pairs, feed classify_claim, surface stats in study/report payloads and markdown, and add dashboard tests for weak positive/non-significant effects."
}
```

```json
{
  "run_id": "2026-07-04T14:33:21+01:00",
  "pattern": "enhancement-plan-full-fairness-fields-triage",
  "duration_s": 420,
  "items_found": 5,
  "actions_taken": 2,
  "escalations": 0,
  "tokens_estimate": 6700,
  "outcome": "report-only",
  "notes": "Inspected dashboard/research fairness gates for enhancement item 6. Current lab fairness checks cover dataset, seed, steps, eval cadence, device, roles, train batch/seq/lr/weight decay/grad clip, data kind, corpus path, and val fraction, but omit synthetic generator/task fields. A temp ResultsDB probe with only data.gen_seed mismatched still returned same_preprocessing=true and verdict single-run candidate better; data.gen_seed was absent from matched_config_fields. Recommended later L2 fix: compare all non-allowlisted flattened config fields per comparison family, include generator/contextual/cancellation fields, expose mismatched keys in API/UI, and add tests that synthetic generator mismatch fails fairness while intended model-only differences pass."
}
```

```json
{
  "run_id": "2026-07-04T14:37:14+01:00",
  "pattern": "enhancement-plan-parameter-matched-analogues-triage",
  "duration_s": 260,
  "items_found": 5,
  "actions_taken": 2,
  "escalations": 0,
  "tokens_estimate": 6200,
  "outcome": "report-only",
  "notes": "Inspected parameter-matched analogue generation under dashboard/model/research protocols. Dashboard analogue builders currently use curated twins or component swaps; qllm.models.model.matched_classical_d_ff exists but is only used by benchmarks/ablation.py. Parameter-count probes at vocab 32 showed quantum-ffn-4q -> classical-small ratio about -0.578 while matched_d_ff=4, quantum-attn-4q -> classical-small ratio about -0.063, two-stream quantum/classical bias ratio about -0.002, and qrnn-small -> gru-small is far from matched for valid recurrent vocabularies. Recommended later L2 fix: add per-family analogue builders/ladders, record analogue type and parameter delta ratio before marking paired-ready, and keep parameter-matched evidence rungs gated by tolerance."
}
```

```json
{
  "run_id": "2026-07-04T14:43:32+01:00",
  "pattern": "enhancement-plan-result-claim-ledger-triage",
  "duration_s": 220,
  "items_found": 5,
  "actions_taken": 2,
  "escalations": 0,
  "tokens_estimate": 5600,
  "outcome": "report-only",
  "notes": "Inspected result-claim-ledger surfaces under dashboard and research-protocol rules. No results/claims.json or docs/CLAIMS.md exists; RESULTS.md stores claim status in prose; result_dashboard_payload creates highlighted summary cards from live result rows, where offline runs have claim_level=null and cards only surface verdict_label. Linked lab comparison rows do use classify_claim for verdict/claim_level, but there is no durable ledger tying highlighted results to supporting runs, limitations, next decisive checks, and current status. Recommended later L2 fix: add a small ledger schema, mirror it into result summaries, and test that highlighted cards cannot imply advantage without protocol claim fields."
}
```

```json
{
  "run_id": "2026-07-04T14:46:33+01:00",
  "pattern": "enhancement-plan-durable-dashboard-queue-triage",
  "duration_s": 180,
  "items_found": 5,
  "actions_taken": 2,
  "escalations": 0,
  "tokens_estimate": 5200,
  "outcome": "report-only",
  "notes": "Inspected durable dashboard queue surfaces. ExperimentQueue creates a process-local queue.Queue and only enqueues IDs during submit or analogue creation; it does not reload queued lab_jobs at startup. lab_jobs persists status/run_key/config but has no worker id, heartbeat, claim timestamp, or stale recovery state. The current dashboard DB has no queued jobs but does have job 42 and matching live_runs row marked running since 2026-06-21T18:36 at step 0, demonstrating missing stale-running recovery. Recommended later L2 fix: DB-backed claim/heartbeat/recovery, startup rehydration of queued jobs, and tests for restart resume plus stale running handling."
}
```

```json
{
  "run_id": "2026-07-04T14:48:58+01:00",
  "pattern": "enhancement-plan-dashboard-local-safety-triage",
  "duration_s": 160,
  "items_found": 5,
  "actions_taken": 2,
  "escalations": 0,
  "tokens_estimate": 5000,
  "outcome": "report-only",
  "notes": "Inspected dashboard local-safety guardrails. run.py defaults to 127.0.0.1 but accepts arbitrary --host without an explicit allow-remote gate. server.py configures wildcard CORS, /api/plot/{name} builds RESULTS_DIR/name without a resolved containment check or 404/400 semantics, and the SPA fallback serves FRONTEND_DIST/full_path if it is a file without checking that the resolved path remains under the frontend dist directory. status.py reports environment/GPU/frontend state but not bind-safety warnings. No tests currently cover CORS restrictions, path traversal rejection, or non-loopback binding warnings. Recommended later L2 fix: localhost-only CORS by default, explicit remote opt-in, safe path resolver for plot/static serving, status warnings, and focused dashboard tests."
}
```

```json
{
  "run_id": "2026-07-04T14:53:04+01:00",
  "pattern": "enhancement-plan-checkpoint-resume-triage",
  "duration_s": 170,
  "items_found": 5,
  "actions_taken": 2,
  "escalations": 0,
  "tokens_estimate": 5200,
  "outcome": "report-only",
  "notes": "Inspected checkpoint/resume surfaces. fit creates TrainState and optimizer state in memory but only writes final params.msgpack and summary.json after the loop; no latest/best checkpoint, optimizer state, current step, RNG state, or resume_from path exists. scripts/train.py exposes config/steps/run-name/tracking/sample but no resume flag. Dashboard runner calls fit with should_cancel and marks jobs done/cancelled, while ResultsDB stores lab job status, live_runs progress, and steps but no checkpoint path or resumed-step metadata. Benchmark scripts skip completed result cells via ResultsDB.exists, which is whole-cell resumability rather than mid-run recovery. Recommended later L2 fix: checkpoint latest and best validation state, restore params/optimizer/RNG/step, expose resume through CLI/dashboard/benchmarks, and record resume metadata in summary JSON plus ResultsDB."
}
```

```json
{
  "run_id": "2026-07-04T14:56:26+01:00",
  "pattern": "enhancement-plan-idempotent-step-logging-triage",
  "duration_s": 150,
  "items_found": 5,
  "actions_taken": 2,
  "escalations": 0,
  "tokens_estimate": 4800,
  "outcome": "report-only",
  "notes": "Inspected idempotent per-step logging. The steps table has only an autoincrement id and no uniqueness constraint on run_key, step, and name; ResultsDB.log_step uses plain INSERT INTO steps. A small temp-DB probe logged train_loss twice for the same run_key and step and fetch_steps returned two rows, while live_runs current_step/last_train_loss reflected only the latest scalar. Dashboard curve readers group all fetch_steps rows and order only by step, so repeated metric points can render or be counted silently. Existing tests cover runs/metrics replacement but not steps idempotence. Recommended later L2 fix: choose replace-in-place, delete-old-curve, or new-attempt-key policy, enforce it in schema/write paths, and add focused dashboard tests for duplicate logging plus monotonic curves."
}
```

```json
{
  "run_id": "2026-07-04T14:59:29+01:00",
  "pattern": "enhancement-plan-generation-contract-triage",
  "duration_s": 175,
  "items_found": 5,
  "actions_taken": 2,
  "escalations": 0,
  "tokens_estimate": 5200,
  "outcome": "report-only",
  "notes": "Inspected generation contract across architectures. qllm.train.loop.generate assumes model.cfg.max_seq_len; transformer and two_stream models carry cfg and passed a tiny initialized-params generation probe, but GRULM and QRNNLM do not expose cfg and fail with AttributeError before sampling. Dashboard model_test_payload marks prompt generation ready for completed text jobs with config and params.msgpack, without checking architecture support, and run_model_test calls generate directly. Existing tests cover transformer generation only; recurrent/two_stream tests cover shape/dispatch and dashboard tests cover artifact capability reporting but not generation execution across architectures. Recommended later L2 fix: pass context_len or explicit model metadata into generate, gate dashboard support by architecture with clear unsupported messages, and add generation smoke tests for transformer, GRU, QRNN, and two-stream where supported."
}
```

```json
{
  "run_id": "2026-07-04T15:05:48+01:00",
  "pattern": "enhancement-plan-dashboard-comparison-warnings-triage",
  "duration_s": 190,
  "items_found": 5,
  "actions_taken": 2,
  "escalations": 0,
  "tokens_estimate": 5600,
  "outcome": "report-only",
  "notes": "Inspected docs/ENHANCEMENT_PLAN.md item 14 plus dashboard comparison, evidence, launch, studies, and dashboard tests under dashboard/research-protocol rules. comparison_research_payload returns fairness, verdict, resource_normalized, and evidence_ladder but no structured warnings or caveats payload; comparison_evidence_ladder can label a single fair anecdote as promising run while multi-seed and ablation/control rungs remain false. The Comparison page renders verdict badges, fairness flags, ladder, and deltas but no first-class warning panel; Launch warns only for candidate_only when an analogue exists, while Studies already warns when a candidate has no analogue. Recommended later L2 fix: add a backend warning builder for one-seed, parameter mismatch, missing controls, two-stream side-info, slow negligible gains, and synthetic config mismatch; expose warnings in API payloads and render them with focused dashboard tests."
}
```

```json
{
  "run_id": "2026-07-04T15:09:07+01:00",
  "pattern": "enhancement-plan-dataset-import-boundaries-triage",
  "duration_s": 180,
  "items_found": 5,
  "actions_taken": 2,
  "escalations": 0,
  "tokens_estimate": 5200,
  "outcome": "report-only",
  "notes": "Inspected docs/ENHANCEMENT_PLAN.md item 15 plus qllm/dashboard/datasets.py, qllm/resultsdb.py lab_datasets schema, Datasets.jsx, server import route, and existing dashboard tests. The importer records source, split, text_column, corpus_path, n_rows, n_chars, preview, and a database ts timestamp, and rejects row_limit outside 1..200000. Missing pieces: source revision, requested row_limit, corpus hash, byte/char cap, truncation state or warnings, and UI display of timestamp/hash/revision/row-limit provenance. A mocked import with three source rows and row_limit=2 returned/stored only n_rows=2 with no truncation flag or warning. Recommended later L2 fix: migrate lab_datasets provenance fields, compute a corpus hash, record row/size cap decisions plus truncation warnings, surface them in the Datasets page, and test reproducible metadata on truncated imports."
}
```

```json
{
  "run_id": "2026-07-04T15:14:21+01:00",
  "pattern": "enhancement-plan-quantum-runtime-telemetry-triage",
  "duration_s": 190,
  "items_found": 5,
  "actions_taken": 2,
  "escalations": 0,
  "tokens_estimate": 5600,
  "outcome": "report-only",
  "notes": "Inspected docs/ENHANCEMENT_PLAN.md item 16 plus qllm/train/loop.py, qllm/research_protocol.py, qllm/dashboard/resources.py, qllm/resultsdb.py, dashboard result/GPU pages, saved summaries, and current DB schema. Existing coverage: quantum_resource_estimate computes queue-time state_dim, token calls, component multiplier, score, and band; resource_ledger_from_config exposes static ledger fields and is tested; dashboard result rows display wall time, qubits, depth, backend, and resource band. Missing pieces: fit only prints first-step JIT time and persists total wall_seconds; summary.json and runs table do not record first-step compile time, steady-state step time, peak memory, or durable resource ledger fields; Research Results sorts by validation perplexity rather than wall/resource cost. Recommended later L2 fix: measure compile and steady-step timings, optionally capture peak memory with None fallback, persist telemetry in summaries/DB/resource metadata, expose sortable dashboard resource-cost columns, and add focused tests."
}
```
