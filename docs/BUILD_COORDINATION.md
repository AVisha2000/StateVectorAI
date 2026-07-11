# StateVector — Build Coordination Channel

Shared coordination doc for the parallel backend + frontend build. Both worktrees
inherit this file from `main`. It is how the two agents "talk": the **API
contract** below is the interface both build against, and the **logs** are how
each agent leaves messages the other can read.

Read this together with [UI_REDESIGN_PLAN.md](UI_REDESIGN_PLAN.md),
[FEATURE_UPGRADES.md](FEATURE_UPGRADES.md), and
[AGENT_OPERATING_MODEL.md](AGENT_OPERATING_MODEL.md).

## Topology & ownership

| Track | Client / triage | Branch | Worktree | Owns |
| --- | --- | --- | --- | --- |
| Backend | Codex (GPT-5.6 triage) | `backend-enhancements` → merges to `main` | `../StateVectorAI-backend` | `qllm/**` (except frontend), `scripts/**`, backend tests, **this file's API contract + Backend log** |
| Frontend | Claude Code (Opus 4.8 apex + Sonnet/Haiku triage) | `ui-redesign` | `../StateVectorAI-ui-redesign` | `qllm/dashboard/frontend/**`, **this file's UI log** |
| Integration | Claude Code apex (later) | short-lived `connect/*` | temporary | wiring, end-to-end verification |

**Disjoint ownership is absolute.** The backend track never edits
`qllm/dashboard/frontend/**`; the frontend track never edits `qllm/**` outside
the frontend. If a change is needed across the line, request it in your log —
do not reach across.

## Protocol — how to talk without merge conflicts

1. Each agent edits **only its own section** of this file (Backend edits the API
   contract + Backend log; UI edits the UI log). Never edit the other's section.
2. To read the other agent's latest messages without merging their code:
   ```
   git fetch origin
   git show origin/backend-enhancements:docs/BUILD_COORDINATION.md   # UI reads backend
   git show origin/ui-redesign:docs/BUILD_COORDINATION.md            # backend reads UI
   ```
3. Append log entries **newest last**, each prefixed with a date and author, e.g.
   `2026-07-12 · backend:`. Keep entries short and actionable.
4. The **API contract** is backend-owned and is the single source of truth for
   endpoints. The frontend builds to it. If the frontend needs a new endpoint or
   a shape change, it writes a request in the UI log; backend answers by updating
   the contract and noting it in the Backend log.
5. Sync cadence: both tracks rebase onto `origin/main` at least at the start of
   each work session so the contract and this protocol stay current.
6. Human gates are unchanged (GPU/QPU, spend, claim promotion, paid providers,
   consequential Git). Neither agent merges to `main` or promotes a claim without
   the user.

---

## API contract (backend-owned)

All routes are under `/api`. Status: `stable` (exists today), `proposed`
(frontend needs it), `building`, `shipped`. The frontend currently consumes the
`stable` set via `qllm/dashboard/frontend/src/lib/hooks.js` + `api.js`.

### Stable (already implemented — confirmed in `qllm/dashboard/server.py` + `api.js`)
| Method | Path | Returns (shape summary) |
| --- | --- | --- |
| GET | `/lab/overview` | queue counts, GPU readiness, evidence warnings |
| GET | `/jobs` | array of jobs: `{id, run_name, status(queued\|running\|done\|error\|cancelled), comparison_role, preset_id, dataset_name, seed, steps, eval_every, model_family, group_id, analogue_state, analogue_job_id, compare_to_job_id, device_target, gpu_reservation, interpretation_warnings, config}` |
| GET | `/jobs/{id}` · `/jobs/{id}/workspace` · `/jobs/{id}/comparison` · `/jobs/{id}/model-graph` · `/jobs/{id}/model-tests` | run detail, twin comparison, model graph, tests |
| POST | `/jobs` · `/jobs/sweep` · `/jobs/{id}/cancel` · `/jobs/{id}/classical-analogue` | queue / sweep / cancel / analogue |
| GET | `/status` | worker/GPU/queue status (shape TBD — see UI request #1) |
| GET | `/datasets` | array: `{name, source, source_type, split, n_rows, n_bytes, n_chars, text_column, ...}` |
| GET | `/presets` · `/config/choices` · `/explore*` · `/scaling-tests*` · `/studies*` · `/suites` · `/runs` · `/live` | see `api.js` |

### Proposed / needed by the frontend (backend to design & confirm shapes)
| Prio | Method | Path | Purpose | Requested for |
| --- | --- | --- | --- | --- |
| P1 | GET (SSE) | `/stream/jobs` (or WS) | live job/queue updates to replace 2–3s polling | Phase 1 shell (`useIsFetching` today) |
| P1 | GET | `/status` **contract** | pin the shape: `{worker, gpu_available, queued, running, runs}` | System surface |
| P2 | GET | `/jobs/{id}/diagnostics` | per-run quantum diagnostics from `qllm/quantum/metrics.py`: `{grad_variance, grad_snr, expressibility_kl, meyer_wallach, scaling_fit?}` | Phase 2 run detail |
| P2 | GET | `/verdicts` · `/verdicts/{id}` | persistent verdict/adjudication store (does not exist yet — verdicts are derived on the fly). Each: claim_level, replication_status, per-dimension scorecard (diagnostics labeled as diagnostics), fairness/controls, caveats | Phase 2 Verdicts |
| P3 | GET/POST | `/designer/circuit` round-trip | validate/build a circuit spec against `registry.py` `BACKEND_TYPES`/`CIRCUIT_ANSATZ_TYPES` | Phase 5 Designer |
| P3 | GET | `/library/*`, `/discover/*` | greenfield research service (human-gated provider/cost) | Phase 4 |

Contract rule: never emit a composite "advantage score"; keep claim-level and
replication distinct; label wall-time as simulator cost. See RESEARCH_PROGRAM.md.

---

## Log — from Backend (Codex appends here)

- 2026-07-11 · backend: _channel created by apex. Awaiting backend session start._

## Log — from UI (Claude appends here)

- 2026-07-11 · ui: Phase 1 foundation shipped on `ui-redesign` (`9c8b26f`): new
  IA shell, design tokens, TanStack Query, Overview/Runs/Datasets/System wired to
  the stable contract; Bench/Verdicts/Atlas/Designer/Discover/Library scaffolded.
  **Requests:** (#1) pin the `/status` shape; (#2) `/stream/jobs` SSE for live
  updates; (#3) `/jobs/{id}/diagnostics` for Phase 2.

## Open decisions / blockers

| # | Item | Owner | Status |
| --- | --- | --- | --- |
| D1 | `/status` response shape | backend | open |
| D2 | Live updates transport: SSE vs WebSocket | backend | open |
| D3 | Persistent verdict store schema | backend | open |
| D4 | Research-service LLM/embedding provider + per-day cost budget (**human-gated**) | user | open |
