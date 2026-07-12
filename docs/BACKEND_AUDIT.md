# StateVector Backend Audit — 2026-07-12

Read-only audit of `qllm/**` (backend + dashboard), `scripts/**`, and `tests/**`
on `main` (post frontend merge). Four parallel passes — security/boundaries,
research integrity, queue/runner correctness, and test coverage/quality — with
the most severe findings spot-verified against the code. No code was changed.

## Verdict

**The backend is in strong shape. No blockers, no critical bugs, and the
research-integrity guarantees are airtight.** The findings are a handful of
Medium hardening items (CSRF on state-changing routes, an advisory-only GPU
gate, an unguarded heartbeat thread), a few Low robustness/defense-in-depth
items, one real correctness/drift risk from duplicated helpers, and some test
gaps. Nothing here says "don't ship"; it's a punch-list to harden a system whose
core design is sound.

Severity legend: **M** medium · **L** low · **T** tech-debt/correctness ·
**TG** test gap · **N** nit.

---

## Findings (ranked)

### M1 — CSRF on state-changing `payload: dict` routes  *(verified)*
`qllm/dashboard/server.py` mutation routes take a raw `payload: dict`
(`api_create_job:503`, `api_create_scaling_sweep:537`, `api_import_hf:444`,
`api_cancel_job`, `api_queue_job_classical_analogue:585`, model-spec routes
`251/267/299`). FastAPI parses these via Starlette's `request.json()`, which does
**not** enforce `Content-Type`. The `enforce_local_request_access` middleware
(`server.py:105-114`) only checks the socket IP, and CORS only blocks *reading*
the response — not *sending* the request.
**Impact:** a malicious page the operator visits while the dashboard runs can
fire a non-preflighted `text/plain` POST to `127.0.0.1` and submit/cancel jobs
or trigger a dataset import (which in loopback mode reads arbitrary local
files/URLs — see L5). The attacker can't read the response, but the state change
and side effects still happen.
**Fix:** on state-changing routes, require `Content-Type: application/json`
(makes the request non-simple → forces a CORS preflight → the loopback-only
origin allowlist blocks it), and/or reject requests whose `Sec-Fetch-Site` is
`cross-site` / whose `Origin` isn't loopback. Small, centralizable middleware.

### M2 — GPU/high-memory reservation is display-only, not an execution gate  *(verified)*
`gpu_reservation.py` + `lab.py:24,153,411` only *read* reservation state for the
UI. `claim_next_lab_job` (`resultsdb.py:1399+`) selects
`WHERE status='queued' AND cancel_requested=0 ORDER BY id LIMIT 1` — no
`device_target`/GPU filter. Safe today only because there's a single in-process
worker (`server.py:121`) running jobs serially. A second process on the same DB
(a stray `uvicorn`, `--reload` double-spawn, a concurrent `queue_smoke.py`) could
claim two GPU jobs at once → OOM. The "exclusive GPU lane" is advisory, not
enforced.
**Fix:** if multi-process is ever real, add GPU exclusivity to the claim SQL
(`AND NOT EXISTS (SELECT 1 FROM lab_jobs WHERE status='running' AND <gpu-required>)`);
otherwise document + enforce the single-worker assumption (e.g. a PID/advisory
lock on startup).

### M3 — Heartbeat thread has no exception handling  *(verified)*
`runner.py:986-995`: `heartbeat_loop` calls `db.heartbeat_lab_job(...)` with no
try/except. A transient `sqlite3.OperationalError` (lock contention) or I/O error
silently kills the daemon thread without setting `ownership_lost`. Lease renewal
then depends only on the training-cadence `on_progress` heartbeat; for slow
steps with a long eval cadence the lease can lapse unobserved (harmless
single-process, but widens the split-brain window of M2 in multi-worker).
**Fix:** wrap the loop body in try/except — retry transient errors, and set
`ownership_lost` (or log + continue) on unexpected ones instead of dying silently.

### T1 — Duplicated `_decode_config` with divergent behavior  *(verified)*
Four copies: `model_tests.py:22` and `gpu_reservation.py:51` check
`job.get("config")` (decoded-dict fast path) then fall back to `config_json`;
`explore.py:21` and `workspace.py:22` only parse `config_json` (workspace also
guards `row is None`). A row carrying an already-decoded `config` dict is handled
differently across dashboard views — a real drift/correctness risk, not just
style. (Also duplicated with drift: `_job_variant` `lab.py:95` vs `explore.py:267`;
`_curve` `queries.py:225` vs `workspace.py:31`; `_artifact_dir` `model_tests.py:32`
vs `diagnostics.py:95`.)
**Fix:** consolidate into one shared `dashboard/_config.py` (and shared
`_curve`/`_artifact_dir`) so all views decode identically.

### L1 — `submit()` duplicate-run_uuid race surfaces a raw IntegrityError
`runner.py:436-465` does a check-then-act across two connections
(`get_lab_job_by_run_uuid` read, then `create_lab_job` insert). The
`UNIQUE(run_uuid)` index prevents dup rows, but a concurrent double-submit makes
the loser hit an unhandled `sqlite3.IntegrityError` → confusing `400 UNIQUE
constraint failed` instead of the intended idempotent "return existing job."
**Fix:** catch `IntegrityError` around the insert and re-fetch by `run_uuid`.

### L2 — All exceptions on mutation routes become HTTP 400
`server.py:131-133` `_payload_error` maps every `except Exception` to 400
(`server.py:504-533`, `536-565`, etc.). No stack-trace leak, but real server bugs
(`AttributeError`, `KeyError`, `sqlite3.OperationalError`) masquerade as client
errors during triage.
**Fix:** `except ValueError → 400`; `except Exception → 500` + server-side log.

### L3 — Schema migration `ALTER TABLE` not lock-protected on fresh-DB race
`resultsdb.py:266-343` runs `PRAGMA table_info` + conditional `ALTER TABLE ADD
COLUMN` outside `BEGIN IMMEDIATE`; `ResultsDB()` is created per request. Two
near-simultaneous first calls on a brand-new/just-upgraded DB could both attempt
the ALTER → `duplicate column name` → unhandled 500. One-time-per-fresh-DB only.
**Fix:** wrap migration in `BEGIN IMMEDIATE`, or catch/ignore duplicate-column.

### L4 — No global request-body size cap
Raw-`dict` POST routes buffer/parse arbitrarily large JSON before per-field
bounds apply. Local-only DoS surface.
**Fix:** a Starlette body-size middleware (per-route or global).

### L5 — Loopback-mode dataset import = arbitrary local-file/URL read  *(by design)*
`POST /api/datasets/hf/import` (`server.py:443`, `datasets.py:151-172`) only
restricts to Hub IDs when `remote_access_enabled()`; in default loopback mode
`source` can be any local path or `http(s)://` URL passed to `load_dataset`.
Matches the trusted-local threat model and is test-covered, but it's a real
local-file-read/SSRF primitive whose only guard is the loopback boundary (and M1).
**Fix (optional hardening):** confine `source` even in loopback mode, or require
an explicit opt-in for non-Hub sources.

### L6 — Forbidden-score guard lives only in the service layer
`_reject_forbidden_score_keys` runs in `verdicts.py:110-112`, but
`resultsdb.py:637 append_verdict_snapshot` (the DB writer) has no guard. Safe
today (single caller), but a future direct DB call could persist an
`advantage_score` key unchecked.
**Fix:** move the recursive guard into `append_verdict_snapshot` (defense-in-depth).

### L7 — `claim_level` field name overloaded across two vocabularies
`research_protocol.classify_claim` returns `claim_level ∈ {invalid, anecdote,
smoke, weak, fragile, quantum-inspired, empirical, …}`; the ledger's `claim_level`
(`claims.py:19-31`) is `{untested … formal}`. They never collide in storage
(`verdicts.py` re-maps the former to `assessment_level`), but the shared name is a
readability trap.
**Fix:** rename `classify_claim`'s output key to `assessment_level`.

### TG1 — ~50 of ~56 route handlers have no end-to-end HTTP test
Only ~6 routes go through `TestClient`; the rest are covered only via their
payload-builder functions — param binding, `response_model` serialization, and
HTTPException status mapping are untested.
**Fix:** one parametrized `TestClient` smoke test hitting each route with minimal
valid input against a temp `QLLM_DB`/`QLLM_RESULTS`.

### TG2 — `status.py` has zero coverage
`environment_status`/`frontend_build_available`/`module_ok`/`command_ok` feed
`/api/status` + `/api/lab/overview` and are untested (jax/nvidia-smi/node checks).
**Fix:** focused test with monkeypatched `jax`/`shutil.which`/`subprocess`.

### TG3 — `verify_changes.py` under-selects dashboard tests
`verify_changes.py:301-308` routes any `qllm/dashboard/**` change to only
`test_dashboard_lab.py`, missing `test_dashboard_security.py`,
`test_openapi_contract.py`, `test_dashboard_verdicts.py`,
`test_dashboard_diagnostics.py`, `test_dashboard_stream.py`. A hook-driven change
to `security.py`/`server.py` won't auto-run the security/contract tests.
**Fix:** expand the dashboard→tests mapping.

### T3 / N1 — watch items
`resultsdb.py` is a 2168-line, 70-method God-object (high blast radius; refactor
candidate). The loopback CORS regex is always-on, so any local web app is
equally trusted (intentional; worth revisiting as data sensitivity grows).

---

## Confirmed strong (no change needed)

- **Research integrity (airtight):** recursive no-composite-score guard
  (`verdicts.py:19,71-80,110`); `claim_level` vs `replication_status` are distinct
  ledger-bound columns; `classify_claim` gates every promotion (single-seed →
  ≤anecdote, underpowered → ≤smoke, dequantized → quantum-inspired, only fully
  gated → empirical) and no diagnostic can inflate a claim; promotion requires a
  manual `claims.yaml` edit (no code write path); simulator wall-time explicitly
  disclaimed as non-QPU (`resources.py:338`); fairness + analogue ladder are
  mandatory pre-claim gates; negative/null outcomes are preserved, not dropped.
- **Security:** loopback default + always-on runtime IP gate (blocks even
  `--host 0.0.0.0`); SSE independently loopback-gated; `resolve_within` path
  confinement at every artifact/dataset/plot/asset read+write, incl. symlink
  escape, applied at both submit and worker time; no `eval/exec/pickle.load`,
  one safe `nvidia-smi` call; arXiv bounds (fixed host, 1–25/req, atomic 50/day,
  10s, 2 MiB) all code-enforced; D4 capabilities hardcoded with no flip path; no
  secrets; parameterized SQL throughout.
- **Queue/runner:** atomic `BEGIN IMMEDIATE` + WHERE-guarded claim; lease
  fencing-token pattern; reservation released on every terminal path
  (done/error/cancelled/stale-recovery); strict checkpoint identity matching;
  atomic + `sha256`-checksummed checkpoint writes; `require_publish_ownership`
  closes the split-brain window.
- **Tests:** OpenAPI drift check is real and wired (`test_openapi_contract.py`);
  suite is hermetic (no live frontend build needed — D6 resolved); RNG seeded;
  migration path tested; no dead code / TODO markers.

---

## Fix plan

**P0 — do soon (small, real value)**
- M1 CSRF: enforce `Content-Type: application/json` (+ optional Origin/Sec-Fetch
  check) on state-changing routes. ~1 middleware, + a test.
- M3 heartbeat: try/except in `heartbeat_loop`. ~5 lines + a test.
- T1 `_decode_config`: consolidate the 4 copies (and `_curve`/`_job_variant`/
  `_artifact_dir`) into shared helpers. Mechanical + regression test.

**P1 — hardening & robustness**
- M2 GPU gate: enforce in claim SQL or lock the single-worker assumption.
- L1 idempotent submit (catch IntegrityError); L2 400-vs-500 split; L6 move the
  forbidden-score guard into the DB writer; TG3 expand `verify_changes.py` routing.

**P2 — coverage & tidy**
- TG1 route-level `TestClient` smoke tests; TG2 `status.py` tests; L3/L4/L5
  hardening; L7 rename; T3 `resultsdb.py` refactor watch.

## Ownership note

Every fix above is in `qllm/**` (backend), owned by the backend track under the
coordination protocol. This document is a report + plan; implementing the fixes
is a separate decision (backend track, or reassign the frontend apex to cross
over with explicit approval). Recommended: land P0 first.
