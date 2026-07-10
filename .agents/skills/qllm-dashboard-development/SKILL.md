---
name: qllm-dashboard-development
description: Build, debug, or review the QLLM FastAPI/React dashboard, queue, model builder, datasets, studies, GPU status, and result views. Use for dashboard UI, API, browser queueing, and dashboard workflow changes.
---

# QLLM Dashboard Development

Use this skill for QLLM Lab work. The dashboard is an experiment console, not a marketing site: preserve dense, scannable workflows for running, comparing, and inspecting research jobs.

## First Steps

1. Locate the repo root by finding `pyproject.toml` with project name `qllm`.
2. Read `qllm/dashboard/README.md`, then read `references/dashboard-map.md`.
3. For backend changes, inspect `qllm/dashboard/server.py` and the helper module that owns the payload.
4. For frontend changes, inspect `qllm/dashboard/frontend/src/api.js`, `App.jsx`, and the page/component being changed.

## Backend Rules

- Keep `server.py` thin: route handlers should delegate payload construction to focused modules.
- Store durable job/run state in `ResultsDB`; avoid parallel ad hoc stores.
- Use `ExperimentQueue` for anything that starts, cancels, or compares runs.
- Return JSON shapes that are easy for pages to render directly; preserve existing keys when possible.
- Convert user-facing exceptions to HTTP 400/404 with clear `detail` messages.

## Frontend Rules

- Keep the operational cockpit feel: compact headings, tables, filters, badges, and direct actions.
- Add API functions in `src/api.js` before wiring pages to new endpoints.
- Preserve existing route/navigation patterns in `App.jsx` and `main.jsx`.
- Show loading, empty, and error states for new data surfaces.
- Avoid decorative landing-page patterns; this UI is for repeated research operations.

## Verification

- Backend-only: run focused tests for dashboard modules, then `pytest tests/test_dashboard_lab.py -q` when relevant.
- Frontend-only: run `npm run build` from `qllm/dashboard/frontend`.
- Queue/API changes: run `python scripts/queue_smoke.py` when the API is available.
- Visual changes: launch the dashboard and inspect the relevant route in a browser when possible.
