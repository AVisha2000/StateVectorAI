# Dashboard Instructions

The root and `qllm/AGENTS.md` guidance remains in force. This file is the
specific contract for the FastAPI/SQLite dashboard and React/Vite frontend.

## Route and boundaries

- Use `$qllm-dashboard-development` for every dashboard API, queue, database
  view, study, model builder, or UI task. Add `$qllm-research-protocol` when a
  view labels or interprets evidence.
- Keep the server a thin local interface over canonical experiment data. Do
  not duplicate scientific calculations in React or fabricate placeholder
  results that look real.
- Preserve API field compatibility across `server.py`, payload builders,
  `frontend/src/api.js`, and consuming components. Update both sides and tests
  together when a contract changes.
- Treat `results/`, SQLite databases, imported datasets, queue rows, and live
  run state as user research artifacts. Do not delete or rewrite them.
- Bind and design for localhost use. Do not broaden CORS, file access, or path
  serving beyond the local research-console threat model.
- Job cancellation, GPU targeting, and queued expensive studies remain
  human-gated. Smoke runs must explicitly target CPU.

## Validation

Backend and API behavior:

```powershell
pytest -q tests/test_dashboard_lab.py
```

Frontend behavior and production build:

```powershell
Push-Location qllm/dashboard/frontend
npm test
npm run build
Pop-Location
```

For UI changes, inspect the rendered route at desktop and narrow widths and
exercise the changed interaction. For queue/API changes, start the dashboard
locally, then run:

```powershell
python scripts/queue_smoke.py --steps 1 --eval-every 1 --device-target cpu
```

Report if a browser or live-server check could not be performed; a successful
build alone does not verify interaction behavior.
