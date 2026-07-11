# Dashboard Map

## Backend Files

- `qllm/dashboard/server.py`: FastAPI routes and static frontend serving.
- `qllm/dashboard/runner.py`: `ExperimentQueue`, job submission, cancellation, analogues.
- `qllm/resultsdb.py`: SQLite schema and persistence helpers.
- `qllm/dashboard/lab.py`: overview, enriched jobs, comparisons, scaling payloads.
- `qllm/dashboard/workspace.py`: run workspace payloads.
- `qllm/dashboard/presets.py`: curated launch presets.
- `qllm/dashboard/model_specs.py`: editable model specs and validation.
- `qllm/dashboard/studies.py`: study creation, queueing, reports.
- `qllm/dashboard/datasets.py`: local dataset registry and Hugging Face imports.
- `qllm/dashboard/status.py`: environment and frontend build status.
- `qllm/dashboard/gpu_reservation.py`: GPU lane status.
- `qllm/dashboard/evidence.py`: evidence summaries.
- `qllm/dashboard/security.py`: loopback access and confined path resolution.
- `qllm/dashboard/resources.py`: queue estimates and resource-ledger views.
- `qllm/dashboard/explore.py`: research-cockpit exploration payloads.

## Frontend Files

- `qllm/dashboard/frontend/src/api.js`: API client.
- `qllm/dashboard/frontend/src/App.jsx`: shell and navigation.
- `qllm/dashboard/frontend/src/main.jsx`: router definitions.
- `qllm/dashboard/frontend/src/styles.css`: global styling.
- `qllm/dashboard/frontend/src/pages/LabOverview.jsx`: main overview.
- `qllm/dashboard/frontend/src/pages/Launch.jsx`: queue new experiments.
- `qllm/dashboard/frontend/src/pages/Jobs.jsx`: job list/management.
- `qllm/dashboard/frontend/src/pages/RunWorkspace.jsx`: live job details.
- `qllm/dashboard/frontend/src/pages/Comparison.jsx`: candidate-vs-baseline comparison.
- `qllm/dashboard/frontend/src/pages/Models.jsx`: model builder/spec browsing.
- `qllm/dashboard/frontend/src/pages/Studies.jsx`: study list and creation.
- `qllm/dashboard/frontend/src/pages/Datasets.jsx`: dataset imports.
- `qllm/dashboard/frontend/src/pages/GPU.jsx`: system/GPU readiness.
- `qllm/dashboard/frontend/src/pages/Explore.jsx`: evidence and research-map exploration.
- `qllm/dashboard/frontend/src/pages/ResearchResults.jsx`: claim-aware results view.
- `qllm/dashboard/frontend/src/appShell.js`: shell navigation and page metadata.
- `qllm/dashboard/frontend/src/chartTheme.js`: shared Recharts theme tokens.
- `qllm/dashboard/frontend/src/exploreView.js`: exploration view-model helpers.
- `qllm/dashboard/frontend/src/components/EvidenceWarnings.jsx`: canonical
  rendering for backend-produced interpretation warnings.

## Common Commands

```powershell
pip install -e ".[dashboard,hf]"
python -m qllm.dashboard.run --port 8000
python -m qllm.dashboard.run --db .tmp/dashboard-qa.db --results .tmp/dashboard-qa-results --data .tmp/dashboard-qa-data --port 8000
python scripts/queue_smoke.py --steps 1 --eval-every 1 --device-target cpu
```

Frontend:

```powershell
cd qllm/dashboard/frontend
npm install
npm test
npm run build
```

## API Patterns

- `GET /api/status`: environment and GPU readiness.
- `GET /api/presets`: launch presets.
- `POST /api/jobs`: queue one job.
- `POST /api/jobs/sweep`: queue qubit/depth sweep.
- `GET /api/jobs/{id}/workspace`: run workspace.
- `GET /api/jobs/{id}/comparison`: comparison payload.
- `POST /api/model-specs/{id}/jobs`: queue editable model spec.
- `GET /api/studies/{id}` and `GET /api/studies/{id}/report`: study detail
  payloads; their browser peers are registered separately in `main.jsx`.
- `POST /api/studies/{id}/queue`: queue study.
- `GET /api/explore`, `/api/explore/domain/{domain_slug}`,
  `/api/explore/dataset/{dataset_name}`, and `/api/explore/task/{task_slug}`:
  research-cockpit payloads. Confirm the corresponding browser route before
  creating links.

## Common Pitfalls

- Updating a page without adding the matching API helper.
- Returning a payload shape that breaks existing pages.
- Forgetting cancellation behavior in queue changes.
- Letting GPU-targeted jobs bypass readiness checks.
- Starting long runs from tests instead of smoke-size jobs.
- Testing a visual route against a persistent database or queueing a job as
  part of browser QA.
- Fixing a chart in one theme while hard-coded tooltip or grid colors regress
  the other theme.
- Reimplementing warning thresholds in React instead of rendering structured
  `interpretation_warnings` from the backend.
- Manually rewriting queue rows or accepting a checkpoint whose run/data
  identity does not match the persisted job.

## Focused Durability And Safety Tests

- `tests/test_durable_runs.py`: claims, leases, recovery, checkpoint identity,
  cancellation, and reservation release.
- `tests/test_dashboard_security.py`: loopback/CORS gates and path containment.
- `tests/test_dashboard_lab.py`: API payloads, warning contracts, studies, and
  queue-facing behavior.
