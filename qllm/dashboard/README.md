# QLLM Lab - self-hosted dashboard

Local experiment console over `results/qllm_results.db`.

The dashboard shows finished leaderboards, live-training progress, durable
queued jobs, public Hugging Face text imports, claim/evidence contracts,
structured warnings, manifests/checkpoints, and resource ledgers.

## Architecture

```text
qllm/dashboard/
  server.py       FastAPI API + confined static frontend
  queries.py      historical run/suite projections
  lab.py          job/comparison/scaling payloads
  studies.py      multi-seed study protocols and reports
  evidence.py     structured warning/durability/resource view models
  presets.py      curated registry-driven model presets
  datasets.py     bounded/provenanced Hugging Face text imports
  runner.py       transactional DB-claimed worker + recovery
  run.py          loopback-safe launcher
  frontend/       React + Vite evidence cockpit
```

Data flow: `fit()` with a `dashboard_db` set writes idempotent per-step curves
to canonical `run_steps` (while retaining the legacy `steps` adapter), progress
to `live_runs`, and final summaries to `runs`. Browser
queued jobs are stored in `lab_jobs`; imported corpora are stored in
`lab_datasets`. Job claims, heartbeats, stale recovery, checkpoints, immutable
run manifests, and idempotent step logs are persisted in SQLite. Existing
databases are upgraded through additive repeatable migrations.

## Run It

```bash
pip install -e ".[dashboard,hf]"

cd qllm/dashboard/frontend
npm install
npm test
npm run build
cd ../../..

python -m qllm.dashboard.run --port 8000
```

Open `http://localhost:8000`.

Queue a CPU-only smoke run:

```bash
python scripts/queue_smoke.py --steps 1 --eval-every 1 --device-target cpu
```

## Pages

- `Overview`: queue health, active work, recent comparisons, warnings, and readiness.
- `Experiments`: filter/open/rerun/compare jobs and inspect durable state.
- `Model Builder`: inspect, validate, save, and queue per-layer model specs.
- `Studies`: create matched multi-seed protocols and open evidence reports.
- `Results`: browse task/study/suite evidence without hiding invalid protocols.
- `Datasets`: import bounded public Hugging Face text data with provenance.
- Run/comparison pages: warnings first, then claim/metric/statistics, complete
  fairness mismatches, analogue limitations, manifests, checkpoints, recovery,
  resources, and backend capabilities.

Research interpretation is computed in Python and rendered by React without
reclassification. See [the researcher guide](../../docs/RESEARCHER_GUIDE.md)
and [engineer guide](../../docs/DEVELOPMENT.md).

The launcher binds to loopback by default. A non-loopback bind requires
`--allow-remote` and explicit `--cors-origin` values; use it only on a trusted
network. Dataset and artifact paths remain confined to configured roots.
