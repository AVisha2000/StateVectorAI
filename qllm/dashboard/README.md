# QLLM Lab - self-hosted dashboard

Local experiment console over `results/qllm_results.db`.

The dashboard shows finished leaderboards, live-training progress, queued
jobs, public Hugging Face text imports, and metric docs.

## Architecture

```text
qllm/dashboard/
  server.py     FastAPI JSON API over ResultsDB + static-serves the built UI
  queries.py    read-only dashboard SQL
  presets.py    curated model presets exposed to the Run tab
  datasets.py   local dataset registry + Hugging Face text import
  runner.py     single-worker local experiment queue
  run.py        launcher (python -m qllm.dashboard.run)
  frontend/     React + Vite app
```

Data flow: `fit()` with a `dashboard_db` set writes per-step curves to
`steps`, progress to `live_runs`, and final summaries to `runs`. Browser
queued jobs are stored in `lab_jobs`; imported corpora are stored in
`lab_datasets`.

## Run It

```bash
pip install -e ".[dashboard,hf]"

cd qllm/dashboard/frontend
npm install
npm run build
cd ../../..

python -m qllm.dashboard.run --port 8000
```

Open `http://localhost:8000`.

Queue a 5-step smoke run:

```bash
python scripts/queue_smoke.py
```

## Pages

- `Run`: queue a curated local experiment.
- `Datasets`: import public Hugging Face text datasets into local corpora.
- `Live runs`: queued jobs, running jobs, cancellation, and live curves.
- `Suites`: card grid of completed suites.
- `Suite`: sortable leaderboard.
- `Run detail`: config, metrics, and training curve.
- `Docs`: presets, dataset assumptions, metrics, and cautious quantum-result interpretation.

## Evidence vocabulary

Dashboard summaries use the same cautious ladder as the repository docs:

- `smoke`: one fair pair or otherwise preliminary evidence.
- `paired analysis`: three or more fair pairs with paired statistics.
- `missing control`: required baseline, frozen/random circuit, Markov twin, or
  analogue evidence is absent.
- `resource caveat`: any quantum result whose quality gain is small relative to
  measured or estimated cost.

Comparison and study routes should show structured warnings before optimistic
labels. Frontend code should not recompute scientific verdicts; it should render
the backend payload, claim status, limitations, and resource ledger.
