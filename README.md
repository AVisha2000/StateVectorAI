# QLLM

Modular quantum-classical hybrid language modeling in JAX/Flax.

This repo is a testbed for evaluating quantum components inside language
models with the same training and evaluation pipeline used for classical
baselines. Quantum modules are selected by config, so comparisons stay
controlled instead of becoming separate experiments.

## What is in the repo

```text
qllm/                 Python package
  classical/          Classical layers and recurrent baselines
  quantum/            Circuits, backends, quantum layers, diagnostics
  data/               Text and synthetic quantum-data tasks
  models/             Model registries and architectures
  train/              Training loop and generation helpers
  dashboard/          Self-hosted results dashboard
benchmarks/           Experiment and plotting harnesses
configs/              Ready-to-run YAML configs
scripts/              Training, comparison, and GPU sanity scripts
tests/                Unit and integration tests
data/input.txt        Tiny Shakespeare corpus
RESULTS.md            Consolidated findings
DATA.md               Dataset notes
GPU_SETUP.md          Local GPU setup guide
GPU_QUEUE.md          Prioritized GPU experiment queue
```

Generated experiment outputs may exist locally under `results/`, `mlruns/`,
and `mlflow.db`. They are useful research artifacts, but they are ignored by
Git so the source tree stays clean.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -U pip
pip install -r requirements-cpu.txt
pip install --no-deps -e .
```

`requirements-cpu.txt` is the pinned, authoritative native CPU development,
dashboard, and Hugging Face profile. `requirements.txt` remains a compatibility
alias for that profile. Every immutable run manifest records Python, package,
JAX backend/device, and precision identity; completed summaries also report
provenance-labelled timing, state dimension, logical circuit-forward estimates,
parameter count, and device-memory support.

For NVIDIA GPU use under WSL, skip the CPU requirements command above and
follow `GPU_SETUP.md` so the pinned CUDA-enabled JAX wheel is installed first.

## Common commands

```bash
# Train the classical baseline
python scripts/train.py --config configs/classical_small.yaml

# Train a 4-qubit VQC FFN variant
python scripts/train.py --config configs/quantum_ffn_4q.yaml

# Run the barren-plateau scaling probe
python benchmarks/scaling_probe.py --qubits 2 4 6 8 10 12 --layers 2 --mlflow

# Compare runs
python scripts/compare_runs.py

# Test
pytest -q
```


## Research workflow

QLLM treats a claim as something to earn, not a dashboard label. Use the
following ladder when planning or reading an experiment:

- A single fair candidate/baseline pair is smoke evidence only.
- Three or more fair pairs can support paired analysis when seeds, data,
  steps, evaluation cadence, device target, and preprocessing are matched.
- Quantum-component comparisons need explicit classical controls: matched
  analogues where possible, frozen/random-circuit controls for trainability,
  Markov controls for synthetic long-memory data, and resource accounting for
  state dimension, circuit calls, wall time, and device/backend.
- Missing controls, side-information metrics, unmatched parameters, or
  negligible gains at high cost should be reported as limitations before any
  positive interpretation.

Quick map from question to starting point:

| Question | Start with |
| --- | --- |
| Does a quantum layer train at all? | `scripts/train.py` with a matched classical config and frozen/random controls. |
| Does monitored quantum data contain long memory? | `DATA.md` screens plus Markov-control synthetic configs. |
| Does a result survive multiple seeds? | Dashboard studies and `qllm.research_protocol.paired_stats`. |
| Is a dashboard comparison safe to read? | Warning panels, fairness mismatches, claim status, and resource ledger fields. |
| How do I add a component? | `docs/DEVELOPMENT.md`. |

```text
data/config -> train loop -> immutable run manifest/checkpoints -> SQLite results DB
          -> dashboard comparisons/studies -> research-protocol claim review
```

## AI-assisted development

Start Codex from the repository root so it discovers the scoped `AGENTS.md`
files, project agents, and on-demand skills. The operating model and adoption
commands are documented in [docs/AGENT_OPERATING_MODEL.md](docs/AGENT_OPERATING_MODEL.md).
Validate the setup with:

```bash
python scripts/check_agent_setup.py
python scripts/verify_changes.py --plan
```

## Dashboard

Double-click `QLLM Portal.bat` from the repo root on Windows to start the
portal and open it in your browser.

If dependencies or the frontend build are missing, double-click
`Setup QLLM Portal.bat` first. It installs the dashboard/Hugging Face Python
extras and builds the React frontend when Node.js is available.

Manual launch:

```bash
cd qllm/dashboard/frontend
npm install
npm run build
cd ../../..

python -m qllm.dashboard.run --port 8000
```

Then open `http://localhost:8000`.

The dashboard is loopback-only by default. Dashboard-managed datasets and run
artifacts are confined to the configured `--data` and `--results` roots. A
non-loopback bind requires `--allow-remote` plus explicit `--cors-origin`
values and prints a trusted-network warning; remote dataset imports accept Hub
repository IDs only, never URLs or server-local paths.

The dashboard is now a local experiment console:

- `Run` queues local training jobs from curated presets.
- `Jobs` shows queued/running/completed jobs; click a row to open the Run Workspace with live curves, model description, artifacts, and classical comparison.
- `Datasets` imports public Hugging Face text datasets into local corpora.
- `Suites` keeps the original read-only leaderboard views.
- `Docs` explains the presets and metrics.
- `GPU` shows whether JAX can see a GPU and blocks GPU-targeted jobs until CUDA-backed JAX is active.

Smoke-test the queue after the dashboard is running:

```bash
python scripts/queue_smoke.py

# Paired quantum-vs-classical smoke
python scripts/queue_smoke.py --preset quantum-ffn-4q --steps 1 --eval-every 1 --seed 22 --run-name pair-smoke --compare
```

For NVIDIA GPU setup, verify the environment before requesting GPU runs:

```bash
python -m pip install -U "jax[cuda13]==0.10.1"
python -c "import jax; print(jax.devices())"
python scripts/check_gpu.py
```

Use `GPU_SETUP.md` and the GPU tab for the full checklist. The portal does
not install CUDA from the browser; it only reports readiness and rejects GPU
jobs if JAX still sees CPU only.

Hugging Face imports are public-only in this version. Paste a dataset id such
as `roneneldan/TinyStories`, choose a split like `train`, choose the text
column, and QLLM Lab writes a local `.txt` corpus under `data/imported/`.

## Current status

The project has working classical baselines, several quantum swap-in modules,
scaling diagnostics, recurrent quantum-memory probes, contextual tasks,
two-stream experiments, and a SQLite-backed dashboard. The next high-value
work is the GPU queue in `GPU_QUEUE.md`, especially the separation flagship and
the two-stream seed-count expansion.
