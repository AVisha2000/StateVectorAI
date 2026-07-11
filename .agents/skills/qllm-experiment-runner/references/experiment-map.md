# QLLM Experiment Map

## Key Files

- `configs/*.yaml`: ready-to-run experiments.
- `scripts/train.py`: single-run CLI entry point.
- `qllm/train/loop.py`: shared train/eval/generation pipeline.
- `qllm/train/artifacts.py`: run manifests, atomic checkpoints, compatibility,
  environment identity, and resume lineage.
- `qllm/resources.py`: static plans and measured runtime/resource ledgers.
- `qllm/resultsdb.py`: UUID-backed metrics and durable dashboard job state.
- `qllm/data/datasets.py`: dataset dispatch for text and synthetic tasks.
- `qllm/models/model.py`: architecture and component dispatch.
- `qllm/dashboard/runner.py`: local experiment queue.
- `qllm/dashboard/presets.py`: curated dashboard presets.
- `RESULTS.md`: consolidated findings.
- `GPU_QUEUE.md`: prioritized GPU experiments.

## Common Commands

```powershell
python scripts/train.py --config configs/classical_small.yaml
python scripts/train.py --config configs/quantum_ffn_4q.yaml --steps 50
python scripts/train.py --config configs/classical_small.yaml --resume-from results/my-run/checkpoints/latest.msgpack --checkpoint-every 10
python benchmarks/scaling_probe.py --qubits 2 4 6 8 --layers 2 --mlflow
python scripts/compare_runs.py
pytest -q
```

Dashboard:

```powershell
python -m qllm.dashboard.run --port 8000
python scripts/queue_smoke.py --steps 1 --eval-every 1 --device-target cpu
```

GPU readiness:

```powershell
python -c "import jax; print(jax.devices())"
python scripts/check_gpu.py
```

## Output Locations

- `results/<run_name>/summary.json`: final summary and history.
- `results/<run_name>/manifest.json`: immutable run/data/environment identity.
- `results/<run_name>/params.msgpack`: trained parameters.
- `results/<run_name>/checkpoints/latest.msgpack`: latest exact resume state.
- `results/<run_name>/checkpoints/best.msgpack`: best-validation checkpoint.
- `results/qllm_results.db`: dashboard SQLite store.
- `mlflow.db`: MLflow backend store.
- `results/.data_cache/`: generated synthetic datasets.

## Experiment Families

- Text baselines: `classical_small.yaml`, `quantum_ffn_4q.yaml`, `quantum_attn_4q.yaml`.
- Quantum-generated sequence tasks: `ising_sharp.yaml`, `ising_markov_sharp.yaml`.
- Memory scaling and recurrent work: `benchmarks/memory_sweep.py`, `benchmarks/recurrent_floor.py`, `benchmarks/qrnn_landscape.py`.
- Contextuality work: `benchmarks/contextual_sweep.py`, `qllm/data/contextual.py`, `qllm/quantum/contextual_cell.py`.
- Two-stream sentence-conditioning work: `benchmarks/two_stream_probe.py`, `qllm/models/two_stream.py`.

## Run Hygiene

- Keep smoke runs small before committing to long sweeps.
- Record exact command, seed list, device, and commit/worktree state when summarizing.
- Record checkpoint source, same-run versus fork mode, parent UUID, completed
  step, and resource-ledger measurement status when resuming or comparing cost.
- Never edit a checkpoint, queue row, or step metric to make a run look complete.
- Do not claim "advantage" from a single run or an unmatched baseline.

## Focused Durability And Resource Tests

- `tests/test_durable_runs.py`: checkpoint compatibility, exact resume,
  idempotent steps, queue leases/recovery, cancellation, and reservation release.
- `tests/test_dashboard_security.py`: checkpoint/path confinement and local access.
- `tests/test_resource_accounting.py`: environment and runtime ledger semantics.
- `tests/test_backend_capabilities.py` and `tests/test_tensorcircuit_mps.py`:
  exact/approximate capability behavior and local MPS execution contracts.
- `tests/test_dependency_profiles.py` plus
  `python scripts/check_dependency_profiles.py --repo .`: pinned-profile drift.
