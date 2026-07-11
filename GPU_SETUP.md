# Running the QLLM testbed on your local GPU

The portal can queue GPU-targeted runs, but it will only enable them once
JAX actually reports a GPU-backed device. The UI does not install CUDA for
you; it reports readiness and blocks unsafe GPU submissions.

The checked-in dependency files are exact top-level pinned profiles, not
hash-locked transitive environments. Clean CI proves the current native CPU and
optional CPU MPS resolutions on Windows/Linux; it does not claim CUDA/driver
compatibility. The WSL profile relationship can be checked without installing
anything:

```bash
python scripts/check_dependency_profiles.py
```

JAX dispatches most project code to whatever device it finds. The
hand-rolled quantum ops are plain `jnp`, so they follow JAX placement.
PennyLane's `default.qubit` path also uses the JAX interface in this repo,
but it is worth checking separately with `scripts/check_gpu.py`.

## 0. Prerequisites

- An NVIDIA GPU with a recent driver. Check the driver and CUDA line:
  ```bash
  nvidia-smi
  ```
- Python 3.11 or 3.12.
- Node.js 18+ if you need to rebuild the dashboard frontend.
- On Windows, the official JAX support table does not list native Windows
  NVIDIA GPU wheels as supported. Use WSL2 if native Windows Python keeps
  reporting CPU only.

## 1. Create an isolated environment

Use a venv or conda; do not install into system Python.

```bash
python3 -m venv ~/.venvs/qllm-wsl
source ~/.venvs/qllm-wsl/bin/activate
```

On this Windows machine, use the checked-in launchers. They target the
`Ubuntu-24.04` WSL distro installed by `Setup GPU WSL.bat`:

1. Double-click `Setup GPU WSL.bat` and approve the Administrator prompt.
2. Reboot if Windows asks.
3. Open Ubuntu once and create the Linux username.
4. Double-click `Setup QLLM GPU in WSL.bat`.
5. Start the GPU-backed portal with `Start QLLM GPU Portal.bat`.

The WSL venv lives in your Linux home directory, not inside the repo
mount, because `/mnt/c` can trip over venv activation-file permissions.

## 2. Install JAX with CUDA support first

`requirements.txt` intentionally selects the native CPU profile and must not be
used in the WSL GPU environment. Install the CUDA variant first, using the current
official JAX guidance for your driver/CUDA setup:

```bash
pip install -U pip
pip install -U "jax[cuda13]==0.10.1"
```

If your driver is not ready for CUDA 13, follow the official JAX install
page for the CUDA 12 option that matches your machine while retaining the
project-compatible JAX `0.10.1` version.

Verify GPU visibility before queueing GPU work:

```bash
python3 -c "import jax; print(jax.devices())"
```

You want to see a CUDA/GPU device, not only `CpuDevice(id=0)`. If it still
shows CPU, the JAX/CUDA/driver versions do not match. Check `nvidia-smi`
and the official JAX installation page before queueing GPU runs.

After any dependency install that touches `jax` or `jaxlib`, rerun the
verification. If it falls back to CPU, reinstall the CUDA-enabled JAX wheel.
The WSL setup script does this once automatically after the dependency
install, but it is still worth remembering if you later install packages by
hand.

## 3. Install the optional WSL GPU profile

```bash
pip install -r requirements-gpu-wsl.txt
pip install --no-deps -e .
```

`requirements-gpu-wsl.txt` is pinned but deliberately excludes `jax` and
`jaxlib`, so it cannot replace the CUDA-enabled wheel. `--no-deps` is
intentional: the profile supplies the project dependencies without asking pip
to resolve the base JAX declaration again. The checked-in
`scripts/setup_wsl_gpu.sh` performs this sequence and verifies GPU visibility.
The static dependency-profile checker also requires this file to equal the CPU
profile minus exactly `jax` and `jaxlib`; only an actual WSL run can prove CUDA
visibility.

## 4. Sanity-check the GPU path

```bash
python3 scripts/check_gpu.py
```

This prints the JAX devices, runs short classical and quantum training
paths, and reports wall time. It is the quickest check that the GPU is
present and the project paths do not crash on this hardware.

## 5. Run tests once on the GPU environment

```bash
pytest -q
```

Everything should behave the same as CPU, but dtype and backend differences
can show up. It is better to catch those before a long sweep.

## 6. Build and start the dashboard

```bash
cd qllm/dashboard/frontend
npm install
npm run build
cd ../../..
python -m qllm.dashboard.run --port 8000
```

Open `http://localhost:8000`, then use the GPU tab. If JAX reports a GPU,
the Run form can use device target `gpu`; otherwise GPU jobs are rejected
before training starts.

Binding to `0.0.0.0` is trusted-network remote mode and requires the explicit
`--allow-remote` flag. It exposes the portal beyond loopback; keep the default
loopback command above unless remote access is genuinely required. Remote mode
also requires one or more explicit `--cors-origin` values; the checked-in WSL
launcher supplies the local Windows browser origins.

## 7. The GPU queue

See `GPU_QUEUE.md` for the prioritized experiment list that was deferred
for GPU compute.
