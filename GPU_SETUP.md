# Running the QLLM testbed on your local GPU

The portal can queue GPU-targeted runs, but it will only enable them once
JAX actually reports a GPU-backed device. The UI does not install CUDA for
you; it reports readiness and blocks unsafe GPU submissions.

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

The pinned `requirements.txt` can reinstall CPU-only `jax` / `jaxlib` wheels
on a plain install. Install the CUDA variant first, using the current
official JAX guidance for your driver/CUDA setup:

```bash
pip install -U pip
pip install -U "jax[cuda13]"
```

If your driver is not ready for CUDA 13, follow the official JAX install
page for the CUDA 12 option that matches your machine.

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

## 3. Install the rest of the project

```bash
pip install \
  flax==0.12.7 \
  optax==0.2.8 \
  pennylane==0.45.0 \
  pennylane-lightning==0.45.0 \
  PyYAML==6.0.3 \
  fastapi>=0.110 \
  uvicorn>=0.29 \
  httpx>=0.27 \
  datasets>=2.20 \
  pytest>=8 \
  hypothesis>=6 \
  matplotlib>=3.8 \
  mlflow>=3
```

Avoid running `pip install -r requirements.txt` after the CUDA JAX install
unless you know it will not replace `jaxlib`. If you do use the pinned
requirements file, rerun the JAX device check immediately afterward.

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

## 7. The GPU queue

See `GPU_QUEUE.md` for the prioritized experiment list that was deferred
for GPU compute.
