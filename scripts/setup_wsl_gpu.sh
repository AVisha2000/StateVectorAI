#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO_ROOT="$(pwd)"
VENV_DIR="${HOME}/.venvs/qllm-wsl"

echo "WSL kernel:"
uname -a

echo
echo "NVIDIA device check:"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is not visible inside WSL. Update the NVIDIA Windows driver and restart WSL."
  exit 1
fi
nvidia-smi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required inside WSL. Install it with: sudo apt update && sudo apt install -y python3 python3-venv python3-pip"
  exit 1
fi

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install -U pip
python -m pip install -U "jax[cuda13]==0.10.1"
python -m pip install -r "${REPO_ROOT}/requirements-gpu-wsl.txt"
# The project metadata declares JAX as a base dependency. --no-deps is
# intentional here because the curated profile above is complete and preserves
# the CUDA-enabled wheel installed first.
python -m pip install --no-deps -e "${REPO_ROOT}"

check_jax_gpu() {
python - <<'PY'
import sys
import jax

devices = jax.devices()
print("JAX version:", jax.__version__)
print("JAX devices:", devices)
if not any(getattr(d, "platform", "") in {"gpu", "cuda"} for d in devices):
    print("JAX still does not see a GPU.", file=sys.stderr)
    raise SystemExit(1)
PY
}

if ! check_jax_gpu; then
  echo
  echo "Dependency install appears to have left JAX on CPU; forcing the CUDA wheel back in."
  python -m pip install -U --force-reinstall "jax[cuda13]==0.10.1"
  check_jax_gpu
fi

python scripts/check_gpu.py
python -m pytest tests/test_dashboard_lab.py -q

echo
echo "GPU-backed QLLM environment is ready."
echo "Trusted-network remote mode (explicitly exposes the portal):"
echo "Run: source ${VENV_DIR}/bin/activate && python -m qllm.dashboard.run --host 0.0.0.0 --allow-remote --cors-origin http://127.0.0.1:8000 --cors-origin http://localhost:8000 --port 8000"
