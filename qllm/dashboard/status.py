"""Environment checks shown by the QLLM Lab portal."""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path


def module_ok(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def command_ok(name: str) -> bool:
    if shutil.which(name) is not None:
        return True
    if name in ("node", "npm"):
        suffix = "node.exe" if name == "node" else "npm.cmd"
        return Path("C:/Program Files/nodejs", suffix).exists()
    return False


def frontend_build_available(frontend_dist: Path) -> bool:
    """Return whether the complete static bundle required by FastAPI exists."""
    return (
        (frontend_dist / "index.html").is_file()
        and (frontend_dist / "assets").is_dir()
    )


def environment_status(frontend_dist: Path) -> dict:
    jax_devices = []
    jax_backend = None
    gpu_ready = False
    try:
        import jax

        devices = jax.devices()
        jax_devices = [
            {"id": str(d), "platform": getattr(d, "platform", "unknown")}
            for d in devices
        ]
        jax_backend = jax.default_backend()
        gpu_ready = any(d.get("platform") in ("gpu", "cuda", "rocm", "tpu")
                        for d in jax_devices)
    except Exception as exc:
        jax_devices = [{"id": f"JAX unavailable: {exc}", "platform": "error"}]

    nvidia_smi = None
    smi_path = shutil.which("nvidia-smi")
    if smi_path:
        try:
            out = subprocess.check_output(
                [
                    smi_path,
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ],
                text=True,
                stderr=subprocess.STDOUT,
                timeout=5,
            ).strip()
            nvidia_smi = {"ok": True, "output": out}
        except Exception as exc:
            nvidia_smi = {"ok": False, "output": str(exc)}

    training = {
        "ok": all(module_ok(m) for m in ("jax", "flax", "optax")),
        "modules": {
            "jax": module_ok("jax"),
            "flax": module_ok("flax"),
            "optax": module_ok("optax"),
        },
        "install": f"{sys.executable} -m pip install -e .",
    }
    hf = {
        "ok": module_ok("datasets"),
        "modules": {"datasets": module_ok("datasets")},
        "install": f"{sys.executable} -m pip install -e .[hf]",
    }
    frontend = {
        "ok": frontend_build_available(frontend_dist),
        "node": command_ok("node"),
        "npm": command_ok("npm"),
        "build": "cd qllm/dashboard/frontend && npm install && npm run build",
    }
    return {
        "ok": training["ok"] and frontend["ok"],
        "python": sys.executable,
        "training": training,
        "huggingface": hf,
        "frontend": frontend,
        "gpu": {
            "ready": gpu_ready,
            "jax_backend": jax_backend,
            "jax_devices": jax_devices,
            "nvidia_smi": nvidia_smi,
            "setup": (
                "Install a CUDA-enabled JAX wheel, then rerun the portal. "
                "See GPU_SETUP.md and https://docs.jax.dev/en/latest/installation.html."
            ),
        },
    }
