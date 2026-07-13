"""Hermetic regression coverage for dashboard environment status probes."""
from __future__ import annotations

import builtins
import sys
import types

import pytest

from qllm.dashboard import status


def _fake_jax(devices, backend="cpu"):
    module = types.ModuleType("jax")
    module.devices = lambda: devices
    module.default_backend = lambda: backend
    return module


def _patch_status_probes(
    monkeypatch,
    *,
    modules=None,
    frontend_ok=True,
    commands=None,
    smi_path=None,
    smi_output="",
):
    modules = modules or {"jax": True, "flax": True, "optax": True, "datasets": True}
    commands = commands or {"node": True, "npm": True}
    monkeypatch.setattr(status, "module_ok", lambda name: modules.get(name, False))
    monkeypatch.setattr(status, "frontend_build_available", lambda _: frontend_ok)
    monkeypatch.setattr(status, "command_ok", lambda name: commands.get(name, False))
    monkeypatch.setattr(
        status.shutil,
        "which",
        lambda name: smi_path if name == "nvidia-smi" else None,
    )
    monkeypatch.setattr(status.subprocess, "check_output", lambda *args, **kwargs: smi_output)


def test_module_ok_uses_importlib_spec_lookup(monkeypatch):
    marker = object()
    monkeypatch.setattr(
        status.importlib.util,
        "find_spec",
        lambda name: marker if name == "present" else None,
    )

    assert status.module_ok("present") is True
    assert status.module_ok("missing") is False


@pytest.mark.parametrize(
    "name, expected_suffix, fallback_exists",
    [("node", "node.exe", True), ("npm", "npm.cmd", True), ("node", "node.exe", False)],
)
def test_command_ok_uses_windows_node_fallback(monkeypatch, name, expected_suffix, fallback_exists):
    calls = []

    class FakePath:
        def exists(self):
            return fallback_exists

    monkeypatch.setattr(status.shutil, "which", lambda _: None)
    monkeypatch.setattr(status, "Path", lambda *parts: calls.append(parts) or FakePath())

    assert status.command_ok(name) is fallback_exists
    assert calls == [("C:/Program Files/nodejs", expected_suffix)]


def test_command_ok_rejects_missing_non_node_command(monkeypatch):
    monkeypatch.setattr(status.shutil, "which", lambda _: None)
    monkeypatch.setattr(status, "Path", lambda *_: pytest.fail("unexpected fallback"))

    assert status.command_ok("git") is False


def test_command_ok_accepts_path_command_without_windows_fallback(monkeypatch):
    monkeypatch.setattr(status.shutil, "which", lambda name: f"C:/tools/{name}.exe")
    monkeypatch.setattr(status, "Path", lambda *_: pytest.fail("unexpected fallback"))

    assert status.command_ok("node") is True


def test_frontend_build_available_requires_index_and_assets(tmp_path):
    dist = tmp_path / "dist"
    assert status.frontend_build_available(dist) is False

    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html>", encoding="utf-8")
    assert status.frontend_build_available(dist) is False

    (dist / "index.html").unlink()
    (dist / "assets").mkdir()
    assert status.frontend_build_available(dist) is False

    (dist / "index.html").write_text("<!doctype html>", encoding="utf-8")
    assert status.frontend_build_available(dist) is True


@pytest.mark.parametrize("platform", ["gpu", "cuda", "rocm", "tpu"])
def test_environment_status_marks_gpu_like_jax_platforms_ready(monkeypatch, tmp_path, platform):
    _patch_status_probes(monkeypatch)
    monkeypatch.setitem(sys.modules, "jax", _fake_jax([types.SimpleNamespace(platform=platform)]))

    result = status.environment_status(tmp_path / "dist")

    assert result["ok"] is True
    assert result["gpu"]["ready"] is True
    assert result["gpu"]["jax_backend"] == "cpu"
    assert result["gpu"]["jax_devices"] == [
        {"id": f"namespace(platform='{platform}')", "platform": platform}
    ]
    assert result["gpu"]["nvidia_smi"] is None
    assert "CUDA-enabled JAX wheel" in result["gpu"]["setup"]


def test_environment_status_reports_jax_import_failure_hermetically(monkeypatch, tmp_path):
    _patch_status_probes(monkeypatch)
    original_import = builtins.__import__

    def fail_jax_import(name, *args, **kwargs):
        if name == "jax":
            raise ImportError("test-only unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_jax_import)
    result = status.environment_status(tmp_path / "dist")

    assert result["gpu"]["ready"] is False
    assert result["gpu"]["jax_backend"] is None
    assert result["gpu"]["jax_devices"] == [
        {"id": "JAX unavailable: test-only unavailable", "platform": "error"}
    ]


def test_environment_status_reports_jax_device_probe_failure(monkeypatch, tmp_path):
    _patch_status_probes(monkeypatch)
    jax = _fake_jax([], backend="should-not-be-used")
    jax.devices = lambda: (_ for _ in ()).throw(RuntimeError("device probe failed"))
    monkeypatch.setitem(sys.modules, "jax", jax)

    result = status.environment_status(tmp_path / "dist")

    assert result["gpu"]["ready"] is False
    assert result["gpu"]["jax_backend"] is None
    assert result["gpu"]["jax_devices"] == [
        {"id": "JAX unavailable: device probe failed", "platform": "error"}
    ]


def test_environment_status_includes_successful_nvidia_smi_probe(monkeypatch, tmp_path):
    output = "NVIDIA RTX, 555.85, 24576 MiB\n"
    _patch_status_probes(
        monkeypatch,
        smi_path="C:/Windows/System32/nvidia-smi.exe",
        smi_output=output,
    )
    calls = []
    monkeypatch.setattr(
        status.subprocess,
        "check_output",
        lambda *args, **kwargs: calls.append((args, kwargs)) or output,
    )
    monkeypatch.setitem(sys.modules, "jax", _fake_jax([types.SimpleNamespace(platform="cpu")]))

    result = status.environment_status(tmp_path / "dist")

    assert result["gpu"]["nvidia_smi"] == {"ok": True, "output": output.strip()}
    assert calls == [
        (
            (
                [
                    "C:/Windows/System32/nvidia-smi.exe",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ],
            ),
            {"text": True, "stderr": status.subprocess.STDOUT, "timeout": 5},
        )
    ]


def test_environment_status_captures_nvidia_smi_failure(monkeypatch, tmp_path):
    _patch_status_probes(monkeypatch, smi_path="C:/nvidia-smi.exe")
    monkeypatch.setattr(
        status.subprocess,
        "check_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("smi failed")),
    )
    monkeypatch.setitem(sys.modules, "jax", _fake_jax([types.SimpleNamespace(platform="cpu")]))

    result = status.environment_status(tmp_path / "dist")

    assert result["gpu"]["nvidia_smi"] == {"ok": False, "output": "smi failed"}


def test_environment_status_dependency_readiness_install_help_and_overall_semantics(monkeypatch, tmp_path):
    _patch_status_probes(
        monkeypatch,
        modules={"jax": True, "flax": False, "optax": True, "datasets": False},
        frontend_ok=True,
        commands={"node": False, "npm": True},
    )
    monkeypatch.setitem(sys.modules, "jax", _fake_jax([types.SimpleNamespace(platform="cpu")]))

    result = status.environment_status(tmp_path / "dist")

    assert result["ok"] is False
    assert result["training"] == {
        "ok": False,
        "modules": {"jax": True, "flax": False, "optax": True},
        "install": f"{sys.executable} -m pip install -e .",
    }
    assert result["huggingface"] == {
        "ok": False,
        "modules": {"datasets": False},
        "install": f"{sys.executable} -m pip install -e .[hf]",
    }
    assert result["frontend"] == {
        "ok": True,
        "node": False,
        "npm": True,
        "build": "cd qllm/dashboard/frontend && npm install && npm run build",
    }


def test_environment_status_overall_ok_ignores_optional_hf_and_gpu(monkeypatch, tmp_path):
    _patch_status_probes(monkeypatch, modules={"jax": True, "flax": True, "optax": True, "datasets": False})
    monkeypatch.setitem(sys.modules, "jax", _fake_jax([types.SimpleNamespace(platform="cpu")]))

    result = status.environment_status(tmp_path / "dist")

    assert result["ok"] is True
    assert result["huggingface"]["ok"] is False
    assert result["gpu"]["ready"] is False
