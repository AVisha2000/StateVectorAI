from __future__ import annotations

from pathlib import Path

import pytest

from scripts import check_dependency_profiles as profiles


PROJECT = """\
[project]
dependencies = [
  "jax>=0.10,<0.11",
  "flax>=0.12",
  "optax>=0.2.4",
  "pennylane==0.45.*",
  "pennylane-lightning==0.45.*",
  "numpy>=2.0",
  "pyyaml>=6",
]

[project.optional-dependencies]
tracking = ["mlflow>=3"]
plots = ["matplotlib>=3.8"]
tc = ["tensorcircuit-ng==1.7.0"]
mps = ["tensorcircuit-ng==1.7.0"]
dashboard = ["fastapi>=0.110", "uvicorn>=0.29", "httpx>=0.27"]
hf = ["datasets>=2.20"]
dev = ["pytest>=8", "hypothesis>=6", "matplotlib>=3.8", "mlflow>=3"]
"""

CPU_PINS = {
    "jax": "0.10.1",
    "jaxlib": "0.10.1",
    "flax": "0.12.7",
    "optax": "0.2.8",
    "orbax-checkpoint": "0.10.3",
    "pennylane": "0.45.0",
    "pennylane-lightning": "0.45.0",
    "numpy": "2.4.4",
    "pyyaml": "6.0.3",
    "mlflow": "3.13.0",
    "matplotlib": "3.10.8",
    "fastapi": "0.138.0",
    "uvicorn": "0.49.0",
    "httpx": "0.28.1",
    "datasets": "5.0.0",
    "pytest": "9.0.3",
    "hypothesis": "6.155.2",
}


def _pins(values: dict[str, str]) -> str:
    return "".join(f"{name}=={version}\n" for name, version in values.items())


def _profile_repo(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text(PROJECT, encoding="utf-8")
    (tmp_path / "requirements-cpu.txt").write_text(_pins(CPU_PINS), encoding="utf-8")
    gpu = {name: version for name, version in CPU_PINS.items() if name not in {"jax", "jaxlib"}}
    (tmp_path / "requirements-gpu-wsl.txt").write_text(_pins(gpu), encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("-r requirements-cpu.txt\n", encoding="utf-8")
    (tmp_path / "requirements-mps.txt").write_text(
        "-r requirements-cpu.txt\ntensorcircuit-ng==1.7.0\n",
        encoding="utf-8",
    )
    return tmp_path


def test_current_repository_dependency_profiles_pass() -> None:
    root = Path(__file__).resolve().parents[1]
    parsed = profiles.validate_repository(root)
    assert parsed["cpu"].resolved["jax"].version == "0.10.1"
    assert parsed["mps"].resolved["tensorcircuit-ng"].version == "1.7.0"


def test_missing_or_out_of_range_cpu_pin_fails(tmp_path: Path) -> None:
    root = _profile_repo(tmp_path)
    bad = dict(CPU_PINS)
    bad.pop("flax")
    (root / "requirements-cpu.txt").write_text(_pins(bad), encoding="utf-8")
    with pytest.raises(profiles.ProfileError, match="missing required distribution 'flax'"):
        profiles.validate_repository(root)

    bad["flax"] = "0.11.0"
    (root / "requirements-cpu.txt").write_text(_pins(bad), encoding="utf-8")
    with pytest.raises(profiles.ProfileError, match="does not satisfy"):
        profiles.validate_repository(root)


def test_required_transitive_compatibility_pin_cannot_drift_away(
    tmp_path: Path,
) -> None:
    root = _profile_repo(tmp_path)
    bad = dict(CPU_PINS)
    bad.pop("orbax-checkpoint")
    (root / "requirements-cpu.txt").write_text(_pins(bad), encoding="utf-8")
    with pytest.raises(profiles.ProfileError, match="compatibility pins"):
        profiles.validate_repository(root)


@pytest.mark.parametrize(
    "line, message",
    [
        ("flax>=0.12\n", "exact NAME==VERSION"),
        ("flax==0.12.7\nflax==0.12.8\n", "duplicates distribution"),
    ],
)
def test_loose_or_duplicate_profile_entries_fail(
    tmp_path: Path, line: str, message: str
) -> None:
    root = _profile_repo(tmp_path)
    (root / "requirements-cpu.txt").write_text(line, encoding="utf-8")
    with pytest.raises(profiles.ProfileError, match=message):
        profiles.validate_repository(root)


def test_gpu_profile_cannot_install_jax_or_drift_from_cpu(tmp_path: Path) -> None:
    root = _profile_repo(tmp_path)
    gpu_path = root / "requirements-gpu-wsl.txt"
    gpu_path.write_text(gpu_path.read_text(encoding="utf-8") + "jax==0.10.1\n", encoding="utf-8")
    with pytest.raises(profiles.ProfileError, match="minus exactly jax/jaxlib"):
        profiles.validate_repository(root)

    _profile_repo(tmp_path)
    text = gpu_path.read_text(encoding="utf-8").replace("flax==0.12.7", "flax==0.12.8")
    gpu_path.write_text(text, encoding="utf-8")
    with pytest.raises(profiles.ProfileError, match="GPU WSL pin drift"):
        profiles.validate_repository(root)


def test_alias_and_mps_profile_drift_fail(tmp_path: Path) -> None:
    root = _profile_repo(tmp_path)
    (root / "requirements.txt").write_text("jax==0.10.1\n", encoding="utf-8")
    with pytest.raises(profiles.ProfileError, match="must contain exactly"):
        profiles.validate_repository(root)

    _profile_repo(tmp_path)
    (root / "requirements-mps.txt").write_text(
        "-r requirements-cpu.txt\ntensorcircuit-ng==1.6.0\n",
        encoding="utf-8",
    )
    with pytest.raises(profiles.ProfileError, match="does not satisfy"):
        profiles.validate_repository(root)


def test_runtime_validation_rejects_missing_and_drifted_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _profile_repo(tmp_path)
    cpu = profiles.validate_repository(root)["cpu"]

    def installed(name: str) -> str:
        if name == "flax":
            return "0.12.6"
        if name == "jaxlib":
            raise profiles.importlib.metadata.PackageNotFoundError(name)
        return cpu.resolved[name].version

    monkeypatch.setattr(profiles.importlib.metadata, "version", installed)
    with pytest.raises(profiles.ProfileError, match="version drift for flax"):
        # Remove the missing package temporarily so deterministic sorting reaches flax.
        reduced = profiles.Profile(cpu.path, cpu.direct, {k: v for k, v in cpu.resolved.items() if k != "jaxlib"}, cpu.includes)
        profiles.validate_runtime(reduced)

    def missing_jaxlib(name: str) -> str:
        if name == "jaxlib":
            raise profiles.importlib.metadata.PackageNotFoundError(name)
        return cpu.resolved[name].version

    monkeypatch.setattr(profiles.importlib.metadata, "version", missing_jaxlib)
    with pytest.raises(profiles.ProfileError, match="missing jaxlib"):
        profiles.validate_runtime(cpu)


def test_cli_static_validation_reports_profile_boundary(capsys: pytest.CaptureFixture[str]) -> None:
    root = Path(__file__).resolve().parents[1]
    assert profiles.main(["--repo", str(root)]) == 0
    output = capsys.readouterr().out
    assert "top-level pins; not a transitive lock" in output


def test_dependency_ci_runs_clean_cpu_matrix_and_required_mps_suite() -> None:
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/dependency-matrix.yml").read_text(
        encoding="utf-8"
    )
    assert "os: [ubuntu-latest, windows-latest]" in workflow
    assert 'python-version: ["3.11", "3.12"]' in workflow
    assert "python -m pip install -r requirements-cpu.txt" in workflow
    assert "python -m pip install -r requirements-mps.txt" in workflow
    assert 'QLLM_REQUIRE_TENSORCIRCUIT_MPS: "1"' in workflow
    assert "tests/test_tensorcircuit_mps.py" in workflow
    assert "scripts/check_gpu.py" not in workflow
