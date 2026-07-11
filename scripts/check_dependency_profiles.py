#!/usr/bin/env python3
"""Validate QLLM's top-level pinned dependency profiles.

This checker deliberately uses only the Python standard library so profile
drift can be rejected before installing project dependencies.  The checked-in
requirements files are top-level pinned profiles, not hash-locked transitive
locks; clean-install CI supplies the compatibility evidence for their current
resolution.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_NAME_SEPARATORS = re.compile(r"[-_.]+")
_PIN_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]+\])?==(?P<version>[^;\s]+)$"
)
_PROJECT_REQUIREMENT_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]+\])?(?P<spec>.*)$"
)
_SELECTED_CPU_EXTRAS = ("tracking", "plots", "dashboard", "hf", "dev")
_MPS_EXTRAS = ("tc", "mps")
_GPU_OMISSIONS = frozenset({"jax", "jaxlib"})
_CPU_COMPATIBILITY_PINS = frozenset({"jaxlib", "orbax-checkpoint"})


class ProfileError(ValueError):
    """A dependency profile cannot prove the repository contract."""


@dataclass(frozen=True)
class RequirementPin:
    name: str
    version: str
    source: Path
    line: int


@dataclass(frozen=True)
class Profile:
    path: Path
    direct: dict[str, RequirementPin]
    resolved: dict[str, RequirementPin]
    includes: tuple[Path, ...]


def canonical_name(name: str) -> str:
    """Return the PEP 503-style normalized distribution name."""
    return _NAME_SEPARATORS.sub("-", name).lower()


def _strip_comment(line: str) -> str:
    # Requirement URLs are intentionally unsupported in pinned profiles, so a
    # literal # always begins a comment here.
    return line.split("#", 1)[0].strip()


def _read_profile(path: Path, stack: tuple[Path, ...] = ()) -> Profile:
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(item.name for item in (*stack, path))
        raise ProfileError(f"recursive requirements include: {chain}")
    if not path.is_file():
        raise ProfileError(f"missing dependency profile: {path}")

    direct: dict[str, RequirementPin] = {}
    resolved: dict[str, RequirementPin] = {}
    includes: list[Path] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = _strip_comment(raw)
        if not line:
            continue
        include_match = re.fullmatch(r"-r\s+(.+)", line)
        if include_match:
            include_path = (path.parent / include_match.group(1).strip()).resolve()
            included = _read_profile(include_path, (*stack, path))
            includes.append(include_path)
            includes.extend(included.includes)
            for name, pin in included.resolved.items():
                if name in resolved:
                    raise ProfileError(
                        f"{path.name}:{line_number} includes duplicate distribution {name!r}"
                    )
                resolved[name] = pin
            continue
        if line.startswith("-"):
            raise ProfileError(
                f"{path.name}:{line_number} unsupported requirements directive {line!r}"
            )
        match = _PIN_RE.fullmatch(line)
        if not match:
            raise ProfileError(
                f"{path.name}:{line_number} must be an exact NAME==VERSION pin; got {line!r}"
            )
        name = canonical_name(match.group("name"))
        if name in direct or name in resolved:
            raise ProfileError(
                f"{path.name}:{line_number} duplicates distribution {name!r}"
            )
        pin = RequirementPin(name, match.group("version"), path, line_number)
        direct[name] = pin
        resolved[name] = pin
    return Profile(path, direct, resolved, tuple(dict.fromkeys(includes)))


def _version_parts(version: str) -> tuple[int, ...]:
    """Parse the numeric release segment used by QLLM's checked-in ranges."""
    release = version.split("+", 1)[0].split("-", 1)[0]
    if not re.fullmatch(r"\d+(?:\.\d+)*", release):
        raise ProfileError(f"unsupported non-numeric pinned version {version!r}")
    return tuple(int(item) for item in release.split("."))


def _compare_versions(left: str, right: str) -> int:
    a = _version_parts(left)
    b = _version_parts(right)
    length = max(len(a), len(b))
    a += (0,) * (length - len(a))
    b += (0,) * (length - len(b))
    return (a > b) - (a < b)


def _satisfies(version: str, spec: str) -> bool:
    spec = spec.strip()
    if not spec:
        return True
    for clause in (item.strip() for item in spec.split(",")):
        match = re.fullmatch(r"(==|>=|<=|>|<)(\d+(?:\.\d+)*(?:\.\*)?)", clause)
        if not match:
            raise ProfileError(f"unsupported project dependency clause {clause!r}")
        operator, target = match.groups()
        if operator == "==" and target.endswith(".*"):
            prefix = target[:-2]
            if not (version == prefix or version.startswith(prefix + ".")):
                return False
            continue
        comparison = _compare_versions(version, target)
        if not {
            "==": comparison == 0,
            ">=": comparison >= 0,
            "<=": comparison <= 0,
            ">": comparison > 0,
            "<": comparison < 0,
        }[operator]:
            return False
    return True


def _project_requirements(values: Iterable[str]) -> list[tuple[str, str]]:
    requirements: list[tuple[str, str]] = []
    for raw in values:
        match = _PROJECT_REQUIREMENT_RE.fullmatch(raw.strip())
        if not match:
            raise ProfileError(f"cannot parse project dependency {raw!r}")
        requirements.append(
            (canonical_name(match.group("name")), match.group("spec").strip())
        )
    return requirements


def _require_coverage(
    profile: Profile,
    requirements: Iterable[tuple[str, str]],
    *,
    label: str,
) -> None:
    for name, spec in requirements:
        pin = profile.resolved.get(name)
        if pin is None:
            raise ProfileError(f"{label} is missing required distribution {name!r}")
        if not _satisfies(pin.version, spec):
            raise ProfileError(
                f"{label} pin {name}=={pin.version} does not satisfy project range {spec!r}"
            )


def validate_repository(repo: str | Path) -> dict[str, Profile]:
    """Validate static profile structure and return the parsed profiles."""
    root = Path(repo).resolve()
    pyproject_path = root / "pyproject.toml"
    if not pyproject_path.is_file():
        raise ProfileError(f"missing project metadata: {pyproject_path}")
    with pyproject_path.open("rb") as handle:
        project = tomllib.load(handle).get("project", {})

    optional = project.get("optional-dependencies", {})
    base_requirements = _project_requirements(project.get("dependencies", []))
    selected_requirements = list(base_requirements)
    for extra in _SELECTED_CPU_EXTRAS:
        if extra not in optional:
            raise ProfileError(f"pyproject is missing selected CPU extra {extra!r}")
        selected_requirements.extend(_project_requirements(optional[extra]))
    mps_requirements: list[tuple[str, str]] = []
    for extra in _MPS_EXTRAS:
        if extra not in optional:
            raise ProfileError(f"pyproject is missing optional backend extra {extra!r}")
        mps_requirements.extend(_project_requirements(optional[extra]))

    profiles = {
        "cpu": _read_profile(root / "requirements-cpu.txt"),
        "gpu-wsl": _read_profile(root / "requirements-gpu-wsl.txt"),
        "alias": _read_profile(root / "requirements.txt"),
        "mps": _read_profile(root / "requirements-mps.txt"),
    }
    cpu = profiles["cpu"]
    gpu = profiles["gpu-wsl"]
    alias = profiles["alias"]
    mps = profiles["mps"]

    _require_coverage(cpu, selected_requirements, label="CPU profile")
    missing_compatibility = sorted(_CPU_COMPATIBILITY_PINS - set(cpu.resolved))
    if missing_compatibility:
        raise ProfileError(
            "CPU profile is missing required compatibility pins: "
            f"{missing_compatibility}"
        )
    expected_gpu = set(cpu.resolved) - _GPU_OMISSIONS
    if set(gpu.resolved) != expected_gpu:
        missing = sorted(expected_gpu - set(gpu.resolved))
        extra = sorted(set(gpu.resolved) - expected_gpu)
        raise ProfileError(
            "GPU WSL profile must equal CPU minus exactly jax/jaxlib; "
            f"missing={missing}, extra={extra}"
        )
    for name in expected_gpu:
        if gpu.resolved[name].version != cpu.resolved[name].version:
            raise ProfileError(
                f"GPU WSL pin drift for {name}: {gpu.resolved[name].version} != "
                f"CPU {cpu.resolved[name].version}"
            )
    if alias.direct or alias.includes != (cpu.path,):
        raise ProfileError("requirements.txt must contain exactly -r requirements-cpu.txt")
    if set(mps.resolved) != set(cpu.resolved) | {
        name for name, _spec in mps_requirements
    }:
        raise ProfileError(
            "MPS profile must extend CPU with exactly the declared optional backend distributions"
        )
    for name, pin in cpu.resolved.items():
        if mps.resolved[name].version != pin.version:
            raise ProfileError(f"MPS profile drifts from CPU pin {name}=={pin.version}")
    _require_coverage(mps, mps_requirements, label="MPS profile")
    return profiles


def validate_runtime(profile: Profile) -> dict[str, str]:
    """Verify installed direct versions against a parsed profile."""
    installed: dict[str, str] = {}
    for name, pin in sorted(profile.resolved.items()):
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ProfileError(
                f"runtime profile is missing {name}=={pin.version}"
            ) from exc
        if version != pin.version:
            raise ProfileError(
                f"runtime version drift for {name}: installed {version}, expected {pin.version}"
            )
        installed[name] = version
    return installed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="repository root")
    parser.add_argument(
        "--runtime-profile",
        choices=("cpu", "mps"),
        help="also require installed versions to match this profile",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        profiles = validate_repository(args.repo)
        print("dependency profiles: PASS (top-level pins; not a transitive lock)")
        for label in ("cpu", "gpu-wsl", "mps"):
            profile = profiles[label]
            print(f"- {label}: {len(profile.resolved)} resolved top-level pins")
        if args.runtime_profile:
            installed = validate_runtime(profiles[args.runtime_profile])
            print(
                f"runtime {args.runtime_profile}: PASS "
                f"({len(installed)} installed versions match)"
            )
    except (OSError, ProfileError, tomllib.TOMLDecodeError) as exc:
        print(f"dependency profiles: FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
