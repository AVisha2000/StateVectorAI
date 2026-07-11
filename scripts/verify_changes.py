#!/usr/bin/env python3
"""Plan and run safe, path-aware verification for the current worktree.

The script is deliberately read-only with respect to Git. It may write only an
ignored fingerprint record beneath ``.tmp/verify-changes``. It never launches a
training command, GPU job, QPU job, commit, push, release, or deployment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


STATE_RELATIVE_PATH = Path(".tmp") / "verify-changes" / "state.json"
MAX_CAPTURE_CHARS = 12_000
AGENT_TESTS = (
    "tests/test_agent_configuration.py",
    "tests/test_verify_changes.py",
)
BENCHMARK_TESTS = (
    "tests/test_ablation.py",
    "tests/test_advantage.py",
    "tests/test_qnlp_suite.py",
    "tests/test_contextual.py",
    "tests/test_interference.py",
    "tests/test_recurrent.py",
    "tests/test_two_stream.py",
)


class VerificationError(RuntimeError):
    """Raised when verification cannot safely be planned or executed."""


@dataclass(frozen=True)
class Change:
    status: str
    path: str
    original_path: str | None = None


@dataclass(frozen=True)
class Check:
    id: str
    argv: tuple[str, ...]
    reason: str


def _normalize_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def parse_porcelain_z(data: bytes) -> list[Change]:
    """Parse ``git status --porcelain=v1 -z`` without losing spaced paths."""
    fields = data.split(b"\0")
    changes: list[Change] = []
    index = 0
    while index < len(fields):
        field = fields[index]
        index += 1
        if not field:
            continue
        if len(field) < 4:
            raise VerificationError("unexpected short record from git status")
        status = field[:2].decode("ascii", errors="replace")
        path = field[3:].decode("utf-8", errors="surrogateescape")
        original: str | None = None
        if "R" in status or "C" in status:
            if index >= len(fields) or not fields[index]:
                raise VerificationError("rename/copy record is missing its original path")
            original = fields[index].decode("utf-8", errors="surrogateescape")
            index += 1
        changes.append(
            Change(
                status=status,
                path=_normalize_path(path),
                original_path=_normalize_path(original) if original else None,
            )
        )
    return changes


def read_changes(repo: Path) -> list[Change]:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
            cwd=repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError as exc:
        raise VerificationError("git executable was not found") from exc
    if completed.returncode:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise VerificationError(f"git status failed: {message or 'unknown error'}")
    return parse_porcelain_z(completed.stdout)


def changed_paths(changes: Iterable[Change]) -> list[str]:
    paths: set[str] = set()
    for change in changes:
        paths.add(_normalize_path(change.path))
        if change.original_path:
            paths.add(_normalize_path(change.original_path))
    return sorted(path for path in paths if path)


def _is_agent_path(path: str) -> bool:
    lowered = path.casefold()
    return (
        lowered.startswith(".agents/")
        or lowered.startswith(".claude/")
        or lowered.startswith(".codex/")
        or lowered.startswith(".github/workflows/")
        or lowered.endswith("/agents.md")
        or lowered.endswith("/claude.md")
        or lowered in {"agents.md", "claude.md", "plans.md"}
        or lowered in {
            "scripts/check_agent_setup.py",
            "scripts/verify_changes.py",
            ".github/dependabot.yml",
            ".github/pull_request_template.md",
            *AGENT_TESTS,
        }
    )


def _project_python(repo: Path) -> str:
    """Prefer the checked-out virtualenv when the launcher Python is bare."""
    candidates = (
        repo / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"),
        repo / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def classify_human_gates(paths: Sequence[str]) -> list[dict[str, Any]]:
    """Return path-based gates that automation is not authorized to approve."""
    buckets: dict[str, set[str]] = {
        "research_claim": set(),
        "gpu": set(),
        "qpu": set(),
        "publish": set(),
    }
    for path in paths:
        normalized = _normalize_path(path)
        lowered = normalized.casefold()
        parts = lowered.split("/")
        name = parts[-1]

        if (
            lowered == "results.md"
            or lowered.startswith("results/")
            or lowered.startswith("studies/") and any(word in name for word in ("report", "result", "claim"))
            or lowered.startswith("docs/") and any(word in name for word in ("result", "claim", "evidence"))
        ):
            buckets["research_claim"].add(normalized)

        if (
            lowered == "gpu_queue.md"
            or any(part in {"gpu", "cuda", "rocm"} for part in parts)
            or any(word in name for word in ("gpu", "cuda", "rocm"))
        ):
            buckets["gpu"].add(normalized)

        if (
            any(part in {"qpu", "hardware", "ibmq", "braket"} for part in parts)
            or any(word in name for word in ("qpu", "ibmq", "braket"))
        ):
            buckets["qpu"].add(normalized)

        if (
            lowered in {".pypirc", "release.md"}
            or any(part in {"release", "deploy", "publishing"} for part in parts)
            or lowered.startswith(".github/workflows/")
            and any(word in name for word in ("release", "publish", "deploy"))
            or any(word in name for word in ("publish", "release", "deploy"))
            and lowered.startswith(("scripts/", ".github/"))
        ):
            buckets["publish"].add(normalized)

    reasons = {
        "research_claim": "Evidence and quantum-advantage claim changes require human review.",
        "gpu": "GPU queue, spend, and execution changes require human approval.",
        "qpu": "QPU or hardware execution changes require human approval.",
        "publish": "Publishing, release, deployment, push, and merge changes require human approval.",
    }
    return [
        {"kind": kind, "paths": sorted(bucket), "reason": reasons[kind]}
        for kind, bucket in buckets.items()
        if bucket
    ]


def select_checks(paths: Sequence[str], repo: Path | None = None) -> list[Check]:
    """Select a minimal deterministic CPU-only check set from changed paths."""
    normalized = sorted({_normalize_path(path) for path in paths})
    root = (repo or Path.cwd()).resolve()
    python = _project_python(root)
    checks: list[Check] = [
        Check(
            "agent-setup",
            (python, "scripts/check_agent_setup.py", "--repo", "."),
            "The repository agent contract must remain valid for every change.",
        )
    ]

    agent_changed = any(_is_agent_path(path) for path in normalized)
    frontend_changed = any(
        path.casefold().startswith("qllm/dashboard/frontend/")
        or path.casefold() == ".github/workflows/dashboard-frontend.yml"
        for path in normalized
    )
    dashboard_backend_changed = any(
        path.casefold().startswith("qllm/dashboard/")
        and not path.casefold().startswith("qllm/dashboard/frontend/")
        and path.casefold().endswith(".py")
        for path in normalized
    )
    dependency_changed = any(
        path.casefold() == "pyproject.toml"
        or Path(path).name.casefold().startswith("requirements")
        and Path(path).suffix.casefold() == ".txt"
        or path.casefold() in {
            "scripts/check_dependency_profiles.py",
            "tests/test_dependency_profiles.py",
            ".github/workflows/dependency-matrix.yml",
        }
        for path in normalized
    )
    python_ci_changed = any(
        path.casefold() == ".github/workflows/ci.yml" for path in normalized
    )
    core_changed = any(
        path.casefold().startswith("qllm/")
        and path.casefold().endswith(".py")
        and not path.casefold().startswith("qllm/dashboard/")
        for path in normalized
    ) or dependency_changed or python_ci_changed
    benchmark_changed = any(
        path.casefold().startswith("benchmarks/") and path.casefold().endswith(".py")
        for path in normalized
    )
    config_changed = any(
        path.casefold().startswith("configs/")
        and Path(path).suffix.casefold() in {".json", ".toml", ".yaml", ".yml"}
        for path in normalized
    )
    script_paths = tuple(
        path
        for path in normalized
        if path.casefold().startswith("scripts/")
        and path.casefold().endswith(".py")
        and not _is_agent_path(path)
        and (root / path).is_file()
    )

    if agent_changed:
        existing_agent_tests = tuple(path for path in AGENT_TESTS if (root / path).is_file())
        if existing_agent_tests:
            checks.append(
                Check(
                    "agent-tests",
                    (python, "-m", "pytest", "-q", *existing_agent_tests),
                    "Agent workflow and non-mutating helper behavior changed.",
                )
            )

    if frontend_changed:
        checks.append(
            Check(
                "dashboard-frontend-tests",
                ("npm", "run", "test", "--prefix", "qllm/dashboard/frontend"),
                "Dashboard frontend behavior tests changed or may be affected.",
            )
        )
        checks.append(
            Check(
                "dashboard-build",
                ("npm", "run", "build", "--prefix", "qllm/dashboard/frontend"),
                "Dashboard frontend code changed.",
            )
        )
    if dashboard_backend_changed:
        checks.append(
            Check(
                "dashboard-tests",
                (python, "-m", "pytest", "-q", "tests/test_dashboard_lab.py"),
                "Dashboard API or queue code changed.",
            )
        )
    if dependency_changed:
        checks.append(
            Check(
                "dependency-profile-static",
                (python, "scripts/check_dependency_profiles.py", "--repo", "."),
                "Dependency metadata or a pinned profile changed; reject profile drift before installation.",
            )
        )
        checks.append(
            Check(
                "dependency-profile-tests",
                (
                    python,
                    "-m",
                    "pytest",
                    "-q",
                    "tests/test_dependency_profiles.py",
                    "tests/test_verify_changes.py",
                ),
                "Dependency profile validation and change-aware routing changed.",
            )
        )
    if config_changed:
        checks.append(
            Check(
                "config-tests",
                (python, "-m", "pytest", "-q", "tests/test_config_data.py"),
                "Experiment configuration files changed.",
            )
        )
    if benchmark_changed:
        available = tuple(path for path in BENCHMARK_TESTS if (root / path).is_file())
        if available:
            checks.append(
                Check(
                    "benchmark-tests",
                    (python, "-m", "pytest", "-q", *available),
                    "Scientific benchmark code changed; run benchmark-facing tests without launching it.",
                )
            )
    if script_paths:
        syntax_program = (
            "import ast,pathlib,sys; "
            "[ast.parse(pathlib.Path(p).read_text(encoding='utf-8'),filename=p) for p in sys.argv[1:]]"
        )
        checks.append(
            Check(
                "script-syntax",
                (python, "-c", syntax_program, *script_paths),
                "Changed operational scripts receive a read-only syntax check; they are never executed here.",
            )
        )
        if "scripts/train.py" in script_paths:
            checks.append(
                Check(
                    "train-entrypoint-tests",
                    (
                        python,
                        "-m",
                        "pytest",
                        "-q",
                        "tests/test_integration.py",
                        "tests/test_config_data.py",
                    ),
                    "The training entrypoint changed, but training itself must not run automatically.",
                )
            )
        if "scripts/compare_runs.py" in script_paths:
            checks.append(
                Check(
                    "comparison-tests",
                    (
                        python,
                        "-m",
                        "pytest",
                        "-q",
                        "tests/test_advantage.py",
                        "tests/test_research_protocol.py",
                    ),
                    "Result comparison behavior changed.",
                )
            )
    if core_changed:
        checks.append(
            Check(
                "python-tests",
                (python, "-m", "pytest", "-q"),
                "Core model, data, training, or dependency code changed.",
            )
        )
    else:
        changed_tests = tuple(
            path
            for path in normalized
            if path.casefold().startswith("tests/test_") and path.casefold().endswith(".py")
            and path not in AGENT_TESTS
            and path != "tests/test_dashboard_lab.py"
            and (root / path).is_file()
        )
        if changed_tests:
            checks.append(
                Check(
                    "focused-tests",
                    (python, "-m", "pytest", "-q", *changed_tests),
                    "Focused test modules changed.",
                )
            )

    # Deduplicate by stable id while retaining the first reason and command.
    unique: dict[str, Check] = {}
    for check in checks:
        unique.setdefault(check.id, check)
    return list(unique.values())


def _hash_path(hasher: Any, root: Path, relative: str) -> None:
    hasher.update(relative.encode("utf-8", errors="surrogateescape"))
    path = root / Path(relative)
    try:
        if path.is_symlink():
            hasher.update(b"SYMLINK\0")
            hasher.update(os.readlink(path).encode("utf-8", errors="surrogateescape"))
        elif path.is_file():
            hasher.update(b"FILE\0")
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    hasher.update(chunk)
        elif path.exists():
            hasher.update(b"OTHER\0")
        else:
            hasher.update(b"MISSING\0")
    except OSError as exc:
        hasher.update(f"ERROR:{type(exc).__name__}:{exc}".encode("utf-8", errors="replace"))


def fingerprint_changes(repo: Path, changes: Sequence[Change]) -> str:
    """Hash statuses, paths, and worktree content for unchanged-state detection."""
    hasher = hashlib.sha256()
    for change in sorted(changes, key=lambda item: (item.path, item.status, item.original_path or "")):
        hasher.update(change.status.encode("ascii", errors="replace"))
        hasher.update(b"\0")
        _hash_path(hasher, repo, change.path)
        hasher.update(b"\0")
        if change.original_path:
            _hash_path(hasher, repo, change.original_path)
        hasher.update(b"\0")
    # File content alone cannot distinguish two different staged versions when
    # a third version is present in the worktree. Include both read-only diffs.
    if (repo / ".git").exists():
        for arguments in (
            ("git", "diff", "--cached", "--binary", "--no-ext-diff", "--"),
            ("git", "diff", "--binary", "--no-ext-diff", "--"),
        ):
            try:
                process = subprocess.Popen(
                    arguments,
                    cwd=repo,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
            except FileNotFoundError as exc:
                raise VerificationError("git executable was not found") from exc
            assert process.stdout is not None
            while chunk := process.stdout.read(1024 * 1024):
                hasher.update(chunk)
            hasher.update(f"returncode={process.wait()}".encode("ascii"))
    return hasher.hexdigest()


def build_plan(repo: Path, changes: Sequence[Change] | None = None) -> dict[str, Any]:
    repo = repo.resolve()
    actual_changes = list(changes) if changes is not None else read_changes(repo)
    paths = changed_paths(actual_changes)
    checks = select_checks(paths, repo)
    return {
        "repo": str(repo),
        "fingerprint": fingerprint_changes(repo, actual_changes),
        "changes": [asdict(change) for change in actual_changes],
        "paths": paths,
        "checks": [asdict(check) | {"argv": list(check.argv)} for check in checks],
        "human_gates": classify_human_gates(paths),
        "policy": {
            "cpu_only": True,
            "git_mutation": False,
            "gpu_or_qpu_launch": False,
            "publish": False,
        },
    }


def _safe_environment(repo: Path) -> dict[str, str]:
    environment = os.environ.copy()
    temp_root = repo / ".tmp" / "verify-changes" / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "NVIDIA_VISIBLE_DEVICES": "none",
            "HIP_VISIBLE_DEVICES": "",
            "ROCR_VISIBLE_DEVICES": "",
            "JAX_PLATFORMS": "cpu",
            "JAX_PLATFORM_NAME": "cpu",
            "QLLM_ALLOW_GPU": "0",
            "QLLM_ALLOW_QPU": "0",
            "TMP": str(temp_root),
            "TEMP": str(temp_root),
            "TMPDIR": str(temp_root),
        }
    )
    return environment


def _resolve_argv(argv: Sequence[str]) -> list[str]:
    result = list(argv)
    if result and result[0] == "npm":
        executable = shutil.which("npm")
        if executable:
            result[0] = executable
    return result


def _trim_output(value: str) -> str:
    if len(value) <= MAX_CAPTURE_CHARS:
        return value
    half = (MAX_CAPTURE_CHARS - 80) // 2
    return value[:half] + "\n... verification output truncated ...\n" + value[-half:]


def run_plan(plan: dict[str, Any], timeout: int = 600) -> dict[str, Any]:
    repo = Path(plan["repo"])
    environment = _safe_environment(repo)
    results: list[dict[str, Any]] = []
    for raw_check in plan["checks"]:
        argv = _resolve_argv(raw_check["argv"])
        started = time.monotonic()
        try:
            completed = subprocess.run(
                argv,
                cwd=repo,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout,
            )
            returncode = completed.returncode
            output = _trim_output(completed.stdout or "")
            error: str | None = None
        except FileNotFoundError as exc:
            returncode = 127
            output = ""
            error = f"required executable was not found: {exc.filename}"
        except subprocess.TimeoutExpired as exc:
            returncode = 124
            raw_output = exc.stdout or ""
            if isinstance(raw_output, bytes):
                raw_output = raw_output.decode("utf-8", errors="replace")
            output = _trim_output(raw_output)
            error = f"check exceeded {timeout} seconds"

        results.append(
            {
                "id": raw_check["id"],
                "argv": raw_check["argv"],
                "returncode": returncode,
                "passed": returncode == 0,
                "duration_seconds": round(time.monotonic() - started, 3),
                "output": output,
                "error": error,
            }
        )

    checks_passed = all(result["passed"] for result in results)
    if not checks_passed:
        status = "failed"
    elif plan["human_gates"]:
        status = "human_review_required"
    else:
        status = "passed"
    return {
        "status": status,
        "ok": status == "passed",
        "fingerprint": plan["fingerprint"],
        "paths": plan["paths"],
        "human_gates": plan["human_gates"],
        "checks": results,
        "policy": plan["policy"],
    }


def load_state(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def save_state(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(result)
    payload["recorded_at_unix"] = int(time.time())
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _read_hook_input() -> dict[str, Any]:
    if sys.stdin.isatty():
        return {}
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return {}
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _hook_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Keep hook output small while retaining actionable failing evidence."""
    check_summaries: list[dict[str, Any]] = []
    for check in result.get("checks", []):
        summary = {
            "id": check.get("id"),
            "passed": check.get("passed"),
            "returncode": check.get("returncode"),
        }
        if not check.get("passed"):
            summary["error"] = check.get("error")
            summary["output"] = _trim_output(str(check.get("output", "")))[-2_000:]
        check_summaries.append(summary)
    return {
        "status": result.get("status"),
        "fingerprint": result.get("fingerprint"),
        "checks": check_summaries,
        "human_gates": result.get("human_gates", []),
    }


def run_hook(
    repo: Path,
    hook_input: dict[str, Any] | None = None,
    *,
    timeout: int = 600,
    state_path: Path | None = None,
    changes: Sequence[Change] | None = None,
) -> dict[str, Any]:
    """Run one Stop-hook verification attempt with an infinite-loop fuse."""
    hook_input = hook_input or {}
    if hook_input.get("stop_hook_active") is True:
        return {
            "decision": "allow",
            "reason": "Stop hook is already active; allowing stop to prevent recursion.",
            "loop_fuse": True,
        }

    repo = repo.resolve()
    target_state = state_path or (repo / STATE_RELATIVE_PATH)
    plan = build_plan(repo, changes)
    prior = load_state(target_state)

    if prior and prior.get("fingerprint") == plan["fingerprint"]:
        if prior.get("status") == "passed":
            return {
                "decision": "allow",
                "reason": "Unchanged worktree fingerprint already passed verification.",
                "verification": _hook_summary(prior),
                "cached": True,
            }
        if prior.get("hook_blocked_once") is True:
            return {
                "decision": "allow",
                "reason": (
                    "Unchanged failing or human-gated state was already blocked once; "
                    "allowing stop so the agent can hand control to the user."
                ),
                "verification": _hook_summary(prior),
                "loop_fuse": True,
            }
        result = prior
    else:
        result = run_plan(plan, timeout=timeout)

    if result.get("status") == "passed":
        result["hook_blocked_once"] = False
        save_state(target_state, result)
        return {
            "decision": "allow",
            "reason": "Safe CPU verification passed.",
            "verification": _hook_summary(result),
        }

    result["hook_blocked_once"] = True
    save_state(target_state, result)
    if result.get("status") == "human_review_required":
        reason = "Human-gated paths changed. Stop automated work and report the required approvals."
    else:
        failed = [item["id"] for item in result.get("checks", []) if not item.get("passed")]
        reason = "Safe verification failed: " + (", ".join(failed) or "unknown check")
    return {"decision": "block", "reason": reason, "verification": _hook_summary(result)}


def render_hook_response(response: dict[str, Any], platform: str) -> dict[str, Any]:
    """Render only the documented Stop-hook decision fields for a client."""
    if platform not in {"codex", "claude"}:
        raise VerificationError(f"unsupported hook platform: {platform}")
    if response.get("decision") == "block":
        return {
            "decision": "block",
            "reason": str(response.get("reason") or "Verification requires another pass."),
        }
    # Both clients document success by omitting a decision. The richer result
    # remains in the ignored verifier state file and direct run_hook callers.
    return {}


def _print_human_plan(plan: dict[str, Any]) -> None:
    print(f"Fingerprint: {plan['fingerprint']}")
    print("Changed paths:")
    for path in plan["paths"]:
        print(f"- {path}")
    print("Safe checks:")
    for check in plan["checks"]:
        print(f"- {check['id']}: {' '.join(check['argv'])}")
    if plan["human_gates"]:
        print("Human gates:")
        for gate in plan["human_gates"]:
            print(f"- {gate['kind']}: {', '.join(gate['paths'])}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("plan", "run", "hook"), help=argparse.SUPPRESS)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--plan", action="store_true", help="print the path-aware verification plan")
    modes.add_argument("--run", action="store_true", help="run the selected CPU-only checks")
    modes.add_argument("--hook", action="store_true", help="run Stop-hook JSON behavior")
    parser.add_argument(
        "--hook-platform",
        choices=("codex", "claude"),
        default="codex",
        help="render Stop-hook output for the invoking client",
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument("--json", action="store_true", help="emit JSON (hook always does)")
    parser.add_argument("--timeout", type=int, default=600, help="seconds allowed per check")
    parser.add_argument(
        "--state-file",
        type=Path,
        help="override ignored state path (primarily for isolated tests)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    selected_flags = [name for name in ("plan", "run", "hook") if getattr(args, name)]
    mode = selected_flags[0] if selected_flags else args.mode
    if mode is None:
        raise SystemExit("choose exactly one mode: --plan, --run, or --hook")
    if args.mode is not None and selected_flags and args.mode != mode:
        raise SystemExit("positional mode conflicts with selected mode flag")
    if args.timeout < 1:
        raise SystemExit("--timeout must be positive")
    repo = args.repo.resolve()
    try:
        if mode == "hook":
            response = render_hook_response(
                run_hook(
                    repo,
                    _read_hook_input(),
                    timeout=args.timeout,
                    state_path=args.state_file,
                ),
                args.hook_platform,
            )
            print(json.dumps(response, indent=2, sort_keys=True))
            return 0

        plan = build_plan(repo)
        if mode == "plan":
            if args.json:
                print(json.dumps(plan, indent=2, sort_keys=True))
            else:
                _print_human_plan(plan)
            return 0

        result = run_plan(plan, timeout=args.timeout)
        state_path = args.state_file or (repo / STATE_RELATIVE_PATH)
        result["hook_blocked_once"] = False
        save_state(state_path, result)
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(f"Verification status: {result['status']}")
            for check in result["checks"]:
                print(f"- {check['id']}: {'PASS' if check['passed'] else 'FAIL'}")
            for gate in result["human_gates"]:
                print(f"- HUMAN GATE {gate['kind']}: {', '.join(gate['paths'])}")
        return {"passed": 0, "failed": 1, "human_review_required": 2}[result["status"]]
    except VerificationError as exc:
        error = {"status": "error", "ok": False, "error": str(exc)}
        if args.json or mode == "hook":
            print(json.dumps(error, indent=2, sort_keys=True))
        else:
            print(f"Verification error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
