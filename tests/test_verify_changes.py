from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts import verify_changes as verifier


def test_porcelain_parser_preserves_dotfiles_spaces_and_renames() -> None:
    changes = verifier.parse_porcelain_z(
        b"?? .agents/skills/new skill/SKILL.md\0R  docs/new name.md\0docs/old name.md\0"
    )
    assert changes == [
        verifier.Change("??", ".agents/skills/new skill/SKILL.md"),
        verifier.Change("R ", "docs/new name.md", "docs/old name.md"),
    ]


def test_path_selection_is_cpu_only_and_never_launches_training() -> None:
    checks = verifier.select_checks(
        [
            ".agents/skills/qllm-agent-workflow/SKILL.md",
            ".claude/agents/verifier.md",
            "qllm/model.py",
        ],
        Path(__file__).resolve().parents[1],
    )
    assert {check.id for check in checks} >= {"agent-setup", "agent-tests", "python-tests"}
    flattened = " ".join(word for check in checks for word in check.argv).casefold()
    for forbidden in ("train.py", "queue_smoke", "qpu", "git commit", "git push"):
        assert forbidden not in flattened


@pytest.mark.parametrize(
    "path",
    [
        ".github/workflows/agent-configuration.yml",
        ".github/workflows/dependency-review.yml",
        ".github/dependabot.yml",
    ],
)
def test_ci_metadata_changes_run_agent_tests(path: str) -> None:
    checks = verifier.select_checks([path], Path(__file__).resolve().parents[1])
    assert {check.id for check in checks} == {"agent-setup", "agent-tests"}


def test_ci_workflow_changes_run_the_checks_they_define() -> None:
    root = Path(__file__).resolve().parents[1]
    python_ids = {
        check.id
        for check in verifier.select_checks([".github/workflows/ci.yml"], root)
    }
    assert python_ids == {"agent-setup", "agent-tests", "python-tests"}

    dashboard_ids = [
        check.id
        for check in verifier.select_checks(
            [".github/workflows/dashboard-frontend.yml"], root
        )
    ]
    assert set(dashboard_ids) == {
        "agent-setup",
        "agent-tests",
        "dashboard-frontend-tests",
        "dashboard-build",
    }
    assert dashboard_ids.index("dashboard-frontend-tests") < dashboard_ids.index(
        "dashboard-build"
    )


def test_sensitive_paths_are_explicit_human_gates() -> None:
    gates = verifier.classify_human_gates(
        [
            "RESULTS.md",
            "GPU_QUEUE.md",
            "configs/qpu_hardware.yaml",
            ".github/workflows/publish.yml",
        ]
    )
    assert {gate["kind"] for gate in gates} == {"research_claim", "gpu", "qpu", "publish"}


def test_research_plans_are_not_mistaken_for_claim_updates() -> None:
    gates = verifier.classify_human_gates(
        ["docs/RESEARCH_PROGRAM.md", "docs/RESEARCH_MAP.yaml"]
    )
    assert gates == []


def test_atlas_ontology_changes_require_human_review(tmp_path: Path) -> None:
    gates = verifier.classify_human_gates(["docs/ATLAS_ONTOLOGY.yaml"])
    assert gates == [
        {
            "kind": "research_ontology",
            "paths": ["docs/ATLAS_ONTOLOGY.yaml"],
            "reason": "Curated Atlas research groupings require human review.",
        }
    ]
    result = verifier.run_plan(
        {
            "repo": str(tmp_path),
            "fingerprint": "atlas-review",
            "paths": ["docs/ATLAS_ONTOLOGY.yaml"],
            "checks": [],
            "human_gates": gates,
            "policy": {},
        }
    )
    assert result["status"] == "human_review_required"
    assert result["ok"] is False


def test_benchmarks_configs_scripts_and_mixed_tests_get_focused_checks() -> None:
    root = Path(__file__).resolve().parents[1]
    checks = verifier.select_checks(
        [
            ".agents/skills/example/SKILL.md",
            "benchmarks/scaling_probe.py",
            "configs/example.yaml",
            "scripts/train.py",
            "tests/test_quantum.py",
        ],
        root,
    )
    ids = {check.id for check in checks}
    assert {
        "agent-tests",
        "benchmark-tests",
        "config-tests",
        "script-syntax",
        "train-entrypoint-tests",
        "focused-tests",
    } <= ids


def test_deleted_script_and_test_are_not_scheduled_for_execution(tmp_path: Path) -> None:
    checks = verifier.select_checks(
        ["scripts/retired_helper.py", "tests/test_retired_helper.py"], tmp_path
    )
    assert {check.id for check in checks} == {"agent-setup"}


def test_frontend_changes_run_behavior_tests_before_build() -> None:
    checks = verifier.select_checks(
        ["qllm/dashboard/frontend/src/modelConfig.js"],
        Path(__file__).resolve().parents[1],
    )
    ids = [check.id for check in checks]
    assert "dashboard-frontend-tests" in ids
    assert "dashboard-build" in ids
    assert ids.index("dashboard-frontend-tests") < ids.index("dashboard-build")


@pytest.mark.parametrize(
    "path",
    [
        "qllm/dashboard/server.py",
        "qllm/dashboard/security.py",
        "qllm/dashboard/runner.py",
        "qllm/dashboard/verdicts.py",
        "qllm/dashboard/diagnostics.py",
        "qllm/dashboard/live_stream.py",
        "qllm/dashboard/status.py",
        "qllm/dashboard/designer.py",
        "qllm/dashboard/atlas.py",
    ],
)
def test_dashboard_backend_changes_run_complete_contract_bundle(path: str) -> None:
    checks = verifier.select_checks(
        [path], Path(__file__).resolve().parents[1]
    )
    dashboard = next(check for check in checks if check.id == "dashboard-tests")
    assert dashboard.argv[0:4] == (
        verifier._project_python(Path(__file__).resolve().parents[1]),
        "-m",
        "pytest",
        "-q",
    )
    assert dashboard.argv[4:] == verifier.DASHBOARD_BACKEND_TESTS


def test_safe_environment_uses_short_system_temp_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    system_temp = tmp_path / "system-temp"
    monkeypatch.setattr(verifier.tempfile, "gettempdir", lambda: str(system_temp))
    environment = verifier._safe_environment(tmp_path)
    selected = Path(environment["TMP"])
    assert selected.parent == system_temp
    assert selected.name.startswith("qllm-v-")
    assert environment["TEMP"] == environment["TMP"]
    assert environment["TMPDIR"] == environment["TMP"]
    assert selected.is_dir()


def test_run_plan_uses_per_check_timeouts_and_honors_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[int] = []

    def completed(*args, timeout: int, **kwargs):
        observed.append(timeout)
        return subprocess.CompletedProcess(args[0], 0, stdout="")

    monkeypatch.setattr(verifier.subprocess, "run", completed)
    plan = {
        "repo": str(tmp_path),
        "fingerprint": "timeouts",
        "paths": [],
        "checks": [
            {"id": "focused", "argv": ["python", "-V"], "reason": "focused"},
            {
                "id": "python-tests",
                "argv": ["python", "-m", "pytest", "-q"],
                "reason": "full suite",
                "timeout_seconds": verifier.FULL_SUITE_TIMEOUT_SECONDS,
            },
        ],
        "human_gates": [],
        "policy": {},
    }

    assert verifier.run_plan(plan)["status"] == "passed"
    assert observed == [
        verifier.DEFAULT_CHECK_TIMEOUT_SECONDS,
        verifier.FULL_SUITE_TIMEOUT_SECONDS,
    ]

    observed.clear()
    assert verifier.run_plan(plan, timeout=37)["status"] == "passed"
    assert observed == [37, 37]


@pytest.mark.parametrize(
    "path",
    [
        "pyproject.toml",
        "requirements.txt",
        "requirements-cpu.txt",
        "requirements-gpu-wsl.txt",
        "requirements-mps.txt",
        "scripts/check_dependency_profiles.py",
        "tests/test_dependency_profiles.py",
        ".github/workflows/dependency-matrix.yml",
    ],
)
def test_dependency_paths_run_profile_validation_and_core_tests(path: str) -> None:
    checks = verifier.select_checks([path], Path(__file__).resolve().parents[1])
    ids = {check.id for check in checks}
    assert {
        "dependency-profile-static",
        "dependency-profile-tests",
        "python-tests",
    } <= ids


def test_fingerprint_changes_when_worktree_content_changes(tmp_path: Path) -> None:
    path = tmp_path / "module.py"
    path.write_text("value = 1\n", encoding="utf-8")
    changes = [verifier.Change(" M", "module.py")]
    first = verifier.fingerprint_changes(tmp_path, changes)
    path.write_text("value = 2\n", encoding="utf-8")
    second = verifier.fingerprint_changes(tmp_path, changes)
    assert first != second


def test_stop_hook_active_immediately_allows_without_verification(tmp_path: Path) -> None:
    result = verifier.run_hook(tmp_path, {"stop_hook_active": True})
    assert result["decision"] == "allow"
    assert result["loop_fuse"] is True


@pytest.mark.parametrize("platform", ["codex", "claude"])
def test_render_hook_response_omits_success_decision(platform: str) -> None:
    assert verifier.render_hook_response({"decision": "allow", "reason": "done"}, platform) == {}


@pytest.mark.parametrize("platform", ["codex", "claude"])
def test_render_hook_response_preserves_only_block_contract(platform: str) -> None:
    rendered = verifier.render_hook_response(
        {"decision": "block", "reason": "Run tests", "verification": {"large": "payload"}},
        platform,
    )
    assert rendered == {"decision": "block", "reason": "Run tests"}


def test_hook_blocks_unchanged_failure_only_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "module.py").write_text("broken = True\n", encoding="utf-8")
    changes = [verifier.Change("??", "module.py")]

    def failed_plan(plan: dict[str, object], timeout: int = 600) -> dict[str, object]:
        return {
            "status": "failed",
            "ok": False,
            "fingerprint": plan["fingerprint"],
            "paths": plan["paths"],
            "human_gates": [],
            "checks": [{"id": "focused-tests", "passed": False}],
            "policy": plan["policy"],
        }

    monkeypatch.setattr(verifier, "run_plan", failed_plan)
    state = tmp_path / ".tmp/verify-state.json"
    first = verifier.run_hook(tmp_path, {}, state_path=state, changes=changes)
    second = verifier.run_hook(tmp_path, {}, state_path=state, changes=changes)

    assert first["decision"] == "block"
    assert second["decision"] == "allow"
    assert second["loop_fuse"] is True


def test_public_mode_flags_are_supported() -> None:
    parser = verifier._build_parser()
    assert parser.parse_args(["--plan"]).plan is True
    assert parser.parse_args(["--run"]).run is True
    assert parser.parse_args(["--hook"]).hook is True
    assert parser.parse_args(["--hook", "--hook-platform", "claude"]).hook_platform == "claude"
    assert parser.parse_args(["--run"]).timeout is None
    assert parser.parse_args(["--run", "--timeout", "37"]).timeout == 37
