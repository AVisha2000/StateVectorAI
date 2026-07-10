from __future__ import annotations

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
        [".agents/skills/qllm-agent-workflow/SKILL.md", "qllm/model.py"],
        Path(__file__).resolve().parents[1],
    )
    assert {check.id for check in checks} >= {"agent-setup", "agent-tests", "python-tests"}
    flattened = " ".join(word for check in checks for word in check.argv).casefold()
    for forbidden in ("train.py", "queue_smoke", "qpu", "git commit", "git push"):
        assert forbidden not in flattened


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
