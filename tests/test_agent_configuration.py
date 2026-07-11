from __future__ import annotations

import re
from pathlib import Path

from scripts.check_agent_setup import (
    EXPECTED_AGENT_MODELS,
    EXPECTED_CLAUDE_AGENT_MODELS,
    READ_ONLY_CODEX_AGENTS,
    READ_ONLY_CLAUDE_AGENTS,
    REQUIRED_AGENT_GUIDES,
    REQUIRED_CLAUDE_AGENTS,
    render_claude_skill_bridge,
    validate_repo,
)


ROOT = Path(__file__).resolve().parents[1]


def _workflow_text(name: str) -> str:
    return (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _valid_fixture(root: Path) -> None:
    guide = "# Agent guide\n\nFollow the local scope, run focused tests, and report verification evidence. " * 2
    for relative in REQUIRED_AGENT_GUIDES:
        _write(root / relative, guide)
        _write(root / Path(relative).with_name("CLAUDE.md"), "@AGENTS.md\n")
    _write(
        root / "PLANS.md",
        "# Plans\n\n## Scope\nBound the work.\n\n## Progress\nTrack work.\n\n"
        "## Decisions\nRecord decisions.\n\n## Verification\nRecord commands and results.\n",
    )
    _write(root / ".gitignore", ".tmp/\nCLAUDE.local.md\n.claude/settings.local.json\n")
    skill_description = "Route example repository work through a safe workflow."
    _write(
        root / ".agents/skills/example-skill/SKILL.md",
        "---\nname: example-skill\n"
        f"description: {skill_description}\n---\n"
        "# Example skill\n\nUse this workflow when changing the example repository.\n",
    )
    _write(
        root / ".agents/skills/example-skill/agents/openai.yaml",
        'interface:\n  display_name: "Example"\n  short_description: "Example repository workflow"\n'
        '  default_prompt: "Use $example-skill to handle this task."\n',
    )
    _write(
        root / ".claude/skills/example-skill/SKILL.md",
        render_claude_skill_bridge("example-skill", skill_description),
    )
    _write(
        root / ".codex/config.toml",
        'model = "gpt-5.6"\n\n[features]\nmulti_agent = true\nhooks = true\n\n'
        "[agents]\nmax_threads = 4\nmax_depth = 1\njob_max_runtime_seconds = 900\n",
    )
    _write(
        root / ".codex/hooks.json",
        '{"hooks":{"Stop":[{"hooks":[{"type":"command",'
        '"command":"python scripts/verify_changes.py --hook",'
        '"commandWindows":"python scripts/verify_changes.py --hook"}]}]}}',
    )
    _write(
        root / ".claude/settings.json",
        '{"hooks":{"Stop":[{"hooks":[{"type":"command","command":"python",'
        '"args":["${CLAUDE_PROJECT_DIR}/scripts/verify_changes.py","--hook",'
        '"--hook-platform","claude","--repo","${CLAUDE_PROJECT_DIR}"],'
        '"timeout":600}]}]}}',
    )
    agent = (
        'name = "{name}"\n'
        'description = "A focused repository specialist"\n'
        'developer_instructions = "Inspect only the assigned scope and return concrete evidence to the parent agent."\n'
        'model = "{model}"\n'
        'model_reasoning_effort = "medium"\n'
        'sandbox_mode = "{sandbox_mode}"\n'
    )

    for name, model in EXPECTED_AGENT_MODELS.items():
        sandbox_mode = "read-only" if name in READ_ONLY_CODEX_AGENTS else "workspace-write"
        _write(
            root / ".codex/agents" / f"{name}.toml",
            agent.format(name=name, model=model, sandbox_mode=sandbox_mode),
        )

    for filename in REQUIRED_CLAUDE_AGENTS:
        name = Path(filename).stem
        read_only = name in READ_ONLY_CLAUDE_AGENTS
        tools = "Read, Glob, Grep" if read_only else "Read, Edit, Write, Glob, Grep, Bash"
        permission_mode = "plan" if read_only else "default"
        _write(
            root / ".claude/agents" / filename,
            "---\n"
            f"name: {name}\n"
            "description: Handle one focused repository role with explicit scope and evidence.\n"
            f"model: {EXPECTED_CLAUDE_AGENT_MODELS[name]}\n"
            f"tools: {tools}\n"
            f"permissionMode: {permission_mode}\n"
            "---\n\n"
            "Read the active repository instructions and handle only the bounded packet. "
            "Preserve user work, respect human gates, and return concrete file and command "
            "evidence to the parent without expanding scope.\n",
        )


def test_repository_agent_configuration_is_valid() -> None:
    assert validate_repo(ROOT) == []


def test_qllm_skill_catalog_covers_dashboard_research_and_issue_closeout() -> None:
    dashboard = (ROOT / ".agents/skills/qllm-dashboard-development/SKILL.md").read_text(
        encoding="utf-8"
    )
    research = (ROOT / ".agents/skills/qllm-research-protocol/SKILL.md").read_text(
        encoding="utf-8"
    )
    verification = (
        ROOT / ".agents/skills/qllm-code-change-verification/references/check-matrix.md"
    ).read_text(encoding="utf-8")
    closeout = (ROOT / ".agents/skills/qllm-issue-closure/SKILL.md").read_text(
        encoding="utf-8"
    )
    closeout_checklist = (
        ROOT
        / ".agents/skills/qllm-issue-closure/references/issue-closure-checklist.md"
    ).read_text(encoding="utf-8")
    model = (ROOT / ".agents/skills/qllm-model-development/SKILL.md").read_text(
        encoding="utf-8"
    )
    experiment = (ROOT / ".agents/skills/qllm-experiment-runner/SKILL.md").read_text(
        encoding="utf-8"
    )
    experiment_ui = (
        ROOT / ".agents/skills/qllm-experiment-runner/agents/openai.yaml"
    ).read_text(encoding="utf-8")

    assert "npm test" in dashboard
    assert "both themes at desktop and narrow" in dashboard
    assert "teacher_forced_side_information" in research
    assert "rerun_required" in research
    assert "npm test" in verification
    assert "require explicit user" in closeout
    assert "gh issue view" in closeout_checklist
    assert "`DatasetBundle`" in model
    assert "flattening compatibility-only" in model
    assert "explicit user approval" in experiment
    assert "conflicting values" in experiment
    assert "remote action only with explicit authorization" in (
        ROOT / ".agents/skills/qllm-issue-closure/agents/openai.yaml"
    ).read_text(encoding="utf-8")
    assert "$qllm-experiment-runner" in experiment_ui


def test_agent_configuration_ci_installs_collection_dependencies() -> None:
    workflow = _workflow_text("agent-configuration.yml")
    assert "python -m pip install pytest==9.0.3 PyYAML==6.0.3" in workflow


def test_github_actions_are_immutable_least_privilege_and_bounded() -> None:
    workflows = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
    assert {path.name for path in workflows} == {
        "agent-configuration.yml",
        "ci.yml",
        "dashboard-frontend.yml",
        "dependency-matrix.yml",
        "dependency-review.yml",
    }
    action_pattern = re.compile(
        r"^\s*(?:-\s+)?uses:\s+[-\w.]+/[-\w.]+@([0-9a-f]{40})\s+#\s+v\d[^\s]*\s*$"
    )
    for path in workflows:
        workflow = path.read_text(encoding="utf-8")
        assert "permissions:\n  contents: read" in workflow
        assert "concurrency:" in workflow
        assert "cancel-in-progress: true" in workflow
        action_lines = [line for line in workflow.splitlines() if "uses:" in line]
        assert action_lines
        assert all(action_pattern.fullmatch(line) for line in action_lines)


def test_default_python_ci_runs_the_complete_cpu_contract() -> None:
    workflow = _workflow_text("ci.yml")
    assert "pull_request:" in workflow
    assert "branches:\n      - main" in workflow
    assert "    paths:" not in workflow
    assert "os: [ubuntu-latest, windows-latest]" in workflow
    assert 'python-version: "3.12"' in workflow
    assert "python -m pip install -r requirements-cpu.txt" in workflow
    assert "python scripts/check_dependency_profiles.py --runtime-profile cpu" in workflow
    assert "python -m compileall -q qllm benchmarks scripts tests" in workflow
    assert 'python -m pytest -q --basetemp "${{ runner.temp }}/pytest-ci"' in workflow


def test_dashboard_and_dependency_automation_contracts() -> None:
    dashboard = _workflow_text("dashboard-frontend.yml")
    assert 'node-version: "22"' in dashboard
    assert "os: [ubuntu-latest, windows-latest]" in dashboard
    assert "working-directory: qllm/dashboard/frontend" in dashboard
    for command in ("npm ci", "npm test", "npm run build"):
        assert f"run: {command}" in dashboard

    dependency_review = _workflow_text("dependency-review.yml")
    assert "pull_request:" in dependency_review
    assert "  push:" not in dependency_review
    assert "actions/dependency-review-action@" in dependency_review
    assert "fail-on-severity: moderate" in dependency_review

    dependabot = (ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")
    for ecosystem in ("github-actions", "pip", "npm"):
        assert f"package-ecosystem: {ecosystem}" in dependabot
    assert dependabot.count("interval: weekly") == 3


def test_valid_minimal_configuration(tmp_path: Path) -> None:
    _valid_fixture(tmp_path)
    assert validate_repo(tmp_path) == []


def test_rejects_duplicate_skill_names_and_legacy_location(tmp_path: Path) -> None:
    _valid_fixture(tmp_path)
    _write(
        tmp_path / ".agents/skills/second-skill/SKILL.md",
        "---\nname: example-skill\n"
        "description: Another long and informative skill description.\n---\n# Duplicate\n",
    )
    _write(
        tmp_path / ".agents/skills/second-skill/agents/openai.yaml",
        'interface:\n  default_prompt: "Use $example-skill for this task."\n',
    )
    _write(tmp_path / ".codex/skills/legacy/SKILL.md", "---\nname: legacy\n---\n")

    errors = validate_repo(tmp_path)
    assert any("duplicate skill name" in error for error in errors)
    assert any("canonical .agents/skills" in error for error in errors)


def test_rejects_stale_agent_keys_and_unsafe_limits(tmp_path: Path) -> None:
    _valid_fixture(tmp_path)
    _write(
        tmp_path / ".codex/agents/explorer.toml",
        'description = "A focused repository explorer"\n'
        'instructions = "This is the stale instruction field and must not be used by custom agents."\n'
        'reasoning_effort = "high"\n',
    )
    _write(
        tmp_path / ".codex/config.toml",
        'model = "gpt-5.6"\n\n[features]\nmulti_agent = false\nhooks = false\n\n'
        "[agents]\nmax_threads = 12\nmax_depth = 3\njob_max_runtime_seconds = 3600\n",
    )

    errors = validate_repo(tmp_path)
    assert any("instructions->developer_instructions" in error for error in errors)
    assert any("reasoning_effort->model_reasoning_effort" in error for error in errors)
    assert any("agents.max_threads must equal 4" in error for error in errors)
    assert any("agents.max_depth must equal 1" in error for error in errors)
    assert any("job_max_runtime_seconds" in error for error in errors)
    assert any("features.multi_agent must be true" in error for error in errors)
    assert any("features.hooks must be true" in error for error in errors)


def test_rejects_root_or_agent_model_substitution(tmp_path: Path) -> None:
    _valid_fixture(tmp_path)
    config = tmp_path / ".codex/config.toml"
    config.write_text(
        config.read_text(encoding="utf-8").replace('model = "gpt-5.6"', 'model = "gpt-5.4"'),
        encoding="utf-8",
    )
    explorer = tmp_path / ".codex/agents/luna_explorer.toml"
    explorer.write_text(
        explorer.read_text(encoding="utf-8").replace(
            'model = "gpt-5.6-luna"', 'model = "gpt-5.6-terra"'
        ),
        encoding="utf-8",
    )

    errors = validate_repo(tmp_path)
    assert any("project root model must be gpt-5.6" in error for error in errors)
    assert any("luna_explorer.toml: model must be gpt-5.6-luna" in error for error in errors)


def test_claude_agent_triage_puts_opus_at_the_top() -> None:
    assert EXPECTED_CLAUDE_AGENT_MODELS["planner"] == "claude-opus-4-8"
    assert EXPECTED_CLAUDE_AGENT_MODELS["verifier"] == "claude-opus-4-8"
    # Every required Claude role has an assigned model tier.
    for filename in REQUIRED_CLAUDE_AGENTS:
        assert Path(filename).stem in EXPECTED_CLAUDE_AGENT_MODELS


def test_rejects_missing_or_substituted_claude_agent_model(tmp_path: Path) -> None:
    _valid_fixture(tmp_path)
    planner = tmp_path / ".claude/agents/planner.md"
    planner.write_text(
        planner.read_text(encoding="utf-8").replace(
            "model: claude-opus-4-8\n", ""
        ),
        encoding="utf-8",
    )
    terra = tmp_path / ".claude/agents/terra-worker.md"
    terra.write_text(
        terra.read_text(encoding="utf-8").replace(
            "model: claude-sonnet-5", "model: claude-opus-4-8"
        ),
        encoding="utf-8",
    )

    errors = validate_repo(tmp_path)
    assert any(
        "planner.md: model is required and must be claude-opus-4-8" in error
        for error in errors
    )
    assert any(
        "terra-worker.md: model must be claude-sonnet-5" in error for error in errors
    )


def test_skill_prompt_must_name_its_skill(tmp_path: Path) -> None:
    _valid_fixture(tmp_path)
    _write(
        tmp_path / ".agents/skills/example-skill/agents/openai.yaml",
        'interface:\n  default_prompt: "Handle this task safely."\n',
    )
    assert any("must mention $example-skill" in error for error in validate_repo(tmp_path))


def test_skill_metadata_and_direct_references_are_validated(tmp_path: Path) -> None:
    _valid_fixture(tmp_path)
    skill = tmp_path / ".agents/skills/example-skill/SKILL.md"
    skill.write_text(
        skill.read_text(encoding="utf-8")
        + "\nRead `references/missing.md` and `references/unlinked.md`.\n",
        encoding="utf-8",
    )
    _write(
        tmp_path / ".agents/skills/example-skill/references/unlinked.md",
        "# Linked after all\n",
    )
    _write(
        tmp_path / ".agents/skills/example-skill/references/orphan.md",
        "# Orphan\n",
    )
    _write(
        tmp_path / ".agents/skills/example-skill/agents/openai.yaml",
        'interface:\n  default_prompt: "Use $example-skill for this task."\n',
    )

    errors = validate_repo(tmp_path)
    assert any("display_name is required" in error for error in errors)
    assert any("short_description must be 25-64" in error for error in errors)
    assert any("referenced resource references/missing.md is missing" in error for error in errors)
    assert any("bundled reference references/orphan.md is not linked" in error for error in errors)


def test_requires_import_only_claude_bridges_and_configs_guide(tmp_path: Path) -> None:
    _valid_fixture(tmp_path)
    _write(tmp_path / "CLAUDE.md", "@AGENTS.md\n\nDuplicated project policy.\n")
    (tmp_path / "configs/AGENTS.md").unlink()

    errors = validate_repo(tmp_path)
    assert any("CLAUDE.md: must contain only @AGENTS.md" in error for error in errors)
    assert any("configs/AGENTS.md: required agent guide is missing" in error for error in errors)


def test_rejects_drifting_claude_skill_bridge(tmp_path: Path) -> None:
    _valid_fixture(tmp_path)
    bridge = tmp_path / ".claude/skills/example-skill/SKILL.md"
    bridge.write_text(
        bridge.read_text(encoding="utf-8")
        .replace("Route example repository work", "Divergent Claude workflow")
        .replace("../../../.agents/skills", "../../copied-skills")
        + "\nAdditional duplicated workflow policy.\n",
        encoding="utf-8",
    )

    errors = validate_repo(tmp_path)
    assert any("description must match canonical skill metadata" in error for error in errors)
    assert any("must reference canonical target" in error for error in errors)
    assert any("must match the exact canonical bridge template" in error for error in errors)


def test_rejects_writable_review_roles_and_invalid_claude_hook(tmp_path: Path) -> None:
    _valid_fixture(tmp_path)
    planner = tmp_path / ".codex/agents/planner.toml"
    planner.write_text(
        planner.read_text(encoding="utf-8").replace(
            'sandbox_mode = "read-only"', 'sandbox_mode = "workspace-write"'
        ),
        encoding="utf-8",
    )
    claude_planner = tmp_path / ".claude/agents/planner.md"
    claude_planner.write_text(
        claude_planner.read_text(encoding="utf-8")
        .replace("tools: Read, Glob, Grep", "tools: Read, Edit, Write")
        .replace("permissionMode: plan", "permissionMode: default"),
        encoding="utf-8",
    )
    _write(tmp_path / ".claude/settings.json", '{"hooks":{"Stop":"malformed"}}')

    errors = validate_repo(tmp_path)
    assert any("planner.toml: read-only role must use sandbox_mode" in error for error in errors)
    assert any("planner.md: read-only role exposes write tools" in error for error in errors)
    assert any("planner.md: read-only role must use permissionMode" in error for error in errors)
    assert any(".claude/settings.json: a command-based Stop hook is required" in error for error in errors)
