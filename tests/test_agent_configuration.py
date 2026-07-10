from __future__ import annotations

from pathlib import Path

from scripts.check_agent_setup import REQUIRED_AGENT_GUIDES, validate_repo


ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _valid_fixture(root: Path) -> None:
    guide = "# Agent guide\n\nFollow the local scope, run focused tests, and report verification evidence. " * 2
    for relative in REQUIRED_AGENT_GUIDES:
        _write(root / relative, guide)
    _write(
        root / "PLANS.md",
        "# Plans\n\n## Scope\nBound the work.\n\n## Progress\nTrack work.\n\n"
        "## Decisions\nRecord decisions.\n\n## Verification\nRecord commands and results.\n",
    )
    _write(root / ".gitignore", ".tmp/\n")
    _write(
        root / ".agents/skills/example-skill/SKILL.md",
        "---\nname: example-skill\n"
        "description: Route example repository work through a safe workflow.\n---\n"
        "# Example skill\n\nUse this workflow when changing the example repository.\n",
    )
    _write(
        root / ".agents/skills/example-skill/agents/openai.yaml",
        'interface:\n  display_name: "Example"\n  short_description: "Example workflow"\n'
        '  default_prompt: "Use $example-skill to handle this task."\n',
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
    agent = (
        'name = "{name}"\n'
        'description = "A focused repository specialist"\n'
        'developer_instructions = "Inspect only the assigned scope and return concrete evidence to the parent agent."\n'
        'model = "{model}"\n'
        'model_reasoning_effort = "medium"\n'
    )
    from scripts.check_agent_setup import EXPECTED_AGENT_MODELS

    for filename, model in EXPECTED_AGENT_MODELS.items():
        _write(
            root / ".codex/agents" / f"{filename}.toml",
            agent.format(name=filename, model=model),
        )


def test_repository_agent_configuration_is_valid() -> None:
    assert validate_repo(ROOT) == []


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


def test_skill_prompt_must_name_its_skill(tmp_path: Path) -> None:
    _valid_fixture(tmp_path)
    _write(
        tmp_path / ".agents/skills/example-skill/agents/openai.yaml",
        'interface:\n  default_prompt: "Handle this task safely."\n',
    )
    assert any("must mention $example-skill" in error for error in validate_repo(tmp_path))
