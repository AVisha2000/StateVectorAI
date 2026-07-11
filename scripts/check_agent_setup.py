#!/usr/bin/env python3
"""Validate the repository's shared Codex and Claude Code operating system.

This checker intentionally uses only the Python standard library so it can run
before the project environment has been installed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tomllib
from pathlib import Path
from typing import Any


REQUIRED_AGENT_GUIDES = (
    "AGENTS.md",
    "qllm/AGENTS.md",
    "qllm/dashboard/AGENTS.md",
    "benchmarks/AGENTS.md",
    "tests/AGENTS.md",
    "docs/AGENTS.md",
    "configs/AGENTS.md",
)
REQUIRED_CUSTOM_AGENTS = (
    "planner.toml",
    "explorer.toml",
    "verifier.toml",
    "terra_worker.toml",
    "luna_explorer.toml",
    "mini_worker.toml",
    "spark_helper.toml",
)
EXPECTED_AGENT_MODELS = {
    "planner": "gpt-5.6",
    "explorer": "gpt-5.6-terra",
    "verifier": "gpt-5.6",
    "terra_worker": "gpt-5.6-terra",
    "luna_explorer": "gpt-5.6-luna",
    "mini_worker": "gpt-5.4-mini",
    "spark_helper": "gpt-5.3-codex-spark",
}
# Claude Code model triage, keyed by kebab-case agent stem. Opus 4.8 sits at the
# top for planning and verification; Sonnet handles discovery and implementation;
# Haiku handles small inventories, mechanical edits, and text-only helpers. This
# mirrors the Codex tiering in EXPECTED_AGENT_MODELS.
EXPECTED_CLAUDE_AGENT_MODELS = {
    "planner": "claude-opus-4-8",
    "verifier": "claude-opus-4-8",
    "explorer": "claude-sonnet-5",
    "terra-worker": "claude-sonnet-5",
    "luna-explorer": "claude-haiku-4-5-20251001",
    "mini-worker": "claude-haiku-4-5-20251001",
    "spark-helper": "claude-haiku-4-5-20251001",
}
IGNORED_PARTS = {
    ".git",
    ".tmp",
    ".venv",
    ".venv-wsl",
    "node_modules",
    "dist",
    "__pycache__",
}
STALE_AGENT_KEYS = {"instructions", "reasoning_effort"}
VALID_REASONING_EFFORT = {"minimal", "low", "medium", "high", "xhigh"}
REQUIRED_CLAUDE_AGENTS = (
    "planner.md",
    "explorer.md",
    "verifier.md",
    "terra-worker.md",
    "luna-explorer.md",
    "mini-worker.md",
    "spark-helper.md",
)
READ_ONLY_CODEX_AGENTS = {
    "planner",
    "explorer",
    "verifier",
    "luna_explorer",
    "spark_helper",
}
WRITING_CODEX_AGENTS = {"terra_worker", "mini_worker"}
READ_ONLY_CLAUDE_AGENTS = {
    "planner",
    "explorer",
    "verifier",
    "luna-explorer",
    "spark-helper",
}
WRITING_CLAUDE_AGENTS = {"terra-worker", "mini-worker"}


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _discover_named_files(root: Path, filename: str) -> list[Path]:
    discovered: list[Path] = []
    for directory, directories, filenames in os.walk(root):
        directories[:] = [name for name in directories if name not in IGNORED_PARTS]
        if filename in filenames:
            discovered.append(Path(directory) / filename)
    return sorted(discovered)


def _read_text(path: Path, errors: list[str], root: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        errors.append(f"{_relative(path, root)}: cannot read UTF-8 text ({exc})")
        return None


def _has_placeholder(text: str) -> bool:
    return bool(re.search(r"\[\s*TODO\b|<\s*TODO\b|TODO:\s", text, re.IGNORECASE))


def _parse_frontmatter(text: str) -> dict[str, str]:
    """Parse the small scalar subset needed from SKILL.md frontmatter."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("missing opening --- frontmatter delimiter")
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration as exc:
        raise ValueError("missing closing --- frontmatter delimiter") from exc

    values: dict[str, str] = {}
    index = 1
    while index < end:
        line = lines[index]
        match = re.match(r"^([A-Za-z0-9_-]+):(?:\s*(.*))?$", line)
        if not match:
            index += 1
            continue
        key, raw_value = match.group(1), (match.group(2) or "").strip()
        if raw_value in {">", "|"}:
            index += 1
            continuation: list[str] = []
            while index < end and (not lines[index].strip() or lines[index][:1].isspace()):
                continuation.append(lines[index].strip())
                index += 1
            values[key] = " ".join(part for part in continuation if part)
            continue
        if len(raw_value) >= 2 and raw_value[0] == raw_value[-1] and raw_value[0] in "\"'":
            raw_value = raw_value[1:-1]
        values[key] = raw_value
        index += 1
    return values


def _yaml_scalar(text: str, key: str) -> str:
    """Read one quoted or unquoted scalar from the small openai.yaml subset."""
    match = re.search(
        rf"(?m)^\s*{re.escape(key)}:\s*(?P<value>[^#\r\n]+?)\s*$", text
    )
    if match is None:
        return ""
    value = match.group("value").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return value.strip()


def render_claude_skill_bridge(name: str, description: str) -> str:
    """Return the only allowed Claude adapter for a canonical project skill."""
    return (
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        "---\n\n"
        "# Canonical QLLM skill bridge\n\n"
        f"Read and follow the [canonical skill](../../../.agents/skills/{name}/SKILL.md)\n"
        "completely before taking task actions. Resolve every relative reference from\n"
        "that canonical skill directory. This adapter exposes Claude Code discovery\n"
        "metadata only; do not fork the workflow here.\n"
    )


def _command_hooks(data: object, event: str) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        return []
    hooks = data.get("hooks")
    if not isinstance(hooks, dict):
        return []
    groups = hooks.get(event)
    if not isinstance(groups, list):
        return []
    commands: list[dict[str, Any]] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        handlers = group.get("hooks")
        if not isinstance(handlers, list):
            continue
        commands.extend(
            handler
            for handler in handlers
            if isinstance(handler, dict) and handler.get("type") == "command"
        )
    return commands


def _validate_agent_guides(root: Path, errors: list[str]) -> None:
    for relative in REQUIRED_AGENT_GUIDES:
        path = root / relative
        if not path.is_file():
            errors.append(f"{relative}: required agent guide is missing")

    plans = root / "PLANS.md"
    if not plans.is_file():
        errors.append("PLANS.md: required durable-plan contract is missing")
    else:
        text = _read_text(plans, errors, root)
        if text is not None:
            lowered = text.casefold()
            for concept in ("scope", "progress", "decision", "verification"):
                if concept not in lowered:
                    errors.append(f"PLANS.md: must describe {concept}")
            if _has_placeholder(text):
                errors.append("PLANS.md: contains an unresolved TODO placeholder")

    for path in _discover_named_files(root, "AGENTS.md"):
        text = _read_text(path, errors, root)
        if text is None:
            continue
        relative = _relative(path, root)
        if len(text.strip()) < 80:
            errors.append(f"{relative}: agent guide is too short to be actionable")
        if not re.search(r"(?m)^#\s+\S", text):
            errors.append(f"{relative}: must contain a Markdown title")
        if _has_placeholder(text):
            errors.append(f"{relative}: contains an unresolved TODO placeholder")


def _validate_claude_instruction_bridges(root: Path, errors: list[str]) -> None:
    guides = _discover_named_files(root, "AGENTS.md")
    for guide in guides:
        bridge = guide.with_name("CLAUDE.md")
        relative = _relative(bridge, root)
        if not bridge.is_file():
            errors.append(f"{relative}: import bridge for {_relative(guide, root)} is missing")
            continue
        text = _read_text(bridge, errors, root)
        if text is not None and text.strip() != "@AGENTS.md":
            errors.append(f"{relative}: must contain only @AGENTS.md")

    for bridge in _discover_named_files(root, "CLAUDE.md"):
        if not bridge.with_name("AGENTS.md").is_file():
            errors.append(
                f"{_relative(bridge, root)}: project instructions must import a sibling AGENTS.md"
            )


def _extract_default_prompt(text: str) -> str | None:
    match = re.search(r"(?m)^\s*default_prompt:\s*(.*?)\s*$", text)
    if not match:
        return None
    value = match.group(1).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return value


def _validate_skills(root: Path, errors: list[str]) -> None:
    canonical = root / ".agents" / "skills"
    if not canonical.is_dir():
        errors.append(".agents/skills: canonical project skill directory is missing")
        return

    legacy = root / ".codex" / "skills"
    if legacy.is_dir() and any(legacy.rglob("SKILL.md")):
        errors.append(".codex/skills: move project skills to canonical .agents/skills")

    seen: dict[str, str] = {}
    skill_files = sorted(canonical.glob("*/SKILL.md"))
    if not skill_files:
        errors.append(".agents/skills: no project skills found")
        return

    for skill_file in skill_files:
        relative = _relative(skill_file, root)
        text = _read_text(skill_file, errors, root)
        if text is None:
            continue
        try:
            metadata = _parse_frontmatter(text)
        except ValueError as exc:
            errors.append(f"{relative}: {exc}")
            continue

        name = metadata.get("name", "").strip()
        description = metadata.get("description", "").strip()
        expected_name = skill_file.parent.name
        if not name:
            errors.append(f"{relative}: frontmatter name is required")
        elif name != expected_name:
            errors.append(
                f"{relative}: frontmatter name {name!r} must match directory {expected_name!r}"
            )
        elif not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
            errors.append(f"{relative}: skill name must be lowercase kebab-case")

        folded = name.casefold()
        if name and folded in seen:
            errors.append(f"{relative}: duplicate skill name also used by {seen[folded]}")
        elif name:
            seen[folded] = relative

        if not description or len(description) < 20:
            errors.append(f"{relative}: frontmatter description must be informative")
        if _has_placeholder(text):
            errors.append(f"{relative}: contains an unresolved TODO placeholder")

        yaml_path = skill_file.parent / "agents" / "openai.yaml"
        if not yaml_path.is_file():
            errors.append(f"{_relative(yaml_path, root)}: skill UI metadata is missing")
            continue
        yaml_text = _read_text(yaml_path, errors, root)
        if yaml_text is None:
            continue
        display_name = _yaml_scalar(yaml_text, "display_name")
        short_description = _yaml_scalar(yaml_text, "short_description")
        prompt = _extract_default_prompt(yaml_text)
        if not display_name:
            errors.append(f"{_relative(yaml_path, root)}: display_name is required")
        if not 25 <= len(short_description) <= 64:
            errors.append(
                f"{_relative(yaml_path, root)}: short_description must be 25-64 characters"
            )
        if not prompt:
            errors.append(f"{_relative(yaml_path, root)}: default_prompt is required")
        elif name and f"${name}" not in prompt:
            errors.append(
                f"{_relative(yaml_path, root)}: default_prompt must mention ${name}"
            )

        referenced = set(
            re.findall(
                r"`((?:references|scripts|assets)/[A-Za-z0-9_.\-/]+)`", text
            )
        )
        for resource in sorted(referenced):
            resource_path = skill_file.parent / Path(resource)
            if not resource_path.is_file():
                errors.append(f"{relative}: referenced resource {resource} is missing")
        references_dir = skill_file.parent / "references"
        if references_dir.is_dir():
            for resource_path in sorted(references_dir.iterdir()):
                if resource_path.is_file():
                    resource = resource_path.relative_to(skill_file.parent).as_posix()
                    if resource not in text:
                        errors.append(
                            f"{relative}: bundled reference {resource} is not linked directly"
                        )


def _validate_claude_skill_bridges(root: Path, errors: list[str]) -> None:
    canonical = root / ".agents" / "skills"
    bridges = root / ".claude" / "skills"
    if not bridges.is_dir():
        errors.append(".claude/skills: Claude Code skill bridge directory is missing")
        return

    canonical_names = {path.parent.name for path in canonical.glob("*/SKILL.md")}
    bridge_names = {path.parent.name for path in bridges.glob("*/SKILL.md")}
    for name in sorted(canonical_names - bridge_names):
        errors.append(f".claude/skills/{name}/SKILL.md: bridge for canonical skill is missing")
    for name in sorted(bridge_names - canonical_names):
        errors.append(f".claude/skills/{name}/SKILL.md: no canonical .agents skill exists")

    for name in sorted(canonical_names & bridge_names):
        canonical_path = canonical / name / "SKILL.md"
        bridge_path = bridges / name / "SKILL.md"
        canonical_text = _read_text(canonical_path, errors, root)
        bridge_text = _read_text(bridge_path, errors, root)
        if canonical_text is None or bridge_text is None:
            continue
        try:
            canonical_metadata = _parse_frontmatter(canonical_text)
            bridge_metadata = _parse_frontmatter(bridge_text)
        except ValueError as exc:
            errors.append(f"{_relative(bridge_path, root)}: {exc}")
            continue
        for key in ("name", "description"):
            if bridge_metadata.get(key) != canonical_metadata.get(key):
                errors.append(
                    f"{_relative(bridge_path, root)}: {key} must match canonical skill metadata"
                )
        target = f"../../../.agents/skills/{name}/SKILL.md"
        if target not in bridge_text:
            errors.append(
                f"{_relative(bridge_path, root)}: must reference canonical target {target}"
            )
        if len(bridge_text) > 1_000:
            errors.append(
                f"{_relative(bridge_path, root)}: bridge is too large; keep workflow in .agents/skills"
            )
        expected = render_claude_skill_bridge(
            canonical_metadata.get("name", ""),
            canonical_metadata.get("description", ""),
        )
        if bridge_text != expected:
            errors.append(
                f"{_relative(bridge_path, root)}: must match the exact canonical bridge template"
            )
        if _has_placeholder(bridge_text):
            errors.append(f"{_relative(bridge_path, root)}: contains an unresolved TODO placeholder")


def _load_toml(path: Path, errors: list[str], root: Path) -> dict[str, Any] | None:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        errors.append(f"{_relative(path, root)}: invalid TOML ({exc})")
        return None


def _validate_codex_config(root: Path, errors: list[str]) -> None:
    config_path = root / ".codex" / "config.toml"
    if not config_path.is_file():
        errors.append(".codex/config.toml: project Codex configuration is missing")
        return
    config = _load_toml(config_path, errors, root)
    if config is None:
        return

    if config.get("model") != "gpt-5.6":
        errors.append(".codex/config.toml: project root model must be gpt-5.6")

    features = config.get("features")
    if not isinstance(features, dict):
        errors.append(".codex/config.toml: [features] table is required")
    else:
        for feature in ("multi_agent", "hooks"):
            if features.get(feature) is not True:
                errors.append(f".codex/config.toml: features.{feature} must be true")

    agents = config.get("agents")
    if not isinstance(agents, dict):
        errors.append(".codex/config.toml: [agents] table is required")
    else:
        if agents.get("max_threads") != 4:
            errors.append(".codex/config.toml: agents.max_threads must equal 4")
        if agents.get("max_depth") != 1:
            errors.append(".codex/config.toml: agents.max_depth must equal 1")
        runtime = agents.get("job_max_runtime_seconds")
        if not isinstance(runtime, int) or isinstance(runtime, bool) or not 1 <= runtime <= 900:
            errors.append(
                ".codex/config.toml: agents.job_max_runtime_seconds must be an integer from 1 to 900"
            )

    hooks_path = root / ".codex" / "hooks.json"
    if not hooks_path.is_file():
        errors.append(".codex/hooks.json: enabled Stop-hook configuration is missing")
        return
    try:
        hooks_data = json.loads(hooks_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f".codex/hooks.json: invalid JSON ({exc})")
        return
    commands = _command_hooks(hooks_data, "Stop")
    if not commands:
        errors.append(".codex/hooks.json: a command-based Stop hook is required")
    else:
        command_text = "\n".join(
            str(command.get(key, ""))
            for command in commands
            for key in ("command", "commandWindows")
        )
        if "scripts/verify_changes.py" not in command_text or "--hook" not in command_text:
            errors.append(
                ".codex/hooks.json: Stop hook must invoke scripts/verify_changes.py --hook"
            )


def _validate_custom_agents(root: Path, errors: list[str]) -> None:
    agents_dir = root / ".codex" / "agents"
    for filename in REQUIRED_CUSTOM_AGENTS:
        if not (agents_dir / filename).is_file():
            errors.append(f".codex/agents/{filename}: required custom agent is missing")

    if not agents_dir.is_dir():
        return
    for path in sorted(agents_dir.glob("*.toml")):
        data = _load_toml(path, errors, root)
        if data is None:
            continue
        relative = _relative(path, root)
        stale = sorted(STALE_AGENT_KEYS.intersection(data))
        if stale:
            replacement = {
                "instructions": "developer_instructions",
                "reasoning_effort": "model_reasoning_effort",
            }
            rendered = ", ".join(f"{key}->{replacement[key]}" for key in stale)
            errors.append(f"{relative}: stale custom-agent keys ({rendered})")
        instructions = data.get("developer_instructions")
        if not isinstance(instructions, str) or len(instructions.strip()) < 40:
            errors.append(f"{relative}: developer_instructions must be actionable")
        elif _has_placeholder(instructions):
            errors.append(f"{relative}: developer_instructions contains a TODO placeholder")
        description = data.get("description")
        if not isinstance(description, str) or len(description.strip()) < 12:
            errors.append(f"{relative}: description must be informative")
        effort = data.get("model_reasoning_effort")
        if effort not in VALID_REASONING_EFFORT:
            errors.append(
                f"{relative}: model_reasoning_effort must be one of "
                + ", ".join(sorted(VALID_REASONING_EFFORT))
            )
        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"{relative}: name must be a non-empty string")
        elif name != path.stem:
            errors.append(f"{relative}: name must match filename stem {path.stem!r}")
        expected_model = EXPECTED_AGENT_MODELS.get(path.stem)
        if expected_model is not None and data.get("model") != expected_model:
            errors.append(f"{relative}: model must be {expected_model}")
        if path.stem in READ_ONLY_CODEX_AGENTS and data.get("sandbox_mode") != "read-only":
            errors.append(f"{relative}: read-only role must use sandbox_mode = 'read-only'")
        if path.stem in WRITING_CODEX_AGENTS and data.get("sandbox_mode") != "workspace-write":
            errors.append(f"{relative}: writing role must use sandbox_mode = 'workspace-write'")


def _validate_claude_settings(root: Path, errors: list[str]) -> None:
    settings_path = root / ".claude" / "settings.json"
    if not settings_path.is_file():
        errors.append(".claude/settings.json: project Claude Code settings are missing")
        return
    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f".claude/settings.json: invalid JSON ({exc})")
        return
    if not isinstance(settings, dict):
        errors.append(".claude/settings.json: top-level JSON value must be an object")
        return
    if settings.get("disableAllHooks") is True:
        errors.append(".claude/settings.json: shared hooks must not be disabled")
    if (root / ".claude" / "hooks.json").exists():
        errors.append(".claude/hooks.json: Claude project hooks belong in settings.json")

    commands = _command_hooks(settings, "Stop")
    if not commands:
        errors.append(".claude/settings.json: a command-based Stop hook is required")
        return

    valid = False
    for command in commands:
        args = command.get("args")
        if not isinstance(args, list) or not all(isinstance(value, str) for value in args):
            continue
        rendered = "\n".join([str(command.get("command", "")), *args])
        timeout = command.get("timeout", 600)
        if (
            "${CLAUDE_PROJECT_DIR}/scripts/verify_changes.py" in rendered
            and "--hook" in args
            and "--hook-platform" in args
            and "claude" in args
            and "--repo" in args
            and "${CLAUDE_PROJECT_DIR}" in args
            and isinstance(timeout, int)
            and not isinstance(timeout, bool)
            and 1 <= timeout <= 600
        ):
            valid = True
            break
    if not valid:
        errors.append(
            ".claude/settings.json: Stop hook must use exec-form args to invoke "
            "verify_changes.py --hook --hook-platform claude from ${CLAUDE_PROJECT_DIR}"
        )


def _validate_claude_agents(root: Path, errors: list[str]) -> None:
    agents_dir = root / ".claude" / "agents"
    for filename in REQUIRED_CLAUDE_AGENTS:
        if not (agents_dir / filename).is_file():
            errors.append(f".claude/agents/{filename}: required Claude role is missing")
    if not agents_dir.is_dir():
        return

    for path in sorted(agents_dir.glob("*.md")):
        relative = _relative(path, root)
        text = _read_text(path, errors, root)
        if text is None:
            continue
        try:
            metadata = _parse_frontmatter(text)
        except ValueError as exc:
            errors.append(f"{relative}: {exc}")
            continue
        name = metadata.get("name", "").strip()
        description = metadata.get("description", "").strip()
        if name != path.stem:
            errors.append(f"{relative}: name must match filename stem {path.stem!r}")
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
            errors.append(f"{relative}: name must be lowercase kebab-case")
        if len(description) < 20:
            errors.append(f"{relative}: description must be informative")
        expected_model = EXPECTED_CLAUDE_AGENT_MODELS.get(name)
        if expected_model is not None:
            actual_model = metadata.get("model", "").strip()
            if not actual_model:
                errors.append(
                    f"{relative}: model is required and must be {expected_model}"
                )
            elif actual_model != expected_model:
                errors.append(f"{relative}: model must be {expected_model}")
        if len(text.strip()) < 160:
            errors.append(f"{relative}: role instructions are too short to be actionable")
        if _has_placeholder(text):
            errors.append(f"{relative}: contains an unresolved TODO placeholder")

        tools = {
            value.strip()
            for value in metadata.get("tools", "").split(",")
            if value.strip()
        }
        permission_mode = metadata.get("permissionMode")
        if name in READ_ONLY_CLAUDE_AGENTS:
            forbidden = {"Edit", "Write", "NotebookEdit"}.intersection(tools)
            if forbidden:
                errors.append(f"{relative}: read-only role exposes write tools {sorted(forbidden)}")
            if permission_mode != "plan":
                errors.append(f"{relative}: read-only role must use permissionMode: plan")
        if name in WRITING_CLAUDE_AGENTS:
            if not {"Edit", "Write"}.issubset(tools):
                errors.append(f"{relative}: writing role must expose Edit and Write")
            if permission_mode != "default":
                errors.append(f"{relative}: writing role must use permissionMode: default")


def validate_repo(root: Path) -> list[str]:
    """Return deterministic validation errors for *root*."""
    root = root.resolve()
    errors: list[str] = []
    _validate_agent_guides(root, errors)
    _validate_claude_instruction_bridges(root, errors)
    _validate_skills(root, errors)
    _validate_claude_skill_bridges(root, errors)
    _validate_codex_config(root, errors)
    _validate_custom_agents(root, errors)
    _validate_claude_settings(root, errors)
    _validate_claude_agents(root, errors)

    gitignore = root / ".gitignore"
    if gitignore.is_file():
        text = _read_text(gitignore, errors, root)
        if text is not None:
            ignored = {line.strip().rstrip("/") for line in text.splitlines()}
            for entry, reason in (
                (".tmp", "verifier state"),
                ("CLAUDE.local.md", "personal Claude instructions"),
                (".claude/settings.local.json", "personal Claude settings"),
            ):
                if entry not in ignored:
                    errors.append(f".gitignore: {entry} must be ignored for {reason}")
    else:
        errors.append(".gitignore: missing (verifier and local agent state require ignores)")
    return sorted(set(errors))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    errors = validate_repo(args.repo)
    result = {"ok": not errors, "error_count": len(errors), "errors": errors}
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif errors:
        print("Agent setup validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
    else:
        print("Agent setup validation passed.")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
