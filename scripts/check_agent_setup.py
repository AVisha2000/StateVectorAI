#!/usr/bin/env python3
"""Validate the repository's Codex instructions, skills, and custom agents.

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
)
REQUIRED_CUSTOM_AGENTS = ("planner.toml", "explorer.toml", "verifier.toml")
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


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


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

    discovered: list[Path] = []
    for directory, directories, filenames in os.walk(root):
        directories[:] = [name for name in directories if name not in IGNORED_PARTS]
        if "AGENTS.md" in filenames:
            discovered.append(Path(directory) / "AGENTS.md")

    for path in sorted(discovered):
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
        prompt = _extract_default_prompt(yaml_text)
        if not prompt:
            errors.append(f"{_relative(yaml_path, root)}: default_prompt is required")
        elif name and f"${name}" not in prompt:
            errors.append(
                f"{_relative(yaml_path, root)}: default_prompt must mention ${name}"
            )


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
    try:
        stop_groups = hooks_data["hooks"]["Stop"]
        commands = [
            hook
            for group in stop_groups
            for hook in group.get("hooks", [])
            if isinstance(hook, dict) and hook.get("type") == "command"
        ]
    except (KeyError, TypeError):
        commands = []
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


def validate_repo(root: Path) -> list[str]:
    """Return deterministic validation errors for *root*."""
    root = root.resolve()
    errors: list[str] = []
    _validate_agent_guides(root, errors)
    _validate_skills(root, errors)
    _validate_codex_config(root, errors)
    _validate_custom_agents(root, errors)

    gitignore = root / ".gitignore"
    if gitignore.is_file():
        text = _read_text(gitignore, errors, root)
        if text is not None and not any(
            line.strip().rstrip("/") == ".tmp" for line in text.splitlines()
        ):
            errors.append(".gitignore: .tmp/ must be ignored for verifier state")
    else:
        errors.append(".gitignore: missing (verifier state requires ignored .tmp/)")
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
