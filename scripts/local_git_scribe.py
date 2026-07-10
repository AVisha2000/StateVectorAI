#!/usr/bin/env python3
"""Ask a loopback-only Ollama model to propose a commit message.

By default the scribe reads only the already-staged diff. It never stages,
commits, pushes, rebases, tags, or otherwise mutates Git state.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Sequence


DEFAULT_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5-coder:3b"
DEFAULT_MAX_CHARS = 60_000
HARD_MAX_CHARS = 120_000
MAX_RESPONSE_BYTES = 256_000
ALLOWED_TYPES = {
    "build",
    "chore",
    "ci",
    "docs",
    "feat",
    "fix",
    "perf",
    "refactor",
    "revert",
    "style",
    "test",
}
ALLOWED_CONFIDENCE = {"low", "medium", "high"}


class ScribeError(RuntimeError):
    """A safe, user-facing local scribe failure."""


def validate_local_url(url: str) -> str:
    """Return a normalized Ollama base URL, rejecting every non-loopback host."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "http":
        raise ValueError("Ollama URL must use http on the local machine")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Ollama URL must not contain credentials, a query, or a fragment")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Ollama URL must include a host")
    if hostname.casefold() != "localhost":
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError as exc:
            raise ValueError("Ollama host must be localhost or a loopback IP") from exc
        if not address.is_loopback:
            raise ValueError("Ollama host must be localhost or a loopback IP")
    path = parsed.path.rstrip("/")
    if path and path != "/api":
        raise ValueError("Ollama base URL path may only be /api")
    netloc = parsed.netloc
    return urllib.parse.urlunsplit(("http", netloc, path, "", "")).rstrip("/")


def _endpoint(base_url: str, endpoint: str) -> str:
    base = validate_local_url(base_url)
    if base.endswith("/api"):
        return f"{base}/{endpoint.lstrip('/')}"
    return f"{base}/api/{endpoint.lstrip('/')}"


def cap_diff(diff: str, max_chars: int = DEFAULT_MAX_CHARS) -> tuple[str, bool]:
    """Bound model input while preserving both the start and end of a diff."""
    if max_chars < 2_000 or max_chars > HARD_MAX_CHARS:
        raise ValueError(f"max_chars must be between 2000 and {HARD_MAX_CHARS}")
    if len(diff) <= max_chars:
        return diff, False
    marker = "\n\n... DIFF TRUNCATED BY LOCAL SCRIBE ...\n\n"
    available = max_chars - len(marker)
    head = int(available * 0.75)
    tail = available - head
    return diff[:head] + marker + diff[-tail:], True


def _run_git(repo: Path, arguments: Sequence[str]) -> bytes:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ScribeError("git executable was not found") from exc
    if completed.returncode:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ScribeError(f"read-only git command failed: {message or 'unknown error'}")
    return completed.stdout


def read_diff(repo: Path, *, include_unstaged: bool = False) -> tuple[str, list[str]]:
    """Read a staged diff by default, using only non-mutating Git commands."""
    if include_unstaged:
        diff_args = ("diff", "--no-ext-diff", "--unified=3", "HEAD", "--")
        names_args = ("diff", "--name-only", "-z", "HEAD", "--")
    else:
        diff_args = ("diff", "--cached", "--no-ext-diff", "--unified=3", "--")
        names_args = ("diff", "--cached", "--name-only", "-z", "--")
    raw_diff = _run_git(repo, diff_args)
    raw_names = _run_git(repo, names_args)
    diff = raw_diff.decode("utf-8", errors="replace")
    files = sorted(
        name.decode("utf-8", errors="replace").replace("\\", "/")
        for name in raw_names.split(b"\0")
        if name
    )
    if not diff.strip():
        source = "tracked staged or unstaged" if include_unstaged else "staged"
        raise ScribeError(f"no {source} diff is available")
    return diff, files


def build_prompt(diff: str, files: Sequence[str], *, truncated: bool) -> str:
    file_lines = "\n".join(f"- {path}" for path in files[:200])
    if len(files) > 200:
        file_lines += f"\n- ... and {len(files) - 200} more files"
    truncation_note = (
        "The diff was truncated. Lower confidence and avoid claims about omitted details."
        if truncated
        else "The complete textual diff is present."
    )
    return f"""You are a commit-message scribe. Analyze only the supplied diff.
Do not claim tests passed unless the diff itself proves that. Do not give shell commands.
Return exactly one JSON object with these fields:
- type: one of {', '.join(sorted(ALLOWED_TYPES))}
- scope: a short lowercase scope or null
- subject: imperative summary without a conventional-commit prefix, at most 72 characters
- body: an array of zero to five concise factual strings
- breaking_change: a concise string or null
- confidence: low, medium, or high

{truncation_note}

Changed files:
{file_lines or '- unavailable'}

Diff:
---
{diff}
---
"""


def _strip_json_fence(value: str) -> str:
    stripped = value.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1) if match else stripped


def normalize_proposal(value: str | dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize the model's structured commit proposal."""
    if isinstance(value, str):
        try:
            parsed = json.loads(_strip_json_fence(value))
        except json.JSONDecodeError as exc:
            raise ScribeError("local model did not return valid JSON") from exc
    else:
        parsed = value
    if not isinstance(parsed, dict):
        raise ScribeError("local model proposal must be a JSON object")

    commit_type = str(parsed.get("type", "")).strip().casefold()
    if commit_type not in ALLOWED_TYPES:
        raise ScribeError("local model returned an unsupported commit type")

    raw_scope = parsed.get("scope")
    scope = None if raw_scope is None else str(raw_scope).strip().casefold()
    if scope == "":
        scope = None
    if scope is not None and not re.fullmatch(r"[a-z0-9][a-z0-9._/-]{0,29}", scope):
        raise ScribeError("local model returned an invalid commit scope")

    subject = " ".join(str(parsed.get("subject", "")).split())
    if not subject or len(subject) > 72 or "\n" in subject:
        raise ScribeError("local model subject must contain 1-72 characters")
    subject = subject.rstrip(".")

    raw_body = parsed.get("body", [])
    if not isinstance(raw_body, list) or len(raw_body) > 5:
        raise ScribeError("local model body must be an array of at most five items")
    body: list[str] = []
    for item in raw_body:
        line = " ".join(str(item).split())
        if not line or len(line) > 240:
            raise ScribeError("each commit body item must contain 1-240 characters")
        body.append(line)

    raw_breaking = parsed.get("breaking_change")
    breaking = None if raw_breaking is None else " ".join(str(raw_breaking).split())
    if breaking == "":
        breaking = None
    if breaking is not None and len(breaking) > 240:
        raise ScribeError("breaking_change must be at most 240 characters")

    confidence = str(parsed.get("confidence", "low")).strip().casefold()
    if confidence not in ALLOWED_CONFIDENCE:
        raise ScribeError("confidence must be low, medium, or high")

    header = f"{commit_type}{f'({scope})' if scope else ''}: {subject}"
    message_parts = [header]
    if body:
        message_parts.append("\n".join(body))
    if breaking:
        message_parts.append(f"BREAKING CHANGE: {breaking}")
    return {
        "type": commit_type,
        "scope": scope,
        "subject": subject,
        "body": body,
        "breaking_change": breaking,
        "confidence": confidence,
        "commit_message": "\n\n".join(message_parts),
    }


def _local_opener() -> urllib.request.OpenerDirector:
    # Explicitly bypass proxy environment variables: requests must stay local.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    return urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())


def _read_json_response(response: Any) -> dict[str, Any]:
    raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ScribeError("Ollama response exceeded the safety limit")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ScribeError("Ollama returned an invalid JSON response") from exc
    if not isinstance(value, dict):
        raise ScribeError("Ollama response must be a JSON object")
    return value


def check_ollama(base_url: str, timeout: float = 3.0) -> dict[str, Any]:
    request = urllib.request.Request(_endpoint(base_url, "tags"), method="GET")
    try:
        with _local_opener().open(request, timeout=timeout) as response:
            payload = _read_json_response(response)
    except (OSError, urllib.error.URLError) as exc:
        raise ScribeError(f"local Ollama service is unavailable: {exc}") from exc
    models = [
        model.get("name")
        for model in payload.get("models", [])
        if isinstance(model, dict) and isinstance(model.get("name"), str)
    ]
    return {"ok": True, "url": validate_local_url(base_url), "models": models}


def request_proposal(
    *,
    base_url: str,
    model: str,
    prompt: str,
    timeout: float = 60.0,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1},
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        _endpoint(base_url, "generate"),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _local_opener().open(request, timeout=timeout) as response:
            result = _read_json_response(response)
    except (OSError, urllib.error.URLError) as exc:
        raise ScribeError(f"local Ollama request failed: {exc}") from exc
    response_text = result.get("response")
    if not isinstance(response_text, str):
        raise ScribeError("Ollama response did not include generated text")
    return normalize_proposal(response_text)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument(
        "--url",
        default=os.environ.get("QLLM_OLLAMA_URL", DEFAULT_URL),
        help="loopback Ollama base URL",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("QLLM_OLLAMA_MODEL", DEFAULT_MODEL),
        help="installed local Ollama model",
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="local request timeout")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help="maximum diff characters sent to the model",
    )
    parser.add_argument(
        "--include-unstaged",
        action="store_true",
        help="explicitly read tracked unstaged changes too; untracked files remain excluded",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="check Ollama availability without reading a diff or generating a proposal",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        validate_local_url(args.url)
        if args.timeout <= 0:
            raise ValueError("timeout must be positive")
        if args.check:
            result = check_ollama(args.url, timeout=min(args.timeout, 10.0))
        else:
            diff, files = read_diff(args.repo.resolve(), include_unstaged=args.include_unstaged)
            bounded_diff, truncated = cap_diff(diff, args.max_chars)
            prompt = build_prompt(bounded_diff, files, truncated=truncated)
            proposal = request_proposal(
                base_url=args.url,
                model=args.model,
                prompt=prompt,
                timeout=args.timeout,
            )
            result = {
                "ok": True,
                "source": "tracked_changes" if args.include_unstaged else "staged_diff",
                "model": args.model,
                "files": files,
                "diff_chars": len(diff),
                "input_chars": len(bounded_diff),
                "truncated": truncated,
                "proposal": proposal,
                "mutated_git": False,
            }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ScribeError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
