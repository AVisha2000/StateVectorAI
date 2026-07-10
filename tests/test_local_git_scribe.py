from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import local_git_scribe as scribe


def test_local_url_accepts_only_loopback_hosts() -> None:
    assert scribe.validate_local_url("http://127.0.0.1:11434") == "http://127.0.0.1:11434"
    assert scribe.validate_local_url("http://localhost:11434/api") == "http://localhost:11434/api"
    assert scribe.validate_local_url("http://[::1]:11434") == "http://[::1]:11434"

    for unsafe in (
        "https://127.0.0.1:11434",
        "http://ollama.example.com:11434",
        "http://192.168.1.20:11434",
        "http://user:password@localhost:11434",
    ):
        with pytest.raises(ValueError):
            scribe.validate_local_url(unsafe)


def test_diff_cap_is_deterministic_and_preserves_both_ends() -> None:
    diff = "start\n" + "x" * 10_000 + "\nend"
    bounded, truncated = scribe.cap_diff(diff, 2_000)
    assert truncated is True
    assert len(bounded) == 2_000
    assert bounded.startswith("start")
    assert bounded.endswith("end")
    assert "DIFF TRUNCATED" in bounded


def test_normalize_proposal_builds_conventional_commit_message() -> None:
    proposal = scribe.normalize_proposal(
        {
            "type": "Test",
            "scope": "Agents",
            "subject": "add deterministic workflow checks.",
            "body": ["Validate project instructions", "Keep Git state unchanged"],
            "breaking_change": None,
            "confidence": "high",
        }
    )
    assert proposal["type"] == "test"
    assert proposal["scope"] == "agents"
    assert proposal["subject"] == "add deterministic workflow checks"
    assert proposal["commit_message"].startswith("test(agents): add deterministic workflow checks")


def test_read_diff_defaults_to_staged_and_uses_only_read_only_git(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_git(repo: Path, arguments: tuple[str, ...]) -> bytes:
        calls.append(arguments)
        if "--name-only" in arguments:
            return b"scripts/example.py\0"
        return b"diff --git a/scripts/example.py b/scripts/example.py\n+safe = True\n"

    monkeypatch.setattr(scribe, "_run_git", fake_git)
    diff, files = scribe.read_diff(Path("."))

    assert "+safe = True" in diff
    assert files == ["scripts/example.py"]
    assert calls == [
        ("diff", "--cached", "--no-ext-diff", "--unified=3", "--"),
        ("diff", "--cached", "--name-only", "-z", "--"),
    ]
    forbidden = {"add", "commit", "push", "reset", "checkout", "tag"}
    assert not forbidden.intersection(word for call in calls for word in call)


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, _limit: int) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class _FakeOpener:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.request: Any = None

    def open(self, request: Any, timeout: float) -> _FakeResponse:
        self.request = request
        assert timeout == 5.0
        return _FakeResponse(self.response)


def test_request_proposal_uses_local_generate_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    response_text = json.dumps(
        {
            "type": "chore",
            "scope": "agents",
            "subject": "tighten local workflow",
            "body": [],
            "breaking_change": None,
            "confidence": "medium",
        }
    )
    opener = _FakeOpener({"response": response_text})
    monkeypatch.setattr(scribe, "_local_opener", lambda: opener)

    proposal = scribe.request_proposal(
        base_url="http://127.0.0.1:11434", model="tiny-local", prompt="bounded", timeout=5.0
    )

    assert proposal["commit_message"] == "chore(agents): tighten local workflow"
    assert opener.request.full_url == "http://127.0.0.1:11434/api/generate"
    payload = json.loads(opener.request.data)
    assert payload["stream"] is False
    assert payload["format"] == "json"


def test_check_mode_does_not_read_a_diff(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        scribe,
        "check_ollama",
        lambda url, timeout: {"ok": True, "url": url, "models": ["tiny-local"]},
    )

    def unexpected_diff(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("--check must not read Git state")

    monkeypatch.setattr(scribe, "read_diff", unexpected_diff)
    assert scribe.main(["--check", "--timeout", "2"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is True
    assert output["models"] == ["tiny-local"]
