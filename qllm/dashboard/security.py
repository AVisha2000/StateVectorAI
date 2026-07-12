"""Local-access and filesystem-boundary policy for the dashboard."""
from __future__ import annotations

import ipaddress
import json
import os
import re
from pathlib import Path
from urllib.parse import urlsplit


LOOPBACK_ORIGIN_REGEX = (
    r"^https?://(?:localhost|127\.0\.0\.1|\[::1\])(?::[0-9]{1,5})?$"
)
_HF_HUB_DATASET_ID = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?"
    r"(?:/[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?)?$"
)


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def is_loopback_host(host: str | None) -> bool:
    value = str(host or "").strip().strip("[]").lower()
    if value == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def validate_cors_origin(origin: str, *, allow_remote: bool) -> str:
    value = str(origin or "").strip().rstrip("/")
    if not value or "*" in value:
        raise ValueError("CORS origins must be explicit and cannot contain wildcards.")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"Invalid CORS origin '{origin}'.")
    if parsed.path or parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ValueError(
            f"CORS origin must contain only scheme, host, and port: '{origin}'."
        )
    if not allow_remote and not is_loopback_host(parsed.hostname):
        raise ValueError("Non-loopback CORS origins require --allow-remote.")
    return value


def configure_access(*, host: str, allow_remote: bool, cors_origins: list[str]) -> None:
    if not is_loopback_host(host) and not allow_remote:
        raise ValueError(
            f"Refusing non-loopback bind '{host}'. Pass --allow-remote to expose "
            "the dashboard deliberately."
        )
    origins = [
        validate_cors_origin(value, allow_remote=allow_remote)
        for value in cors_origins
    ]
    if allow_remote and not origins:
        raise ValueError("--allow-remote requires at least one explicit --cors-origin.")
    os.environ["QLLM_ALLOW_REMOTE"] = "1" if allow_remote else "0"
    os.environ["QLLM_CORS_ORIGINS"] = json.dumps(origins)


def remote_access_enabled() -> bool:
    return _truthy(os.environ.get("QLLM_ALLOW_REMOTE"))


def client_access_allowed(client_host: str | None) -> bool:
    return (
        remote_access_enabled()
        or client_host == "testclient"
        or is_loopback_host(client_host)
    )


def configured_cors_origins() -> list[str]:
    raw = os.environ.get("QLLM_CORS_ORIGINS", "[]")
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("QLLM_CORS_ORIGINS must be a JSON list.") from exc
    if not isinstance(values, list):
        raise RuntimeError("QLLM_CORS_ORIGINS must be a JSON list.")
    allow_remote = remote_access_enabled()
    return [
        validate_cors_origin(str(value), allow_remote=allow_remote)
        for value in values
    ]


def request_origin_allowed(origin: str) -> bool:
    """Return whether an unsafe browser request may originate from *origin*."""
    value = str(origin or "").strip().rstrip("/")
    return bool(
        re.fullmatch(LOOPBACK_ORIGIN_REGEX, value)
        or value in configured_cors_origins()
    )


def json_media_type(content_type: str | None) -> bool:
    """Accept JSON and structured-suffix JSON request media types."""
    media_type = str(content_type or "").split(";", 1)[0].strip().lower()
    return media_type == "application/json" or (
        media_type.startswith("application/") and media_type.endswith("+json")
    )


def access_status() -> dict:
    remote = remote_access_enabled()
    return {
        "mode": "remote" if remote else "loopback-only",
        "remote_access": remote,
        "cors_origins": configured_cors_origins(),
        "warning": (
            "REMOTE ACCESS ENABLED: the dashboard can expose local research data."
            if remote
            else None
        ),
    }


def is_direct_dataset_url(source: str) -> bool:
    return str(source or "").strip().lower().startswith(
        ("http://", "https://", "hf://")
    )


def is_hf_hub_dataset_id(source: str) -> bool:
    """Accept only conservative Hub repo IDs for unauthenticated remote import."""
    value = str(source or "").strip()
    return bool(_HF_HUB_DATASET_ID.fullmatch(value)) and not any(
        part in {".", ".."} for part in value.split("/")
    )


def resolve_within(
    root: str | Path,
    value: str | Path,
    *,
    label: str = "path",
    allow_absolute: bool = True,
) -> Path:
    """Resolve *value* and require it to remain inside *root*, including symlinks."""
    root_path = Path(root).resolve()
    raw = str(value)
    if not raw or "\x00" in raw:
        raise ValueError(f"Invalid {label}.")
    candidate_input = Path(raw)
    if candidate_input.is_absolute() and not allow_absolute:
        raise ValueError(f"Absolute {label} values are not allowed.")
    candidate = (
        candidate_input.resolve()
        if candidate_input.is_absolute()
        else (root_path / candidate_input).resolve()
    )
    try:
        candidate.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(f"{label.capitalize()} must stay within '{root_path}'.") from exc
    return candidate


def resolve_data_path(
    root: str | Path,
    value: str | Path,
    *,
    label: str,
) -> Path:
    """Authorize dashboard corpus paths with a legacy ``data/`` adapter.

    Absolute paths must already be beneath the configured root. Relative paths
    historically used either cwd-relative ``data/...`` or data-root-relative
    ``imported/...`` forms; both normalize to the selected data root.
    """
    root_path = Path(root).resolve()
    candidate = Path(value)
    if candidate.is_absolute():
        return resolve_within(root_path, candidate, label=label)
    cwd_candidate = candidate.resolve()
    try:
        cwd_candidate.relative_to(root_path)
    except ValueError:
        parts = candidate.parts
        if parts and parts[0].lower() == "data":
            parts = parts[1:]
        candidate = root_path.joinpath(*parts)
    else:
        candidate = cwd_candidate
    return resolve_within(root_path, candidate, label=label)


def resolve_web_asset(root: str | Path, value: str) -> Path:
    raw = str(value or "")
    if "\\" in raw or Path(raw).is_absolute() or raw.startswith(("/", "//")):
        raise ValueError("Invalid asset path.")
    return resolve_within(root, raw, label="asset path", allow_absolute=False)
