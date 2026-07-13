from __future__ import annotations

import importlib
import re
import sys
import types
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from qllm.dashboard.model_tests import _artifact_dir, model_test_payload
from qllm.dashboard.runner import ExperimentQueue
from qllm.dashboard import run as dashboard_run
from qllm.dashboard.static_frontend import mount_frontend
from qllm.dashboard.security import (
    access_status,
    client_access_allowed,
    configure_access,
    configured_cors_origins,
    is_loopback_host,
    is_hf_hub_dataset_id,
    remote_access_enabled,
    resolve_data_path,
    resolve_web_asset,
    resolve_within,
)


@pytest.fixture(autouse=True)
def _local_access(monkeypatch):
    monkeypatch.setenv("QLLM_ALLOW_REMOTE", "0")
    monkeypatch.setenv("QLLM_CORS_ORIGINS", "[]")


def test_loopback_detection_and_default_client_boundary(monkeypatch):
    assert is_loopback_host("localhost")
    assert is_loopback_host("127.0.0.1")
    assert is_loopback_host("::1")
    assert client_access_allowed("127.0.0.1")
    assert not client_access_allowed(None)
    assert not client_access_allowed("192.0.2.10")
    assert access_status()["mode"] == "loopback-only"
    for value in ("0", "false", "off", "no"):
        monkeypatch.setenv("QLLM_ALLOW_REMOTE", value)
        assert remote_access_enabled() is False


def test_loopback_cors_regex_is_anchored_and_rejects_hostile_origins():
    from qllm.dashboard.security import LOOPBACK_ORIGIN_REGEX

    for origin in (
        "http://localhost:5173",
        "https://127.0.0.1:8000",
        "http://[::1]:9000",
    ):
        assert re.fullmatch(LOOPBACK_ORIGIN_REGEX, origin)
    for origin in (
        "null",
        "https://evil.example",
        "http://localhost.evil.example:5173",
    ):
        assert re.fullmatch(LOOPBACK_ORIGIN_REGEX, origin) is None


def test_remote_bind_requires_opt_in_and_explicit_origins(monkeypatch):
    with pytest.raises(ValueError, match="Refusing non-loopback bind"):
        configure_access(host="0.0.0.0", allow_remote=False, cors_origins=[])
    with pytest.raises(ValueError, match="at least one explicit"):
        configure_access(host="0.0.0.0", allow_remote=True, cors_origins=[])
    with pytest.raises(ValueError, match="wildcards"):
        configure_access(
            host="0.0.0.0", allow_remote=True, cors_origins=["http://*"]
        )

    configure_access(
        host="0.0.0.0",
        allow_remote=True,
        cors_origins=["https://lab.example:8443"],
    )
    assert configured_cors_origins() == ["https://lab.example:8443"]
    assert client_access_allowed("192.0.2.10")
    assert access_status()["warning"].startswith("REMOTE ACCESS ENABLED")


def test_launcher_refuses_unguarded_bind_and_warns_for_remote(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["qllm-dashboard", "--host", "0.0.0.0", "--results", str(tmp_path)],
    )
    with pytest.raises(SystemExit) as exc:
        dashboard_run.main()
    assert exc.value.code == 2

    called = {}

    def fake_run(app, **kwargs):
        called.update({"app": app, **kwargs})

    monkeypatch.setitem(sys.modules, "uvicorn", types.SimpleNamespace(run=fake_run))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qllm-dashboard",
            "--host",
            "0.0.0.0",
            "--results",
            str(tmp_path),
            "--allow-remote",
            "--cors-origin",
            "https://lab.example",
        ],
    )
    dashboard_run.main()
    assert called["host"] == "0.0.0.0"
    assert "WARNING: REMOTE ACCESS ENABLED" in capsys.readouterr().out


def test_path_confinement_rejects_traversal_absolute_and_backslash(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    assert resolve_web_asset(root, "plot.png") == (root / "plot.png").resolve()
    for unsafe in ("../secret.png", "/etc/passwd", r"..\secret.png"):
        with pytest.raises(ValueError):
            resolve_web_asset(root, unsafe)
    with pytest.raises(ValueError, match="must stay within"):
        resolve_within(root, tmp_path / "outside.msgpack", label="checkpoint")


def test_remote_dataset_ids_exclude_local_paths_and_schemes():
    assert is_hf_hub_dataset_id("organization/public-dataset")
    assert is_hf_hub_dataset_id("public_dataset")
    for source in (
        "../private",
        r"..\private",
        "/etc/passwd",
        r"C:\private",
        "file:///etc/passwd",
        "https://example.invalid/data.txt",
        "org/name/extra",
    ):
        assert not is_hf_hub_dataset_id(source)


def test_path_confinement_rejects_symlink_escape_when_supported(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = root / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable in this Windows environment")
    with pytest.raises(ValueError, match="must stay within"):
        resolve_within(root, link / "artifact.json", label="artifact")
    with pytest.raises(ValueError, match="must stay within"):
        resolve_data_path(root, "escape/imported.txt", label="dataset import")

    queue = ExperimentQueue(
        str(tmp_path / "symlink.db"), start_worker=False, results_dir=root
    )
    run_dir = root / "run"
    run_dir.mkdir()
    (run_dir / "checkpoints").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="must stay within"):
        queue._authorized_artifact_layout(run_dir)

    other_run = root / "other-run"
    other_run.mkdir()
    manifest_target = other_run / "manifest.json"
    manifest_target.write_text("outside this run")
    manifest_run = root / "manifest-run"
    manifest_run.mkdir()
    (manifest_run / "manifest.json").symlink_to(manifest_target)
    with pytest.raises(ValueError, match="must stay within"):
        queue._authorized_artifact_layout(manifest_run)


def test_queue_rejects_submitted_and_persisted_paths_outside_results(tmp_path):
    root = tmp_path / "results"
    queue = ExperimentQueue(
        str(tmp_path / "queue.db"), start_worker=False, results_dir=root
    )
    with pytest.raises(ValueError, match="Artifact directory must stay within"):
        queue.submit(
            "classical-small",
            "default-text",
            "unsafe",
            0,
            1,
            1,
            device_target="cpu",
            artifact_dir=str(tmp_path / "outside"),
        )
    assert queue._recoverable_checkpoint(
        {"artifact_dir": str(tmp_path / "legacy-outside")}
    ) is None
    with pytest.raises(ValueError, match="Persisted artifact directory"):
        _artifact_dir(root, {"artifact_dir": str(tmp_path / "legacy-outside")})


def test_queue_rejects_dataset_corpus_outside_configured_data_root(
    monkeypatch, tmp_path
):
    data_root = tmp_path / "data"
    queue = ExperimentQueue(
        str(tmp_path / "dataset-queue.db"),
        start_worker=False,
        results_dir=tmp_path / "results",
        data_dir=data_root,
    )
    outside = tmp_path / "private.txt"
    outside.write_text("private")
    monkeypatch.setattr(
        "qllm.dashboard.runner.get_dataset",
        lambda _db, _name: {"corpus_path": str(outside)},
    )
    with pytest.raises(ValueError, match="Dataset corpus path must stay within"):
        queue.submit("classical-small", "unsafe", "unsafe-data", 0, 1, 1)


def test_custom_data_root_materializes_default_dataset_under_that_root(tmp_path):
    data_root = tmp_path / "corpora"
    queue = ExperimentQueue(
        str(tmp_path / "custom-data.db"),
        start_worker=False,
        results_dir=tmp_path / "results",
        data_dir=data_root,
    )
    job = queue.submit("classical-small", "default-text", "custom-data", 0, 1, 1)
    assert Path(job["config"]["data.corpus_path"]) == (
        data_root / "input.txt"
    ).resolve()


def test_persisted_unsafe_corpus_is_not_tested_or_executed(tmp_path):
    data_root = tmp_path / "data"
    results_root = tmp_path / "results"
    queue = ExperimentQueue(
        str(tmp_path / "persisted-data.db"),
        start_worker=False,
        results_dir=results_root,
        data_dir=data_root,
    )
    job = queue.submit("classical-small", "default-text", "persisted-data", 0, 1, 1)
    bad_config = dict(job["config"])
    bad_config["data.corpus_path"] = str(tmp_path / "private.txt")
    queue.db().update_lab_job(job["id"], config=bad_config)

    capability = model_test_payload(
        queue.db(), job["id"], results_dir=results_root, data_dir=data_root
    )
    assert capability["supported_tests"]["prompt_generation"] is False
    assert any("must stay within" in reason for reason in capability["unsupported_reasons"])

    queue._run_one(job["id"])
    failed = queue.get(job["id"])
    assert failed["status"] == "error"
    assert "Persisted dataset corpus path must stay within" in failed["error"]


def test_remote_mode_blocks_direct_url_imports_but_keeps_dataset_ids(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("QLLM_DB", str(tmp_path / "server.db"))
    monkeypatch.setenv("QLLM_RESULTS", str(tmp_path / "results"))
    monkeypatch.setenv("QLLM_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("QLLM_ALLOW_REMOTE", "1")
    monkeypatch.setenv("QLLM_CORS_ORIGINS", '["https://lab.example"]')
    monkeypatch.delitem(sys.modules, "qllm.dashboard.server", raising=False)
    server = importlib.import_module("qllm.dashboard.server")
    seen = []

    def fake_import(store, **kwargs):
        seen.append(kwargs["source"])
        return {"source": kwargs["source"]}

    monkeypatch.setattr(server, "import_hf_text_dataset", fake_import)
    client = TestClient(server.app)
    if server.FRONTEND_MOUNTED:
        assert client.get("/").status_code == 200
    allowed = client.options(
        "/api/health",
        headers={
            "Origin": "https://lab.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    denied = client.options(
        "/api/health",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert allowed.headers["access-control-allow-origin"] == "https://lab.example"
    assert "access-control-allow-origin" not in denied.headers
    for source in (
        "http://example.invalid/data.txt",
        "https://example.invalid/data.txt",
        "hf://datasets/example/data.txt",
        "../server-private",
        r"C:\server-private",
        "file:///server-private",
        "data",
        "README.md",
    ):
        with pytest.raises(HTTPException, match="Hub dataset ID"):
            server.api_import_hf({"source": source, "text_column": "text"})
    assert server.api_import_hf(
        {"source": "org/public-dataset", "text_column": "text"}
    ) == {"source": "org/public-dataset"}
    assert seen == ["org/public-dataset"]

    monkeypatch.setenv("QLLM_ALLOW_REMOTE", "0")
    monkeypatch.setenv("QLLM_CORS_ORIGINS", "[]")
    assert server.api_import_hf(
        {"source": "https://example.invalid/data.txt", "text_column": "text"}
    ) == {"source": "https://example.invalid/data.txt"}
    assert seen == ["org/public-dataset", "https://example.invalid/data.txt"]
    server.QUEUE.close()


def test_api_mutation_request_policy_and_payload_error_classification(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("QLLM_DB", str(tmp_path / "server.db"))
    monkeypatch.setenv("QLLM_RESULTS", str(tmp_path / "results"))
    monkeypatch.setenv("QLLM_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("QLLM_DISABLE_WORKER", "1")
    monkeypatch.delitem(sys.modules, "qllm.dashboard.server", raising=False)
    server = importlib.import_module("qllm.dashboard.server")
    calls = []

    def fake_submit(**kwargs):
        calls.append(kwargs)
        return {"queued": True}

    monkeypatch.setattr(server.QUEUE, "submit", fake_submit)
    client = TestClient(server.app)

    cross_site = client.post(
        "/api/jobs",
        content='{"preset_id":"classical-small"}',
        headers={
            "Content-Type": "text/plain",
            "Origin": "https://evil.example",
            "Sec-Fetch-Site": "cross-site",
        },
    )
    assert cross_site.status_code == 403
    assert calls == []

    hostile_origin = client.post(
        "/api/jobs",
        json={"preset_id": "classical-small"},
        headers={"Origin": "https://evil.example"},
    )
    assert hostile_origin.status_code == 403
    assert calls == []

    wrong_media_type = client.post(
        "/api/jobs",
        content='{"preset_id":"classical-small"}',
        headers={"Content-Type": "text/plain"},
    )
    assert wrong_media_type.status_code == 415
    assert calls == []

    declared_oversized = client.post(
        "/api/jobs",
        content=b"{}",
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(server.MAX_API_MUTATION_BODY_BYTES + 1),
        },
    )
    assert declared_oversized.status_code == 413
    assert calls == []

    base_payload = b'{"preset_id":"classical-small"}'
    exact_limit_payload = base_payload + b" " * (
        server.MAX_API_MUTATION_BODY_BYTES - len(base_payload)
    )
    exact_limit = client.post(
        "/api/jobs",
        content=exact_limit_payload,
        headers={"Content-Type": "application/json"},
    )
    assert exact_limit.status_code == 200
    assert len(calls) == 1

    def oversized_chunks():
        yield base_payload
        yield b" " * server.MAX_API_MUTATION_BODY_BYTES

    chunked_oversized = client.post(
        "/api/jobs",
        content=oversized_chunks(),
        headers={
            "Content-Type": "application/json",
            "Transfer-Encoding": "chunked",
        },
    )
    assert chunked_oversized.status_code == 413
    assert len(calls) == 1

    bodyless_cross_site = client.post(
        "/api/jobs/1/cancel", headers={"Sec-Fetch-Site": "cross-site"}
    )
    assert bodyless_cross_site.status_code == 403
    assert len(calls) == 1

    accepted = client.post(
        "/api/jobs",
        json={"preset_id": "classical-small"},
        headers={"Origin": "http://localhost:5173"},
    )
    assert accepted.status_code == 200
    assert accepted.json() == {"queued": True}
    assert len(calls) == 2

    monkeypatch.setenv("QLLM_ALLOW_REMOTE", "1")
    monkeypatch.setenv("QLLM_CORS_ORIGINS", '["https://allowed.example"]')
    allowlisted_remote = client.post(
        "/api/jobs",
        json={"preset_id": "classical-small"},
        headers={
            "Origin": "https://allowed.example",
            "Sec-Fetch-Site": "cross-site",
        },
    )
    assert allowlisted_remote.status_code == 200
    assert len(calls) == 3
    monkeypatch.setenv("QLLM_ALLOW_REMOTE", "0")
    monkeypatch.setenv("QLLM_CORS_ORIGINS", "[]")

    def invalid_submit(**_kwargs):
        raise ValueError("invalid request")

    monkeypatch.setattr(server.QUEUE, "submit", invalid_submit)
    invalid = client.post("/api/jobs", json={"preset_id": "classical-small"})
    assert invalid.status_code == 400
    assert invalid.json()["detail"] == "invalid request"

    def broken_submit(**_kwargs):
        raise RuntimeError("unexpected implementation detail")

    monkeypatch.setattr(server.QUEUE, "submit", broken_submit)
    broken = client.post("/api/jobs", json={"preset_id": "classical-small"})
    assert broken.status_code == 500
    assert broken.json()["detail"] == "Internal server error."
    assert "unexpected implementation detail" not in broken.text
    server.QUEUE.close()


def test_frontend_mount_skips_missing_or_partial_build(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()

    assert mount_frontend(FastAPI(), dist) is False

    (dist / "index.html").write_text("<main>QLLM</main>", encoding="utf-8")
    assert mount_frontend(FastAPI(), dist) is False

    assets = dist / "assets"
    assets.mkdir()
    (assets / "app.js").write_text("console.log('qllm')", encoding="utf-8")
    complete_app = FastAPI()
    assert mount_frontend(complete_app, dist) is True
    complete_client = TestClient(complete_app)
    assert complete_client.get("/").text == "<main>QLLM</main>"
    assert complete_client.get("/assets/app.js").status_code == 200


def test_frontend_mount_tolerates_build_race(monkeypatch, tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<main>QLLM</main>", encoding="utf-8")

    def missing_assets(*_args, **_kwargs):
        raise RuntimeError("Directory does not exist")

    monkeypatch.setattr(
        "qllm.dashboard.static_frontend.StaticFiles", missing_assets
    )
    assert mount_frontend(FastAPI(), dist) is False


def test_wsl_launcher_uses_explicit_remote_gate():
    launcher = Path("Start QLLM GPU Portal.bat").read_text()
    assert "--host 0.0.0.0" in launcher
    assert "--allow-remote" in launcher
    assert "--cors-origin http://127.0.0.1:8000" in launcher
    setup_script = Path("scripts/setup_wsl_gpu.sh").read_text()
    assert "--allow-remote" in setup_script
    assert "--cors-origin http://127.0.0.1:8000" in setup_script
    assert 'jax[cuda13]==0.10.1' in setup_script
