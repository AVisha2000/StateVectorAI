#!/usr/bin/env python3
"""Start the local QLLM Lab portal and open it in a browser."""
from __future__ import annotations

import importlib.util
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HOST = "127.0.0.1"
PORT = 8000
URL = f"http://{HOST}:{PORT}"


def _port_open() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((HOST, PORT)) == 0


def _health_ok() -> bool:
    try:
        with urllib.request.urlopen(f"{URL}/api/health", timeout=1.5) as resp:
            return resp.status == 200
    except Exception:
        return False


def _require(module: str, package_hint: str) -> None:
    if importlib.util.find_spec(module) is None:
        raise SystemExit(
            f"Missing dependency: {module}\n"
            f"Install it with:\n"
            f"  {sys.executable} -m pip install {package_hint}\n"
        )


def main() -> None:
    print("QLLM Lab portal launcher")
    print(f"Project: {ROOT}")
    if _health_ok():
        print(f"Portal is already running at {URL}")
        webbrowser.open(URL)
        return
    if _port_open():
        raise SystemExit(
            f"Port {PORT} is already in use, but it is not the QLLM portal.\n"
            "Close the other process or change the launcher port."
        )

    _require("fastapi", "fastapi uvicorn httpx")
    _require("uvicorn", "uvicorn")

    dist = ROOT / "qllm" / "dashboard" / "frontend" / "dist" / "index.html"
    if not dist.exists():
        print("WARNING: built frontend not found.")
        print("Build it with: cd qllm\\dashboard\\frontend && npm install && npm run build")
        print("Starting API anyway.")

    cmd = [
        sys.executable,
        "-m",
        "qllm.dashboard.run",
        "--host",
        HOST,
        "--port",
        str(PORT),
    ]
    log = ROOT / ".portal.log"
    print(f"Starting portal at {URL}")
    print(f"Log: {log}")
    with log.open("a", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=fh,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )

    for _ in range(40):
        if _health_ok():
            print(f"Portal is ready: {URL}")
            webbrowser.open(URL)
            print("Leave this window open while using the portal.")
            print("Press Ctrl+C here to stop it.")
            try:
                proc.wait()
            except KeyboardInterrupt:
                print("Stopping portal...")
                proc.terminate()
            return
        if proc.poll() is not None:
            raise SystemExit(
                f"Portal exited early with code {proc.returncode}. Check {log}."
            )
        time.sleep(0.5)

    raise SystemExit(f"Portal did not become ready. Check {log}.")


if __name__ == "__main__":
    main()
