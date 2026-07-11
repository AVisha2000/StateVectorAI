#!/usr/bin/env python3
"""Install portal dependencies and build the frontend when Node is available."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NODE_DIR = Path("C:/Program Files/nodejs")


def run(cmd: list[str], cwd: Path = ROOT) -> int:
    print("+", " ".join(cmd))
    return subprocess.call(cmd, cwd=cwd)


def npm_cmd() -> list[str] | None:
    found = shutil.which("npm")
    if found:
        return [found]
    npm = NODE_DIR / "npm.cmd"
    if npm.exists():
        return [str(npm)]
    return None


def main() -> int:
    print("QLLM Lab setup")
    print(f"Python: {sys.executable}")
    code = run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-r",
        str(ROOT / "requirements-cpu.txt"),
    ])
    if code == 0:
        code = run([
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "-e",
            str(ROOT),
        ])
    if code != 0:
        print()
        print("Python dependency install failed.")
        print("On Windows this is often long-path support. Two practical fixes:")
        print("  1. Move this repo to a shorter path, for example C:\\qllm")
        print("  2. Enable Windows Long Paths, then rerun this setup.")
        return code

    frontend = ROOT / "qllm" / "dashboard" / "frontend"
    npm = npm_cmd()
    if npm is None:
        print()
        print("npm was not found on PATH, so the React UI was not rebuilt.")
        print("Install Node.js 18+ from https://nodejs.org, then rerun setup.")
        return 0

    if NODE_DIR.exists():
        import os
        os.environ["PATH"] = f"{NODE_DIR};{os.environ.get('PATH', '')}"

    if run([*npm, "install"], cwd=frontend) != 0:
        return 1
    if run([*npm, "run", "build"], cwd=frontend) != 0:
        return 1

    print()
    print("QLLM Lab setup complete. Double-click QLLM Portal.bat to start.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
