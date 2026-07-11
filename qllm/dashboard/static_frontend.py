"""Optional static frontend mounting for the local dashboard API."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .security import resolve_web_asset
from .status import frontend_build_available


def mount_frontend(application: FastAPI, frontend_dist: Path) -> bool:
    """Mount a complete frontend build without making it an API prerequisite."""
    if not frontend_build_available(frontend_dist):
        return False

    application.mount(
        "/assets",
        StaticFiles(directory=frontend_dist / "assets"),
        name="assets",
    )

    @application.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path:
            try:
                candidate = resolve_web_asset(frontend_dist, full_path)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if candidate.is_file():
                return FileResponse(candidate)
        return FileResponse(resolve_web_asset(frontend_dist, "index.html"))

    return True
