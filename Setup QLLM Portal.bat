@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
) else (
  set "PY=python"
)

"%PY%" "scripts\setup_portal.py"
echo.
echo Setup finished. Press any key to close this window.
pause >nul
