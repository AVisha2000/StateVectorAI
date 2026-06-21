@echo off
setlocal

set "WSL_DISTRO=Ubuntu-24.04"

where wsl >nul 2>&1
if %errorlevel% neq 0 (
  echo WSL is not installed or no distro has been initialized yet.
  echo Run "Setup GPU WSL.bat" and "Setup QLLM GPU in WSL.bat" first.
  pause
  exit /b 1
)

wsl -d %WSL_DISTRO% -e sh -lc "exit 0" >nul 2>&1
if %errorlevel% neq 0 (
  echo %WSL_DISTRO% is not installed or has not been initialized yet.
  echo Run "Setup GPU WSL.bat" and "Setup QLLM GPU in WSL.bat" first.
  pause
  exit /b 1
)

for /f "usebackq delims=" %%i in (`wsl -d %WSL_DISTRO% wslpath -a "%CD%"`) do set WSL_CWD=%%i

echo Starting GPU-backed QLLM portal from WSL...
echo Open http://127.0.0.1:8000 after the server starts.
wsl -d %WSL_DISTRO% bash -lc "cd \"%WSL_CWD%\" && source ~/.venvs/qllm-wsl/bin/activate && python -m qllm.dashboard.run --host 0.0.0.0 --port 8000"
