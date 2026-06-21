@echo off
setlocal

set "WSL_DISTRO=Ubuntu-24.04"

where wsl >nul 2>&1
if %errorlevel% neq 0 (
  echo WSL was not found. Run "Setup GPU WSL.bat" first.
  pause
  exit /b 1
)

wsl -d %WSL_DISTRO% -e sh -lc "exit 0" >nul 2>&1
if %errorlevel% neq 0 (
  echo %WSL_DISTRO% is not installed or has not been initialized yet.
  echo Run "Setup GPU WSL.bat", reboot if asked, then open Ubuntu once.
  pause
  exit /b 1
)

for /f "usebackq delims=" %%i in (`wsl -d %WSL_DISTRO% wslpath -a "%CD%"`) do set WSL_CWD=%%i

echo Setting up QLLM GPU environment in WSL...
wsl -d %WSL_DISTRO% bash -lc "cd \"%WSL_CWD%\" && bash scripts/setup_wsl_gpu.sh"
if %errorlevel% neq 0 (
  echo.
  echo WSL GPU setup failed. Check the output above.
  pause
  exit /b 1
)

echo.
echo WSL GPU setup finished.
echo Start the GPU-backed portal with:
echo.
echo   Start QLLM GPU Portal.bat
echo.
pause
