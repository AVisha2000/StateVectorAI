@echo off
setlocal

net session >nul 2>&1
if %errorlevel% neq 0 (
  echo Requesting Administrator permission to install Windows Subsystem for Linux...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b
)

echo.
echo Installing WSL with Ubuntu. Windows may ask for a reboot.
echo After reboot, open Ubuntu once, create the Linux username, then run:
echo.
echo   Setup QLLM GPU in WSL.bat
echo.
wsl --install -d Ubuntu

echo.
echo If WSL reports that a reboot is required, reboot before continuing.
pause
