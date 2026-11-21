@echo off
setlocal
cd /d "%~dp0"
set "FLOW_PY="
py -3.12 -c "import sys" >nul 2>&1
if not errorlevel 1 set "FLOW_PY=py -3.12"
if not defined FLOW_PY (
    py -3.11 -c "import sys" >nul 2>&1
    if not errorlevel 1 set "FLOW_PY=py -3.11"
)
if not defined FLOW_PY set "FLOW_PY=python"
set "FLOW_PROFILE=%~1"
if not defined FLOW_PROFILE set "FLOW_PROFILE=app"
%FLOW_PY% scripts\install_platform.py --profile "%FLOW_PROFILE%"
if errorlevel 1 (
    echo Installation interrompue. Conserver le message d'erreur ci-dessus.
    pause
    exit /b 1
)
pause
