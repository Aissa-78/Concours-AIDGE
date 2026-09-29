@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo FlowSense n'est pas encore installe sur ce PC.
    echo Double-cliquez d'abord sur Installer_Windows.bat.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" "src\app.py"
if errorlevel 1 (
    echo.
    echo FlowSense s'est arrete avec une erreur. Copiez le message ci-dessus.
    pause
    exit /b 1
)
