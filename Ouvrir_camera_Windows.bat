@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-colibry-hw\Scripts\python.exe" (
    echo Lancer Installer_outils_Windows.bat hardware dans le terminal.
    pause
    exit /b 1
)
".venv-colibry-hw\Scripts\python.exe" scripts\view_colibry_camera.py --load-sram
if errorlevel 1 (
    echo Voir le guide Windows Linux : SDK, firmware prive et pilote USB requis.
    pause
    exit /b 1
)
