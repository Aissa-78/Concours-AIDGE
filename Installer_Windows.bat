@echo off
setlocal
cd /d "%~dp0"

set "FLOW_PY="
py -3.11 -c "import sys" >nul 2>&1
if not errorlevel 1 set "FLOW_PY=py -3.11"
if not defined FLOW_PY (
    py -3.12 -c "import sys" >nul 2>&1
    if not errorlevel 1 set "FLOW_PY=py -3.12"
)
if not defined FLOW_PY (
    python -c "import sys; raise SystemExit(sys.version_info[:2] not in ((3,11),(3,12)))" >nul 2>&1
    if not errorlevel 1 set "FLOW_PY=python"
)
if not defined FLOW_PY goto missing_python

if not exist ".venv\Scripts\python.exe" (
    echo Creation de l'environnement Python...
    %FLOW_PY% -m venv ".venv"
    if errorlevel 1 goto failure
)

echo Installation des dependances. Cela peut prendre plusieurs minutes...
".venv\Scripts\python.exe" -m pip install -r "requirements.txt"
if errorlevel 1 goto failure
".venv\Scripts\python.exe" -c "import tkinter, torch, ultralytics, imageio_ffmpeg; print('FlowSense pret')"
if errorlevel 1 goto failure

echo.
echo Installation terminee. Double-cliquez maintenant sur FlowSense_Windows.bat.
pause
exit /b 0

:missing_python
echo Python 3.11 ou 3.12 est necessaire. Installez-le depuis python.org.
echo Pendant l'installation, cochez Add python.exe to PATH.
pause
exit /b 1

:failure
echo.
echo L'installation a echoue. Copiez le message d'erreur affiche ci-dessus.
pause
exit /b 1
