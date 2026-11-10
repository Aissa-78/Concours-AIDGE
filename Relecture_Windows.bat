@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Lancez d'abord Installer_Windows.bat.
    pause
    exit /b 1
)
if not exist "dataset\profile_mix_v1\portes\manifest.csv" (
    echo Les images du projet ne sont pas dans le ZIP GitHub.
    echo Recuperez le dossier dataset/profile_mix_v1 dans le partage prive du groupe.
    echo Voir Documentation/Installation_Windows.md.
    pause
    exit /b 1
)

echo 1 - Verifier la validation restante
echo 2 - Corriger les 72 images d'entrainement local
echo 3 - Revoir toutes les images, meme celles deja validees
choice /c 123 /n /m "Votre choix (1/2/3) : "
if errorlevel 3 goto review_all
if errorlevel 2 goto review_train
set "FLOW_REVIEW_ARGS=--split validation"
goto launch

:review_train
set "FLOW_REVIEW_ARGS=--split train"
goto launch

:review_all
set "FLOW_REVIEW_ARGS=--split all --include-reviewed"

:launch
".venv\Scripts\python.exe" "scripts\review_profile_dataset.py" %FLOW_REVIEW_ARGS%
if errorlevel 1 (
    echo Une erreur est survenue. Copiez le message ci-dessus.
)
pause
