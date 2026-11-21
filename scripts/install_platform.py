"""Installe un environnement séparé, sans toucher aux pilotes USB ni au SDK."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).absolute().parent.parent


def installation_plan(profile, system, root=ROOT):
    if system not in ('Windows', 'Linux'):
        raise ValueError('Cet installateur vise Windows/Linux. Conserver les environnements du Mac.')
    if profile == 'aidge' and system == 'Windows':
        raise ValueError('Pour Aidge, ouvrir Ubuntu/WSL et exécuter bash Installer_Linux.sh aidge. Compilation native : voir le guide.')
    if profile not in ('app', 'aidge', 'hardware'):
        raise ValueError('Profil inconnu.')
    names = ({'app': '.venv', 'hardware': '.venv-colibry-hw'} if system == 'Windows'
             else {'app': '.venv-linux', 'hardware': '.venv-colibry-linux', 'aidge': '.venv-aidge-linux'})
    folder = root / names[profile]
    python = folder / ('Scripts/python.exe' if system == 'Windows' else 'bin/python')
    commands = [[str(python), '-m', 'pip', 'install', '--upgrade', 'pip']]
    if profile == 'app':
        commands += [[str(python), '-m', 'pip', 'install', 'torch', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cpu'],
                     [str(python), '-m', 'pip', 'install', '-r', str(root / 'requirements.txt')],
                     [str(python), str(root / 'scripts/check_setup.py')]]
    elif profile == 'hardware':
        commands += [[str(python), '-m', 'pip', 'install', '-r', str(root / 'requirements-colibry-hardware.txt')],
                     [str(python), '-c', 'import tkinter, PIL, numpy, pyftdi, usb, libusb_package, elftools; print("Imports caméra OK ; USB pas encore testé.")']]
    else:
        commands += [[str(python), '-m', 'pip', 'install', 'aidge-core', 'aidge-backend-cpu', 'aidge-onnx', 'numpy', 'onnx'],
                     [str(python), str(root / 'scripts/check_setup.py'), '--aidge-only'],
                     [str(python), str(root / 'scripts/check_aidge_training.py')]]
    return folder, python, commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['app', 'hardware', 'aidge'], default='app')
    parser.add_argument('--dry-run', action='store_true', help='Afficher le plan sans rien installer.')
    args = parser.parse_args()
    try:
        folder, python, commands = installation_plan(args.profile, platform.system())
    except ValueError as error:
        parser.error(str(error))
    plan = {'environment': str(folder), 'python': str(python), 'commands': commands,
            'drivers_modified': False, 'sdk_downloaded': False}
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return
    if sys.version_info[:2] not in ((3, 11), (3, 12)):
        parser.error('Lancer cet installateur avec Python 3.11 ou 3.12.')
    marker = folder / 'flowsense_platform.json'
    if folder.exists():
        if not python.is_file() or (folder / ('bin/python' if platform.system() == 'Windows' else 'Scripts/python.exe')).exists():
            raise SystemExit('Environnement incompatible : conserver ce dossier sous un autre nom puis relancer. Ne pas copier les venv du Mac.')
        if marker.exists() and json.loads(marker.read_text())['system'] != platform.system():
            raise SystemExit('Environnement créé sur un autre système : utiliser un dossier local.')
    else:
        subprocess.run([sys.executable, '-m', 'venv', str(folder)], check=True)
    # Vérifier le véritable interpréteur, y compris un venv copié accidentellement.
    actual = json.loads(subprocess.check_output([str(python), '-c', 'import json,platform,sys; print(json.dumps([platform.system(),list(sys.version_info[:2])]))'], text=True))
    if actual[0] != platform.system() or tuple(actual[1]) not in ((3, 11), (3, 12)):
        raise SystemExit('Interpréteur de cet environnement incompatible.')
    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)
    marker.write_text(json.dumps({'system': platform.system(), 'profile': args.profile}, indent=2))
    with (folder / 'installed_versions.txt').open('w') as handle:
        subprocess.run([str(python), '-m', 'pip', 'freeze'], stdout=handle, check=True)
    print('Installation et contrôles logiciels terminés. La carte et les données privées restent à vérifier.')


if __name__ == '__main__':
    main()
