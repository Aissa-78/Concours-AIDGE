"""Diagnostic sans installation : dependances, interpretation Python et donnees."""

import argparse
from importlib import import_module
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def check_modules(names):
    errors = []
    for name in names:
        try:
            import_module(name)
            print(f"OK : {name}")
        except (ImportError, OSError, RuntimeError) as error:
            errors.append(name)
            print(f"ABSENT / INUTILISABLE : {name} ({error})")
    return errors


def missing_data(root):
    required = ("dataset/profile_mix_v1/SOURCE.json", "dataset/profile_mix_v1/portes/manifest.csv",
                "dataset/profile_mix_v1/coco/manifest.csv")
    return [name for name in required if not (root/name).is_file()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aidge-only", action="store_true", help="Verifier le Python Aidge separe, pas l'environnement interface.")
    parser.add_argument("--no-gui", action="store_true", help="Ne pas verifier tkinter sur un serveur de tests.")
    parser.add_argument("--data", action="store_true", help="Verifier aussi la presence du dataset prive de profil.")
    args = parser.parse_args()
    print(f"Python : {sys.executable}\nVersion : {sys.version.split()[0]}\nProjet : {ROOT}")
    modules = ["aidge_core", "aidge_backend_cpu", "aidge_onnx"] if args.aidge_only else [
        "torch", "torchvision", "numpy", "cv2", "PIL", "onnx", "ultralytics", "imageio_ffmpeg", "requests"]
    if not args.aidge_only and not args.no_gui:
        modules.append("tkinter")
    errors = check_modules(modules)
    if args.data:
        absent = missing_data(ROOT)
        for path in absent:
            print(f"DONNEES ABSENTES : {path}")
        errors.extend(absent)
    if errors:
        print("Diagnostic incomplet : voir les lignes ci-dessus. Le code GitHub ne contient ni les donnees privees ni les environnements du Mac.")
        raise SystemExit(1)
    print("Verification terminee. Cela ne valide ni l'affichage interactif, ni ColibryNPU, ni les 5 mW.")


if __name__ == "__main__":
    main()
