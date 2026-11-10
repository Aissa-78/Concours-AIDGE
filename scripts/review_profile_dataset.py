"""Ouvre la relecture des images de profil dans une copie, sans toucher au bilan.

Par défaut : les 34 images de validation locale. S marque une annotation comme
vérifiée par l'utilisateur. Les prochaines ouvertures reprennent les restantes.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def prepare_review_copy(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("La copie doit être dans un dossier distinct, sans inclusion réciproque.")
    required = ("SOURCE.json", "portes/manifest.csv", "coco/manifest.csv")
    if not all((source/name).is_file() for name in required):
        raise FileNotFoundError("Dataset mixte source incomplet.")
    marker = destination/"REVIEW.json"
    if destination.exists():
        if not marker.is_file():
            raise ValueError("Dossier existant non reconnu : choisir une autre copie, sans l'écraser.")
        provenance = json.loads(marker.read_text())
        if provenance["source_dataset"] != str(source):
            # Une copie partagee Mac/Windows conserve le chemin de son poste
            # d'origine. Autoriser la reprise si sa source est byte-a-byte identique.
            inventory = provenance.get("original_sha256", {})
            if not inventory:
                raise ValueError("Origine de la copie non verifiable.")
            for relative, expected in inventory.items():
                path = (source/relative).resolve()
                if not path.is_relative_to(source) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    raise ValueError("La source locale ne correspond pas a celle de la copie partagee ; aucune correction ecrasee.")
        if not all((destination/name).is_file() for name in required):
            raise ValueError("Copie de relecture incomplète.")
        return False
    # Copie réelle, pas de liens : modifier un label ne change jamais la source.
    shutil.copytree(source, destination)
    inventory = {path.relative_to(source).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in source.rglob("*") if path.is_file()}
    marker.write_text(json.dumps({"source_dataset": str(source), "original_sha256": inventory,
        "purpose": "Relecture humaine sur copie, anciens scores conservés. Aucun entraînement automatique.",
        "review_status": "Consulter reviewed=oui dans portes/manifest.csv ; la copie n'est pas validée d'office.",
        "provenance_note": "SOURCE.json conserve la provenance initiale. Les labels actuels de cette copie et le manifest font foi pour les futures corrections.",
        "validation_is_not_training": True}, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT/"dataset/profile_mix_v1")
    parser.add_argument("--review-dir", type=Path, default=ROOT/"dataset/profile_mix_human_v1")
    parser.add_argument("--split", choices=("validation", "train", "all"), default="validation")
    parser.add_argument("--include-reviewed", action="store_true", help="Afficher aussi les annotations déjà sauvées avec S.")
    parser.add_argument("--prepare-only", action="store_true", help="Préparer la copie sans ouvrir de fenêtre.")
    args = parser.parse_args()
    try:
        created = prepare_review_copy(args.source, args.review_dir)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"{error}\nConsulter Documentation/Installation_Windows.md pour les fichiers a partager.") from error
    dataset = args.review_dir/"portes"
    with (dataset/"manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if (args.split == "all" or row["split"] == args.split)
                and (args.include_reviewed or row["reviewed"] != "oui")]
    print(f"{'Copie créée' if created else 'Copie conservée, reprise'} : {args.review_dir}", flush=True)
    print(f"{len(selected)} images à revoir ({args.split}). Les données et le modèle précédents restent inchangés.", flush=True)
    if not selected:
        print("Aucune image restante. Pour revoir les images sauvees : --include-reviewed. Pour le train : --split train.", flush=True)
    if args.prepare_only or not selected:
        return
    command = [sys.executable, str(ROOT/"scripts/review_detection_dataset.py"),
               "--dataset-dir", str(dataset.absolute()), "--split", args.split]
    if args.include_reviewed:
        command.append("--include-reviewed")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
