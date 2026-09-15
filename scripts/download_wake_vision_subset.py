"""Télécharge un petit sous-ensemble Wake Vision pour FlowSense.

Le script construit localement les dossiers attendus par ``src/train.py`` :
``personne`` lorsque Wake Vision indique ``person=1`` et ``vide`` lorsque
``person=0``. Les images ne sont jamais ajoutées à Git.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import ssl
from urllib.request import Request, urlopen

import certifi


DATASET = "Harvard-Edge/Wake-Vision"
ROWS_API = "https://datasets-server.huggingface.co/rows"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = {0: "vide", 1: "personne"}
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Créer un sous-ensemble local personne/vide depuis Wake Vision."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--train-per-class",
        type=int,
        default=100,
        help="Nombre cible d'images de chaque classe dans train.",
    )
    parser.add_argument(
        "--validation-per-class",
        type=int,
        default=25,
        help="Nombre cible d'images de chaque classe dans validation.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=30,
        help="Nombre maximal de pages de 100 annotations parcourues par split.",
    )
    return parser.parse_args()


def existing_image_count(directory: Path) -> int:
    """Compte les images déjà présentes afin de compléter, sans les effacer."""

    return sum(path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS for path in directory.iterdir())


def fetch_rows(split: str, offset: int) -> list[dict]:
    """Lit 100 annotations publiques à partir d'un offset Wake Vision."""

    url = f"{ROWS_API}?dataset={DATASET}&config=default&split={split}&offset={offset}&length=100"
    request = Request(url, headers={"User-Agent": "FlowSense-dataset-builder/1.0"})
    with urlopen(request, timeout=60, context=SSL_CONTEXT) as response:
        payload = json.load(response)
    return payload["rows"]


def download_image(url: str, destination: Path) -> None:
    """Télécharge une image publique dans le dossier local demandé."""

    request = Request(url, headers={"User-Agent": "FlowSense-dataset-builder/1.0"})
    with urlopen(request, timeout=60, context=SSL_CONTEXT) as response:
        destination.write_bytes(response.read())


def fill_split(
    source_split: str,
    destination_split: str,
    data_dir: Path,
    target_per_class: int,
    max_pages: int,
) -> None:
    """Complète les deux classes d'un split sans supprimer de fichiers existants."""

    destinations = {
        label: data_dir / destination_split / class_name
        for label, class_name in CLASS_NAMES.items()
    }
    for directory in destinations.values():
        directory.mkdir(parents=True, exist_ok=True)

    counts = {label: existing_image_count(directory) for label, directory in destinations.items()}
    needed = {label: max(target_per_class - count, 0) for label, count in counts.items()}
    print(f"{destination_split} avant téléchargement : vide={counts[0]}, personne={counts[1]}")

    if not any(needed.values()):
        print(f"{destination_split} contient déjà assez d'images.")
        return

    for page in range(max_pages):
        if not any(needed.values()):
            break

        rows = fetch_rows(source_split, offset=page * 100)
        if not rows:
            break

        for item in rows:
            row_index = item["row_idx"]
            row = item["row"]
            label = int(row["person"])
            if label not in CLASS_NAMES or needed[label] == 0:
                continue

            image_url = row["image"]["src"]
            destination = destinations[label] / f"wakevision_{source_split}_{row_index}.jpg"
            if destination.exists():
                continue

            try:
                download_image(image_url, destination)
            except OSError as error:
                print(f"Image ignorée ({row_index}) : {error}")
                continue

            needed[label] -= 1
            counts[label] += 1
            print(
                f"{destination_split}: {CLASS_NAMES[label]} "
                f"{counts[label]}/{target_per_class}"
            )

    if any(needed.values()):
        print(
            f"Attention : le maximum de pages a été atteint pour {destination_split}. "
            f"Il manque vide={needed[0]}, personne={needed[1]}."
        )


def main() -> None:
    args = parse_args()
    if args.train_per_class < 1 or args.validation_per_class < 1 or args.max_pages < 1:
        raise ValueError("Les quantités et --max-pages doivent être supérieurs à 0.")

    print(f"Source : {DATASET} (Wake Vision, CC BY 4.0)")
    fill_split(
        source_split="train_quality",
        destination_split="train",
        data_dir=args.data_dir,
        target_per_class=args.train_per_class,
        max_pages=args.max_pages,
    )
    fill_split(
        source_split="validation",
        destination_split="validation",
        data_dir=args.data_dir,
        target_per_class=args.validation_per_class,
        max_pages=args.max_pages,
    )


if __name__ == "__main__":
    main()
