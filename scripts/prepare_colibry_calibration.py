"""Prépare des images réelles, avec le même prétraitement que l'entraînement.

Exemple :
  ./.venv/bin/python scripts/prepare_colibry_calibration.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


IMAGE_SIZE = 96


def preprocess(path: Path) -> np.ndarray:
    frame = cv2.imread(str(path))
    if frame is None:
        raise ValueError(f"Image illisible : {path}")
    height, width = frame.shape[:2]
    scale = min(IMAGE_SIZE / width, IMAGE_SIZE / height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    left = (IMAGE_SIZE - new_width) // 2
    top = (IMAGE_SIZE - new_height) // 2
    canvas = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 114, dtype=np.uint8)
    canvas[top:top + new_height, left:left + new_width] = cv2.resize(
        frame, (new_width, new_height), interpolation=cv2.INTER_AREA
    )
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/detection_proposals"))
    parser.add_argument("--max-images", type=int, default=32)
    parser.add_argument(
        "--output", type=Path, default=Path("runs/tiny_heatmap_relu/calibration.npz")
    )
    args = parser.parse_args()
    if args.max_images < 1:
        raise SystemExit("--max-images doit être positif.")
    manifest = args.data_dir / "manifest.csv"
    if not manifest.is_file():
        raise SystemExit(f"Manifest introuvable : {manifest}")
    with manifest.open(newline="", encoding="utf-8") as source:
        rows = [row for row in csv.DictReader(source) if row["split"] == "train"]
    if not rows:
        raise SystemExit("Aucune image d'entraînement dans le manifest.")

    # Répartir les échantillons sur tout le manifest (plusieurs vidéos).
    indices = np.linspace(0, len(rows) - 1, min(args.max_images, len(rows)), dtype=int)
    selected = [rows[int(index)] for index in indices]
    images = np.concatenate(
        [preprocess(args.data_dir / row["image"]) for row in selected], axis=0
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, images=images)
    print(f"Calibration : {len(images)} images réelles, forme {images.shape}")
    print(f"Fichier : {args.output}")
    print("Ces images règlent les échelles INT8 ; elles ne réentraînent pas le modèle.")


if __name__ == "__main__":
    main()
