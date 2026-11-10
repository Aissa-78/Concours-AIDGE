"""Images, centres et augmentations synchronisés pour la carte V2."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from camera_preprocess import prepare_camera_frame

# Contenu 80x80, centré dans l'entrée 128x128. Les convolutions sans padding
# ont un champ de 52 pixels, stride 4 : leurs centres sont 26 + 4*i.
# Ils couvrent exactement les cellules du contenu (marge 24 + centre 2 + 4*i).
IMAGE_SIZE = 80
INPUT_SIZE = 128
GRID_SIZE = 20


def read_manifest(root: Path):
    with (root / "manifest.csv").open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows or any(row["reviewed"] not in ("oui", "annotation_coco", "annotation_wepdtof") for row in rows):
        raise ValueError("Chaque image doit être revue ou porter la provenance de ses annotations officielles.")
    for row in rows:
        for key in ("image", "label"):
            path = (root / row[key]).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                raise ValueError(f"Fichier absent ou hors dataset : {path}")
    return rows


def read_sample(root: Path, row):
    """Conserve les rectangles humains ; cible = centre du rectangle visible."""
    import cv2

    frame = cv2.imread(str(root / row["image"]))
    if frame is None:
        raise ValueError(f"Image illisible : {row['image']}")
    # Les anciens essais restent en 80x80 utile ; le nouveau lot caméra est 160x120.
    # Même conversion gris/réduction pour les portes, COCO et WEPDTOF.
    canvas, geometry = prepare_camera_frame(frame, width=IMAGE_SIZE, height=IMAGE_SIZE)
    rw, rh, left, top = geometry.resized_width, geometry.resized_height, geometry.left, geometry.top
    centers, boxes = [], []
    for line in (root / row["label"]).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5 or parts[0] != "0":
            raise ValueError(f"Annotation invalide : {row['label']}")
        cx, cy, bw, bh = map(float, parts[1:])
        if not np.isfinite([cx, cy, bw, bh]).all() or not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
            raise ValueError(f"Coordonnées invalides : {row['label']}")
        if min(cx - bw / 2, cy - bh / 2) < -1e-4 or max(cx + bw / 2, cy + bh / 2) > 1.0001:
            raise ValueError(f"Rectangle hors image : {row['label']}")
        centers.append(((cx * rw + left) / IMAGE_SIZE, (cy * rh + top) / IMAGE_SIZE))
        boxes.append((cx, cy, bw, bh))
    image = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
    margin = (INPUT_SIZE - IMAGE_SIZE) // 2
    image[margin:margin + IMAGE_SIZE, margin:margin + IMAGE_SIZE] = canvas / 255.0
    return image[None], centers, boxes


def target_heatmap(centers, grid_size=GRID_SIZE, sigma=1.0):
    yy, xx = np.mgrid[:grid_size, :grid_size].astype(np.float32)
    target = np.zeros((grid_size, grid_size), dtype=np.float32)
    for cx, cy in centers:
        gx, gy = cx * grid_size - 0.5, cy * grid_size - 0.5
        target = np.maximum(target, np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * sigma ** 2)))
    return target[None]


def augment(image, target, rng):
    """Les mêmes rotations/miroirs sont appliqués à l'image et à sa cible."""
    turns = int(rng.integers(4))
    image = np.rot90(image, turns, axes=(-2, -1))
    target = np.rot90(target, turns, axes=(-2, -1))
    if rng.random() < 0.5:
        image, target = image[..., ::-1], target[..., ::-1]
    contrast, offset = rng.uniform(0.8, 1.2), rng.uniform(-0.08, 0.08)
    image = image * contrast + offset
    if rng.random() < 0.3:
        image = image + rng.normal(0, 0.012, image.shape)
    return np.ascontiguousarray(np.clip(image, 0, 1), dtype=np.float32), np.ascontiguousarray(target)
