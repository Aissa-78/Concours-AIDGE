"""Mesure une première qualité de localisation sur les images validées.

La métrique compare les pics de la carte 20x20 aux centres des rectangles
humains. Elle ne mesure PAS le suivi ni le comptage entrée/sortie en vidéo.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from tiny_heatmap import TinyHeatmap
from train_tiny_heatmap import DetectionFrames, GRID_SIZE, IMAGE_SIZE


def ground_truth_centers(data_dir: Path, row: dict[str, str]) -> list[tuple[float, float]]:
    frame = cv2.imread(str(data_dir / row["image"]))
    if frame is None:
        raise RuntimeError(f"Image illisible : {row['image']}")
    height, width = frame.shape[:2]
    scale = min(IMAGE_SIZE / width, IMAGE_SIZE / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    left = (IMAGE_SIZE - resized_width) // 2
    top = (IMAGE_SIZE - resized_height) // 2
    centers = []
    for line in (data_dir / row["label"]).read_text(encoding="utf-8").splitlines():
        _, center_x, center_y, _, _ = line.split()
        grid_x = (float(center_x) * resized_width + left) * GRID_SIZE / IMAGE_SIZE - 0.5
        grid_y = (float(center_y) * resized_height + top) * GRID_SIZE / IMAGE_SIZE - 0.5
        centers.append((grid_x, grid_y))
    return centers


def find_peaks(scores: np.ndarray, threshold: float, separation: float) -> list[tuple[float, float]]:
    local_maximum = cv2.dilate(scores, np.ones((3, 3), dtype=np.uint8))
    candidates = np.argwhere((scores >= threshold) & (scores >= local_maximum - 1e-6))
    candidates = sorted(candidates, key=lambda point: float(scores[tuple(point)]), reverse=True)
    peaks = []
    for y, x in candidates:
        if all(math.hypot(float(x) - px, float(y) - py) > separation for px, py in peaks):
            peaks.append((float(x), float(y)))
    return peaks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/detection_proposals"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--match-radius", type=float, default=2.5)
    args = parser.parse_args()
    if not 0 < args.threshold < 1 or args.match_radius <= 0:
        raise SystemExit("Seuil doit être dans (0,1) et rayon positif.")
    with (args.data_dir / "manifest.csv").open(newline="", encoding="utf-8") as source:
        rows = [row for row in csv.DictReader(source) if row["split"] == "validation"]
    if not rows or any(row["reviewed"] != "oui" for row in rows):
        raise SystemExit("Les images de validation doivent toutes être vérifiées.")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = TinyHeatmap().eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    dataset = DetectionFrames(args.data_dir, rows)
    true_positive = false_positive = false_negative = 0
    empty_images = empty_false_alarms = 0
    torch.set_num_threads(min(4, torch.get_num_threads()))
    with torch.no_grad():
        for index, row in enumerate(rows):
            image, _ = dataset[index]
            scores = torch.sigmoid(model(image.unsqueeze(0)))[0, 0].numpy()
            predictions = find_peaks(scores, args.threshold, separation=2.0)
            truth = ground_truth_centers(args.data_dir, row)
            if not truth:
                empty_images += 1
                if predictions:
                    empty_false_alarms += 1
            unmatched = set(range(len(truth)))
            for px, py in predictions:
                matches = sorted(
                    (math.hypot(px - truth[i][0], py - truth[i][1]), i)
                    for i in unmatched
                )
                if matches and matches[0][0] <= args.match_radius:
                    true_positive += 1
                    unmatched.remove(matches[0][1])
                else:
                    false_positive += 1
            false_negative += len(unmatched)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    print(f"Validation : {len(rows)} images, {true_positive + false_negative} personnes annotées")
    print(f"Seuil={args.threshold:.2f}, rayon de correspondance={args.match_radius:.1f} cases")
    print(f"Correspondances={true_positive}, faux positifs={false_positive}, oublis={false_negative}")
    print(f"Précision={precision:.1%}, rappel={recall:.1%}, F1={f1:.1%}")
    print(f"Fausses alertes sur images vides : {empty_false_alarms}/{empty_images}")
    print("Métrique indicative de centres ; pas une mesure de comptage vidéo.")


if __name__ == "__main__":
    main()
