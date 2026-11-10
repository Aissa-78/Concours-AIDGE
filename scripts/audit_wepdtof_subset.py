"""Contrôle le lot local et montre les images négatives à vérifier visuellement."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from heatmap_v2_data import read_manifest, read_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset/wepdtof_subset")
    args = parser.parse_args()
    root = args.dataset_dir
    rows = read_manifest(root)
    counts, scenes, hashes, panels = Counter(), {"train": set(), "validation": set()}, {"train": set(), "validation": set()}, []
    for index, row in enumerate(rows, 1):
        image, centers, boxes = read_sample(root, row)
        if image.shape != (1, 128, 128) or not np.isfinite(image).all():
            raise ValueError("Image préparée invalide.")
        counts[row["split"] + "_images"] += 1
        counts[row["split"] + "_presences"] += len(centers)
        scenes[row["split"]].add(row["video"])
        hashes[row["split"]].add(hashlib.sha256(image.tobytes()).hexdigest())
        if boxes:
            continue
        counts[row["split"] + "_negative"] += 1
        frame = cv2.imread(str(root / row["image"]))
        frame = cv2.resize(frame, (200, 200), interpolation=cv2.INTER_AREA)
        panel = np.full((235, 200, 3), 248, np.uint8)
        panel[35:] = frame
        cv2.putText(panel, f"{index:03d} - {row['video'][8:]}", (4, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (20, 20, 20), 1)
        panels.append(panel)
    if scenes["train"] & scenes["validation"] or hashes["train"] & hashes["validation"]:
        raise ValueError("Scène ou image commune à train et validation.")
    if panels:
        pages = []
        for offset in range(0, len(panels), 24):
            page = list(panels[offset:offset+24])
            while len(page) % 6:
                page.append(np.full_like(page[0], 248))
            name = f"negatives_a_verifier_{offset//24+1:02d}.jpg"
            cv2.imwrite(str(root / name), np.concatenate([
                np.concatenate(page[i:i+6], axis=1) for i in range(0, len(page), 6)], axis=0))
            pages.append(name)
        while len(panels) % 6:
            panels.append(np.full_like(panels[0], 248))
        sheet = np.concatenate([np.concatenate(panels[i:i+6], axis=1) for i in range(0, len(panels), 6)], axis=0)
        cv2.imwrite(str(root / "negatives_a_verifier.jpg"), sheet)
    result = {"counts": dict(counts), "scene_overlap": False, "prepared_image_overlap": False,
              "negative_preview_pages": pages if panels else [],
              "human_review_complete": all(row["reviewed"] == "oui" for row in rows),
              "note": "Format et séparation vérifiés ; ceci ne certifie pas visuellement chaque annotation."}
    (root / "AUDIT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
