"""Prévisualise WEPDTOF officiel sans télécharger le ZIP de 5 Go."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np

from wepdtof_archive import RangeArchive

URL = "https://www.bu.edu/vip/files/WEPDTOF.zip"


def member_image(archive, member):
    result = cv2.imdecode(np.frombuffer(archive.read(member), np.uint8), cv2.IMREAD_COLOR)
    if result is None:
        raise ValueError(f"Image illisible : {member}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/wepdtof_preview"))
    args = parser.parse_args()
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    panels, summary = [], []
    with RangeArchive(URL) as remote, zipfile.ZipFile(remote) as archive:
        annotations = sorted(n for n in archive.namelist()
                             if n.startswith("WEPDTOF/annotations/") and n.endswith(".json"))
        for member in annotations:
            scene = Path(member).stem
            metadata = root / f"{scene}.json"
            if not metadata.exists():
                metadata.write_bytes(archive.read(member))
            data = json.loads(metadata.read_text())
            entries = data["images"]
            sample = entries[len(entries) // 2]
            print(scene, "images", len(entries), "annotations", len(data["annotations"]),
                  "sample", sample, "first_box", data["annotations"][:1], flush=True)
            names = archive.namelist()
            expected = f"WEPDTOF/frames/{scene}/{Path(sample['file_name']).name}"
            if expected not in names:
                raise ValueError(f"Image introuvable dans le ZIP : {expected}")
            image_path = root / f"{scene}.jpg"
            if image_path.exists():
                frame = cv2.imread(str(image_path))
            else:
                frame = member_image(archive, expected)
                cv2.imwrite(str(image_path), frame)
            roi_path = root / f"{scene}_roi.png"
            if not roi_path.exists():
                roi_path.write_bytes(archive.read(f"WEPDTOF/ROI_maps/{scene}.png"))
            counts = {}
            for ann in data["annotations"]:
                counts[ann["image_id"]] = counts.get(ann["image_id"], 0) + 1
            summary.append({"scene": scene, "images": len(entries), "annotations": len(data["annotations"]),
                            "min_people": min(counts.get(image["id"], 0) for image in entries),
                            "max_people": max(counts.get(image["id"], 0) for image in entries),
                            "sample_member": expected})
            width, height = 320, 240
            scale = min(width / frame.shape[1], height / frame.shape[0])
            small = cv2.resize(frame, (round(frame.shape[1] * scale), round(frame.shape[0] * scale)))
            panel = np.full((height + 35, width, 3), 245, np.uint8)
            top = 35 + (height - small.shape[0]) // 2
            left = (width - small.shape[1]) // 2
            panel[top:top + small.shape[0], left:left + small.shape[1]] = small
            cv2.putText(panel, scene, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1)
            panels.append(panel)
        summary_path = root / "scenes.json"
        summary_path.write_text(json.dumps({"source": URL, "etag": remote.etag,
                                             "download_bytes": remote.transferred, "scenes": summary},
                                            indent=2), encoding="utf-8")
        for start in range(0, len(panels), 4):
            while len(panels[start:start + 4]) < 4:
                panels.append(np.full_like(panels[0], 245))
        sheet = np.concatenate([np.concatenate(panels[start:start + 4], axis=1)
                                for start in range(0, len(panels), 4)], axis=0)
        cv2.imwrite(str(root / "planche.jpg"), sheet)
        print(f"Planche : {root / 'planche.jpg'} ; transfert : {remote.transferred / 1e6:.1f} Mo", flush=True)


if __name__ == "__main__":
    main()
