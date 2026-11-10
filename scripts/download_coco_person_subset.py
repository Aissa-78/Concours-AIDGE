"""Sous-ensemble de détection Person de COCO 2017 officiel, avec négatifs.

Utilise instances_val2017.json et construit deux splits locaux disjoints par
identifiant COCO. Aucun score COCO officiel n'est revendiqué. Les images et
leurs licences d'origine restent locales. Le carnet ne fournit pas de lien
vers une archive ASYGN spécifique : ce script extrait la catégorie person
directement de COCO officiel.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import random
import threading
import zipfile

import cv2
import requests

BASE = "https://s3.amazonaws.com/images.cocodataset.org"
LOCAL = threading.local()


def download(url, path):
    if path.is_file() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
    for attempt in range(3):
        try:
            with LOCAL.session.get(url, timeout=(20, 90), stream=True) as response:
                response.raise_for_status()
                partial = path.with_suffix(path.suffix + ".part")
                with partial.open("wb") as target:
                    for chunk in response.iter_content(1024 * 1024):
                        target.write(chunk)
                partial.replace(path)
            return
        except requests.RequestException:
            if attempt == 2:
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/coco_person_subset"))
    parser.add_argument("--annotations-zip", type=Path)
    parser.add_argument("--train-images", type=int, default=500)
    parser.add_argument("--validation-images", type=int, default=125)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if min(args.train_images, args.validation_images, args.workers) < 1:
        raise SystemExit("Les nombres d'images et de workers doivent être positifs.")
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    archive = args.annotations_zip or root / "annotations_trainval2017.zip"
    download(BASE + "/annotations/annotations_trainval2017.zip", archive)
    with zipfile.ZipFile(archive) as source:
        data = json.loads(source.read("annotations/instances_val2017.json"))
    person_id = next(cat["id"] for cat in data["categories"] if cat["name"] == "person")
    annotations = defaultdict(list)
    for annotation in data["annotations"]:
        if annotation["category_id"] == person_id:
            annotations[annotation["image_id"]].append(annotation)
    positives, negatives = [], []
    for image in data["images"]:
        boxes = annotations[image["id"]]
        if not boxes:
            negatives.append(image)
        # Exclure les foules et les scènes contenant des personnes minuscules :
        # on ne supprime jamais seulement une personne pour la traiter en fond.
        elif len(boxes) <= 3 and all(
            not box.get("iscrowd", 0)
            and box["bbox"][2] / image["width"] >= 0.05
            and box["bbox"][3] / image["height"] >= 0.15
            for box in boxes
        ):
            positives.append(image)
    rng = random.Random(71)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    p_offset = n_offset = 0
    selected = []
    for split, count in (("train", args.train_images), ("validation", args.validation_images)):
        negative_count = max(1, round(count * 0.2))
        positive_count = count - negative_count
        if p_offset + positive_count > len(positives) or n_offset + negative_count > len(negatives):
            raise SystemExit("Pas assez d'images répondant aux filtres ; réduire les tailles demandées.")
        chosen = positives[p_offset:p_offset + positive_count] + negatives[n_offset:n_offset + negative_count]
        p_offset += positive_count
        n_offset += negative_count
        selected.extend((split, image) for image in chosen)

    def fetch(item):
        split, info = item
        relative = Path(split) / "images" / info["file_name"]
        download(BASE + "/val2017/" + info["file_name"], root / relative)
        frame = cv2.imread(str(root / relative))
        if frame is None or frame.shape[:2] != (info["height"], info["width"]):
            raise ValueError(f"Image COCO illisible ou dimensions différentes : {relative}")
        label = Path(split) / "labels" / f"{info['id']:012d}.txt"
        (root / label).parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for annotation in annotations[info["id"]]:
            x, y, width, height = annotation["bbox"]
            # Quelques rectangles COCO dépassent légèrement l'image.
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(info["width"], x + width), min(info["height"], y + height)
            if x2 <= x1 or y2 <= y1:
                raise ValueError("Rectangle COCO dégénéré")
            lines.append(f"0 {(x1+x2)/2/info['width']:.8f} {(y1+y2)/2/info['height']:.8f} "
                         f"{(x2-x1)/info['width']:.8f} {(y2-y1)/info['height']:.8f}")
        (root / label).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return {"split": split, "video": f"COCO2017-{info['id']}", "frame": 0,
                "time_seconds": 0, "image": relative.as_posix(), "label": label.as_posix(),
                "reviewed": "annotation_coco", "coco_id": info["id"], "license_id": info["license"]}

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(fetch, selected), 1):
            rows.append(row)
            if i % 25 == 0 or i == len(selected):
                print(f"COCO : {i}/{len(selected)} images avec annotations", flush=True)
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "SOURCE.json").write_text(json.dumps({
        "dataset": "COCO 2017, catégorie person, extraction de val2017",
        "source": "https://cocodataset.org/#download", "hosting": BASE,
        "seed": 71, "local_train": args.train_images, "local_validation": args.validation_images,
        "negative_fraction": 0.2, "selection": "1 à 3 personnes non-crowd, largeur >=5%, hauteur >=15%, ou image sans personne",
        "annotation_sha256": hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
        "licenses": data["licenses"], "images": [info for _, info in selected],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Sous-ensemble prêt : {root}", flush=True)


if __name__ == "__main__":
    main()
