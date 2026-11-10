"""Prépare des copies 160x120 grises pour les portes et les images publiques.

Les sources, les splits et le statut de revue ne sont pas modifiés. Les rectangles
sont transformés dans les fichiers labels, jamais dessinés dans les images train.
Ce lot est destiné au prochain modèle QQVGA, pas aux anciens modèles 80x80.
"""

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from camera_preprocess import prepare_camera_frame
from heatmap_v2_data import read_manifest


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-data", type=Path, default=ROOT / "dataset/detection_proposals")
    parser.add_argument("--public-data", type=Path, default=ROOT / "dataset/wepdtof_expanded_v2")
    parser.add_argument("--test-data", type=Path, default=ROOT / "dataset/porte_B_test")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dataset/camera_qqvga_v1")
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    args = parser.parse_args()
    if not (0 < args.width <= 320 and 0 < args.height <= 240):
        raise SystemExit("La résolution doit être positive et ne pas dépasser 320x240.")
    if args.output_dir.exists():
        raise SystemExit("Dossier existant : choisir un autre --output-dir, sans écraser les originaux.")
    sources = [("portes", args.local_data), ("wepdtof", args.public_data), ("porte_B_test", args.test_data)]
    manifests = [(name, source, read_manifest(source)) for name, source in sources]
    args.output_dir.mkdir(parents=True)
    previews = args.output_dir / "apercus"
    previews.mkdir()
    summary, cards = {}, []
    for name, source, original_rows in manifests:
        destination = args.output_dir / name
        destination.mkdir()
        rows, provenance, counts = [], [], {}
        for index, original in enumerate(original_rows):
            if original["split"] not in ("train", "validation"):
                raise ValueError("Split inconnu.")
            if name == "porte_B_test" and original["split"] != "validation":
                raise ValueError("La porte B doit rester exclusivement hors train.")
            frame = cv2.imread(str(source / original["image"]))
            gray, geometry = prepare_camera_frame(frame, args.width, args.height)
            boxes = []
            for line in (source / original["label"]).read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5 or parts[0] != "0":
                    raise ValueError(f"Label invalide : {original['label']}")
                boxes.append(geometry.transform_box(tuple(map(float, parts[1:]))))
            stem = f"{index:04d}_{Path(original['image']).stem}"
            image_path = Path(original["split"]) / "images" / f"{stem}.png"
            label_path = Path(original["split"]) / "labels" / f"{stem}.txt"
            (destination / image_path).parent.mkdir(parents=True, exist_ok=True)
            (destination / label_path).parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(destination / image_path), gray):
                raise RuntimeError("Échec d'enregistrement d'une image grise.")
            (destination / label_path).write_text("".join("0 " + " ".join(f"{value:.8f}" for value in box) + "\n" for box in boxes), encoding="utf-8")
            row = {**original, "image": image_path.as_posix(), "label": label_path.as_posix()}
            rows.append(row)
            count = counts.setdefault(original["split"], {"images": 0, "presences": 0, "empty": 0})
            count["images"] += 1
            count["presences"] += len(boxes)
            count["empty"] += not boxes
            provenance.append({"image": image_path.as_posix(), "source_image": original["image"],
                               "source_label": original["label"], "source_image_sha256": sha256(source / original["image"]),
                               "source_label_sha256": sha256(source / original["label"]), "geometry": vars(geometry)})
            # Les cadres sont uniquement sur ces aperçus, pas dans le dataset.
            if index < 4:
                preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                for cx, cy, bw, bh in boxes:
                    cv2.rectangle(preview, (round((cx-bw/2)*args.width), round((cy-bh/2)*args.height)),
                                  (round((cx+bw/2)*args.width), round((cy+bh/2)*args.height)), (30, 220, 30), 1)
                preview_name = f"{name}_{index+1:02d}.png"
                cv2.imwrite(str(previews / preview_name), preview)
                cards.append(f"<article><h2>{html.escape(name)} - {index+1}</h2><p>{len(boxes)} présence(s) annotée(s), {original['split']}</p>"
                             f"<img src='apercus/{preview_name}' alt='Gris basse résolution, annotation verte'></article>")
        with (destination / "manifest.csv").open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (destination / "SOURCE.json").write_text(json.dumps({"source_dataset": str(source.resolve()),
            "source_manifest_sha256": sha256(source / "manifest.csv"), "color": "grayscale", "width": args.width,
            "height": args.height, "resize": "INTER_AREA + letterbox gris 114, sans recadrage ni étirement",
            "reviewed": "Statut du jeu source conservé, transformation géométrique automatique ; pas de nouvelle revue humaine.",
            "rows": provenance}, ensure_ascii=False, indent=2), encoding="utf-8")
        summary[name] = counts
        print(f"{name} : {counts}", flush=True)
    (args.output_dir / "SUMMARY.json").write_text(json.dumps({"width": args.width, "height": args.height,
        "grayscale": True, "datasets": summary,
        "note": "Copies préparées pour un futur modèle QQVGA ; les scripts V2 existants réduisent encore à 80x80 utile. Ne pas confondre préparation d'images et réentraînement du réseau."}, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "rapport.html").write_text("""<!doctype html><html lang='fr'><meta charset='utf-8'>
<title>Préparation des images caméra</title><style>body{font:17px/1.5 system-ui;max-width:1100px;margin:30px auto;padding:0 20px}main{display:flex;flex-wrap:wrap;gap:20px}article{width:320px}img{width:320px;image-rendering:pixelated}h2{font-size:18px}</style>
<h1>Images caméra : niveaux de gris et basse résolution</h1><p>Rectangle vert = annotation transformée, pas détection du réseau.
Les véritables images train/validation sont grises et sans rectangle dessiné. Les proportions et les splits sont conservés.
La porte B reste hors apprentissage. Le prétraitement ne garantit pas un gain de détection.</p><main>""" + "".join(cards) + "</main></html>", encoding="utf-8")
    print(f"Copies prêtes : {args.output_dir} ; ouvrir rapport.html pour les voir.")


if __name__ == "__main__":
    main()
