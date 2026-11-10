"""Montre les annotations locales très superposées, sans les corriger.

Une superposition n'est pas une preuve de doublon : décision humaine nécessaire.
"""

import argparse
import html
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from person_detection_math import box_iou


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT/"runs/person_qqvga_full_v2")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"runs/person_annotation_audit_v1")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("Choisir un nouveau dossier d'audit pour conserver les aperçus existants.")
    selected = []
    for split in ("train", "validation"):
        for record in json.loads((args.run_dir/f"{split}.json").read_text()):
            if record["source"] != "portes":
                continue
            xy = [np.array([cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2]) for cx, cy, bw, bh in record["boxes"]]
            overlaps = [(i+1, j+1, float(box_iou(a, b))) for i, a in enumerate(xy) for j, b in enumerate(xy)
                        if i < j and box_iou(a, b) > .5]
            if overlaps:
                selected.append({"record": record, "xy": xy, "overlaps": overlaps})
    args.output_dir.mkdir(parents=True)
    cards, metadata = [], []
    for index, selected_record in enumerate(selected, 1):
        record = selected_record["record"]
        root = Path(record["root"])
        source = json.loads((root/"SOURCE.json").read_text())
        mapping = next(row for row in source["rows"] if row["image"] == record["row"]["image"])
        original = Path(source["source_dataset"])/mapping["source_image"]
        image = cv2.imread(str(original))
        geometry = mapping["geometry"]
        if image is None:
            raise ValueError(f"Image source illisible : {original}")
        height, width = image.shape[:2]
        colors = [(0, 180, 0), (0, 110, 255), (240, 90, 0)]
        for label, rectangle in enumerate(selected_record["xy"], 1):
            x1, y1, x2, y2 = rectangle*np.array([160, 120, 160, 120])
            x1, x2 = (np.array([x1, x2])-geometry["left"])/geometry["resized_width"]*width
            y1, y2 = (np.array([y1, y2])-geometry["top"])/geometry["resized_height"]*height
            color = colors[(label-1) % len(colors)]
            cv2.rectangle(image, (round(x1), round(y1)), (round(x2), round(y2)), color, 3)
            cv2.putText(image, f"Annotation {label}", (max(0, round(x1)), max(20, round(y1)-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
        path = args.output_dir/f"{index:02d}.jpg"
        cv2.imwrite(str(path), image)
        cards.append(f"<article><h2>Image {index} — {html.escape(record['row']['image'])}</h2>"
                     f"<p>Rectangles superposés : {html.escape(str(selected_record['overlaps']))}</p>"
                     f"<img src='{path.name}' style='max-width:100%' alt='Annotations à confirmer'></article>")
        metadata.append({"source_image": str(original), "source_label": str(Path(source["source_dataset"])/mapping["source_label"]),
                         "split": record["row"]["split"], "overlaps": selected_record["overlaps"], "preview": str(path)})
    (args.output_dir/"audit.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir/"rapport.html").write_text("<!doctype html><html lang='fr'><meta charset='utf-8'>"
        "<title>Annotations à vérifier</title><body style='font:17px/1.5 system-ui;max-width:1000px;margin:auto'>"
        "<h1>Annotations à confirmer</h1><p>Aucune annotation n'a été modifiée. Une superposition peut aussi correspondre à deux personnes distinctes.</p>"
        +"".join(cards)+"</body></html>", encoding="utf-8")
    print(f"{len(selected)} images signalées, aucune annotation modifiée : {args.output_dir/'rapport.html'}")


if __name__ == "__main__":
    main()
