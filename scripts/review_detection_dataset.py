"""Corrige les rectangles proposés par YOLO, image par image.

Commandes dans la fenêtre :
- glisser avec le clic gauche : ajouter un rectangle ;
- clic droit sur un rectangle : le supprimer ;
- S : enregistrer et marquer l'image comme vérifiée ;
- N : image suivante sans valider ; P : précédente ; Q : quitter.

Par défaut, la validation est revue en premier : elle servira ensuite à
mesurer le modèle sur des vidéos qui ne figurent pas dans l'entraînement.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("dataset/detection_proposals")
    )
    parser.add_argument("--split", choices=["train", "validation", "all"], default="validation")
    parser.add_argument("--video", help="Ne revoir qu'une vidéo (nom de fichier complet).")
    parser.add_argument("--include-reviewed", action="store_true")
    parser.add_argument(
        "--start-index", type=int, default=1,
        help="Commencer à cette image de la sélection (numéro affiché, à partir de 1).",
    )
    return parser.parse_args()


def read_boxes(label_path: Path, width: int, height: int) -> list[list[float]]:
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5 or fields[0] != "0":
            raise ValueError(f"Étiquette invalide : {label_path} : {line}")
        _, cx, cy, bw, bh = map(float, fields)
        boxes.append([
            (cx - bw / 2) * width, (cy - bh / 2) * height,
            (cx + bw / 2) * width, (cy + bh / 2) * height,
        ])
    return boxes


def write_boxes(label_path: Path, boxes: list[list[float]], width: int, height: int) -> None:
    lines = []
    for x1, y1, x2, y2 in boxes:
        x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
        y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue
        lines.append(
            f"0 {(x1 + x2) / (2 * width):.6f} {(y1 + y2) / (2 * height):.6f} "
            f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
        )
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    args = arguments()
    manifest_path = args.dataset_dir / "manifest.csv"
    if not manifest_path.is_file():
        raise SystemExit("Manifest introuvable. Lancez d'abord prepare_detection_dataset.py.")
    with manifest_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames
        rows = list(reader)
    selected = [
        index for index, row in enumerate(rows)
        if (args.split == "all" or row["split"] == args.split)
        and (not args.video or row["video"] == args.video)
        and (args.include_reviewed or row["reviewed"] != "oui")
    ]
    if not selected:
        print("Aucune image à revoir avec ces filtres.")
        return
    if not 1 <= args.start_index <= len(selected):
        raise SystemExit(f"--start-index doit être entre 1 et {len(selected)}.")

    window = "FlowSense - annotation"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    position = args.start_index - 1
    while 0 <= position < len(selected):
        row_index = selected[position]
        row = rows[row_index]
        image = cv2.imread(str(args.dataset_dir / row["image"]))
        if image is None:
            raise SystemExit(f"Image illisible : {row['image']}")
        height, width = image.shape[:2]
        label_path = args.dataset_dir / row["label"]
        state = {"boxes": read_boxes(label_path, width, height), "start": None}
        scale = min(1.0, 1100 / width, 760 / height)
        display_size = (round(width * scale), round(height * scale))

        def mouse(event, x, y, flags, userdata):
            original_x = x / scale
            original_y = y / scale
            if event == cv2.EVENT_LBUTTONDOWN:
                state["start"] = (original_x, original_y)
            elif event == cv2.EVENT_LBUTTONUP and state["start"] is not None:
                x1, y1 = state["start"]
                if abs(x1 - original_x) >= 3 and abs(y1 - original_y) >= 3:
                    state["boxes"].append([
                        min(x1, original_x), min(y1, original_y),
                        max(x1, original_x), max(y1, original_y),
                    ])
                state["start"] = None
            elif event == cv2.EVENT_RBUTTONDOWN:
                candidates = [
                    (index, (x2 - x1) * (y2 - y1))
                    for index, (x1, y1, x2, y2) in enumerate(state["boxes"])
                    if x1 <= original_x <= x2 and y1 <= original_y <= y2
                ]
                if candidates:
                    # Si deux rectangles se superposent, retirer le plus petit.
                    index = min(candidates, key=lambda item: item[1])[0]
                    state["boxes"].pop(index)

        cv2.setMouseCallback(window, mouse)
        while True:
            canvas = image.copy()
            for box_index, (x1, y1, x2, y2) in enumerate(state["boxes"], start=1):
                cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(
                    canvas, str(box_index), (int(x1), max(20, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )
            cv2.imshow(window, cv2.resize(canvas, display_size))
            cv2.setWindowTitle(
                window,
                f"{position + 1}/{len(selected)} | {row['video']} frame {row['frame']} "
                f"| {len(state['boxes'])} rectangles | S=sauver N=suivant P=precedent Q=quitter",
            )
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("s"), ord("S")):
                write_boxes(label_path, state["boxes"], width, height)
                row["reviewed"] = "oui"
                with manifest_path.open("w", newline="", encoding="utf-8") as output:
                    writer = csv.DictWriter(output, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"Vérifiée : {row['image']} ({len(state['boxes'])} rectangles)")
                position += 1
                break
            if key in (ord("n"), ord("N")):
                position += 1
                break
            if key in (ord("p"), ord("P")):
                position = max(0, position - 1)
                break
            if key in (ord("q"), ord("Q")):
                cv2.destroyAllWindows()
                return
    cv2.destroyAllWindows()
    print("Relecture terminée.")


if __name__ == "__main__":
    main()
