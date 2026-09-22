"""Compte les entrées et sorties dans une vidéo avec un réglage de porte.

Le modèle détecte et suit les personnes. Le réglage JSON, créé par
``calibrate_gate.py``, indique où se trouve la porte pour cette caméra.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO


PERSON_CLASS_ID = 0  # Classe "person" dans le modèle COCO utilisé par YOLO.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compter les entrées et sorties d'une vidéo.")
    parser.add_argument("--video", type=Path, required=True, help="Vidéo MP4 à analyser.")
    parser.add_argument("--config", type=Path, required=True, help="Réglage JSON de la porte.")
    parser.add_argument("--model", default="yolo26n.pt", help="Modèle YOLO à utiliser.")
    parser.add_argument("--confidence", type=float, default=0.35, help="Seuil de confiance YOLO.")
    parser.add_argument("--image-size", type=int, default=640, help="Taille d'analyse YOLO.")
    parser.add_argument(
        "--cooldown-frames",
        type=int,
        default=30,
        help="Nombre minimal d'images entre deux comptes du même ID.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/count_video"),
        help="Dossier qui recevra la vidéo annotée.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Réglage introuvable : {path}")
    try:
        config = json.loads(path.read_text())
        line_1 = config["gate"]["line_1"]
        line_2 = config["gate"]["line_2"]
        inside_reference = config["inside_reference"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"Réglage invalide : {path}") from error
    for point in [*line_1, *line_2, inside_reference]:
        if len(point) != 2 or not all(0 <= coordinate <= 1 for coordinate in point):
            raise ValueError("Les coordonnées du réglage doivent être comprises entre 0 et 1.")
    return config


def to_pixels(point: list[float], width: int, height: int) -> tuple[int, int]:
    return round(point[0] * width), round(point[1] * height)


class Gate:
    """Transforme les coordonnées du réglage en zones intérieure, porte, extérieure."""

    def __init__(self, config: dict, width: int, height: int) -> None:
        self.line_1 = tuple(to_pixels(point, width, height) for point in config["gate"]["line_1"])
        self.line_2 = tuple(to_pixels(point, width, height) for point in config["gate"]["line_2"])
        self.inside_reference = to_pixels(config["inside_reference"], width, height)
        self.mid_start = (
            (self.line_1[0][0] + self.line_2[0][0]) / 2,
            (self.line_1[0][1] + self.line_2[0][1]) / 2,
        )
        self.mid_end = (
            (self.line_1[1][0] + self.line_2[1][0]) / 2,
            (self.line_1[1][1] + self.line_2[1][1]) / 2,
        )
        self.length = ((self.mid_end[0] - self.mid_start[0]) ** 2 + (self.mid_end[1] - self.mid_start[1]) ** 2) ** 0.5
        if self.length == 0:
            raise ValueError("Les deux extrémités de la porte ne peuvent pas être identiques.")
        self.half_band = max(abs(self.signed_distance(self.line_1[0])), abs(self.signed_distance(self.line_2[0])), 8.0)
        self.inside_sign = 1 if self.signed_distance(self.inside_reference) >= 0 else -1

    def signed_distance(self, point: tuple[float, float]) -> float:
        x, y = point
        return ((self.mid_end[0] - self.mid_start[0]) * (y - self.mid_start[1]) - (self.mid_end[1] - self.mid_start[1]) * (x - self.mid_start[0])) / self.length

    def zone(self, point: tuple[float, float]) -> str:
        relative_distance = self.signed_distance(point) * self.inside_sign
        if relative_distance > self.half_band:
            return "INTERIEUR"
        if relative_distance < -self.half_band:
            return "EXTERIEUR"
        return "PORTE"

    def draw(self, image) -> None:
        cv2.line(image, self.line_1[0], self.line_1[1], (0, 255, 255), 3)
        cv2.line(image, self.line_2[0], self.line_2[1], (0, 255, 255), 3)
        cv2.circle(image, self.inside_reference, 6, (0, 255, 0), -1)
        cv2.putText(image, "INTERIEUR", (self.inside_reference[0] + 8, self.inside_reference[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(f"Vidéo introuvable : {args.video}")
    if not 0 <= args.confidence <= 1:
        raise ValueError("--confidence doit être compris entre 0 et 1.")
    if args.cooldown_frames < 0:
        raise ValueError("--cooldown-frames doit être positif ou nul.")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise ValueError("Vidéo illisible. Utilise de préférence un fichier MP4.")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    gate = Gate(load_config(args.config), width, height)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.video.stem}_count.mp4"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Impossible de créer la vidéo : {output_path}")

    entries = 0
    exits = 0
    stable_zone: dict[int, str] = {}
    last_count_frame: dict[int, int] = {}
    events: list[tuple[int, int, str]] = []

    try:
        model = YOLO(args.model)
        for frame_number, result in enumerate(model.track(
            source=str(args.video),
            stream=True,
            tracker="bytetrack.yaml",
            classes=[PERSON_CLASS_ID],
            conf=args.confidence,
            imgsz=args.image_size,
            verbose=False,
        )):
            annotated = result.plot()
            gate.draw(annotated)
            if result.boxes.id is not None:
                for box, track_id in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.id.int().tolist()):
                    x1, _y1, x2, y2 = box
                    foot = ((x1 + x2) / 2, y2)
                    current_zone = gate.zone(foot)
                    if current_zone == "PORTE":
                        continue
                    previous_zone = stable_zone.get(track_id)
                    enough_time = frame_number - last_count_frame.get(track_id, -args.cooldown_frames - 1) > args.cooldown_frames
                    if previous_zone and previous_zone != current_zone and enough_time:
                        if previous_zone == "EXTERIEUR" and current_zone == "INTERIEUR":
                            entries += 1
                            events.append((frame_number, track_id, "ENTREE"))
                            last_count_frame[track_id] = frame_number
                        elif previous_zone == "INTERIEUR" and current_zone == "EXTERIEUR":
                            exits += 1
                            events.append((frame_number, track_id, "SORTIE"))
                            last_count_frame[track_id] = frame_number
                    stable_zone[track_id] = current_zone

            cv2.rectangle(annotated, (10, 10), (265, 92), (0, 0, 0), -1)
            cv2.putText(annotated, f"Entrees : {entries}", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(annotated, f"Sorties : {exits}", (20, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            writer.write(annotated)
    finally:
        writer.release()

    print(f"Entrees comptees : {entries}")
    print(f"Sorties comptees : {exits}")
    print(f"Evenements : {events}")
    print(f"Video avec compteur : {output_path}")


if __name__ == "__main__":
    main()
