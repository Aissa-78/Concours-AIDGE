"""Crée un réglage de porte en cliquant directement sur une image ou une vidéo.

Le réglage est indépendant de l'IA : il indique simplement où se trouve la
porte et quel côté est l'intérieur. Il est donc fait une seule fois après la
pose d'une caméra fixe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


POINT_LABELS = (
    "1/5 - début de la première ligne",
    "2/5 - fin de la première ligne",
    "3/5 - début de la deuxième ligne",
    "4/5 - fin de la deuxième ligne",
    "5/5 - un point du côté INTÉRIEUR",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cliquer la zone de porte et enregistrer son réglage."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Image JPG ou PNG à calibrer.")
    source.add_argument("--video", type=Path, help="Vidéo MP4 à calibrer.")
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Image de la vidéo à afficher (0 = première image).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Fichier JSON de réglage à créer.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Nom lisible de cette porte (par défaut : nom du fichier JSON).",
    )
    return parser.parse_args()


def load_frame(args: argparse.Namespace):
    if args.image:
        image = cv2.imread(str(args.image))
        if image is None:
            raise ValueError(f"Image illisible : {args.image}")
        return image

    if args.frame < 0:
        raise ValueError("--frame doit être positif ou nul.")
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise ValueError(f"Vidéo illisible : {args.video}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, image = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"Image {args.frame} introuvable dans : {args.video}")
    return image


def draw_calibration(image, points: list[tuple[int, int]]):
    canvas = image.copy()
    for index, point in enumerate(points, start=1):
        color = (0, 255, 0) if index == 5 else (0, 255, 255)
        cv2.circle(canvas, point, 7, color, -1)
        cv2.putText(
            canvas,
            str(index),
            (point[0] + 9, point[1] - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    if len(points) >= 2:
        cv2.line(canvas, points[0], points[1], (0, 255, 255), 3)
    if len(points) >= 4:
        cv2.line(canvas, points[2], points[3], (0, 255, 255), 3)

    if len(points) < 5:
        instruction = POINT_LABELS[len(points)]
    else:
        instruction = "Prêt : S enregistrer | U annuler dernier point | R recommencer | Q quitter"
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        instruction,
        (10, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def normalized(point: tuple[int, int], width: int, height: int) -> list[float]:
    return [round(point[0] / width, 6), round(point[1] / height, 6)]


def save_config(args: argparse.Namespace, points: list[tuple[int, int]], width: int, height: int) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "name": args.name or args.output.stem,
        "description": "Zone de porte réglée une fois pour une caméra fixe.",
        "coordinates": "normalised_0_to_1",
        "gate": {
            "line_1": [normalized(points[0], width, height), normalized(points[1], width, height)],
            "line_2": [normalized(points[2], width, height), normalized(points[3], width, height)],
        },
        "inside_reference": normalized(points[4], width, height),
    }
    args.output.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    print(f"Réglage enregistré : {args.output}")


def main() -> None:
    args = parse_args()
    image = load_frame(args)
    height, width = image.shape[:2]
    points: list[tuple[int, int]] = []
    window_name = "FlowSense - calibrage de la porte"

    def on_click(event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 5:
            points.append((x, y))

    print("Clique les 5 points indiqués en haut de la fenêtre.")
    print("Après le 5e point : S = enregistrer, U = annuler, R = recommencer, Q = quitter.")
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_click)

    while True:
        cv2.imshow(window_name, draw_calibration(image, points))
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            points.clear()
        if key == ord("u") and points:
            points.pop()
        if key == ord("s"):
            if len(points) != 5:
                print("Il faut cliquer les 5 points avant d'enregistrer.")
            else:
                save_config(args, points, width, height)
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
