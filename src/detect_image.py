"""Détecte les personnes sur une image avec un YOLO nano déjà entraîné.

Ce script est une démonstration de la première brique du futur compteur :
il dessine un rectangle autour de chaque personne trouvée. Il ne suit pas
encore les personnes dans une vidéo et ne compte pas les entrées/sorties.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


PERSON_CLASS_ID = 0  # Dans le modèle COCO utilisé ici, 0 correspond à "person".


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dessiner les rectangles YOLO autour des personnes d'une image."
    )
    parser.add_argument("--image", type=Path, required=True, help="Image JPG ou PNG à analyser.")
    parser.add_argument(
        "--model",
        default="yolo26n.pt",
        help="Modèle YOLO. Il est téléchargé automatiquement au premier lancement.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/detect_image"),
        help="Dossier qui recevra l'image annotée.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Confiance minimale entre 0 et 1.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=320,
        help="Taille d'analyse : plus petite = plus rapide, mais moins précise.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"Image introuvable : {args.image}")
    if not 0 <= args.confidence <= 1:
        raise ValueError("--confidence doit être compris entre 0 et 1.")

    # Certaines photos de téléphone ou WhatsApp sont des JPEG progressifs que
    # le lecteur interne d'Ultralytics refuse, alors qu'OpenCV sait les lire.
    # On charge donc l'image nous-mêmes et on donne directement ses pixels à
    # YOLO. Cela rend le script plus fiable pour les photos apportées par
    # l'équipe.
    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError(
            "Image illisible. Utilise une image JPG ou PNG qui s'ouvre dans Aperçu."
        )

    # Le modèle nano connaît déjà de nombreux objets. On lui demande seulement
    # de conserver la classe "person" pour notre scénario FlowSense.
    model = YOLO(args.model)
    predictions = model.predict(
        source=image,
        classes=[PERSON_CLASS_ID],
        conf=args.confidence,
        imgsz=args.image_size,
        verbose=False,
    )
    if not predictions:
        raise RuntimeError("YOLO n'a renvoyé aucun résultat pour cette image.")
    result = predictions[0]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.image.stem}_yolo.jpg"
    if not cv2.imwrite(str(output_path), result.plot()):
        raise RuntimeError(f"Impossible d'enregistrer l'image : {output_path}")

    print(f"Personnes détectées : {len(result.boxes)}")
    print(f"Image avec rectangles : {output_path}")


if __name__ == "__main__":
    main()
