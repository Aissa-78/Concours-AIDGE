"""Prédit PERSONNE ou VIDE pour une image avec le modèle entraîné."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from model import PersonPresenceCNN
from preprocess import CLASS_NAMES, build_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tester une image personne / vide.")
    parser.add_argument("--image", type=Path, required=True, help="Image JPG ou PNG à tester.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/best_personne_vide_cnn.pt"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"Image introuvable : {args.image}")
    if not args.model_path.is_file():
        raise FileNotFoundError(
            f"Modèle introuvable : {args.model_path}. Lance d'abord src/train.py."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)

    model = PersonPresenceCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # La préparation est volontairement la même qu'à l'entraînement.
    with Image.open(args.image) as image:
        image_tensor = build_transform()(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        prediction_index = probabilities.argmax(dim=1).item()
        confidence = probabilities[0, prediction_index].item()

    label = CLASS_NAMES[prediction_index]
    print("PERSONNE" if label == "personne" else "VIDE")
    print(f"Confiance : {confidence:.1%}")


if __name__ == "__main__":
    main()
