"""Entraîne le premier classifieur FlowSense : vide (0) ou personne (1)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from model import PersonPresenceCNN
from preprocess import CLASS_NAMES, IMAGE_SIZE, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entraîner le CNN personne / vide.")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/best_personne_vide_cnn.pt"),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="0 est le choix le plus fiable pour commencer sur macOS.",
    )
    return parser.parse_args()


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> int:
    """Retourne le nombre de prédictions correctes dans un lot."""

    predictions = logits.argmax(dim=1)
    return (predictions == labels).sum().item()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Exécute une époque d'entraînement ou de validation.

    Si ``optimizer`` est fourni, les poids sont modifiés : c'est
    l'entraînement. Sans optimizer, seuls les résultats sont mesurés : c'est
    la validation.
    """

    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_correct = 0
    total_images = 0

    # Aucun gradient n'est nécessaire pendant la validation : cela économise
    # mémoire et temps de calcul.
    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if is_training:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += compute_accuracy(logits, labels)
            total_images += batch_size

    return total_loss / total_images, total_correct / total_images


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs et --batch-size doivent être supérieurs à 0.")

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = load_dataset(args.data_dir / "train")
    validation_dataset = load_dataset(args.data_dir / "validation")
    if len(train_dataset) == 0 or len(validation_dataset) == 0:
        raise ValueError("Les dossiers train et validation doivent chacun contenir des images.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = PersonPresenceCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    best_validation_accuracy = -1.0

    print(f"Appareil utilisé : {device}")
    print(f"Images train : {len(train_dataset)} | validation : {len(validation_dataset)}")
    print(f"Classes : 0={CLASS_NAMES[0]}, 1={CLASS_NAMES[1]}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        validation_loss, validation_accuracy = run_epoch(
            model, validation_loader, criterion, device
        )

        print(
            f"Époque {epoch:02d}/{args.epochs} | "
            f"train loss={train_loss:.4f}, accuracy={train_accuracy:.2%} | "
            f"validation loss={validation_loss:.4f}, accuracy={validation_accuracy:.2%}"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "image_size": IMAGE_SIZE,
                },
                args.model_path,
            )
            print(f"  Meilleur modèle sauvegardé : {args.model_path}")

    print(f"Entraînement terminé. Meilleure accuracy validation : {best_validation_accuracy:.2%}")


if __name__ == "__main__":
    main()
