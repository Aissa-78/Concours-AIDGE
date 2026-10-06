"""Entraîne la carte de positions sur les rectangles du dataset de détection.

Tant que les étiquettes YOLO n'ont pas été corrigées, le résultat est une
expérience de reproduction du professeur (YOLO), pas une mesure de qualité
réelle sur des personnes.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from tiny_heatmap import TinyHeatmap


IMAGE_SIZE = 96
GRID_SIZE = 20


class DetectionFrames(Dataset):
    def __init__(self, root: Path, rows: list[dict[str, str]]) -> None:
        self.root = root
        self.rows = rows
        self.grid_x, self.grid_y = np.meshgrid(
            np.arange(GRID_SIZE, dtype=np.float32),
            np.arange(GRID_SIZE, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        frame = cv2.imread(str(self.root / row["image"]))
        if frame is None:
            raise RuntimeError(f"Image illisible : {row['image']}")
        height, width = frame.shape[:2]
        scale = min(IMAGE_SIZE / width, IMAGE_SIZE / height)
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
        left = (IMAGE_SIZE - new_width) // 2
        top = (IMAGE_SIZE - new_height) // 2
        canvas = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 114, dtype=np.uint8)
        canvas[top:top + new_height, left:left + new_width] = cv2.resize(
            frame, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float() / 255.0

        target = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        label_path = self.root / row["label"]
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5 or parts[0] != "0":
                raise ValueError(f"Étiquette invalide : {label_path}")
            center_x = float(parts[1]) * new_width + left
            center_y = float(parts[2]) * new_height + top
            grid_x = center_x * GRID_SIZE / IMAGE_SIZE - 0.5
            grid_y = center_y * GRID_SIZE / IMAGE_SIZE - 0.5
            distance_squared = (self.grid_x - grid_x) ** 2 + (self.grid_y - grid_y) ** 2
            peak = np.exp(-distance_squared / (2 * 1.0 ** 2))
            target = np.maximum(target, peak)
        return image, torch.from_numpy(target[None, :, :])


def run_epoch(
    model: nn.Module, loader: DataLoader, loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    model.train(optimizer is not None)
    total = 0.0
    for images, targets in loader:
        with torch.set_grad_enabled(optimizer is not None):
            scores = model(images)
            loss = loss_fn(scores, targets)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total += float(loss.item()) * len(images)
    return total / len(loader.dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("dataset/detection_proposals")
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("models/tiny_heatmap_pseudo.pt")
    )
    parser.add_argument(
        "--allow-unreviewed-validation", action="store_true",
        help="Seulement pour un essai technique : la validation n'est alors pas une vérité terrain.",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise SystemExit("Époques, taille de lot et taux d'apprentissage doivent être positifs.")

    with (args.data_dir / "manifest.csv").open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    if not train_rows or not validation_rows:
        raise SystemExit("Il faut des images dans train ET validation.")
    train_videos = {row["video"] for row in train_rows}
    validation_videos = {row["video"] for row in validation_rows}
    if train_videos & validation_videos:
        raise SystemExit("Une même vidéo se trouve dans train et validation.")
    unreviewed_validation = sum(row["reviewed"] != "oui" for row in validation_rows)
    if unreviewed_validation and not args.allow_unreviewed_validation:
        raise SystemExit(
            f"{unreviewed_validation} images de validation restent à vérifier. "
            "Utilisez scripts/review_detection_dataset.py. Pour un essai technique "
            "seulement, ajoutez --allow-unreviewed-validation."
        )
    if unreviewed_validation:
        print("ATTENTION : validation sur propositions YOLO non corrigées.")
        print("La loss ci-dessous n'est PAS une mesure de qualité réelle.")
    if any(row["reviewed"] != "oui" for row in train_rows):
        print("ATTENTION : des étiquettes d'entraînement sont encore des propositions YOLO.")

    torch.manual_seed(7)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    train_loader = DataLoader(
        DetectionFrames(args.data_dir, train_rows), batch_size=args.batch_size,
        shuffle=True, num_workers=0,
    )
    validation_loader = DataLoader(
        DetectionFrames(args.data_dir, validation_rows), batch_size=args.batch_size,
        shuffle=False, num_workers=0,
    )
    model = TinyHeatmap()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15.0]))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_validation_loss = float("inf")
    print(f"Train : {len(train_rows)} images de {len(train_videos)} vidéos")
    print(f"Validation : {len(validation_rows)} images de {len(validation_videos)} vidéos")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer)
        validation_loss = run_epoch(model, validation_loader, loss_fn, None)
        print(
            f"Époque {epoch:02d}/{args.epochs} : "
            f"loss train={train_loss:.4f}, validation={validation_loss:.4f}",
            flush=True,
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "image_size": IMAGE_SIZE,
                "grid_size": GRID_SIZE,
                "teacher_labels_unreviewed": unreviewed_validation > 0,
                "best_validation_loss": best_validation_loss,
            }, args.checkpoint)
    print(f"Meilleur checkpoint : {args.checkpoint}")
    if unreviewed_validation:
        print("Corriger la validation avant de mesurer la qualité réelle.")
    else:
        print("Validation corrigée : mesurez maintenant les erreurs de localisation.")


if __name__ == "__main__":
    main()
