"""CNN léger pour classifier une image en deux classes : vide ou personne."""

from __future__ import annotations

import torch
from torch import nn


class PersonPresenceCNN(nn.Module):
    """Petit réseau convolutif prévu pour des images RGB de 64 x 64 pixels.

    La sortie contient deux *logits* : indice 0 pour ``vide`` et indice 1 pour
    ``personne``. La fonction de perte ``CrossEntropyLoss`` appliquée pendant
    l'entraînement transforme ces logits en probabilités de façon adaptée.
    """

    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # 3 canaux RGB -> 16 cartes de caractéristiques, taille 64 x 64.
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            # Réduit 64 x 64 en 32 x 32.
            nn.MaxPool2d(kernel_size=2),
            # Recherche des motifs plus complexes dans les 16 cartes précédentes.
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            # Réduit 32 x 32 en 16 x 16.
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            # 32 cartes de 16 x 16 deviennent un seul vecteur de valeurs.
            nn.Flatten(),
            # Deux scores finaux : [score_vide, score_personne].
            nn.Linear(in_features=32 * 16 * 16, out_features=2),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Calcule les deux scores de classe pour un lot d'images."""

        features = self.features(images)
        return self.classifier(features)
