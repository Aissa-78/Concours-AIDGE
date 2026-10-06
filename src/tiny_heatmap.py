"""Petit réseau expérimental pour produire une carte de positions.

Le réseau n'est pas entraîné : les valeurs de la carte ne signifient pas encore
qu'une personne a été détectée. Ce module sert d'abord au test PyTorch/Aidge.
"""

from torch import nn


class TinyHeatmap(nn.Module):
    """Convertit une image RGB 96x96 en 20x20 scores bruts."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3),
            nn.Hardtanh(0, 1),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(8, 16, kernel_size=3),
            nn.Hardtanh(0, 1),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 16, kernel_size=3),
            nn.Hardtanh(0, 1),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, image):
        return self.layers(image)
