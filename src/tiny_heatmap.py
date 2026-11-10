"""Petit réseau produisant une carte de positions de personnes."""

from torch import nn


class TinyHeatmap(nn.Module):
    """Convertit une image RGB 96x96 en 20x20 scores bruts."""

    def __init__(self, activation: str = "hardtanh") -> None:
        super().__init__()
        if activation not in ("hardtanh", "relu"):
            raise ValueError(f"Activation inconnue : {activation}")
        self.activation = activation

        def make_activation() -> nn.Module:
            return nn.Hardtanh(0, 1) if activation == "hardtanh" else nn.ReLU()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3),
            make_activation(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(8, 16, kernel_size=3),
            make_activation(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 16, kernel_size=3),
            make_activation(),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, image):
        return self.layers(image)
