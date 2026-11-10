"""Architecture V2 : initialisation/export avec PyTorch, apprentissage dans Aidge."""

from torch import nn


class HeatmapV2(nn.Module):
    """Image grise 128x128 -> 20x20 logits, contexte de 52x52 pixels.

    Convolutions sans padding interne pour l'apprentissage Aidge CPU.
    Les poids seront entraînés avec la rétropropagation Aidge, pas par cette définition.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 8, 9), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 12, 9), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(12, 16, 7), nn.ReLU(),
            nn.Conv2d(16, 1, 1),
        )
        nn.init.constant_(self.layers[-1].bias, -2.0)

    def forward(self, image):
        return self.layers(image)
