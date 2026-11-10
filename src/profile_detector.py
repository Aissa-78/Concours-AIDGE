"""Candidat QVGA profils, opérateurs Conv/ReLU et contexte couvrant le corps.

PyTorch définit/exporte le graphe et fournit le contrôle des gradients.
L'apprentissage des poids sera exécuté dans Aidge CPU, pas dans PyTorch.
"""

import torch
from torch import nn

from profile_training_data import WIDTH, HEIGHT, STRIDE


class ProfileDetector(nn.Module):
    def __init__(self, channels=24, blocks=3):
        super().__init__()
        layers = []
        for before, after in ((1, 8), (8, 12), (12, 16), (16, channels)):
            layers.extend([nn.Conv2d(before, after, 3, stride=2, padding=1), nn.ReLU()])
        # Champ réceptif 223 pixels avec trois blocs 5x5 à stride global 16.
        # Plus de contexte pour distinguer un corps entier d'un morceau de décor.
        for _ in range(blocks):
            layers.extend([nn.Conv2d(channels, channels, 5, padding=2), nn.ReLU()])
        layers.append(nn.Conv2d(channels, 5, 1))
        self.layers = nn.Sequential(*layers)
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
                nn.init.normal_(layer.bias, mean=0., std=.001)
        with torch.no_grad():
            self.layers[-1].weight.mul_(.01)
            self.layers[-1].bias[:4].fill_(3.)
            self.layers[-1].bias[4] = -4.

    def forward(self, image):
        return self.layers(image)
