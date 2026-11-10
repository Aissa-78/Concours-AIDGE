"""Détecteur de personnes QQVGA, inspiré du principe séparable de BikeCounter.

Définition/initialisation et référence de gradients seulement : l'apprentissage
est exécuté dans Aidge. Aucun poids appris sur les vélos n'est repris.
"""

from torch import nn
import torch


class PersonDetectorQQVGA(nn.Module):
    stride = 4

    def __init__(self, channels=16, blocks=6, training_dense=False, residual=False):
        super().__init__()
        self.depthwise_indices = []
        self.training_dense = training_dense
        self.residual = residual
        layers = [nn.Conv2d(1, 8, 3, stride=2, padding=1), nn.ReLU()]

        def depthwise(count, kernel=5, stride=1):
            self.depthwise_indices.append(len(layers))
            layers.extend([nn.Conv2d(count, count, kernel, stride=stride, padding=kernel//2,
                                     groups=1 if training_dense else count), nn.ReLU()])

        depthwise(8, stride=2)
        layers.extend([nn.Conv2d(8, channels, 1), nn.ReLU()])
        for _ in range(blocks):
            depthwise(channels)
            layers.extend([nn.Conv2d(channels, channels, 1), nn.ReLU()])
        layers.append(nn.Conv2d(channels, 5, 1))
        self.layers = nn.Sequential(*layers)
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
                # Valeurs distinctes : évite la déduplication ONNX de biais nuls
                # pourtant indépendants pendant l'apprentissage.
                nn.init.normal_(layer.bias, mean=0., std=.001)
        if residual:
            # La branche démarre petite : les détails du chemin direct restent
            # accessibles, sans amplification à chaque addition résiduelle.
            with torch.no_grad():
                for index in range(8, len(self.layers)-1, 4):
                    self.layers[index].weight.mul_(.1)
                    self.layers[index].bias.mul_(.1)
        if training_dense:
            with torch.no_grad():
                for name, mask in self.parameter_masks().items():
                    dict(self.named_parameters())[name].mul_(mask)
        with torch.no_grad():
            # Quatre distances au rectangle, puis logit de présence.
            self.layers[-1].weight.mul_(.01)
            self.layers[-1].bias[:4].fill_(3.)
            self.layers[-1].bias[4] = -4.

    def parameter_masks(self):
        masks = {}
        if self.training_dense:
            for index in self.depthwise_indices:
                layer = self.layers[index]
                mask = torch.zeros_like(layer.weight)
                diagonal = torch.arange(layer.out_channels)
                mask[diagonal, diagonal] = 1
                masks[f"layers.{index}.weight"] = mask
        return masks

    def copy_from_compact(self, compact):
        """Les deux graphes calculent la même fonction, avec les mêmes paramètres utiles."""
        if not self.training_dense or compact.training_dense or self.residual != compact.residual:
            raise ValueError("Copie attendue : compact vers dense contraint.")
        with torch.no_grad():
            for index, (destination, source) in enumerate(zip(self.layers, compact.layers)):
                if not isinstance(destination, nn.Conv2d):
                    continue
                destination.bias.copy_(source.bias)
                if index in self.depthwise_indices:
                    destination.weight.zero_()
                    diagonal = torch.arange(source.out_channels)
                    destination.weight[diagonal, diagonal] = source.weight[:, 0]
                else:
                    destination.weight.copy_(source.weight)

    def compact_state(self, dense_arrays):
        """Poids NumPy du graphe dense -> state_dict compact exportable."""
        if self.training_dense:
            raise ValueError("Appeler cette méthode sur le modèle compact.")
        result = {}
        for name in self.state_dict():
            array = dense_arrays[name.replace(".", "_")]
            if name.endswith("weight") and int(name.split(".")[1]) in self.depthwise_indices:
                diagonal = torch.arange(array.shape[0]).numpy()
                array = array[diagonal, diagonal][:, None]
            result[name] = torch.from_numpy(array.copy())
        return result

    def forward(self, image):
        if self.residual:
            value = self.layers[:6](image)
            for start in range(6, len(self.layers)-1, 4):
                branch = self.layers[start+2](self.layers[start+1](self.layers[start](value)))
                value = self.layers[start+3](value+branch)
            return self.layers[-1](value)
        return self.layers(image)


class ConvPersonDetectorQQVGA(PersonDetectorQQVGA):
    """Variante à convolutions ordinaires : plus de capacité, sans nouvel opérateur.

    22 757 paramètres avec 16 canaux et trois blocs ; l'export et la puissance
    doivent être vérifiés, pas déduits du seul nombre de paramètres.
    """

    def __init__(self, channels=16, blocks=3, training_dense=False, spatial_kernel=5):
        nn.Module.__init__(self)
        self.depthwise_indices = []
        self.training_dense = training_dense
        self.residual = False
        layers = [nn.Conv2d(1, 8, 5, stride=2, padding=2), nn.ReLU(),
                  nn.Conv2d(8, channels, 5, stride=2, padding=2), nn.ReLU()]
        for _ in range(blocks):
            layers.extend([nn.Conv2d(channels, channels, spatial_kernel, padding=spatial_kernel//2), nn.ReLU()])
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


def make_person_detector(channels=16, blocks=6, training_dense=False, residual=False, architecture="separable"):
    if architecture in ("regular", "regular_context"):
        if residual:
            raise ValueError("La variante régulière n'utilise pas de connexion résiduelle.")
        return ConvPersonDetectorQQVGA(channels, blocks, training_dense=training_dense,
                                      spatial_kernel=7 if architecture == "regular_context" else 5)
    if architecture != "separable":
        raise ValueError("Architecture de détecteur inconnue.")
    return PersonDetectorQQVGA(channels, blocks, training_dense=training_dense, residual=residual)
