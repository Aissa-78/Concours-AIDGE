"""Protocole QQVGA : séparation des sources, mélange et augmentations des boîtes.

NumPy uniquement, pour fonctionner aussi dans l'environnement Aidge sans OpenCV.
"""

import numpy as np

from person_detection_math import targets_from_boxes


def assert_disjoint(records):
    """Refuse une image ou une vidéo commune entre apprentissage et validation."""
    groups = {}
    for record in records:
        split = record["row"]["split"]
        if split not in ("train", "validation"):
            raise ValueError("Seuls train et validation sont admis dans ce protocole.")
        # Les segments _2 de WEPDTOF représentent le même lieu.
        video = record["row"]["video"]
        if record["source"] == "wepdtof":
            video = video.removesuffix("_2")
        groups.setdefault(split, {"images": set(), "videos": set()})
        groups[split]["images"].add(record["image_sha256"])
        groups[split]["videos"].add((record["source"], video))
    if set(groups) != {"train", "validation"}:
        raise ValueError("Les deux groupes train et validation sont nécessaires.")
    for key in ("images", "videos"):
        if groups["train"][key] & groups["validation"][key]:
            raise ValueError(f"Fuite entre train et validation : {key} en commun.")


def balanced_order(sources, rng):
    """Utilise TOUS les exemples ; répète la petite source pour obtenir 50/50."""
    sources = np.asarray(sources)
    names = sorted(set(sources.tolist()))
    if len(names) != 2:
        raise ValueError("Ce protocole attend exactement deux sources.")
    groups = [np.flatnonzero(sources == name) for name in names]
    size = max(map(len, groups))
    orders = []
    for group in groups:
        order = rng.permutation(group)
        if len(order) < size:
            order = np.concatenate([order, rng.choice(group, size-len(order), replace=True)])
        orders.append(rng.permutation(order))
    # Chaque batch de taille paire contient autant de chaque source.
    return np.stack(orders, axis=1).reshape(-1)


def augment_sample(image, boxes, rng, geometric=True, photometric=True):
    """Miroirs synchronisés image/boîtes ; pas de rotation 90° sur 160x120.

    Les cibles sont RECALCULÉES depuis les boîtes transformées : inverser une
    heatmap ne suffit pas, car les distances gauche/droite doivent aussi changer.
    """
    if image.shape != (1, 120, 160):
        raise ValueError("Entrée QQVGA grise attendue.")
    # Garder les coordonnées en float64 : .475 * 40 est une frontière de
    # cellule, que l'arrondi float32 peut déplacer d'une cellule entière.
    transformed = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    value = image.copy()
    if geometric:
        horizontal, vertical = rng.random(2) < .5
        if horizontal:
            value = value[..., ::-1]
            transformed[:, 0] = 1-transformed[:, 0]
        if vertical:
            value = value[:, ::-1, :]
            transformed[:, 1] = 1-transformed[:, 1]
    try:
        targets = targets_from_boxes(transformed)
    except ValueError:
        # Une très petite boîte peut ne plus contenir son ancre après miroir.
        # On garde alors l'exemple original entier, sans supprimer une personne.
        value, transformed = image.copy(), np.asarray(boxes).reshape(-1, 4).copy()
        targets = targets_from_boxes(transformed)
    if photometric:
        value = value*rng.uniform(.85, 1.15)+rng.uniform(-.06, .06)
        if rng.random() < .2:
            padded = np.pad(value, ((0, 0), (1, 1), (1, 1)), mode="edge")
            value = sum(padded[:, dy:dy+120, dx:dx+160] for dy in range(3) for dx in range(3))/9
        if rng.random() < .2:
            value = value+rng.normal(0, .008, value.shape)
    return np.ascontiguousarray(np.clip(value, 0, 1), dtype=np.float32), targets, transformed


def selection_key(metrics_by_source, loss):
    """Priorité au moins bon F1 de validation, puis à la moyenne et à la loss."""
    scores = [metrics["f1"] for metrics in metrics_by_source.values()]
    return min(scores), float(np.mean(scores)), -float(loss)
