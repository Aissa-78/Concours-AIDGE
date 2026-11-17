"""Géométrie et augmentations QVGA, sans dépendance PyTorch ou OpenCV."""

import numpy as np
from person_detection_math import targets_from_boxes

WIDTH, HEIGHT, STRIDE = 320, 240, 16


def augment_profile(image, boxes, rng, enabled=True, width=WIDTH, height=HEIGHT, stride=STRIDE):
    """Miroir horizontal et lumière ; jamais de personnes la tête en bas."""
    if image.shape != (1, height, width):
        raise ValueError(f"Une image grise {width}x{height} est requise.")
    value = image.copy()
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    original_boxes = boxes.copy()
    if enabled and rng.random() < .5:
        value = value[..., ::-1]
        boxes[:, 0] = 1-boxes[:, 0]
    if enabled:
        value = value*rng.uniform(.85, 1.15)+rng.uniform(-.06, .06)
        if rng.random() < .2:
            padded = np.pad(value, ((0, 0), (1, 1), (1, 1)), mode="edge")
            value = sum(padded[:, dy:dy+height, dx:dx+width] for dy in range(3) for dx in range(3))/9
        if rng.random() < .2:
            value += rng.normal(0, .008, value.shape)
    try:
        targets = targets_from_boxes(boxes, width, height, stride)
    except ValueError:
        # Une boîte étroite peut perdre son ancre après miroir : garder l'original.
        value, boxes = image.copy(), original_boxes
        targets = targets_from_boxes(boxes, width, height, stride)
    return np.ascontiguousarray(np.clip(value, 0, 1), dtype=np.float32), targets, boxes
