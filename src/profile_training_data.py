"""Géométrie et augmentations QVGA, sans dépendance PyTorch ou OpenCV."""

import numpy as np
from person_detection_math import targets_from_boxes

WIDTH, HEIGHT, STRIDE = 320, 240, 16


def augment_profile(image, boxes, rng, enabled=True):
    """Miroir horizontal et lumière ; jamais de personnes la tête en bas."""
    if image.shape != (1, HEIGHT, WIDTH):
        raise ValueError("Une image grise 320x240 est requise.")
    value = image.copy()
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    if enabled and rng.random() < .5:
        value = value[..., ::-1]
        boxes[:, 0] = 1-boxes[:, 0]
    if enabled:
        value = value*rng.uniform(.85, 1.15)+rng.uniform(-.06, .06)
        if rng.random() < .2:
            padded = np.pad(value, ((0, 0), (1, 1), (1, 1)), mode="edge")
            value = sum(padded[:, dy:dy+HEIGHT, dx:dx+WIDTH] for dy in range(3) for dx in range(3))/9
        if rng.random() < .2:
            value += rng.normal(0, .008, value.shape)
    targets = targets_from_boxes(boxes, WIDTH, HEIGHT, STRIDE)
    return np.ascontiguousarray(np.clip(value, 0, 1), dtype=np.float32), targets, boxes
