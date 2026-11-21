"""Géométrie et augmentations QVGA, sans dépendance PyTorch ou OpenCV."""

import numpy as np
from person_detection_math import targets_from_boxes

WIDTH, HEIGHT, STRIDE = 320, 240, 16


def zoom_translate(image, boxes, scale, offset_x, offset_y, width, height, stride):
    """Même transformation pour les pixels et les rectangles ; fond gris.

    Les coordonnées des bords suivent x' = scale*x + offset_x (en pixels).
    Une personne complètement hors champ disparaît ; une petite partie encore
    visible n'est jamais effacée de l'annotation : rejeter ce recadrage.
    """
    if image.shape != (1, height, width) or not np.isfinite([scale, offset_x, offset_y]).all() or scale <= 0:
        raise ValueError('Transformation géométrique invalide.')
    transformed = []
    for cx, cy, bw, bh in np.asarray(boxes, dtype=np.float64).reshape(-1, 4):
        full = np.array([(cx-bw/2)*width, (cy-bh/2)*height,
                         (cx+bw/2)*width, (cy+bh/2)*height])*scale
        full += [offset_x, offset_y, offset_x, offset_y]
        clipped = np.clip(full, [0, 0, 0, 0], [width, height, width, height])
        x1, y1, x2, y2 = clipped
        if x2 <= x1 or y2 <= y1:
            continue
        visible = (x2-x1)*(y2-y1) / ((full[2]-full[0])*(full[3]-full[1]))
        if x2-x1 < 4 or y2-y1 < 8 or visible < .2:
            raise ValueError('Personne encore visible mais trop petite après recadrage.')
        transformed.append(((x1+x2)/(2*width), (y1+y2)/(2*height),
                            (x2-x1)/width, (y2-y1)/height))
    if len(boxes) and not transformed:
        raise ValueError('Conserver au moins une personne dans une image positive.')
    transformed = np.asarray(transformed, dtype=np.float64).reshape(-1, 4)
    targets_from_boxes(transformed, width, height, stride)
    # Échantillonnage bilinéaire NumPy : fonctionne dans le venv Aidge sans OpenCV.
    yy, xx = np.mgrid[:height, :width]
    sx, sy = (xx+.5-offset_x)/scale-.5, (yy+.5-offset_y)/scale-.5
    inside = (sx >= -.5) & (sx < width-.5) & (sy >= -.5) & (sy < height-.5)
    sx, sy = np.clip(sx, 0, width-1), np.clip(sy, 0, height-1)
    x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
    x1, y1 = np.minimum(x0+1, width-1), np.minimum(y0+1, height-1)
    dx, dy = sx-x0, sy-y0
    plane = image[0]
    value = (plane[y0, x0]*(1-dx)*(1-dy) + plane[y0, x1]*dx*(1-dy)
             + plane[y1, x0]*(1-dx)*dy + plane[y1, x1]*dx*dy)
    return np.where(inside, value, .5)[None].astype(np.float32), transformed


def augment_profile(image, boxes, rng, enabled=True, width=WIDTH, height=HEIGHT, stride=STRIDE,
                    spatial=False):
    """Miroir et lumière ; zoom/recadrage facultatifs, uniquement pour le train."""
    if image.shape != (1, height, width):
        raise ValueError(f"Une image grise {width}x{height} est requise.")
    value = image.copy()
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    original_boxes = boxes.copy()
    if enabled and rng.random() < .5:
        value = value[..., ::-1]
        boxes[:, 0] = 1-boxes[:, 0]
    if enabled and spatial and rng.random() < .5:
        for _ in range(8):
            scale = rng.uniform(1., 1.8)
            offset_x = (1-scale)*width/2 + rng.uniform(-.15, .15)*width
            offset_y = (1-scale)*height/2 + rng.uniform(-.15, .15)*height
            try:
                cropped, cropped_boxes = zoom_translate(value, boxes, scale, offset_x, offset_y,
                                                        width, height, stride)
            except ValueError:
                continue
            value, boxes = cropped, cropped_boxes
            break
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
