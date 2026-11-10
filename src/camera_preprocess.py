"""Même simulation d'image caméra pour nos vidéos et les images publiques.

Réduit la résolution et supprime la couleur, sans flou ou JPEG artificiel.
Les bandes grises conservent les proportions et évitent de couper les personnes.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraGeometry:
    width: int
    height: int
    resized_width: int
    resized_height: int
    left: int
    top: int

    def transform_box(self, box):
        """Rectangle YOLO normalisé d'origine -> rectangle dans l'image préparée."""
        cx, cy, bw, bh = map(float, box)
        values = np.array([cx, cy, bw, bh])
        if not np.isfinite(values).all() or not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
            raise ValueError("Rectangle non fini ou coordonnées invalides.")
        if min(cx - bw/2, cy - bh/2) < -1e-4 or max(cx + bw/2, cy + bh/2) > 1.0001:
            raise ValueError("Rectangle hors de l'image source.")
        return ((cx * self.resized_width + self.left) / self.width,
                (cy * self.resized_height + self.top) / self.height,
                bw * self.resized_width / self.width,
                bh * self.resized_height / self.height)


def prepare_camera_frame(frame, width=160, height=120):
    """Image uint8 BGR/grise -> gris uint8 (H,W) + transformation des annotations.

    Dimensions choisies pour le scénario ; limite QVGA 320x240 du carnet.
    OpenCV reste importé à l'appel pour ne pas l'imposer à l'environnement Aidge.
    """
    import cv2

    if not isinstance(width, int) or not isinstance(height, int) or not (1 <= width <= 320 and 1 <= height <= 240):
        raise ValueError("Résolution positive limitée à 320x240 (largeur x hauteur).")
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim not in (2, 3) or min(frame.shape[:2]) == 0:
        raise ValueError("Image uint8 non vide nécessaire.")
    if frame.ndim == 2:
        gray = frame
    elif frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Enlever les couleurs : un seul canal.
    elif frame.shape[2] == 4:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError("Image grise, BGR ou BGRA nécessaire.")
    factor = min(width / frame.shape[1], height / frame.shape[0])
    rw, rh = max(1, round(frame.shape[1] * factor)), max(1, round(frame.shape[0] * factor))
    left, top = (width - rw) // 2, (height - rh) // 2
    resized = cv2.resize(gray, (rw, rh), interpolation=cv2.INTER_AREA)  # Simuler les pixels d'une caméra basse résolution.
    canvas = np.full((height, width), 114, dtype=np.uint8)
    canvas[top:top+rh, left:left+rw] = resized
    return canvas, CameraGeometry(width, height, rw, rh, left, top)
