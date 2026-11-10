"""Métriques de centres en coordonnées normalisées, indépendantes de la grille."""

import math
import numpy as np


def sigmoid(logits):
    return 1 / (1 + np.exp(-np.clip(logits, -80, 80)))


def find_centers(scores, threshold=0.5, separation=0.1):
    height, width = scores.shape
    padded = np.pad(scores, 1, constant_values=-np.inf)
    maxima = np.maximum.reduce([
        padded[dy:dy + height, dx:dx + width] for dy in range(3) for dx in range(3)
    ])
    candidates = np.argwhere((scores >= threshold) & (scores >= maxima - 1e-6))
    candidates = sorted(candidates, key=lambda point: float(scores[tuple(point)]), reverse=True)
    peaks = []
    for y, x in candidates:
        center = ((float(x) + 0.5) / width, (float(y) + 0.5) / height)
        if all(math.dist(center, other) > separation + 1e-12 for other in peaks):
            peaks.append(center)
    return peaks


def evaluate_logits(logits, truths, threshold=0.5, radius=0.125):
    return evaluate_scores(sigmoid(logits), truths, threshold, radius)


def evaluate_scores(scores, truths, threshold=0.5, radius=0.125):
    if scores.ndim != 4 or scores.shape[1] != 1 or len(scores) != len(truths):
        raise ValueError("Une carte (N,1,H,W) et une liste de centres par image sont nécessaires.")
    if not np.isfinite(scores).all() or not 0 < threshold < 1 or radius <= 0:
        raise ValueError("Sorties non finies ou paramètres de métrique invalides.")
    details = []
    for score, truth in zip(scores[:, 0], truths):
        predictions = find_centers(score, threshold)
        unmatched = set(range(len(truth)))
        pairs, false = [], []
        for pi, point in enumerate(predictions):
            matches = sorted((math.dist(point, truth[i]), i) for i in unmatched)
            if matches and matches[0][0] <= radius:
                pairs.append((pi, matches[0][1]))
                unmatched.remove(matches[0][1])
            else:
                false.append(pi)
        details.append({"tp": len(pairs), "fp": len(false), "fn": len(unmatched),
                        "predictions": predictions, "pairs": pairs, "false": false,
                        "missed": sorted(unmatched)})
    tp, fp, fn = (sum(row[key] for row in details) for key in ("tp", "fp", "fn"))
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0}, details


def weighted_bce(logits, target, positive_weight=5.0):
    """BCE stable sur logits, avec sa dérivée exacte pour scheduler.backward()."""
    loss = (1 - target) * np.logaddexp(0, logits) + positive_weight * target * np.logaddexp(0, -logits)
    gradient = ((1 + (positive_weight - 1) * target) * sigmoid(logits) - positive_weight * target) / target.size
    return float(loss.mean()), gradient.astype(np.float32)
