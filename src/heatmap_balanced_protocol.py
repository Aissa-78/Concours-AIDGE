"""Protocole de mélange et de sélection, testable sans installer Aidge."""

import numpy as np


def balanced_batches(local_count, public_count, batch_size, rng):
    """Chaque image locale une fois, autant de publiques ; chaque batch est 50/50.

    Les indices publics sont décalés pour indexer [images locales, publiques].
    Si le public est plus petit, son échantillonnage autorise la répétition.
    """
    if min(local_count, public_count) < 1 or batch_size < 2 or batch_size % 2:
        raise ValueError("Deux sources non vides et un batch pair >= 2 sont nécessaires.")
    local = rng.permutation(local_count)
    public = rng.choice(public_count, local_count, replace=public_count < local_count) + local_count
    half = batch_size // 2
    return [rng.permutation(np.concatenate((local[i:i+half], public[i:i+half])))
            for i in range(0, local_count, half)]


def selection_key(local_f1, public_f1, local_loss, public_loss):
    """Priorité au domaine le plus faible, puis moyenne, puis loss moyenne."""
    values = np.array([local_f1, public_f1, local_loss, public_loss], dtype=float)
    if not np.isfinite(values).all() or not np.logical_and(values[:2] >= 0, values[:2] <= 1).all():
        raise ValueError("Scores finis entre 0 et 1 et losses finies nécessaires.")
    if min(local_loss, public_loss) < 0:
        raise ValueError("Les losses doivent être positives ou nulles.")
    return (min(local_f1, public_f1), (local_f1 + public_f1) / 2,
            -(local_loss + public_loss) / 2)
