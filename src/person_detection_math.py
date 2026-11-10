"""Cibles, loss analytique et rectangles pour QQVGA ; aucun moteur d'apprentissage."""

import numpy as np

WIDTH, HEIGHT, STRIDE = 160, 120, 4


def hard_negative_weights(values, weights, positive):
    """Concentre la pénalité du fond sur les fausses réponses les plus fortes.

    Chaque image vide garde 16 négatifs difficiles ; sinon 16 par centre vrai.
    Les centres positifs (poids négatif nul) ne peuvent jamais être sélectionnés.
    """
    selected = np.zeros_like(weights)
    for i in range(len(weights)):
        valid = np.flatnonzero(weights[i].ravel() > 0)
        count = min(len(valid), max(16, int(positive[i].sum())*16))
        if count:
            local = values[i].ravel()[valid]
            indices = valid[np.argpartition(local, len(local)-count)[-count:]]
            selected[i].ravel()[indices] = weights[i].ravel()[indices]
    return selected


def targets_from_boxes(boxes, width=WIDTH, height=HEIGHT, stride=STRIDE):
    """Un centre par personne et quatre distances aux bords, en cellules de grille."""
    if min(width, height, stride) < 1 or width % stride or height % stride:
        raise ValueError("Dimensions positives divisibles par le stride requises.")
    heat = np.zeros((1, height//stride, width//stride), np.float32)
    distances = np.zeros((4, *heat.shape[1:]), np.float32)
    mask = np.zeros_like(heat)
    yy, xx = np.mgrid[:heat.shape[1], :heat.shape[2]]
    for cx, cy, bw, bh in boxes:
        if not np.isfinite([cx, cy, bw, bh]).all() or not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
            raise ValueError("Coordonnées de rectangle invalides.")
        if min(cx-bw/2, cy-bh/2) < -1e-4 or max(cx+bw/2, cy+bh/2) > 1.0001:
            raise ValueError("Rectangle hors image.")
        x, y = min(int(cx*width/stride), heat.shape[2]-1), min(int(cy*height/stride), heat.shape[1]-1)
        if mask[0, y, x]:
            raise ValueError("Deux personnes occupent la même cellule : choix non adapté à ce test.")
        anchor_x, anchor_y = (x+.5)*stride, (y+.5)*stride
        box = np.array([(cx-bw/2)*width, (cy-bh/2)*height, (cx+bw/2)*width, (cy+bh/2)*height])
        values = np.array([anchor_x-box[0], anchor_y-box[1], box[2]-anchor_x, box[3]-anchor_y])/stride
        if values.min() < 0:
            raise ValueError("Rectangle trop petit pour contenir le point d'ancrage.")
        distances[:, y, x] = values
        mask[0, y, x] = 1
        heat[0] = np.maximum(heat[0], np.exp(-((xx-x)**2+(yy-y)**2)/(2*.8**2)))
    return heat, distances, mask


def detection_loss(output, heat, distances, mask, classification="focal"):
    """Focal loss de centres + SmoothL1 des quatre distances et gradient exact."""
    z = output[:, 4:5]
    p = 1/(1+np.exp(-np.clip(z, -80, 80)))
    log_p, log_n = -np.logaddexp(0, -z), -np.logaddexp(0, z)
    positive = (heat == 1).astype(np.float32)
    negative_weight = (1-heat)**4 * (1-positive)
    count = max(float(positive.sum()), 1.)
    focal = -(positive*(1-p)**2*log_p + negative_weight*p**2*log_n).sum()/count
    gradient_z = (positive*(2*p*(1-p)**2*log_p-(1-p)**3)
                  + negative_weight*(-2*p**2*(1-p)*log_n+p**3))/count
    classification_name = "focal"
    if classification in ("balanced_bce", "hard_bce"):
        # Équilibrer les deux groupes séparément : 1–3 centres de personnes
        # ne doivent pas se retrouver noyés parmi environ 1 200 cellules de fond.
        if classification == "hard_bce":
            negative_weight = hard_negative_weights(-negative_weight*log_n, negative_weight, positive)
        negative_count = max(float(negative_weight.sum()), 1.)
        focal = -(positive*log_p).sum()/count-(negative_weight*log_n).sum()/negative_count
        gradient_z = positive*(p-1)/count+negative_weight*p/negative_count
        classification_name = classification
    elif classification != "focal":
        raise ValueError("Loss de présence inconnue.")
    delta = output[:, :4]-distances
    absolute = np.abs(delta)
    denominator = max(float(mask.sum())*4, 1.)
    box_loss = (np.where(absolute < 1, .5*delta**2, absolute-.5)*mask).sum()/denominator
    gradient_box = np.clip(delta, -1, 1)*mask/denominator
    gradient = np.concatenate([gradient_box, gradient_z], axis=1).astype(np.float32)
    return float(focal+box_loss), gradient, {classification_name: float(focal), "box": float(box_loss)}


def box_iou(first, second):
    left, top = np.maximum(first[:2], second[:2])
    right, bottom = np.minimum(first[2:], second[2:])
    intersection = max(0., right-left)*max(0., bottom-top)
    area1 = max(0., first[2]-first[0])*max(0., first[3]-first[1])
    area2 = max(0., second[2]-second[0])*max(0., second[3]-second[1])
    return intersection/max(area1+area2-intersection, 1e-8)


def decode_boxes(output, threshold=.5, nms_iou=.5, local_maximum=False, width=WIDTH, height=HEIGHT, stride=STRIDE):
    logits = output[4]
    scores = 1/(1+np.exp(-np.clip(logits, -80, 80)))
    candidates = []
    candidates_mask = scores >= threshold
    if local_maximum:
        # Le réseau apprend les dimensions aux centres uniquement. Décoder les
        # cellules voisines non entraînées produit sinon de faux petits rectangles.
        padded = np.pad(scores, 1, constant_values=-np.inf)
        maxima = np.maximum.reduce([padded[dy:dy+scores.shape[0], dx:dx+scores.shape[1]]
                                    for dy in range(3) for dx in range(3)])
        candidates_mask &= scores == maxima
    for y, x in np.argwhere(candidates_mask):
        l, t, r, b = np.maximum(output[:4, y, x], 0)*stride
        ax, ay = (x+.5)*stride, (y+.5)*stride
        box = np.array([max(0., ax-l), max(0., ay-t), min(width, ax+r), min(height, ay+b)])
        if box[2] > box[0] and box[3] > box[1]:
            candidates.append((float(scores[y, x]), box))
    candidates.sort(key=lambda item: item[0], reverse=True)
    kept = []
    for candidate in candidates:
        if all(box_iou(candidate[1], other[1]) <= nms_iou for other in kept):
            kept.append(candidate)
    return kept


def evaluate_boxes(outputs, truths, threshold=.5, local_maximum=False, width=WIDTH, height=HEIGHT, stride=STRIDE):
    if outputs.ndim != 4 or tuple(outputs.shape[1:]) != (5, height//stride, width//stride) or len(outputs) != len(truths):
        raise ValueError("La forme des sorties doit correspondre à la géométrie et aux N annotations.")
    if not np.isfinite(outputs).all() or not 0 < threshold < 1:
        raise ValueError("Sorties finies et seuil entre 0 et 1 requis.")
    details = []
    for output, boxes in zip(outputs, truths):
        predictions = decode_boxes(output, threshold, local_maximum=local_maximum, width=width, height=height, stride=stride)
        remaining = set(range(len(boxes)))
        truth_pixels = [np.array([(cx-bw/2)*width, (cy-bh/2)*height,
                                 (cx+bw/2)*width, (cy+bh/2)*height]) for cx, cy, bw, bh in boxes]
        tp, decoded = 0, []
        for score, prediction in predictions:
            matched = None
            matches = sorted(((box_iou(prediction, truth_pixels[i]), i) for i in remaining), reverse=True)
            if matches and matches[0][0] >= .5:
                tp += 1
                matched = matches[0][1]
                remaining.remove(matched)
            decoded.append({"score": score, "box": prediction.tolist(), "matched_truth": matched})
        details.append({"tp": tp, "fp": len(predictions)-tp, "fn": len(remaining),
                        "predictions": decoded, "missed": sorted(remaining)})
    tp, fp, fn = (sum(row[key] for row in details) for key in ("tp", "fp", "fn"))
    return {"tp": tp, "fp": fp, "fn": fn, "precision": tp/max(tp+fp, 1),
            "recall": tp/max(tp+fn, 1), "f1": 2*tp/max(2*tp+fp+fn, 1)}, details
