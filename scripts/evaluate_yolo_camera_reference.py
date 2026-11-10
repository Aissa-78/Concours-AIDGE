"""Repère logiciel YOLO sur les mêmes pixels caméra que le petit détecteur.

Pas d'entraînement, de téléchargement ni de remplacement du compteur. Cette
référence préentraînée plus lourde n'est pas un candidat Colibry validé.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from person_detection_math import box_iou


def common_nms(predictions):
    """NMS explicite, y compris si le modèle YOLO est de type end-to-end."""
    kept = []
    for candidate in sorted(predictions, key=lambda pair: pair[0], reverse=True):
        if all(box_iou(candidate[1], previous[1]) <= .5 for previous in kept):
            kept.append(candidate)
    return kept


def match_boxes(predictions, boxes, width=160, height=120, return_details=False):
    remaining = set(range(len(boxes)))
    truth = [np.array([(cx-w/2)*width, (cy-h/2)*height, (cx+w/2)*width, (cy+h/2)*height])
             for cx, cy, w, h in boxes]
    tp, decoded = 0, []
    for _, prediction in sorted(predictions, key=lambda pair: pair[0], reverse=True):
        matched = None
        matches = sorted(((box_iou(prediction, truth[i]), i) for i in remaining), reverse=True)
        if matches and matches[0][0] >= .5:
            tp += 1
            matched = matches[0][1]
            remaining.remove(matched)
        decoded.append({"box": np.asarray(prediction).tolist(), "matched_truth": matched})
    result = {"tp": tp, "fp": len(predictions)-tp, "fn": len(remaining)}
    if return_details:
        return {**result, "predictions": decoded, "missed": sorted(remaining)}
    return result


def summarize(details):
    tp, fp, fn = (sum(d[key] for d in details) for key in ("tp", "fp", "fn"))
    return {"tp": tp, "fp": fp, "fn": fn, "precision": tp/max(tp+fp, 1),
            "recall": tp/max(tp+fn, 1), "f1": 2*tp/max(2*tp+fp+fn, 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-run", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=ROOT/"yolo26n.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inference-size", type=int, choices=(160, 320, 640), default=160,
                        help="Taille interne de YOLO ; 320/640 agrandissent les mêmes pixels, sans ajout de détail.")
    args = parser.parse_args()
    if not args.weights.is_file():
        raise SystemExit("Poids locaux absents : aucun téléchargement automatique autorisé par ce script.")
    if args.output_dir.exists():
        raise SystemExit("Choisir un nouveau dossier pour conserver les évaluations.")
    records = json.loads((args.validation_run/"validation.json").read_text())
    with np.load(args.validation_run/"validation.npz") as values:
        images = values["images"]
    if images.shape != (len(records), 1, 120, 160):
        raise ValueError("Images caméra 160x120 grises requises.")
    torch.set_num_threads(2)
    model = YOLO(str(args.weights.resolve()))
    details, predictions = [], []
    for offset in range(0, len(images), 8):
        # Trois canaux identiques : aucune information couleur n'est réintroduite.
        frames = [cv2.cvtColor(np.round(x[0]*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                  for x in images[offset:offset+8]]
        results = model.predict(source=frames, classes=[0], conf=.5, iou=.5,
                                imgsz=args.inference_size, device="cpu", verbose=False)
        if len(results) != len(frames):
            raise RuntimeError("Nombre de sorties inattendu.")
        for result, record in zip(results, records[offset:offset+8]):
            if tuple(result.orig_shape) != (120, 160):
                raise RuntimeError("Les coordonnées ne sont plus dans l'image caméra.")
            pred = [(float(score), box) for score, box in zip(result.boxes.conf.cpu().numpy(),
                                                              result.boxes.xyxy.cpu().numpy())]
            pred = common_nms(pred)
            details.append(match_boxes(pred, record["boxes"]))
            predictions.append([{"score": score, "box": box.tolist()} for score, box in pred])
    args.output_dir.mkdir(parents=True)
    report = {"weights": str(args.weights.resolve()), "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
              "input": "Mêmes pixels gris 160x120, répétés sur trois canaux ; letterbox interne YOLO possible.",
              "threshold": .5, "nms_iou": .5, "match_iou": .5,
              "yolo_internal_size": args.inference_size,
              "images": len(records), "instances": sum(len(r["boxes"]) for r in records),
              "overall": summarize(details),
              "by_source": {source: summarize([d for d, r in zip(details, records) if r["source"] == source])
                            for source in ("portes", "wepdtof")},
              "predictions": predictions, "trained_here": False, "aidge_executed": False,
              "hardware_validated": False, "power_validated": False,
              "note": "Référence préentraînée plus lourde, pas un modèle proposé pour la carte. Même validation de développement, pas un test aveugle. Les labels locaux ont été proposés initialement par YOLO puis relus : ce repère n'est pas indépendant de leur construction."}
    (args.output_dir/"metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("images", "instances", "overall", "by_source", "note")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
