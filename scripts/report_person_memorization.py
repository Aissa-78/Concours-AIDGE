"""Reconstruit le modèle séparable appris, vérifie ses sorties et illustre le test."""

import argparse
import html
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from person_detector_qqvga import PersonDetectorQQVGA
from person_detection_math import evaluate_boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT/"runs/person_qqvga_mem20_v1")
    parser.add_argument("--aidge-python", type=Path, default=ROOT/".venv-aidge-colibry"/("Scripts/python.exe" if os.name == "nt" else "bin/python"))
    args = parser.parse_args()
    run = args.run_dir.resolve()
    info = json.loads((run/"dataset.json").read_text())
    training = json.loads((run/"training.json").read_text())
    with np.load(run/"best_dense.npz") as source:
        weights = dict(source)
    with np.load(run/"parameter_masks.npz") as masks:
        for name, mask in masks.items():
            if np.any(weights[name]*(1-mask) != 0):
                raise RuntimeError("La contrainte depthwise a été violée, conversion refusée.")
    with np.load(run/"dataset.npz") as source:
        images = source["images"]
    torch.set_num_threads(4)
    compact = PersonDetectorQQVGA(info["channels"], info["blocks"]).eval()
    compact.load_state_dict(compact.compact_state(weights))
    with torch.no_grad():
        reference = compact(torch.from_numpy(images)).numpy()
    dense_outputs = np.load(run/"best_logits.npy")
    if not np.allclose(reference, dense_outputs, atol=5e-5, rtol=5e-4):
        raise RuntimeError("Le modèle compact diffère du modèle dense appris.")
    np.savez_compressed(run/"best.npz", **{name.replace(".", "_"): value.numpy() for name, value in compact.state_dict().items()})
    torch.onnx.export(compact, torch.from_numpy(images[:1]), str(run/"best.onnx"),
                      input_names=["image"], output_names=["detections"], opset_version=17, dynamo=False)
    environment = os.environ.copy()
    environment.update(MPLBACKEND="Agg", MPLCONFIGDIR=str(run/".matplotlib"))
    subprocess.run([str(args.aidge_python.absolute()), str(ROOT/"scripts/infer_person_qqvga_aidge.py"),
                    "--model", str(run/"best.onnx"), "--inputs", str(run/"dataset.npz"),
                    "--output", str(run/"compact_logits.npy")], cwd=ROOT, env=environment, check=True)
    outputs = np.load(run/"compact_logits.npy")
    if not np.allclose(outputs, dense_outputs, atol=5e-5, rtol=5e-4):
        raise RuntimeError("L'inférence Aidge depthwise diffère de l'apprentissage dense contraint.")
    truths = [row["boxes"] for row in info["rows"]]
    metrics, details = evaluate_boxes(outputs, truths)
    result = {"images": len(truths), "annotated_presences": info["presences"], "metrics": metrics,
              "threshold": .5, "iou_threshold": .5, "selected_epoch": training["selected_epoch"],
              "memorization_passed": training["memorization_passed"],
              "compact_dense_max_difference": float(np.max(np.abs(outputs-dense_outputs))),
              "generalization_tested": False, "hardware_tested": False,
              "note": "Même lot pour apprendre et mesurer. Ce score ne prouve ni 90% en situation réelle ni 5mW."}
    quantized_note = ""
    if (run/"quantization.npz").is_file():
        with np.load(run/"quantization.npz") as quantized:
            if not np.allclose(quantized["aidge_float"], outputs, atol=5e-5, rtol=5e-4):
                raise RuntimeError("Le contrôle INT8 ne correspond pas aux sorties du modèle courant.")
            integer_metrics, _ = evaluate_boxes(quantized["aidge_int8"], truths)
        result["int8_cpu_metrics"] = integer_metrics
        result["calibration_images"] = 20
        quantized_note = (f"<p>INT8 signé sur CPU : {integer_metrics['tp']} rectangles retrouvés, "
                          f"{integer_metrics['fp']} supplémentaires, {integer_metrics['fn']} manqués. "
                          "Calibration sur les mêmes 20 images : pas un test indépendant ni une validation UINT8 matérielle.</p>")
    result["bitstreams_present"] = [str(path) for path in run.glob("export*/export*/dnn/aidge_model_bitstream.nnpa")]
    (run/"metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    previews = run/"apercus"
    previews.mkdir(exist_ok=True)
    cards = []
    for index, (record, detail) in enumerate(zip(info["rows"], details)):
        image = cv2.imread(str(Path(record["root"])/record["row"]["image"]))
        image = cv2.resize(image, (480, 360), interpolation=cv2.INTER_NEAREST)
        for j, (cx, cy, bw, bh) in enumerate(record["boxes"]):
            color = (30, 30, 240) if j in detail["missed"] else (255, 255, 255)
            cv2.rectangle(image, (round((cx-bw/2)*480), round((cy-bh/2)*360)),
                          (round((cx+bw/2)*480), round((cy+bh/2)*360)), color, 1)
        for prediction in detail["predictions"]:
            x1, y1, x2, y2 = (round(x*3) for x in prediction["box"])
            color = (0, 180, 0) if prediction["matched_truth"] is not None else (0, 140, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"personne {prediction['score']:.2f}", (x1, max(13, y1-4)), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
        cv2.imwrite(str(previews/f"{index+1:02d}.png"), image)
        cards.append(f"<article><h2>{index+1:02d} - {html.escape(record['source'])}</h2>"
                     f"<p>Retrouvés {detail['tp']} / supplémentaires {detail['fp']} / manqués {detail['fn']}</p>"
                     f"<img src='apercus/{index+1:02d}.png' alt='Rectangles sur une image de mémorisation'></article>")
    (run/"rapport.html").write_text(f"""<!doctype html><html lang='fr'><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>Test technique QQVGA</title>
<style>body{{font:17px/1.5 system-ui;max-width:1100px;margin:30px auto;padding:0 20px}}main{{display:flex;flex-wrap:wrap;gap:20px}}article{{width:480px}}img{{max-width:100%}}h2{{font-size:18px}}.warning{{background:#fff1d6;padding:16px}}</style>
<h1>Détecteur QQVGA - test sur 20 images apprises</h1>
<p class='warning'>Ceci est un test de mémorisation. Les mêmes images servent à apprendre et à mesurer.
Ce n'est PAS une mesure de fiabilité sur une autre porte ni une validation matérielle.</p>
<p>Précision {metrics['precision']:.1%} - rappel {metrics['recall']:.1%} - F1 {metrics['f1']:.1%}.
{metrics['tp']} rectangles retrouvés, {metrics['fp']} supplémentaires, {metrics['fn']} manqués.
Seuil fixé à 0,5 ; une correspondance exige IoU ≥ 0,5. Époque retenue : {training['selected_epoch']}.</p>
{quantized_note}
<p>Annotations blanches, annotations manquées rouges, prédictions correspondantes vertes, prédictions hors correspondance orange.
Le score affiché près d'un rectangle est la sortie du réseau, pas une probabilité de fiabilité validée.</p><main>{''.join(cards)}</main></html>""", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Rapport : {run/'rapport.html'}")


if __name__ == "__main__":
    main()
