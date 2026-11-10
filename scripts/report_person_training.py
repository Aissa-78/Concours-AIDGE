"""Exporte le checkpoint compact et affiche ses erreurs de validation QQVGA.

Vérifie que la conversion dense -> depthwise conserve les sorties Aidge sur
train ET validation. Ne modifie ni les annotations ni le modèle de l'interface.
"""

import argparse
import html
from functools import partial
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
from person_detector_qqvga import make_person_detector
from person_detection_math import evaluate_boxes, detection_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT/"runs/person_qqvga_full_v1")
    parser.add_argument("--aidge-python", type=Path, default=ROOT/".venv-aidge-colibry"/("Scripts/python.exe" if os.name == "nt" else "bin/python"))
    args = parser.parse_args()
    run = args.run_dir.resolve()
    info = json.loads((run/"dataset.json").read_text())
    metrics_function = partial(evaluate_boxes, local_maximum=info.get("local_maximum", False))
    training = json.loads((run/"training.json").read_text())
    weights = dict(np.load(run/"best_dense.npz"))
    with np.load(run/"parameter_masks.npz") as masks:
        for name, mask in masks.items():
            if np.any(weights[name]*(1-mask) != 0):
                raise RuntimeError("Poids hors diagonale non nuls, conversion refusée.")
    torch.set_num_threads(4)
    compact = make_person_detector(info["channels"], info["blocks"], residual=info.get("residual", False), architecture=info.get("architecture", "separable")).eval()
    compact.load_state_dict(compact.compact_state(weights))
    torch.onnx.export(compact, torch.zeros(1, 1, 120, 160), str(run/"best.onnx"),
                      input_names=["image"], output_names=["detections"], opset_version=17, dynamo=False)
    np.savez_compressed(run/"best.npz", **{name.replace(".", "_"): p.numpy() for name, p in compact.state_dict().items()})
    environment = os.environ.copy()
    environment.update(MPLBACKEND="Agg", MPLCONFIGDIR=str(run/".matplotlib"))
    result, validation_details = {}, None
    for split in ("train", "validation"):
        subprocess.run([str(args.aidge_python.absolute()), str(ROOT/"scripts/infer_person_qqvga_aidge.py"),
                        "--model", str(run/"best.onnx"), "--inputs", str(run/f"{split}.npz"),
                        "--output", str(run/f"compact_{split}_logits.npy"), "--batch-size", "8"],
                       cwd=ROOT, env=environment, check=True)
        outputs = np.load(run/f"compact_{split}_logits.npy")
        dense_outputs = np.load(run/f"best_{split}_logits.npy")
        if not np.allclose(outputs, dense_outputs, atol=5e-5, rtol=5e-4):
            raise RuntimeError("La conversion change les prédictions du checkpoint.")
        records = json.loads((run/f"{split}.json").read_text())
        metrics, details = metrics_function(outputs, [row["boxes"] for row in records])
        with np.load(run/f"{split}.npz") as prepared:
            loss, _, components = detection_loss(outputs, prepared["heat"], prepared["distances"], prepared["masks"], classification=info.get("classification", "focal"))
            scores = 1/(1+np.exp(-np.clip(outputs[:, 4:5], -80, 80)))
            positive_scores = scores[prepared["masks"] > 0]
        by_source = {}
        for source in ("portes", "wepdtof"):
            indices = [i for i, row in enumerate(records) if row["source"] == source]
            values, _ = metrics_function(outputs[indices], [records[i]["boxes"] for i in indices])
            by_source[source] = {"images": len(indices), "metrics": values}
            if split == "validation" and values != training["validation_metrics"][source]:
                raise RuntimeError("Le rapport ne correspond pas au checkpoint sélectionné.")
        result[split] = {"images": len(records), "metrics": metrics, "by_source": by_source,
                         "loss": loss, "loss_components": components,
                         "annotated_center_score_mean": float(positive_scores.mean()),
                         "compact_dense_max_difference": float(np.max(np.abs(outputs-dense_outputs)))}
        if split == "validation":
            validation_details = details
    result.update(selected_epoch=training["selected_epoch"], threshold=.5, iou_threshold=.5,
                  no_new_update_selected=training["selected_epoch"] == 0,
                  classification=info.get("classification", "focal"), local_maximum=info.get("local_maximum", False),
                  validation_used_for_model_selection=True, final_blind_test=False,
                  door_B_used=False, hardware_tested=False, counting_tested=False,
                  note="Les 98 images ne servent pas aux gradients, mais servent au choix du checkpoint ; pas une nouvelle porte aveugle.")
    if (run/"quantization.npz").exists():
        with np.load(run/"quantization.npz") as quantized:
            np.testing.assert_allclose(quantized["aidge_float"], np.load(run/"compact_validation_logits.npy"), atol=5e-5, rtol=5e-4)
            integer_metrics, _ = metrics_function(quantized["aidge_int8"], [row["boxes"] for row in records])
            result["int8_cpu_validation_metrics"] = integer_metrics
            result["int8_note"] = "INT8 signé CPU ; calibration train seulement, UINT8 et carte non validés."
    (run/"metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    previews = run/"apercus"
    previews.mkdir(exist_ok=True)
    cards = []
    with np.load(run/"validation.npz") as values:
        images = values["images"]
    for index, (record, detail) in enumerate(zip(records, validation_details)):
        image = cv2.cvtColor(np.round(images[index, 0]*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        image = cv2.resize(image, (480, 360), interpolation=cv2.INTER_NEAREST)
        for j, (cx, cy, bw, bh) in enumerate(record["boxes"]):
            color = (30, 30, 240) if j in detail["missed"] else (255, 255, 255)
            cv2.rectangle(image, (round((cx-bw/2)*480), round((cy-bh/2)*360)),
                          (round((cx+bw/2)*480), round((cy+bh/2)*360)), color, 1)
        for prediction in detail["predictions"]:
            x1, y1, x2, y2 = (round(value*3) for value in prediction["box"])
            color = (0, 180, 0) if prediction["matched_truth"] is not None else (0, 140, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"personne {prediction['score']:.2f}", (x1, max(13, y1-4)), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
        cv2.imwrite(str(previews/f"{index+1:03d}.png"), image)
        cards.append(f"<article><h2>{index+1:03d} — {html.escape(record['source'])}</h2>"
                     f"<p>{html.escape(record['row']['video'])}<br>Retrouvés : {detail['tp']} ; "
                     f"supplémentaires : {detail['fp']} ; manqués : {detail['fn']}</p>"
                     f"<img loading='lazy' src='apercus/{index+1:03d}.png' alt='Erreurs de validation'></article>")
    rows = []
    for source, values in result["validation"]["by_source"].items():
        m = values["metrics"]
        rows.append(f"<tr><td>{html.escape(source)}</td><td>{values['images']}</td><td>{m['precision']:.1%}</td>"
                    f"<td>{m['recall']:.1%}</td><td>{m['f1']:.1%}</td><td>{m['tp']}</td><td>{m['fp']}</td><td>{m['fn']}</td></tr>")
    integer_note = ""
    if "int8_cpu_validation_metrics" in result:
        m = result["int8_cpu_validation_metrics"]
        integer_note = f"<p>INT8 signé CPU, validation globale : F1 {m['f1']:.1%}, {m['tp']} retrouvés, {m['fp']} supplémentaires, {m['fn']} manqués. Calibration sur le train uniquement.</p>"
    initialization_note = "Départ avec des poids aléatoires."
    if training.get("initial_checkpoint"):
        initialization_note = ("Poursuite depuis un checkpoint précédent ; les poids sont conservés, "
                               "mais Adam est réinitialisé. Les numéros d'époque concernent cette reprise seulement.")
    if training["selected_epoch"] == 0:
        initialization_note += (" Attention : le checkpoint retenu est l'état initial de cet essai, avant toute nouvelle mise à jour. "
                                "Aucun des états appris ensuite n'a amélioré le critère de sélection.")
    total = result["validation"]["metrics"]
    annotated = total["tp"]+total["fn"]
    quality_note = (f"<p class='warning'>Résultat global : {total['tp']} présence annotée retrouvée sur {annotated}, "
                    f"{total['fn']} manquées, {total['fp']} détections supplémentaires. F1 : {total['f1']:.1%}. "
                    "La précision seule peut afficher 100 % quand une seule proposition est correcte : "
                    "cela ne signifie pas que toutes les personnes sont détectées.</p>")
    (run/"rapport.html").write_text(f"""<!doctype html><html lang='fr'><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>QQVGA — validation du lot complet</title>
<style>body{{font:17px/1.5 system-ui;max-width:1100px;margin:30px auto;padding:0 20px}}main{{display:flex;flex-wrap:wrap;gap:20px}}article{{width:480px}}img{{max-width:100%}}h2{{font-size:18px}}.warning{{background:#fff1d6;padding:16px}}table{{border-collapse:collapse}}td,th{{padding:8px;border:1px solid #ccc}}</style>
<h1>Détecteur QQVGA — apprentissage sur {result['train']['images']} images</h1>
<p>160 × 120 gris, {info['parameters_compact']} paramètres, architecture {html.escape(info.get('architecture', 'separable'))}, connexions résiduelles : {info.get('residual', False)}. Forward/backward Aidge FP32, loss/Adam NumPy.
{training['epochs_completed']} époques terminées ; checkpoint retenu : époque {training['selected_epoch']}.</p>
<p>{initialization_note}</p>
<p>Loss de présence : {html.escape(info.get('classification', 'focal'))}. Sélection des pics 3×3 : {info.get('local_maximum', False)}.</p>
<p class='warning'>Les 98 images ci-dessous ne servent pas à apprendre les poids. Elles servent à choisir le meilleur checkpoint :
ce n'est donc pas un test final aveugle. Porte B exclue. Ni compteur vidéo, ni UINT8 matériel, ni consommation 5 mW validés.</p>
{quality_note}
<table><thead><tr><th>Validation</th><th>Images</th><th>Précision</th><th>Rappel</th><th>F1</th><th>Retrouvés</th><th>Supplémentaires</th><th>Manqués</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p>Train global (images apprises) : F1 {result['train']['metrics']['f1']:.1%}. Ces métriques comparent des rectangles (IoU ≥ 0,5),
pas seulement leurs centres. Ne pas les comparer directement aux anciens scores du modèle heatmap.</p>
{integer_note}<p>Seuil de détection fixé à 0,5 avant l'essai. Annotations blanches ; annotations manquées rouges ; prédictions correctes vertes ;
prédictions sans correspondance orange. Le score près du rectangle n'est pas la fiabilité mesurée du modèle.</p>
<main>{''.join(cards)}</main></html>""", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Rapport : {run/'rapport.html'}")


if __name__ == "__main__":
    main()
