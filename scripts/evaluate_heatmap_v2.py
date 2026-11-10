"""Évalue le modèle appris avec Aidge face aux deux anciens modèles, sans ajuster le test."""

from __future__ import annotations

import argparse
import csv
import html
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from heatmap_v2 import HeatmapV2
from heatmap_v2_data import read_manifest, read_sample
from heatmap_metrics import evaluate_logits, evaluate_scores
from tiny_heatmap import TinyHeatmap
from train_tiny_heatmap import DetectionFrames
from evaluate_tiny_heatmap import ground_truth_centers


def draw_panel(frame, title, result, truth, content_size):
    height, width = frame.shape[:2]
    scale = min(440 / height, 360 / width)
    dw, dh = round(width * scale), round(height * scale)
    image = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)
    resize = min(content_size / width, content_size / height)
    rw, rh = max(1, round(width * resize)), max(1, round(height * resize))
    left, top = (content_size - rw) // 2, (content_size - rh) // 2

    def point(center):
        return (round((center[0] * content_size - left) / rw * dw),
                round((center[1] * content_size - top) / rh * dh))
    for i, center in enumerate(truth):
        color = (30, 30, 240) if i in result["missed"] else (255, 255, 255)
        cv2.drawMarker(image, point(center), color, cv2.MARKER_CROSS, 14, 2)
    for i, center in enumerate(result["predictions"]):
        color = (0, 165, 255) if i in result["false"] else (30, 220, 30)
        cv2.circle(image, point(center), 8, color, 2)
    panel = np.full((dh + 65, dw, 3), 248, np.uint8)
    panel[65:] = image
    cv2.putText(panel, title, (7, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    cv2.putText(panel, f"OK:{result['tp']} Hors:{result['fp']} Manques:{result['fn']}",
                (7, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1)
    return panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/heatmap_v2")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset/porte_B_test")
    parser.add_argument("--int8", action="store_true", help="Comparer aussi le graphe INT8 réel sur CPU, pas le bitstream.")
    parser.add_argument("--calibrated", action="store_true", help="Montrer aussi le seuil choisi sur la validation locale.")
    parser.add_argument("--reference-run-dir", type=Path, help="Autre essai V2 à comparer au même seuil fixe 0,5.")
    parser.add_argument("--model-label", help="Nom lisible du nouvel essai dans les comparaisons.")
    parser.add_argument("--reference-label", help="Nom lisible de l'essai de référence.")
    parser.add_argument("--aidge-python", type=Path, default=ROOT / ".venv-aidge-colibry" /
                        ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
    args = parser.parse_args()
    run = args.run_dir.resolve()
    output = run / ("evaluation_" + args.data_dir.name)
    output.mkdir(parents=True, exist_ok=True)
    rows = [row for row in read_manifest(args.data_dir) if row["split"] == "validation"]
    source_info = json.loads((run / "dataset.json").read_text(encoding="utf-8"))
    model_title = args.model_label or ("V2 " + source_info.get("pretrain_name", "coco").upper() + " " +
                                     str(source_info["sources"]["coco"]["splits"]["train"]["images"]))
    train_videos = {row["video"] for source in source_info["sources"].values()
                    for row in source["splits"]["train"]["rows"]}
    if {row["video"] for row in rows} & train_videos:
        raise SystemExit("Une vidéo de test apparaît dans l'entraînement.")
    samples = [read_sample(args.data_dir, row) for row in rows]
    if not samples:
        raise SystemExit("Aucune image de validation dans ce dataset.")
    images = np.stack([sample[0] for sample in samples])
    with np.load(run / "dataset.npz") as prepared:
        train_hashes = {hashlib.sha256(image.tobytes()).hexdigest()
                        for name in ("local_train_images", "coco_train_images") for image in prepared[name]}
    if any(hashlib.sha256(image.tobytes()).hexdigest() in train_hashes for image in images):
        raise SystemExit("Une image de test apparaît dans l'entraînement.")
    truths_v2 = [sample[1] for sample in samples]
    np.savez_compressed(output / "inputs.npz", images=images)
    environment = os.environ.copy()
    environment.update(MPLBACKEND="Agg", MPLCONFIGDIR=str(run / ".matplotlib"))
    subprocess.run([str(args.aidge_python.absolute()), str(ROOT / "scripts/train_heatmap_v2_aidge.py"),
                    "--run-dir", str(run), "--predict-inputs", str(output / "inputs.npz"),
                    "--predict-output", str(output / "logits.npy")],
                   cwd=ROOT, env=environment, check=True)
    logits_v2 = np.load(output / "logits.npy")
    torch.set_num_threads(4)
    model_v2 = HeatmapV2().eval()
    with np.load(run / "best.npz") as weights:
        state = {name: torch.from_numpy(weights[name.replace(".", "_")].copy())
                 for name in model_v2.state_dict()}
    model_v2.load_state_dict(state)
    with torch.no_grad():
        reference_v2 = model_v2(torch.from_numpy(images)).numpy()
    if not np.allclose(logits_v2, reference_v2, atol=2e-5, rtol=2e-4):
        raise RuntimeError("Les poids appris Aidge, le modèle ONNX et la référence diffèrent.")
    metrics_v2, details_v2 = evaluate_logits(logits_v2, truths_v2)
    dataset = DetectionFrames(args.data_dir, rows)
    rgb = torch.stack([dataset[i][0] for i in range(len(rows))])
    truths_old = [[((x + 0.5) / 20, (y + 0.5) / 20)
                   for x, y in ground_truth_centers(args.data_dir, row)] for row in rows]
    metrics, details = {"v2_aidge": metrics_v2}, {"v2_aidge": details_v2}
    reference_name = None
    reference_run = None
    if args.reference_run_dir:
        reference_run = args.reference_run_dir.resolve()
        reference_info = json.loads((reference_run / "dataset.json").read_text(encoding="utf-8"))
        if any(reference_info[key] != source_info[key] for key in ("input_size", "content_size", "grid_size")):
            raise SystemExit("Le prétraitement de l'essai de référence est différent.")
        reference_videos = {row["video"] for source in reference_info["sources"].values()
                            for row in source["splits"]["train"]["rows"]}
        if {row["video"] for row in rows} & reference_videos:
            raise SystemExit("Une vidéo évaluée apparaît dans l'entraînement de la référence.")
        with np.load(reference_run / "dataset.npz") as prepared:
            reference_hashes = {hashlib.sha256(image.tobytes()).hexdigest()
                                for name in ("local_train_images", "coco_train_images") for image in prepared[name]}
        if any(hashlib.sha256(image.tobytes()).hexdigest() in reference_hashes for image in images):
            raise SystemExit("Une image évaluée apparaît dans l'entraînement de la référence.")
        subprocess.run([str(args.aidge_python.absolute()), str(ROOT / "scripts/train_heatmap_v2_aidge.py"),
                        "--run-dir", str(reference_run), "--predict-inputs", str(output / "inputs.npz"),
                        "--predict-output", str(output / "reference_logits.npy")],
                       cwd=ROOT, env=environment, check=True)
        reference_name = (reference_info.get("pretrain_name", "coco").upper() + " " +
                          str(reference_info["sources"]["coco"]["splits"]["train"]["images"]) + " im.")
        if args.reference_label:
            reference_name = args.reference_label
        reference_logits = np.load(output / "reference_logits.npy")
        metrics["reference_v2_fp32"], details["reference_v2_fp32"] = evaluate_logits(reference_logits, truths_v2)
    calibrated_threshold = None
    if args.calibrated:
        calibration = json.loads((run / "threshold.json").read_text(encoding="utf-8"))
        if calibration["source"] != "local_validation_only" or calibration["model_sha256"] != hashlib.sha256((run / "best.onnx").read_bytes()).hexdigest():
            raise SystemExit("Le seuil ne correspond pas au modèle ou à la validation locale.")
        calibrated_threshold = calibration["selected"]["threshold"]
        metrics["v2_fp32_seuil_validation"], details["v2_fp32_seuil_validation"] = evaluate_logits(logits_v2, truths_v2, calibrated_threshold)
    if args.int8:
        subprocess.run([str(args.aidge_python.absolute()), str(ROOT / "scripts/run_aidge_quantization.py"),
                        "--onnx", str(run / "best.onnx"), "--inputs", str(output / "inputs.npz"),
                        "--calibration", str(run / "calibration.npz"),
                        "--output", str(output / "quantization.npz")], cwd=ROOT, env=environment, check=True)
        with np.load(output / "quantization.npz") as results:
            if not np.allclose(logits_v2, results["aidge_float"], atol=2e-5, rtol=2e-4):
                raise RuntimeError("Les deux chargements Aidge FP32 diffèrent.")
            metrics["v2_int8_cpu"], details["v2_int8_cpu"] = evaluate_logits(results["aidge_int8"], truths_v2)
            if calibrated_threshold is not None:
                metrics["v2_int8_seuil_validation"], details["v2_int8_seuil_validation"] = evaluate_logits(results["aidge_int8"], truths_v2, calibrated_threshold)
    for name, checkpoint in (("ancien_relu", "tiny_heatmap_relu.pt"),
                             ("ancien_hardtanh", "tiny_heatmap_corrige.pt")):
        cp = torch.load(ROOT / "models" / checkpoint, map_location="cpu", weights_only=True)
        model = TinyHeatmap(cp.get("activation", "hardtanh")).eval()
        model.load_state_dict(cp["model_state_dict"])
        with torch.no_grad():
            # Même inférence image par image et sigmoid que l'évaluateur historique.
            # Les plateaux Hardtanh sont sensibles aux infimes arrondis d'un calcul par lots.
            scores = torch.cat([torch.sigmoid(model(image[None])) for image in rgb]).numpy()
        metrics[name], details[name] = evaluate_scores(scores, truths_old)
    result = {"images": len(rows), "annotated_presences": sum(map(len, truths_v2)),
              "threshold": 0.5, "radius_normalized": 0.125,
              "validation_selected_threshold": calibrated_threshold,
              "pretraining_dataset": source_info.get("pretrain_name", "coco"),
              "public_train_images": source_info["sources"]["coco"]["splits"]["train"]["images"],
              "model_label": model_title, "reference_label": reference_name,
              "model_sha256": hashlib.sha256((run / "best.onnx").read_bytes()).hexdigest(),
              "reference_run": str(reference_run) if reference_run else None,
              "reference_model_sha256": hashlib.sha256((reference_run / "best.onnx").read_bytes()).hexdigest() if reference_run else None,
              "model_selected_without_this_test": args.data_dir.name == "porte_B_test",
              "aidge_pytorch_max_difference": float(np.max(np.abs(logits_v2 - reference_v2))),
              "metrics": metrics, "note": "Comparaison FP32 de centres ; ni comptage vidéo ni validation matérielle."}
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    views = output / "apercus"
    views.mkdir(exist_ok=True)
    cards, csv_rows = [], []
    displayed = "v2_fp32_seuil_validation" if args.calibrated else "v2_aidge"
    displayed_int8 = "v2_int8_seuil_validation" if args.calibrated else "v2_int8_cpu"
    for i, row in enumerate(rows):
        frame = cv2.imread(str(args.data_dir / row["image"]))
        panels = [draw_panel(frame, "Ancien ReLU", details["ancien_relu"][i], truths_old[i], 96),
                  draw_panel(frame, model_title, details[displayed][i], truths_v2[i], 80)]
        if reference_run is not None:
            reference_title = reference_name if args.reference_label else f"V2 {reference_name}"
            panels.insert(1, draw_panel(frame, reference_title, details["reference_v2_fp32"][i], truths_v2[i], 80))
        if args.int8:
            int8_title = (model_title.split()[0] + " INT8 CPU") if args.model_label else "V2 INT8 CPU"
            panels.append(draw_panel(frame, int8_title, details[displayed_int8][i], truths_v2[i], 80))
        cv2.imwrite(str(views / f"{i+1:02d}.jpg"), np.concatenate(panels, axis=1))
        record = {"image": row["image"]}
        for name, values in details.items():
            record.update({name + "_" + key: values[i][key] for key in ("tp", "fp", "fn")})
        csv_rows.append(record)
        cards.append(f"<article><h2>{html.escape(Path(row['image']).name)}</h2>"
                     f"<img loading='lazy' src='apercus/{i+1:02d}.jpg'></article>")
    with (output / "details.csv").open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    labels = {"v2_aidge": "V2 Aidge FP32 (seuil 0,5)", "v2_int8_cpu": "V2 INT8 CPU (seuil 0,5)",
              "v2_fp32_seuil_validation": "V2 FP32 (seuil réglé sur validation)",
              "v2_int8_seuil_validation": "V2 INT8 CPU (seuil réglé sur validation)",
              "ancien_relu": "Ancien ReLU FP32", "ancien_hardtanh": "Ancien Hardtanh FP32"}
    labels["reference_v2_fp32"] = f"V2 {reference_name} FP32 de référence (seuil 0,5)"
    if args.reference_label:
        labels["reference_v2_fp32"] = f"{reference_name} FP32 de référence (seuil 0,5)"
    labels["v2_aidge"] = f"{model_title} Aidge FP32 (seuil 0,5)"
    if args.model_label:
        labels["v2_int8_cpu"] = f"{model_title} INT8 CPU (seuil 0,5)"
    pretraining_name = html.escape(source_info.get("pretrain_name", "coco").upper())
    table = "".join(f"<tr><th>{html.escape(labels[name])}</th><td>{m['tp']}</td><td>{m['fp']}</td>"
                    f"<td>{m['fn']}</td><td>{m['f1']:.1%}</td></tr>" for name, m in metrics.items()
                    if not (calibrated_threshold == 0.5 and name.endswith("seuil_validation")))
    conclusion = ("Au seuil fixe 0,5, ce nouvel essai fait moins bien que l'ancien ReLU sur ces images : il ne le remplace pas automatiquement."
                  if metrics_v2["f1"] < metrics["ancien_relu"]["f1"] else
                  "Au seuil fixe 0,5, le F1 du nouvel essai est au moins égal à celui de l'ancien ReLU sur ces images ; cela reste à confirmer ailleurs.")
    reference_comparison = ""
    if reference_run is not None:
        delta = 100 * (metrics_v2["f1"] - metrics["reference_v2_fp32"]["f1"])
        comparison_title = reference_name if args.reference_label else f"V2 {reference_name}"
        reference_comparison = (f"<p>Par rapport à l'essai {html.escape(comparison_title)} : "
                                f"{delta:+.1f} points de F1 au même seuil 0,5. "
                                "Une comparaison sur ce petit jeu ne suffit pas à démontrer une généralisation.</p>")
    report = f"""<!doctype html><html lang='fr'><meta charset='utf-8'><title>FlowSense V2</title>
<style>body{{font:17px/1.5 system-ui;max-width:1000px;margin:35px auto;padding:0 20px;color:#202530}}
table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:10px}}img{{max-width:100%}}
article{{border-top:1px solid #ccc;margin-top:25px}}h2{{font-size:18px}}</style>
<h1>Essai {html.escape(model_title)} entraîné avec Aidge</h1><p>{len(rows)} images et {result['annotated_presences']} présences annotées.</p>
<p><strong>{conclusion}</strong> Il ne s'agit pas d'un pourcentage de passages correctement comptés.</p>
{reference_comparison}
<table><tr><th>Modèle / exécution</th><th>Centres localisés</th><th>Points hors tolérance</th><th>Centres manqués</th><th>F1</th></tr>{table}</table>
<p>Comparaison au seuil fixe 0,5 ; seuil V2 choisi sur validation locale : {calibrated_threshold if args.calibrated else 'non utilisé'}.
Les aperçus V2 utilisent ce dernier si présent. Rayon de correspondance égal à 12,5 % du côté de l'image préparée.
Les cibles sont les centres des rectangles. Un point sur une tête trop loin du centre peut être hors tolérance.</p>
<p>V2 : gris, contexte élargi, augmentations, préentraînement {pretraining_name} et adaptation aux portes.
Les calculs et gradients de l'apprentissage sont exécutés dans Aidge ; Adam et la loss sont calculés en NumPy.
Ce résultat ne mesure pas encore l'UINT8 final, le comptage vidéo ni la consommation sur carte.</p>
<p>Cercle vert : correspondance. Orange : point hors tolérance. Croix rouge : centre manqué.</p>{''.join(cards)}</html>"""
    (output / "rapport.html").write_text(report, encoding="utf-8")
    for name, measure in metrics.items():
        print(name, measure)
    print(f"Rapport : {output / 'rapport.html'}")


if __name__ == "__main__":
    main()
