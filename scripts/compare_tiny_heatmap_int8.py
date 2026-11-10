"""Compare la localisation FP32/INT8, sans réentraîner ni ajuster le seuil.

Depuis la racine : ./.venv/bin/python scripts/compare_tiny_heatmap_int8.py
Sous Windows : .venv\\Scripts\\python.exe scripts/compare_tiny_heatmap_int8.py
Un environnement Aidge compatible doit aussi être installé sur la machine.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate_tiny_heatmap import find_peaks, ground_truth_centers
from tiny_heatmap import TinyHeatmap
from train_tiny_heatmap import DetectionFrames
from prepare_colibry_calibration import preprocess


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_centers(scores, truth, threshold, radius):
    predictions = find_peaks(scores, threshold, separation=2.0)
    unmatched = set(range(len(truth)))
    matched, extras = [], []
    # Même correspondance gloutonne et même ordre de pics que l'évaluateur existant.
    for index, (px, py) in enumerate(predictions):
        matches = sorted(
            (math.hypot(px - truth[i][0], py - truth[i][1]), i) for i in unmatched
        )
        if matches and matches[0][0] <= radius:
            matched.append(index)
            unmatched.remove(matches[0][1])
        else:
            extras.append(index)
    return {
        "tp": len(matched), "fp": len(extras), "fn": len(unmatched),
        "predictions": predictions, "matched": matched, "extras": extras,
        "missed": sorted(unmatched),
    }


def summarize(results, truths):
    tp, fp, fn = (sum(row[field] for row in results) for field in ("tp", "fp", "fn"))
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0,
        "recall": tp / (tp + fn) if tp + fn else 0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
        "empty_images": sum(not truth for truth in truths),
        "empty_false_alarms": sum(
            not truth and bool(result["predictions"])
            for truth, result in zip(truths, results)
        ),
    }


def render_comparison(source, truth, fp32, int8, output):
    frame = cv2.imread(str(source))
    height, width = frame.shape[:2]
    factor = min(96 / width, 96 / height)
    rw, rh = max(1, round(width * factor)), max(1, round(height * factor))
    left, top = (96 - rw) // 2, (96 - rh) // 2
    display_scale = min(440 / width, 480 / height)
    dw, dh = round(width * display_scale), round(height * display_scale)
    base = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)

    def point(center):
        # Inverse exact de la position sur la grille après letterboxing 96x96.
        gx, gy = center
        return (round(((gx + 0.5) * 96 / 20 - left) / rw * dw),
                round(((gy + 0.5) * 96 / 20 - top) / rh * dh))

    panels = []
    for title, result in (("Annotations", None), ("FP32", fp32), ("INT8 CPU", int8)):
        image = base.copy()
        for i, center in enumerate(truth):
            color = (255, 255, 255) if result is None or i not in result["missed"] else (40, 40, 240)
            cv2.drawMarker(image, point(center), color, cv2.MARKER_CROSS, 16, 2)
        if result:
            for i, center in enumerate(result["predictions"]):
                color = (40, 215, 40) if i in result["matched"] else (0, 165, 255)
                cv2.circle(image, point(center), 9, color, 2)
        panel = np.full((dh + 72, dw, 3), 248, dtype=np.uint8)
        panel[72:] = image
        cv2.putText(panel, title, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
        detail = f"{len(truth)} presence(s)" if result is None else (
            f"OK:{result['tp']}  Hors:{result['fp']}  Manques:{result['fn']}"
        )
        text_width = cv2.getTextSize(detail, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
        font_scale = min(0.42, 0.42 * (dw - 16) / max(text_width, 1))
        cv2.putText(panel, detail, (8, 52), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20), 1)
        panels.append(panel)
    if not cv2.imwrite(str(output), np.concatenate(panels, axis=1)):
        raise RuntimeError(f"Impossible d'enregistrer l'aperçu : {output}")


def write_report(directory, rows, truths, results, summary):
    views = directory / "apercus"
    views.mkdir(exist_ok=True)
    details = []
    for i, (row, truth, fp32, int8) in enumerate(zip(rows, truths, results["fp32"], results["int8"])):
        preview = views / f"{i + 1:02d}.jpg"
        render_comparison(Path(summary["dataset"]) / row["image"], truth, fp32, int8, preview)
        details.append({
            "image": row["image"], "video": row["video"], "time_seconds": row["time_seconds"],
            "annotations": len(truth),
            **{f"{name}_{key}": result[key] for name, result in (("fp32", fp32), ("int8", int8))
               for key in ("tp", "fp", "fn")},
            "apercu": f"apercus/{preview.name}",
        })
    with (directory / "details.csv").open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    (directory / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    before, after = summary["fp32"], summary["int8"]
    lines = [
        "# Comparaison avant/après quantification — porte B", "",
        f"{len(rows)} images, {sum(map(len, truths))} présences annotées. Les mêmes personnes "
        "peuvent apparaître sur plusieurs images.", "",
        "| Mesure | FP32 | INT8 dans Aidge CPU |", "|---|---:|---:|",
        f"| Présences correctement localisées | {before['tp']} | {after['tp']} |",
        f"| Centres annotés manqués | {before['fn']} | {after['fn']} |",
        f"| Points supplémentaires ou hors tolérance | {before['fp']} | {after['fp']} |",
        f"| Précision | {before['precision']:.1%} | {after['precision']:.1%} |",
        f"| Rappel | {before['recall']:.1%} | {after['recall']:.1%} |",
        f"| F1 | {before['f1']:.1%} | {after['f1']:.1%} |", "",
        "La précision indique la part des détections qui correspondent à une annotation. "
        "Le rappel indique la part des présences retrouvées. Le F1 combine les deux ; "
        "ce n'est pas un pourcentage d'images correctement classées.", "",
        f"Différence de F1 après INT8 : {(after['f1'] - before['f1']) * 100:+.1f} points.", "",
        "## Ce qui a été vérifié", "",
        f"- Même checkpoint ReLU et même prétraitement RGB 96×96 dans les deux versions.",
        f"- {summary['calibration_images']} images de calibration issues uniquement du train, "
        "sans vidéo commune ni image prétraitée identique aux images évaluées.",
        f"- Même seuil {summary['threshold']} et même rayon {summary['match_radius']} cases ; "
        "aucun réglage de seuil sur la porte B.",
        f"- Écart maximal PyTorch/Aidge FP32 : {summary['float_max_abs_difference']:.2e}.",
        "- Calculs des convolutions contrôlés : activations et poids INT8, biais et accumulation INT32. "
        "Les adaptateurs d'entrée/sortie restent flottants.", "",
        "Le test porte sur les centres de personnes dans des images, d'après les annotations. "
        "Il ne mesure ni les entrées/sorties d'une vidéo, ni le bitstream exécuté sur Colibry. "
        "La caméra, la consommation et le firmware restent à tester.", "",
        "Un point sur la tête peut être compté hors tolérance si l'annotation vise le centre "
        "du corps. Dans ce cas, il génère à la fois un point sans correspondance et un centre "
        "annoté manqué. Les nombres de faux positifs et d'oublis décrivent donc cette règle "
        "de localisation, et ne signifient pas toujours que le modèle n'a pas vu la personne.", "",
        "## Images à examiner", "",
        "Croix blanche : annotation. Cercle vert : correspondance trouvée. "
        "Cercle orange : point sans correspondance dans la tolérance. "
        "Croix rouge : centre annoté manqué.", "",
    ]
    # Montrer d'abord les cas où les deux versions se distinguent, puis les erreurs communes.
    order = sorted(range(len(details)), key=lambda i: (
        any(details[i][f"fp32_{k}"] != details[i][f"int8_{k}"] for k in ("tp", "fp", "fn")),
        details[i]["int8_fp"] + details[i]["int8_fn"],
    ), reverse=True)
    cards = []
    for i in order:
        detail = details[i]
        name = Path(detail["image"]).name
        lines.extend([f"### {i + 1:02d} — {name}", "", f"![Comparaison]({detail['apercu']})", ""])
        cards.append(
            f"<article><h2>{i + 1:02d} — {html.escape(name)}</h2>"
            f"<p>FP32 : {detail['fp32_tp']} correspondance(s), {detail['fp32_fp']} point(s) hors tolérance, "
            f"{detail['fp32_fn']} centre(s) manqué(s). INT8 : {detail['int8_tp']} correspondance(s), "
            f"{detail['int8_fp']} point(s) hors tolérance, {detail['int8_fn']} centre(s) manqué(s).</p>"
            f"<a href='{detail['apercu']}'><img loading='lazy' src='{detail['apercu']}' "
            "alt='Annotations, détections FP32 et INT8 côte à côte'></a></article>"
        )
    (directory / "rapport.md").write_text("\n".join(lines), encoding="utf-8")
    table_rows = "".join(
        f"<tr><th>{label}</th><td>{before[key]}</td><td>{after[key]}</td></tr>"
        for label, key in (("Présences correctement localisées", "tp"), ("Centres annotés manqués", "fn"),
                           ("Points supplémentaires ou hors tolérance", "fp"))
    )
    content = f"""<!doctype html><html lang="fr"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FlowSense — comparaison FP32 / INT8</title>
<style>body{{font:17px/1.55 system-ui,sans-serif;max-width:1100px;margin:36px auto;padding:0 18px;color:#202530}}
table{{border-collapse:collapse}}th,td{{border:1px solid #ccc;padding:10px 16px;text-align:left}}
article{{border-top:1px solid #ddd;margin-top:36px;padding-top:12px}}h2{{font-size:19px;overflow-wrap:anywhere}}img{{width:100%;height:auto}}
.note{{background:#f2f5f8;padding:16px;border-radius:8px}}a{{color:#124c9c}}</style>
<h1>Avant et après le passage en INT8</h1>
<p>{len(rows)} images de la porte B ; {sum(map(len, truths))} présences annotées.
Les mêmes personnes peuvent apparaître sur plusieurs images.</p>
<table><tr><th>Mesure</th><th>FP32</th><th>INT8 CPU</th></tr>{table_rows}
<tr><th>F1</th><td>{before['f1']:.1%}</td><td>{after['f1']:.1%}</td></tr></table>
<p>F1 : score qui combine les personnes retrouvées et les fausses détections.
Différence : {(after['f1']-before['f1'])*100:+.1f} points après INT8.</p>
<p class="note">Même modèle, mêmes images, même seuil ({summary['threshold']}). Calibration :
{summary['calibration_images']} images de l'entraînement, sans vidéo commune avec le test.
Les convolutions INT8 et l'accumulation INT32 ont été contrôlées. Ce test exécute Aidge sur CPU,
pas le bitstream Colibry. Il ne mesure pas le comptage vidéo ni la consommation.</p>
<p>Croix blanche : annotation. Cercle vert : correspondance. Cercle orange : point sans correspondance.
Croix rouge : centre annoté manqué. Un point sur une tête peut être hors tolérance si
l'annotation vise le centre du corps : il compte alors comme faux positif et comme oubli.
Les aperçus permettent de vérifier ce cas. Cliquer sur une image pour l'agrandir.</p>
{''.join(cards)}<p><a href="rapport.md">Rapport détaillé</a> · <a href="details.csv">Résultats par image</a>
· <a href="metrics.json">Mesures et paramètres</a></p></html>"""
    (directory / "rapport.html").write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/tiny_heatmap_relu.pt")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/tiny_heatmap_relu")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset/porte_B_test")
    parser.add_argument("--calibration-data-dir", type=Path, default=ROOT / "dataset/detection_proposals")
    parser.add_argument("--calibration-images", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/comparaison_int8_porte_B")
    interpreter = ROOT / ".venv-aidge-colibry" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    parser.add_argument("--aidge-python", type=Path, default=interpreter)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--match-radius", type=float, default=2.5)
    args = parser.parse_args()
    if not 0 < args.threshold < 1 or args.match_radius <= 0 or args.calibration_images < 1:
        raise SystemExit("Seuil, rayon ou nombre d'images invalide.")
    onnx = args.run_dir.resolve() / "tiny_heatmap.onnx"
    for path in (args.checkpoint, onnx, args.aidge_python):
        if not path.is_file():
            raise SystemExit(f"Fichier absent : {path}")
    with (args.data_dir / "manifest.csv").open(newline="", encoding="utf-8") as source:
        rows = [row for row in csv.DictReader(source) if row["split"] == "validation"]
    with (args.calibration_data_dir / "manifest.csv").open(newline="", encoding="utf-8") as source:
        train_rows = [row for row in csv.DictReader(source) if row["split"] == "train"]
    if not rows or any(row["reviewed"] != "oui" for row in rows):
        raise SystemExit("Toutes les images de test doivent être vérifiées.")
    if not train_rows:
        raise SystemExit("Aucune image train pour la calibration.")
    if {row["video"] for row in rows} & {row["video"] for row in train_rows}:
        raise SystemExit("Une vidéo de test apparaît dans la calibration : comparaison refusée.")

    cp = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = TinyHeatmap(activation=cp.get("activation", "hardtanh")).eval()
    model.load_state_dict(cp["model_state_dict"])
    torch.set_num_threads(min(4, torch.get_num_threads()))
    dataset = DetectionFrames(args.data_dir, rows)
    inputs = torch.stack([dataset[i][0] for i in range(len(rows))])
    with torch.no_grad():
        reference = model(inputs).numpy()
    truths = [ground_truth_centers(args.data_dir, row) for row in rows]
    selected = [train_rows[int(i)] for i in np.linspace(
        0, len(train_rows) - 1, min(args.calibration_images, len(train_rows)), dtype=int
    )]
    calibration = np.concatenate([
        preprocess(args.calibration_data_dir / row["image"]) for row in selected
    ])
    test_hashes = {hashlib.sha256(image.tobytes()).hexdigest() for image in inputs.numpy()}
    if any(hashlib.sha256(image.tobytes()).hexdigest() in test_hashes for image in calibration):
        raise SystemExit("Image identique dans la calibration et le test : comparaison refusée.")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "inputs.npz", images=inputs.numpy(), pytorch_float=reference)
    np.savez_compressed(output / "calibration.npz", images=calibration)
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["MPLCONFIGDIR"] = str(output / ".matplotlib")
    print(f"Test : {len(rows)} images ; calibration séparée : {len(calibration)} images.", flush=True)
    subprocess.run([
        # Conserver le chemin du venv : resolve() suivrait le lien vers le Python
        # système sous macOS/Linux et ferait perdre ses paquets Aidge.
        str(args.aidge_python.absolute()), str(ROOT / "scripts/run_aidge_quantization.py"),
        "--onnx", str(onnx), "--inputs", str(output / "inputs.npz"),
        "--calibration", str(output / "calibration.npz"), "--output", str(output / "aidge_predictions.npz"),
    ], cwd=ROOT, env=environment, check=True)
    with np.load(output / "aidge_predictions.npz") as predictions:
        floating, integer = predictions["aidge_float"], predictions["aidge_int8"]
    if floating.shape != reference.shape or integer.shape != reference.shape:
        raise SystemExit("Formes des prédictions incompatibles.")
    if not np.allclose(floating, reference, rtol=1e-4, atol=1e-5):
        raise SystemExit("ONNX/Aidge et checkpoint PyTorch diffèrent : comparaison arrêtée.")
    results = {}
    for name, logits in (("fp32", reference), ("int8", integer)):
        scores = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
        results[name] = [compare_centers(score[0], truth, args.threshold, args.match_radius)
                         for score, truth in zip(scores, truths)]
    runtime = json.loads((output / "aidge_predictions.json").read_text(encoding="utf-8"))
    summary = {
        "dataset": str(args.data_dir.resolve()), "images": len(rows),
        "annotated_presences": sum(map(len, truths)), "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint), "onnx_sha256": sha256(onnx),
        "test_manifest_sha256": sha256(args.data_dir / "manifest.csv"),
        "calibration_images": len(calibration),
        "calibration_source_images": [row["image"] for row in selected],
        "calibration_source_dataset": str(args.calibration_data_dir.resolve()),
        "threshold": args.threshold, "match_radius": args.match_radius, "separation": 2.0,
        "float_max_abs_difference": float(np.max(np.abs(floating - reference))),
        "quantization_mean_abs_difference": float(np.mean(np.abs(integer - reference))),
        "quantization_max_abs_difference": float(np.max(np.abs(integer - reference))),
        "fp32": summarize(results["fp32"], truths), "int8": summarize(results["int8"], truths),
        "runtime": runtime,
    }
    write_report(output, rows, truths, results, summary)
    for name in ("fp32", "int8"):
        result = summary[name]
        print(f"{name.upper()} : localisées={result['tp']}, points hors tolérance={result['fp']}, "
              f"centres manqués={result['fn']}, F1={result['f1']:.1%}")
    print(f"Rapport à ouvrir : {output / 'rapport.html'}")


if __name__ == "__main__":
    main()
