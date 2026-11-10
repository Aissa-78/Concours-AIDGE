"""Compare les pixels d'origine, QVGA et QQVGA sans réentraîner le détecteur.

Même validation corrigée, mêmes poids locaux, taille interne YOLO 640 fixe,
seuil principal 0,5 et NMS/IoU 0,5. Aucun modèle de l'interface n'est remplacé.
"""

import argparse
import hashlib
import html
from importlib.metadata import version
import json
import math
from pathlib import Path, PureWindowsPath
import sys
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from camera_preprocess import CameraGeometry, prepare_camera_frame
from evaluate_yolo_camera_reference import common_nms, match_boxes, summarize

CONDITIONS = (
    ("original_color", "Original — couleur", None, False),
    ("original_gray", "Original — gris", None, True),
    ("native_camera_color", "Format caméra sans réduction — couleur", "native", False),
    ("native_camera_gray", "Format caméra sans réduction — gris", "native", True),
    ("qvga_color", "320 × 240 — couleur", (320, 240), False),
    ("qvga_gray", "320 × 240 — gris", (320, 240), True),
    ("qqvga_color", "160 × 120 — couleur", (160, 120), False),
    ("qqvga_gray", "160 × 120 — gris", (160, 120), True),
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relocate_path(value, project_root=ROOT):
    """Retrouve un artefact partagé Mac/Windows ; les hashes restent contrôlés."""
    original = Path(value)
    if original.exists():
        return original
    parts = PureWindowsPath(value).parts if "\\" in str(value) else original.parts
    for index, component in enumerate(parts):
        if component in ("dataset", "runs"):
            candidate = project_root.joinpath(*parts[index:]).resolve()
            if candidate.is_relative_to(project_root.resolve()) and candidate.exists():
                return candidate
    raise FileNotFoundError(f"Fichier/dossier local absent : {value}. Copier les données locales ou préparer le jeu sur ce poste.")


def prepare_variant(frame, boxes, size, gray):
    height, width = frame.shape[:2]
    if size is None:
        geometry = CameraGeometry(width, height, width, height, 0, 0)
        image = frame.copy()
    elif size == "native":
        # Bandes 4:3 sans réduction ni étirement : contrôle de l'effet du cadrage.
        k = max(math.ceil(width/4), math.ceil(height/3))
        geometry = CameraGeometry(4*k, 3*k, width, height, (4*k-width)//2, (3*k-height)//2)
        image = np.full((3*k, 4*k, 3), 114, np.uint8)
        image[geometry.top:geometry.top+height, geometry.left:geometry.left+width] = frame
    else:
        gray_image, geometry = prepare_camera_frame(frame, width=size[0], height=size[1])
        if gray:
            image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        else:
            image = np.full((size[1], size[0], 3), 114, np.uint8)
            resized = cv2.resize(frame, (geometry.resized_width, geometry.resized_height), interpolation=cv2.INTER_AREA)
            image[geometry.top:geometry.top+geometry.resized_height,
                  geometry.left:geometry.left+geometry.resized_width] = resized
    if gray and size in (None, "native"):
        image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return image, [geometry.transform_box(box) for box in boxes]


def load_samples(run):
    records = json.loads((run/"validation.json").read_text())
    sources, samples = {}, []
    with np.load(run/"validation.npz") as prepared:
        pixels = prepared["images"]
    if pixels.shape != (len(records), 1, 120, 160):
        raise ValueError("Validation caméra 160x120 requise pour contrôler les pixels.")
    for index, record in enumerate(records):
        root = relocate_path(record["root"])
        if root not in sources:
            metadata = json.loads((root/"SOURCE.json").read_text())
            if sha256(relocate_path(metadata["source_dataset"])/"manifest.csv") != metadata["source_manifest_sha256"]:
                raise ValueError("Manifest source modifié depuis la préparation.")
            sources[root] = metadata, {r["image"]: r for r in metadata["rows"]}
        metadata, mappings = sources[root]
        mapping = mappings[record["row"]["image"]]
        source_root = relocate_path(metadata["source_dataset"])
        image_path, label_path = source_root/mapping["source_image"], source_root/mapping["source_label"]
        if any(sha256(path) != expected for path, expected in (
            (image_path, mapping["source_image_sha256"]), (label_path, mapping["source_label_sha256"]),
            (root/record["row"]["image"], record["image_sha256"]),
            (root/record["row"]["label"], record["label_sha256"]),
        )):
            raise ValueError(f"Source modifiée : {image_path}")
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Source illisible : {image_path}")
        boxes = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5 or parts[0] != "0":
                raise ValueError(f"Annotation invalide : {label_path}")
            boxes.append(tuple(map(float, parts[1:])))
        reduced, transformed = prepare_variant(frame, boxes, (160, 120), True)
        np.testing.assert_allclose(np.asarray(transformed).reshape(-1, 4),
                                   np.asarray(record["boxes"]).reshape(-1, 4), atol=5.1e-9, rtol=0)
        np.testing.assert_array_equal(reduced[:, :, 0], np.round(pixels[index, 0]*255).astype(np.uint8))
        samples.append({"source": record["source"], "image": str(image_path), "label": str(label_path),
                        "image_sha256": mapping["source_image_sha256"],
                        "label_sha256": mapping["source_label_sha256"],
                        "shape": list(frame.shape[:2]), "frame": frame, "boxes": boxes})
    return samples


def threshold_diagnostics(predictions, samples):
    """Diagnostic complémentaire, pas choix automatique d'un nouveau seuil."""
    if len(predictions) != len(samples):
        raise ValueError("Une prédiction par image requise.")
    report = {}
    for threshold in (.25, .1):
        details = [match_boxes([(p["score"], np.asarray(p["box"])) for p in row["detections_above_005"]
                                if p["score"] >= threshold], row["truth"],
                               width=row["width"], height=row["height"])
                   for row in predictions]
        report[str(threshold)] = {"overall": summarize(details),
                                 "by_source": {source: summarize([d for d, sample in zip(details, samples)
                                                                  if sample["source"] == source])
                                               for source in ("portes", "wepdtof")}}
    return report


def write_report(output, metadata, conditions):
    rows = []
    for condition in conditions:
        m = condition["metrics"]
        rows.append(f"<tr><td>{html.escape(condition['label'])}</td><td>{m['precision']:.1%}</td>"
                    f"<td>{m['recall']:.1%}</td><td>{m['f1']:.1%}</td><td>{m['tp']}</td><td>{m['fp']}</td><td>{m['fn']}</td>"
                    f"<td>{condition['by_source']['portes']['f1']:.1%}</td><td>{condition['by_source']['wepdtof']['f1']:.1%}</td></tr>")
    cards = []
    for index in metadata["preview_indices"]:
        figures = "".join(f"<figure><figcaption>{html.escape(c['label'])}</figcaption>"
                          f"<img loading='lazy' src='apercus/{c['id']}/{index+1:03d}.jpg' alt='Résultat de détection'></figure>"
                          for c in conditions if not c["id"].startswith("native_"))
        sample = metadata["samples"][index]
        cards.append(f"<article><h3>Image {index+1} — {html.escape(sample['source'])} — {html.escape(Path(sample['image']).name)}</h3>"
                     f"<div class='grid'>{figures}</div></article>")
    text = """<!doctype html><html lang='fr'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Diagnostic de résolution — FlowSense</title><style>body{font:17px/1.5 system-ui;max-width:1350px;margin:30px auto;padding:0 20px}table{border-collapse:collapse}th,td{border:1px solid #ccc;padding:8px;text-align:right}td:first-child,th:first-child{text-align:left}.scroll{overflow:auto}.warning{background:#fff1d6;padding:16px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}figure{margin:0}img{max-width:100%;max-height:340px;object-fit:contain}article{border-top:1px solid #ccc;margin-top:30px;padding-top:10px}</style>
<h1>Où perd-on les détections ?</h1><p>Mêmes 98 images de validation et 137 présences annotées corrigées. Aucun entraînement.</p>
<p>YOLO préentraîné local, taille interne 640 fixe, rect=False, seuil 0,5, NMS 0,5 et correspondance IoU ≥ 0,5.
Le gris est répété sur trois canaux identiques pour ce détecteur de référence. Un agrandissement interne n'ajoute aucun détail perdu.</p>
<p class='warning'>Référence logicielle, pas modèle Aidge ni validation Colibry/UINT8/5 mW. La taille interne 640 est uniquement celle de ce diagnostic.
Cette validation a déjà servi au développement. Les annotations locales ont d'abord été proposées par YOLO puis relues : le repère local n'est pas entièrement indépendant.
Un bon résultat ne validerait ni le suivi ni le comptage vidéo.</p>
<h2>Résultats au seuil fixé avant le test</h2><div class='scroll'><table><tr><th>Pixels disponibles</th><th>Précision</th><th>Rappel</th><th>F1</th><th>Retrouvées</th><th>En trop</th><th>Manquées</th><th>F1 portes</th><th>F1 public</th></tr>
"""+"".join(rows)+"""</table></div><p>Comparer couleur/gris à résolution identique pour l'effet du gris ; comparer 320×240 et 160×120 pour celui de la réduction avec le même format caméra.
Les lignes « format caméra sans réduction » ajoutent seulement des bandes 4:3 aux pixels d'origine, afin de contrôler l'effet du cadrage.
Le passage depuis les originaux change donc aussi le cadrage/bandes : ne pas attribuer tous les écarts à un seul facteur.</p>
<h2>Exemples communs</h2><p>Les mêmes indices sont affichés pour chaque condition. Cadres blancs : annotations ; verts : détections appariées ; orange : non appariées. Ces aperçus ne remplacent pas les scores sur les 98 images.</p>
"""+"".join(cards)+"</html>"
    secondary_rows = []
    for condition in conditions:
        for threshold, value in condition["threshold_diagnostics"].items():
            m = value["overall"]
            secondary_rows.append(f"<tr><td>{html.escape(condition['label'])}</td><td>{threshold}</td><td>{m['f1']:.1%}</td>"
                                  f"<td>{m['tp']}</td><td>{m['fp']}</td><td>{m['fn']}</td></tr>")
    keyed = {c["id"]: c for c in conditions}
    qvga, qqvga = keyed["qvga_gray"]["metrics"], keyed["qqvga_gray"]["metrics"]
    diagnosis = (f"<p class='warning'>Sur ce lot et ce détecteur, le gris 320 × 240 donne un F1 de {qvga['f1']:.1%}, "
                 f"contre {qqvga['f1']:.1%} en gris 160 × 120 au même seuil 0,5. "
                 "Cela justifie de tester 320 × 240 pour le prochain petit modèle, sans garantir le même gain pour celui-ci. "
                 "La résolution seule ne résout pas les erreurs restantes. Mémoire, calcul et énergie restent à vérifier.</p>")
    secondary = ("<h2>Diagnostic complémentaire : effet du seuil</h2><p>Calculé après le test principal, à partir des prédictions sauvegardées, sans réentraîner. "
                 "Ce n'est pas un gain de modèle ni une validation aveugle d'un seuil choisi. Aucun réglage du compteur existant n'a été changé.</p>"
                 "<div class='scroll'><table><tr><th>Pixels disponibles</th><th>Seuil</th><th>F1</th><th>Retrouvées</th><th>En trop</th><th>Manquées</th></tr>"
                 +"".join(secondary_rows)+"</table></div><p>Une précision de 100 % avec des personnes manquées ne signifie pas 100 % de réussite. "
                 "Le seuil sélectionné sur ce lot devra encore être évalué sur des séquences indépendantes.</p>")
    text = text.replace("<h2>Exemples communs</h2>", diagnosis+secondary+"<h2>Exemples communs</h2>")
    (output/"rapport.html").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-run", type=Path, default=ROOT/"runs/person_qqvga_hard_context_full_v3")
    parser.add_argument("--weights", type=Path, default=ROOT/"yolo26n.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or not args.weights.is_file():
        raise SystemExit("Nouveau dossier et poids locaux requis ; aucun téléchargement ni écrasement.")
    samples = load_samples(args.validation_run)
    torch.set_num_threads(2)
    model = YOLO(str(args.weights.resolve()))
    args.output_dir.mkdir(parents=True)
    preview_indices = []
    for source in ("portes", "wepdtof"):
        candidates = [i for i, r in enumerate(samples) if r["source"] == source and r["boxes"]]
        preview_indices.extend(candidates[i] for i in np.linspace(0, len(candidates)-1, min(4, len(candidates)), dtype=int))
    metadata = {"weights": str(args.weights.resolve()), "weights_sha256": sha256(args.weights),
                "versions": {"torch": str(torch.__version__), "ultralytics": version("ultralytics"),
                             "numpy": np.__version__, "opencv": cv2.__version__},
                "validation_run": str(args.validation_run.resolve()), "validation_manifest_sha256": sha256(args.validation_run/"validation.json"),
                "images": len(samples), "instances": sum(len(r["boxes"]) for r in samples),
                "threshold": .5, "match_iou": .5, "nms_iou": .5, "inference_size": 640, "rect": False,
                "additional_thresholds": {"values": [.25, .1], "posthoc_diagnostic": True, "automatically_selected": False},
                "preview_indices": preview_indices, "samples": [{k: v for k, v in s.items() if k != "frame"} for s in samples],
                "training_performed": False, "blind_test": False, "aidge_validated": False,
                "hardware_validated": False, "power_validated": False,
                "note": "Comparaison de développement sur les mêmes sources ; pas de validation finale de comptage."}
    results = []
    for name, title, size, gray in CONDITIONS:
        start = time.monotonic()
        details, predictions = [], []
        preview_dir = args.output_dir/"apercus"/name
        preview_dir.mkdir(parents=True)
        for offset in range(0, len(samples), 8):
            batch = samples[offset:offset+8]
            prepared = [prepare_variant(s["frame"], s["boxes"], size, gray) for s in batch]
            output = model.predict(source=[v[0] for v in prepared], classes=[0], conf=.05, iou=.5,
                                   imgsz=640, rect=False, device="cpu", verbose=False)
            if len(output) != len(batch):
                raise RuntimeError("Nombre de sorties inattendu.")
            for index, (detected, (frame, truth)) in enumerate(zip(output, prepared), offset):
                h, w = frame.shape[:2]
                if tuple(detected.orig_shape) != (h, w):
                    raise RuntimeError("Coordonnées de sortie incohérentes.")
                all_pred = common_nms([(float(score), box) for score, box in zip(detected.boxes.conf.cpu().numpy(), detected.boxes.xyxy.cpu().numpy())])
                pred = [(score, box) for score, box in all_pred if score >= .5]
                detail = match_boxes(pred, truth, width=w, height=h, return_details=True)
                details.append(detail)
                predictions.append({"width": w, "height": h, "truth": truth,
                                    "detections_above_005": [{"score": score, "box": box.tolist()} for score, box in all_pred]})
                if index in preview_indices:
                    preview = frame.copy()
                    for cx, cy, bw, bh in truth:
                        cv2.rectangle(preview, (round((cx-bw/2)*w), round((cy-bh/2)*h)),
                                      (round((cx+bw/2)*w), round((cy+bh/2)*h)), (255, 255, 255), max(1, w//400))
                    for item in detail["predictions"]:
                        x1, y1, x2, y2 = [round(x) for x in item["box"]]
                        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 180, 0) if item["matched_truth"] is not None else (0, 140, 255), max(1, w//250))
                    # Agrandissement d'affichage seulement, après mesure.
                    display = cv2.resize(preview, (480, max(1, round(h*480/w))), interpolation=cv2.INTER_NEAREST)
                    if not cv2.imwrite(str(preview_dir/f"{index+1:03d}.jpg"), display):
                        raise RuntimeError("Échec d'enregistrement d'un aperçu.")
        result = {"id": name, "label": title, "grayscale": gray, "metrics": summarize(details),
                  "by_source": {source: summarize([d for d, s in zip(details, samples) if s["source"] == source]) for source in ("portes", "wepdtof")},
                  "seconds": time.monotonic()-start, "details": details, "predictions": predictions}
        result["threshold_diagnostics"] = threshold_diagnostics(predictions, samples)
        results.append(result)
        (args.output_dir/f"{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{title} : {result['metrics']}, {result['seconds']:.1f}s", flush=True)
    (args.output_dir/"comparison.json").write_text(json.dumps({**metadata, "conditions": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(args.output_dir, metadata, results)
    print(f"Rapport : {args.output_dir/'rapport.html'}", flush=True)


if __name__ == "__main__":
    main()
