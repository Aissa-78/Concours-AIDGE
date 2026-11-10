"""Compare des essais QQVGA sur les mêmes images/annotations de validation.

Post-traitement commun : seuil 0,5, pics 3x3, NMS/IoU 0,5. Les métriques du
protocole de chaque essai sont aussi conservées pour ne pas réécrire l'histoire.
"""

import argparse
import html
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from person_detection_math import evaluate_boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-run", type=Path,
                        help="Réévaluer toutes les prédictions avec la même version corrigée des annotations, sans changer les pixels.")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("Conserver le bilan existant ; choisir un nouveau dossier de comparaison.")
    reference, result = None, []
    corrected_records = (json.loads((args.validation_run/"validation.json").read_text())
                         if args.validation_run else None)
    corrected_images = None
    if args.validation_run:
        with np.load(args.validation_run/"validation.npz") as values:
            corrected_images = values["images"]
    for run in args.runs:
        original_records = json.loads((run/"validation.json").read_text())
        changed_labels = 0
        if corrected_records is not None:
            pixels = [(r["source"], r["image_sha256"]) for r in original_records]
            if pixels != [(r["source"], r["image_sha256"]) for r in corrected_records]:
                raise ValueError("Les images ou leur ordre diffèrent : impossible de réutiliser les prédictions.")
            with np.load(run/"validation.npz") as values:
                if not np.array_equal(values["images"], corrected_images):
                    raise ValueError("Les pixels préparés diffèrent ; refaire l'inférence avant de comparer.")
            changed_labels = sum(a["boxes"] != b["boxes"] for a, b in zip(original_records, corrected_records))
            records = corrected_records
        else:
            records = original_records
        fingerprint = [(r["image_sha256"], r["label_sha256"], r["boxes"]) for r in records]
        if reference is None:
            reference = fingerprint
        elif fingerprint != reference:
            raise ValueError("Les validations diffèrent : ne pas mélanger ces résultats.")
        info = json.loads((run/"dataset.json").read_text())
        trained = json.loads((run/"training.json").read_text())
        measured = json.loads((run/"metrics.json").read_text())
        outputs = np.load(run/"compact_validation_logits.npy")
        overall, _ = evaluate_boxes(outputs, [r["boxes"] for r in records], local_maximum=True)
        by_source = {}
        for source in ("portes", "wepdtof"):
            indices = [i for i, r in enumerate(records) if r["source"] == source]
            by_source[source], _ = evaluate_boxes(outputs[indices], [records[i]["boxes"] for i in indices], local_maximum=True)
        result.append({"run": str(run.resolve()), "name": run.name,
                       "train_images": measured["train"]["images"], "parameters": info["parameters_compact"],
                       "architecture": info.get("architecture", "separable"), "residual": info.get("residual", False),
                       "classification": info.get("classification", "focal"), "augmentation": trained["augmentations"],
                       "selected_epoch": trained["selected_epoch"], "common_metrics": overall,
                       "initial_checkpoint": trained.get("initial_checkpoint"),
                       "no_new_update_selected": trained["selected_epoch"] == 0,
                       "reevaluated_label_images": changed_labels,
                       "common_by_source": by_source, "original_metrics": measured["validation"]["metrics"]})
    args.output_dir.mkdir(parents=True)
    metadata = {"validation_images": len(reference), "validation_instances": sum(len(r["boxes"]) for r in records),
                "validation_annotation_run": str(args.validation_run.resolve()) if args.validation_run else None,
                "threshold": .5, "iou_threshold": .5,
                "common_local_maximum": True, "runs": result, "blind_test": False,
                "note": "Développement sur une validation réutilisée. Les lots train et plusieurs réglages peuvent différer ; pas une étude d'ablation exhaustive."}
    (args.output_dir/"comparison.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for r in result:
        m = r["common_metrics"]
        link = os.path.relpath(Path(r["run"])/"rapport.html", args.output_dir.resolve())
        rows.append(f"<tr><td><a href='{html.escape(link)}'>{html.escape(r['name'])}</a></td>"
                    f"<td>{r['train_images']}</td><td>{r['selected_epoch']}</td><td>{r['parameters']}</td><td>{html.escape(r['classification'])}</td>"
                    f"<td>{r['common_by_source']['portes']['f1']:.1%}</td><td>{r['common_by_source']['wepdtof']['f1']:.1%}</td>"
                    f"<td>{m['f1']:.1%}</td><td>{m['tp']}</td><td>{m['fp']}</td><td>{m['fn']}</td></tr>")
    annotation_note = (f"<p>Annotations communes corrigées : {metadata['validation_instances']} présences sur {len(reference)} images. "
                       "Les prédictions anciennes sont réévaluées sans changer les pixels ni réentraîner les poids. "
                       "Les rapports individuels restent inchangés et peuvent donc afficher les anciens chiffres.</p>"
                       if args.validation_run else "")
    (args.output_dir/"rapport.html").write_text("""<!doctype html><html lang='fr'><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>Comparaison des détecteurs</title>
<style>body{font:17px/1.5 system-ui;max-width:1400px;margin:30px auto;padding:0 16px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}.scroll{overflow:auto}.warning{padding:16px;background:#fff1d6}</style>
<h1>Détecteurs QQVGA — comparaison de développement</h1>
<p class='warning'>Les mêmes 98 images et les mêmes annotations servent à cette comparaison. Elles sont hors apprentissage,
mais déjà utilisées pour choisir et régler les modèles : ce n'est pas un test aveugle sur une nouvelle porte.
Les nombres d'images train et certains réglages diffèrent ; ne pas attribuer tout changement à une seule cause.</p>
<p>Post-traitement commun : seuil 0,5, pics locaux 3×3, NMS 0,5 ; une détection correcte exige IoU ≥ 0,5.
Les rapports individuels gardent le post-traitement de leur protocole. Aucun seuil n'a été choisi sur la porte B.</p>
"""+annotation_note+"<div class='scroll'><table><tr><th>Essai</th><th>Images du lot train</th><th>Époque retenue</th><th>Paramètres</th><th>Loss présence</th><th>F1 portes</th><th>F1 public</th><th>F1 global</th><th>Retrouvées</th><th>Supplémentaires</th><th>Manquées</th></tr>"
        +"".join(rows)+"</table></div><p>Une époque retenue égale à 0 signifie qu'aucune nouvelle mise à jour de l'essai n'a été retenue ; en cas de poursuite, ce sont les poids du checkpoint de départ. Une présence annotée n'est pas nécessairement une personne distincte : la même personne peut apparaître sur plusieurs frames. Aucun résultat ne valide encore le comptage, l'UINT8 sur carte ni les 5 mW.</p></html>", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Comparaison : {args.output_dir/'rapport.html'}")


if __name__ == "__main__":
    main()
