"""Crée une copie des annotations avec les deux corrections confirmées.

Décisions humaines du 30/10/2026, pas un nettoyage automatique des chevauchements.
Les images et annotations originales sont conservées à l'identique.
"""

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "testvideo_jour33_f00120": [
        [0., .099933, .472526, .199281, .283261],
        [0., .479332, .647727, .214033, .691560],
        [0., .498218, .596239, .173264, .589717],
    ],
    "testvideo_jour33_f00210": [
        [0., .198683, .612243, .397365, .767921],
        [0., .193494, .453316, .386885, .478710],
    ],
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_confirmed_boxes(stem, labels, width, height):
    if (width, height) != (832, 464):
        raise ValueError("Dimensions différentes de celles de l'aperçu confirmé.")
    np.testing.assert_allclose(labels, EXPECTED[stem], rtol=0, atol=1e-9)
    if stem.endswith("f00120"):
        return labels[:2].copy(), "Suppression de l'annotation 3, doublon de la 2."
    # Personne du bas seulement. L'annotation 2 sur celle du haut reste inchangée.
    result = labels.copy()
    x1, y1, x2, y2 = 0., 176., 320., 464.
    result[0] = [0., (x1+x2)/(2*width), (y1+y2)/(2*height), (x2-x1)/width, (y2-y1)/height]
    return result, "Annotation 1 resserrée sur la personne du bas ; annotation 2 conservée."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT/"dataset/detection_proposals")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"dataset/detection_corrected_20261030")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("Dossier déjà présent : aucune version existante n'est écrasée.")
    with (args.source/"manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    original_hashes = {str(path.relative_to(args.source)): digest(path)
                       for split in ("train", "validation")
                       for path in (args.source/split).rglob("*") if path.is_file()}
    original_hashes["manifest.csv"] = digest(args.source/"manifest.csv")
    validation_before = sum(len((args.source/r["label"]).read_text(encoding="utf-8").splitlines())
                            for r in rows if r["split"] == "validation")
    if validation_before != 84:
        raise ValueError("La version source n'est plus celle de l'audit confirmé (84 présences attendues).")
    corrections = []
    # Vérifier toutes les cibles avant d'écrire une nouvelle version.
    for stem in EXPECTED:
        selected = [r for r in rows if Path(r["label"]).stem == stem]
        if len(selected) != 1 or selected[0]["split"] != "validation":
            raise ValueError(f"Cible de validation absente ou ambiguë : {stem}")
        row = selected[0]
        labels = np.loadtxt(args.source/row["label"], ndmin=2)
        image = cv2.imread(str(args.source/row["image"]))
        if image is None:
            raise ValueError(f"Image illisible : {row['image']}")
        corrected, reason = apply_confirmed_boxes(stem, labels, image.shape[1], image.shape[0])
        corrections.append((row, labels, corrected, reason))
    args.output_dir.mkdir(parents=True)
    for split in ("train", "validation"):
        shutil.copytree(args.source/split, args.output_dir/split)
    previews = args.output_dir/"apercus_corrections"
    previews.mkdir()
    history, cards = [], []
    for row, labels, corrected, reason in corrections:
        path = args.output_dir/row["label"]
        path.write_text("".join("0 "+" ".join(f"{v:.10f}" for v in box[1:])+"\n" for box in corrected), encoding="utf-8")
        row.update(proposals=str(len(corrected)), suspicious_overlap="non")
        image = cv2.imread(str(args.output_dir/row["image"]))
        height, width = image.shape[:2]
        for index, (_, cx, cy, bw, bh) in enumerate(corrected):
            color = [(0, 180, 0), (0, 110, 255)][index % 2]
            cv2.rectangle(image, (round((cx-bw/2)*width), round((cy-bh/2)*height)),
                          (round((cx+bw/2)*width), round((cy+bh/2)*height)), color, 3)
        preview = previews/(Path(row["image"]).stem+".jpg")
        if not cv2.imwrite(str(preview), image):
            raise RuntimeError("Échec d'enregistrement d'un aperçu.")
        history.append({"label": row["label"], "split": "validation", "before": labels.tolist(),
                        "after": corrected.tolist(), "reason": reason,
                        "before_sha256": original_hashes[row["label"]], "after_sha256": digest(path)})
        cards.append(f"<h2>{html.escape(Path(row['image']).stem)}</h2><p>{html.escape(reason)}</p>"
                     f"<img style='max-width:100%' src='apercus_corrections/{preview.name}' alt='Annotations corrigées'>")
    with (args.output_dir/"manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    changed = [name for name, sha in original_hashes.items() if name != "manifest.csv" and digest(args.output_dir/name) != sha]
    expected = [item[0]["label"] for item in corrections]
    if set(changed) != set(expected):
        raise RuntimeError("Modification inattendue dans la copie.")
    for name, sha in original_hashes.items():
        if digest(args.source/name) != sha:
            raise RuntimeError("La source a changé pendant la copie ; vérifier avant utilisation.")
    metadata = {"source": str(args.source.resolve()), "approval_date": "2026-10-30",
                "approval": "Utilisateur : Oui, les deux corrections",
                "original_files_unchanged": True, "train_unchanged": True,
                "corrections": history, "local_validation_instances_before": validation_before,
                "local_validation_instances_after": validation_before-1,
                "note": "Deux corrections humaines, pas une revue exhaustive. Les anciens rapports gardent leurs annotations d'origine."}
    (args.output_dir/"CORRECTIONS.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir/"rapport.html").write_text("<!doctype html><html lang='fr'><meta charset='utf-8'>"
        "<title>Annotations corrigées</title><body style='font:17px/1.5 system-ui;max-width:950px;margin:30px auto;padding:15px'>"
        "<h1>Deux corrections confirmées</h1><p>Originaux conservés. Train inchangé. Validation locale : 83 présences au lieu de 84. "
        "Les cadres sont uniquement sur ces aperçus, pas dans les images d'entraînement.</p>"+"".join(cards)+"</body></html>", encoding="utf-8")
    print(f"Deux annotations corrigées ; originaux et train inchangés : {args.output_dir}")


if __name__ == "__main__":
    main()
