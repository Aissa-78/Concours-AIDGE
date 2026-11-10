"""Ajoute 320 images de vidéos publiques, sans modifier les validations existantes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np

from inspect_wepdtof import URL, member_image
from prepare_wepdtof_subset import preview, scene_group, select_scene
from wepdtof_archive import RangeArchive

# Les scènes sans fenêtre négative fiable n'en reçoivent pas artificiellement.
# Le total ajouté conserve 20 % de négatifs : 64 sur 320.
PLAN = (("jewelry_store", 20, .4), ("tech_store", 30, .4),
        ("warehouse", 40, .4), ("exhibition_setup", 54, 1/3),
        ("printing_store", 50, .2), ("repair_store", 50, 0),
        ("it_office", 40, 0), ("large_office", 36, 0))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("dataset/wepdtof_subset"))
    parser.add_argument("--preview-dir", type=Path, default=Path("dataset/wepdtof_preview"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/wepdtof_expanded_v2"))
    parser.add_argument("--seed", type=int, default=93)
    parser.add_argument("--resume", action="store_true", help="Reprendre seulement un téléchargement incomplet identique.")
    args = parser.parse_args()
    root = args.output_dir
    if root.exists() and (not args.resume or (root / "SOURCE.json").exists()):
        raise SystemExit("Dossier existant : choisir un nouveau nom. --resume ne reprend qu'un lot incomplet.")
    with (args.base_dir / "manifest.csv").open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    source_info = json.loads((args.base_dir / "SOURCE.json").read_text(encoding="utf-8"))
    reserved = {row["video"].removeprefix("WEPDTOF-") for row in rows if row["split"] == "validation"}
    if {scene_group(scene) for scene, _, _ in PLAN} & reserved:
        raise ValueError("Une nouvelle vidéo appartient à un lieu réservé à la validation.")
    old_ids = {row["source_image_id"] for row in source_info["rows"]}
    selected = []
    for scene, count, negative_fraction in PLAN:
        data = json.loads((args.preview_dir / f"{scene}.json").read_text())
        roi = cv2.imread(str(args.preview_dir / f"{scene}_roi.png"), cv2.IMREAD_GRAYSCALE)
        chosen = select_scene(data, roi, count, np.random.default_rng(args.seed),
                              scales=(.35, .5, .65), exclude_image_ids=old_ids,
                              negative_fraction=negative_fraction)
        for ordinal, image, crop, (boxes, polygons) in chosen:
            selected.append({"scene": scene, "ordinal": ordinal, "source_image": image,
                             "crop_xyxy": crop, "boxes": boxes, "polygons": polygons})
        print(f"{scene} : +{count} images, {sum(not c[3][0] for c in chosen)} négatives", flush=True)
    snapshot = {"seed": args.seed, "base_source_sha256": digest(args.base_dir / "SOURCE.json"),
                "base_manifest_sha256": digest(args.base_dir / "manifest.csv"), "selected": selected}
    root.mkdir(parents=True, exist_ok=True)
    selection_path = root / "SELECTION.json"
    if selection_path.exists():
        if json.loads(selection_path.read_text()) != json.loads(json.dumps(snapshot)):
            raise ValueError("La sélection ou le jeu parent a changé ; reprise refusée.")
    else:
        selection_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in rows:
        row["origin"] = "lot_v1_conserve"
        for key in ("image", "label"):
            source, destination = args.base_dir / row[key], root / row[key]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copy2(source, destination)
            if digest(source) != digest(destination):
                raise ValueError("Un fichier parent copié diffère ; reprise refusée pour préserver les corrections.")
    source_rows = list(source_info["rows"])
    with RangeArchive(URL, max_bytes=384 * 1024 * 1024) as remote, zipfile.ZipFile(remote) as archive:
        for index, item in enumerate(selected, 1):
            scene, image, crop = item["scene"], item["source_image"], item["crop_xyxy"]
            stem = image["id"]
            if Path(stem).name != stem:
                raise ValueError("Identifiant source non sûr.")
            image_path = Path("train/images") / f"{stem}.jpg"
            label_path = Path("train/labels") / f"{stem}.txt"
            member = f"WEPDTOF/frames/{scene}/{Path(image['file_name']).name}"
            if not (root / image_path).exists():
                raw = member_image(archive, member)
                if raw.shape[:2] != (image["height"], image["width"]):
                    raise ValueError("Dimensions source différentes des annotations.")
                x1, y1, x2, y2 = crop
                frame = raw[y1:y2, x1:x2]
                ok, encoded = cv2.imencode(".jpg", frame)
                if not ok:
                    raise ValueError("Impossible d'encoder un recadrage.")
                partial = (root / image_path).with_suffix(".jpg.part")
                partial.write_bytes(encoded.tobytes())
                partial.replace(root / image_path)
            frame = cv2.imread(str(root / image_path))
            if frame is None or frame.shape[:2] != (crop[3]-crop[1], crop[2]-crop[0]):
                raise ValueError("Recadrage local illisible ou incorrect.")
            label_text = "".join("0 " + " ".join(f"{v:.8f}" for v in box) + "\n" for box in item["boxes"])
            if (root / label_path).exists() and (root / label_path).read_text() != label_text:
                raise ValueError("Une annotation a été modifiée pendant la reprise : arrêt sans l'écraser.")
            (root / label_path).write_text(label_text, encoding="utf-8")
            rows.append({"split": "train", "video": "WEPDTOF-" + scene_group(scene),
                         "frame": int(stem.rsplit("_", 1)[-1]), "time_seconds": "",
                         "image": image_path.as_posix(), "label": label_path.as_posix(),
                         "reviewed": "annotation_wepdtof", "origin": "ajout_v2"})
            source_rows.append({"image": image_path.as_posix(), "archive_member": member,
                                "source_image_id": image["id"], "crop_xyxy": crop,
                                "source_polygons_in_crop": item["polygons"],
                                "image_file_sha256": digest(root / image_path)})
            if index % 20 == 0:
                print(f"Ajouts : {index}/{len(selected)} ; {remote.transferred / 1e6:.1f} Mo transférés", flush=True)
        transferred, etag = remote.transferred, remote.etag
    fields = list(rows[0])
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (root / "apercus").mkdir(exist_ok=True)
    cards, examples = [], {}
    for index, row in enumerate(rows, 1):
        boxes = [tuple(map(float, line.split()[1:])) for line in (root / row["label"]).read_text().splitlines()]
        frame = cv2.imread(str(root / row["image"]))
        scene = row["video"].removeprefix("WEPDTOF-")
        panel = preview(frame, boxes, f"{index:03d} - {scene}")
        cv2.imwrite(str(root / "apercus" / f"{index:03d}.jpg"), panel)
        if row["origin"] == "ajout_v2" and scene not in examples and boxes:
            examples[scene] = panel
        cards.append(f"<article><h2>{index:03d} — {html.escape(Path(row['image']).stem)}</h2>"
                     f"<p>{row['split']} · {row['origin']} · {len(boxes)} personne(s)</p>"
                     f"<img loading='lazy' src='apercus/{index:03d}.jpg' alt='Annotation source convertie'></article>")
    panels = list(examples.values())
    while len(panels) % 4:
        panels.append(np.full_like(panels[0], 248))
    cv2.imwrite(str(root / "nouveaux_exemples.jpg"), np.concatenate([
        np.concatenate(panels[i:i+4], axis=1) for i in range(0, len(panels), 4)], axis=0))
    (root / "rapport.html").write_text("""<!doctype html><html lang='fr'><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>WEPDTOF : lot augmenté</title>
<style>body{font:17px/1.5 system-ui;max-width:1200px;margin:30px auto;padding:0 20px}main{display:flex;flex-wrap:wrap;gap:15px}article{width:300px}img{width:100%}h2{font-size:16px}</style>
<h1>Lot augmenté de vues du dessus</h1><p>Les images et annotations du premier lot sont conservées,
y compris sa validation. Les ajouts proviennent uniquement des scènes d'apprentissage.
Vert : annotation d'origine convertie, pas une prédiction du modèle. Les recadrages ne corrigent pas le fisheye.</p><main>"""
        + "".join(cards) + "</main></html>", encoding="utf-8")
    info = {**source_info, "dataset": "WEPDTOF, lot augmenté local ; pas le benchmark officiel",
            "parent_dataset": str(args.base_dir.resolve()), "parent_source_sha256": digest(args.base_dir / "SOURCE.json"),
            "seed": args.seed, "transferred_bytes": transferred, "archive_etag": etag,
            "train_scenes": [scene for scene, _, _ in PLAN], "rows": source_rows,
            "selection": "Lot parent intégral conservé. Ajouts : fenêtres fixes à 35/50/65% du petit côté, entièrement en ROI, 0-3 personnes, cinq indices minimum entre frames d'un clip, aucune frame parent réutilisée.",
            "added_images": len(selected), "validation_unchanged": True}
    (root / "SOURCE.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Lot augmenté prêt : {root}, {sum(r['split']=='train' for r in rows)} train / {sum(r['split']=='validation' for r in rows)} validation", flush=True)


if __name__ == "__main__":
    main()
