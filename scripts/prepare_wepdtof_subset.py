"""Petit lot de vues du dessus WEPDTOF, avec labels et séparation par scène.

Lire les conditions d'utilisation du fournisseur avant un autre usage que la
recherche non commerciale. Ce programme prépare les données ; il n'entraîne
pas le réseau et ne modifie pas les jeux de portes existants.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import html
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np

from inspect_wepdtof import URL, member_image
from wepdtof_archive import RangeArchive

TRAIN_SCENES = ("jewelry_store", "tech_store", "warehouse", "exhibition_setup")
VALIDATION_SCENES = ("empty_store", "convenience_store")


def scene_group(scene):
    """Deux clips d'un même lieu ne doivent jamais être dans deux splits."""
    return scene[:-2] if scene.endswith("_2") else scene


def rotated_polygon(box):
    """Format officiel [cx,cy,w,h,angle horaire], angle depuis la verticale."""
    cx, cy, width, height, degrees = map(float, box)
    if not np.isfinite([cx, cy, width, height, degrees]).all() or min(width, height) <= 0:
        raise ValueError("Rectangle WEPDTOF invalide.")
    angle = np.deg2rad(degrees)
    rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    corners = np.array([[-width/2, -height/2], [width/2, -height/2],
                        [width/2, height/2], [-width/2, height/2]])
    return np.ascontiguousarray(corners @ rotation + [cx, cy], dtype=np.float32)


def crop_annotations(annotations, crop, max_people=3):
    """Conserve toutes les personnes du crop, ou rejette le crop entier.

    Ne transforme pas une personne trop petite/coupée en exemple de fond.
    Les fenêtres sont fixes, indépendantes des positions des personnes.
    """
    x1, y1, x2, y2 = crop
    width, height = x2 - x1, y2 - y1
    window = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
    boxes, polygons = [], []
    for annotation in annotations:
        if annotation.get("category_id") != 1 or annotation.get("iscrowd", 0) or annotation.get("ignore", False):
            return None
        polygon = rotated_polygon(annotation["bbox"])
        area, clipped = cv2.intersectConvexConvex(polygon, window)
        if area <= 1e-4:
            continue
        if area < annotation["bbox"][2] * annotation["bbox"][3] * 0.7:
            return None
        points = clipped.reshape(-1, 2) - [x1, y1]
        low, high = points.min(axis=0), points.max(axis=0)
        low = np.maximum(low, 0)
        high = np.minimum(high, [width, height])
        bw, bh = (high - low) / [width, height]
        if min(bw, bh) < 0.06 or max(bw, bh) < 0.15:
            return None
        cx, cy = (low + high) / [2 * width, 2 * height]
        boxes.append((float(cx), float(cy), float(bw), float(bh)))
        polygons.append(points.tolist())
    if len(boxes) > max_people:
        return None
    return boxes, polygons


def fixed_crops(width, height, roi, scales=(0.5,)):
    """Fenêtres carrées à 25 positions fixes, uniquement en zone annotée."""
    if roi.shape != (height, width):
        raise ValueError("Le masque ROI ne correspond pas aux dimensions de la scène.")
    for scale in scales:
        if not 0 < scale <= 1:
            raise ValueError("Échelle de recadrage invalide.")
        size = round(min(width, height) * scale)
        for cy in (0.3, 0.4, 0.5, 0.6, 0.7):
            for cx in (0.3, 0.4, 0.5, 0.6, 0.7):
                x, y = round(cx * width - size/2), round(cy * height - size/2)
                if min(x, y) < 0 or x + size > width or y + size > height:
                    continue
                if np.all(roi[y:y + size, x:x + size] > 127):
                    yield (x, y, x + size, y + size)


def select_scene(data, roi, count, rng, *, scales=(0.5,), exclude_image_ids=(),
                 min_gap=5, negative_fraction=0.2):
    if not 0 <= negative_fraction <= 1 or min_gap < 1:
        raise ValueError("Paramètres de sélection invalides.")
    by_image = defaultdict(list)
    for annotation in data["annotations"]:
        by_image[annotation["image_id"]].append(annotation)
    candidates = {"positive": [], "negative": []}
    images = sorted(data["images"], key=lambda image: image["file_name"])
    excluded = set(exclude_image_ids)
    used = {i for i, image in enumerate(images) if image["id"] in excluded}
    for ordinal, image in enumerate(images):
        if ordinal in used or any(abs(ordinal - previous) < min_gap for previous in used):
            continue
        for crop in fixed_crops(image["width"], image["height"], roi, scales):
            labels = crop_annotations(by_image[image["id"]], crop)
            if labels is not None:
                candidates["positive" if labels[0] else "negative"].append(
                    (ordinal, image, crop, labels))
    for values in candidates.values():
        rng.shuffle(values)
    selected = []
    negative_count = round(count * negative_fraction)
    for kind, requested in (("positive", count - negative_count), ("negative", negative_count)):
        current = []
        if requested == 0:
            continue
        for candidate in candidates[kind]:
            ordinal = candidate[0]
            if ordinal in used or any(abs(ordinal - previous) < min_gap for previous in used):
                continue
            current.append(candidate)
            used.add(ordinal)
            if len(current) == requested:
                break
        if len(current) < requested:
            raise ValueError(f"Pas assez de fenêtres {kind} espacées : {len(current)}/{requested}.")
        selected.extend(current)
    return sorted(selected, key=lambda item: item[0])


def preview(frame, boxes, title):
    size = 300
    panel = np.full((size + 50, size, 3), 248, np.uint8)
    small = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    for cx, cy, bw, bh in boxes:
        low = (round((cx-bw/2)*size), round((cy-bh/2)*size))
        high = (round((cx+bw/2)*size), round((cy+bh/2)*size))
        cv2.rectangle(small, low, high, (0, 230, 0), 2)
        cv2.circle(small, (round(cx*size), round(cy*size)), 3, (0, 230, 255), -1)
    panel[50:] = small
    cv2.putText(panel, title, (7, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1)
    cv2.putText(panel, f"{len(boxes)} personne(s) annotee(s)", (7, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
    return panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path, default=Path("dataset/wepdtof_preview"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/wepdtof_subset"))
    parser.add_argument("--images-per-scene", type=int, default=20)
    args = parser.parse_args()
    if args.images_per_scene < 5:
        raise SystemExit("Prévoir au moins 5 images par scène.")
    root = args.output_dir
    if root.exists():
        raise SystemExit("Le dossier existe déjà : choisir un autre --output-dir pour préserver les images.")
    if {scene_group(s) for s in TRAIN_SCENES} & {scene_group(s) for s in VALIDATION_SCENES}:
        raise SystemExit("Même lieu dans les deux groupes.")
    rng = np.random.default_rng(71)
    selection = []
    for split, scenes in (("train", TRAIN_SCENES), ("validation", VALIDATION_SCENES)):
        for scene in scenes:
            metadata = args.preview_dir / f"{scene}.json"
            data = json.loads(metadata.read_text(encoding="utf-8"))
            roi = cv2.imread(str(args.preview_dir / f"{scene}_roi.png"), cv2.IMREAD_GRAYSCALE)
            chosen = select_scene(data, roi, args.images_per_scene, rng)
            selection.extend((split, scene, candidate) for candidate in chosen)
            print(f"Sélection {scene} : {len(chosen)} fenêtres ; {sum(not c[3][0] for c in chosen)} vides", flush=True)
    root.mkdir(parents=True)
    (root / "apercus").mkdir()
    rows, source_rows, panels, cards = [], [], [], []
    hashes = {"train": set(), "validation": set()}
    with RangeArchive(URL) as remote, zipfile.ZipFile(remote) as archive:
        for index, (split, scene, candidate) in enumerate(selection, 1):
            _, image, crop, (boxes, polygons) = candidate
            member = f"WEPDTOF/frames/{scene}/{Path(image['file_name']).name}"
            raw = member_image(archive, member)
            if raw.shape[:2] != (image["height"], image["width"]):
                raise ValueError("Dimensions d'image différentes des annotations.")
            x1, y1, x2, y2 = crop
            frame = raw[y1:y2, x1:x2]
            digest = hashlib.sha256(frame.tobytes()).hexdigest()
            hashes[split].add(digest)
            stem = f"{scene}_{image['id'].rsplit('_', 1)[-1]}"
            image_path = Path(split) / "images" / f"{stem}.jpg"
            label_path = Path(split) / "labels" / f"{stem}.txt"
            (root / image_path).parent.mkdir(parents=True, exist_ok=True)
            (root / label_path).parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(root / image_path), frame):
                raise ValueError("Impossible d'enregistrer l'image recadrée.")
            (root / label_path).write_text("".join("0 " + " ".join(f"{v:.8f}" for v in box) + "\n" for box in boxes), encoding="utf-8")
            rows.append({"split": split, "video": "WEPDTOF-" + scene_group(scene),
                         "frame": int(stem.rsplit("_", 1)[-1]), "time_seconds": "",
                         "image": image_path.as_posix(), "label": label_path.as_posix(),
                         "reviewed": "annotation_wepdtof"})
            source_rows.append({"image": image_path.as_posix(), "archive_member": member,
                                "crop_xyxy": crop, "source_image_id": image["id"],
                                "source_polygons_in_crop": polygons, "decoded_crop_sha256": digest})
            panel = preview(frame, boxes, f"{index:03d} - {scene}")
            cv2.imwrite(str(root / "apercus" / f"{index:03d}.jpg"), panel)
            if not any(s == scene for s, _ in panels) and boxes:
                panels.append((scene, panel))
            cards.append(f"<article><h2>{index:03d} — {html.escape(stem)}</h2><p>{split} · {len(boxes)} personne(s)</p>"
                         f"<a href='apercus/{index:03d}.jpg'><img loading='lazy' src='apercus/{index:03d}.jpg' alt='Annotation source convertie'></a></article>")
            if index % 10 == 0:
                print(f"WEPDTOF : {index}/{len(selection)} ; {remote.transferred / 1e6:.1f} Mo transférés", flush=True)
        if hashes["train"] & hashes["validation"]:
            raise ValueError("Image identique dans train et validation : lot non validé.")
        with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (root / "SOURCE.json").write_text(json.dumps({
            "dataset": "WEPDTOF, sous-ensemble local recadré ; pas le benchmark officiel",
            "source": "https://vip.bu.edu/projects/vsns/cossy/datasets/wepdtof/",
            "archive_url": URL, "archive_etag": remote.etag, "transferred_bytes": remote.transferred,
            "usage": "Le fournisseur réserve ce jeu à la recherche non commerciale et demande de citer Tezcan et al., WACV 2022. Vérifier ses conditions avant tout autre usage.",
            "license_page": "https://vip.bu.edu/projects/vsns/cossy/datasets/wepdtof/download/",
            "seed": 71, "train_scenes": TRAIN_SCENES, "validation_scenes": VALIDATION_SCENES,
            "annotation_format": "Sources : cx,cy,w,h,angle horaire. Local : enveloppe horizontale du polygone visible recadré.",
            "selection": "Fenêtres fixes, entièrement dans la ROI ; 0 à 3 personnes ; rejet du crop si personne trop petite ou trop coupée ; 20% sans personne ; 5 indices de frame entre deux images.",
            "review": "Annotations originales converties, non revues intégralement par l'équipe ; pas de marque oui automatique.",
            "rows": source_rows,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    values = [panel for _, panel in panels]
    while len(values) % 3:
        values.append(np.full_like(values[0], 248))
    cv2.imwrite(str(root / "planche.jpg"), np.concatenate([
        np.concatenate(values[offset:offset+3], axis=1) for offset in range(0, len(values), 3)], axis=0))
    (root / "rapport.html").write_text("""<!doctype html><html lang='fr'><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>Vues du dessus — WEPDTOF</title>
<style>body{font:17px/1.5 system-ui;max-width:1100px;margin:30px auto;padding:0 20px}main{display:flex;flex-wrap:wrap;gap:20px}article{max-width:320px}img{max-width:100%}h2{font-size:16px}</style>
<h1>Lot public de vues du dessus</h1><p>Annotations WEPDTOF originales converties en rectangles horizontaux.
Les recadrages réduisent le champ, mais ne corrigent pas la déformation fisheye.
Ces images ne sont pas des captures de notre porte et ne prouvent pas une amélioration du modèle.</p>
<p>Vert : rectangle converti. Jaune : centre visé. Les scènes de validation sont distinctes de l'apprentissage.
Cette galerie décrit les données, pas les performances d'un modèle entraîné.</p><main>""" + "".join(cards) + "</main></html>", encoding="utf-8")
    print(f"Lot prêt : {root} ; rapport : {root / 'rapport.html'}", flush=True)


if __name__ == "__main__":
    main()
