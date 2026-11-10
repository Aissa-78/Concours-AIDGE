"""Prépare puis finalise un lot COCO corps entiers, sans entraîner de modèle.

La présélection utilise les keypoints officiels. L'orientation est une
heuristique suivie d'une revue visuelle ; ce n'est pas une classe COCO.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import html
import json
from pathlib import Path
import random
import re
import shutil
import sys
import zipfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from coco_sideview_selection import ALLOWED_LICENSES, inspect_image
from camera_preprocess import prepare_camera_frame
from download_coco_person_subset import BASE, download

HUMAN_WORDS = re.compile(r"\b(man|men|woman|women|person|people|boy|girl|child|children|pedestrian|someone|crowd)\b", re.I)
WALKABLE = re.compile(r"\b(street|sidewalk|hallway|corridor|door|building|entrance|room|kitchen|store|shop|parking|station|path|stairs|stairway)\b", re.I)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def phash(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coefficients = cv2.dct(cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32))[:8, :8].ravel()[1:]
    result = 0
    for value in coefficients > np.median(coefficients):
        result = (result << 1) | int(value)
    return result


def build_plan(args):
    if args.output_dir.exists():
        raise SystemExit("Dossier déjà présent : choisir un nouveau nom.")
    with zipfile.ZipFile(args.annotations_zip) as archive:
        data = json.loads(archive.read("annotations/person_keypoints_train2017.json"))
        by_image, captions = defaultdict(list), defaultdict(list)
        for annotation in data["annotations"]:
            by_image[annotation["image_id"]].append(annotation)
        for caption in json.loads(archive.read("annotations/captions_train2017.json"))["annotations"]:
            captions[caption["image_id"]].append(caption["caption"])
    previous = ROOT/"dataset/coco_person_subset/SOURCE.json"
    excluded = {r["id"] for r in json.loads(previous.read_text())["images"]} if previous.exists() else set()
    pools, negatives = defaultdict(list), []
    licenses = {r["id"]: r for r in data["licenses"]}
    for image in data["images"]:
        if image["id"] in excluded or image["license"] not in ALLOWED_LICENSES:
            continue
        anns, texts = by_image[image["id"]], captions[image["id"]]
        if not anns:
            if WALKABLE.search(" ".join(texts)) and not HUMAN_WORDS.search(" ".join(texts)):
                negatives.append({"image": image, "annotations": [], "captions": texts,
                                  "kind": "vide", "selection_score": 0.})
            continue
        result = inspect_image(image, anns, texts)
        if result:
            pools[result["orientation"]].append({"image": image, "annotations": anns,
                                                  "captions": texts, "kind": result["orientation"], **result})
    rng = random.Random(args.seed)
    candidates = []
    for kind, limit in (("profil_probable", 190), ("autre_vue_debout", 90)):
        pool = pools[kind]
        rng.shuffle(pool)
        pool.sort(key=lambda row: row["selection_score"], reverse=True)
        candidates.extend(pool[:limit])
    rng.shuffle(negatives)
    candidates.extend(negatives[:70])
    for index, row in enumerate(candidates, 1):
        row.update(review_id=f"{index:03d}", license=licenses[row["image"]["license"]],
                   official_split="train2017", url=BASE+"/train2017/"+row["image"]["file_name"])
    args.output_dir.mkdir(parents=True)
    dump(args.output_dir/"plan.json", {"seed": args.seed, "candidates": candidates,
         "annotation_archive": str(args.annotations_zip.resolve()), "annotation_archive_sha256": digest(args.annotations_zip),
         "previous_ids_excluded": sorted(excluded), "available": {**{key: len(value) for key, value in pools.items()}, "vide": len(negatives)},
         "orientation_note": "Profil probable d'après les keypoints, à contrôler visuellement. Pas de métadonnée COCO garantissant cet angle.",
         "licenses": list(licenses.values())})
    print({"candidates": len(candidates), "kinds": dict(Counter(r["kind"] for r in candidates))}, flush=True)


def fetch_and_preview(args):
    plan = json.loads((args.output_dir/"plan.json").read_text())
    candidates = plan["candidates"]
    image_root = args.output_dir/"images"
    image_root.mkdir(exist_ok=True)
    def fetch(row):
        path = image_root/row["image"]["file_name"]
        download(row["url"], path)
        frame = cv2.imread(str(path))
        if frame is None or frame.shape[:2] != (row["image"]["height"], row["image"]["width"]):
            raise ValueError(f"Image illisible ou dimensions incorrectes : {path}")
        return {**row, "sha256": digest(path), "phash": phash(frame)}
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, row in enumerate(pool.map(fetch, candidates), 1):
            records.append(row)
            if index % 25 == 0 or index == len(candidates):
                print(f"Images téléchargées et vérifiées : {index}/{len(candidates)}", flush=True)
    dump(args.output_dir/"downloaded.json", {**{k: v for k, v in plan.items() if k != "candidates"}, "candidates": records})
    sheets = args.output_dir/"planches"
    sheets.mkdir(exist_ok=True)
    for start in range(0, len(records), 20):
        sheet = np.full((4*270, 5*330, 3), 245, np.uint8)
        for offset, row in enumerate(records[start:start+20]):
            frame = cv2.imread(str(image_root/row["image"]["file_name"]))
            h, w = frame.shape[:2]
            for ann in row["annotations"]:
                x, y, bw, bh = ann["bbox"]
                cv2.rectangle(frame, (round(x), round(y)), (round(x+bw), round(y+bh)), (30, 210, 30), max(1, w//300))
            scale = min(320/w, 235/h)
            resized = cv2.resize(frame, (round(w*scale), round(h*scale)), interpolation=cv2.INTER_AREA)
            top, left = (offset//5)*270+28, (offset%5)*330+5
            sheet[top:top+resized.shape[0], left:left+resized.shape[1]] = resized
            label = row["review_id"]+" "+row["kind"].replace("profil_probable", "profil?").replace("autre_vue_debout", "debout")
            cv2.putText(sheet, label, (left, top-8), cv2.FONT_HERSHEY_SIMPLEX, .53, (25, 25, 25), 1)
        if not cv2.imwrite(str(sheets/f"{start//20+1:02d}.jpg"), sheet):
            raise RuntimeError("Échec de création d'une planche.")
    print(f"Planches de revue : {sheets}")


def finalize(args):
    if not args.final_dir or args.final_dir.exists() or not args.decisions:
        raise SystemExit("Un nouveau --final-dir et un fichier --decisions sont requis.")
    source = json.loads((args.output_dir/"downloaded.json").read_text())
    decisions = json.loads(args.decisions.read_text())
    excluded = set(decisions["rejected_review_ids"])
    accepted = set(decisions["accepted_review_ids"])
    known = {row["review_id"] for row in source["candidates"]}
    if accepted & excluded or accepted | excluded != known:
        raise SystemExit("Chaque candidat doit avoir une décision explicite unique : accepté ou rejeté.")
    candidates = [row for row in source["candidates"] if row["review_id"] in accepted]
    # Écarter les doublons proches avant la répartition, sans regarder de score de modèle.
    unique, duplicates = [], []
    for row in candidates:
        previous = next((other for other in unique if row["sha256"] == other["sha256"]
                         or (row["phash"] ^ other["phash"]).bit_count() <= 4), None)
        if previous:
            duplicates.append({"rejected": row["review_id"], "kept": previous["review_id"]})
        else:
            unique.append(row)
    rng = random.Random(source["seed"])
    positives = [r for r in unique if r["kind"] != "vide"]
    negatives = [r for r in unique if r["kind"] == "vide"]
    rng.shuffle(positives)
    rng.shuffle(negatives)
    # Garder la majorité de profils suggérés, sans supprimer toutes les autres vues.
    positives.sort(key=lambda r: (r["kind"] == "profil_probable", r["selection_score"]), reverse=True)
    if len(positives) < args.positive_count or len(negatives) < args.negative_count:
        raise SystemExit(f"Lot insuffisant après revue : {len(positives)} positifs, {len(negatives)} négatifs. Ne pas compléter avec des cas douteux.")
    positives, negatives = positives[:args.positive_count], negatives[:args.negative_count]
    rng.shuffle(positives)
    rng.shuffle(negatives)
    pos_cut, neg_cut = len(positives)*4//5, len(negatives)*4//5
    selected = [("train", r) for r in positives[:pos_cut]+negatives[:neg_cut]]+[("validation", r) for r in positives[pos_cut:]+negatives[neg_cut:]]
    split_counts = Counter(split for split, _ in selected)
    args.final_dir.mkdir(parents=True)
    camera = args.final_dir/"camera_320"
    previews = args.final_dir/"apercus"
    previews.mkdir()
    rows, camera_rows, camera_source_rows, provenance, cards = [], [], [], [], []
    for index, (split, row) in enumerate(selected, 1):
        info = row["image"]
        original = args.output_dir/"images"/info["file_name"]
        if digest(original) != row["sha256"]:
            raise ValueError("Image candidate modifiée depuis la revue.")
        relative = Path(split)/"images"/info["file_name"]
        label = Path(split)/"labels"/(Path(info["file_name"]).stem+".txt")
        for folder in (args.final_dir/relative.parent, args.final_dir/label.parent, camera/relative.parent, camera/label.parent):
            folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, args.final_dir/relative)
        frame = cv2.imread(str(original))
        h, w = frame.shape[:2]
        boxes = []
        for annotation in row["annotations"]:
            x, y, bw, bh = annotation["bbox"]
            x1, y1, x2, y2 = max(0, x), max(0, y), min(w, x+bw), min(h, y+bh)
            if x2 <= x1 or y2 <= y1:
                raise ValueError("Annotation dégénérée.")
            boxes.append(((x1+x2)/(2*w), (y1+y2)/(2*h), (x2-x1)/w, (y2-y1)/h))
        write_labels(args.final_dir/label, boxes)
        gray, geometry = prepare_camera_frame(frame, width=320, height=240)
        transformed = [geometry.transform_box(box) for box in boxes]
        camera_image = relative.with_suffix(".png")
        if not cv2.imwrite(str(camera/camera_image), gray):
            raise RuntimeError("Échec de préparation caméra.")
        write_labels(camera/label, transformed)
        entry = {"split": split, "video": f"COCO2017-{info['id']}", "frame": 0, "time_seconds": 0,
                 "image": relative.as_posix(), "label": label.as_posix(), "reviewed": "annotation_coco",
                 "coco_id": info["id"], "license_id": info["license"], "view_hint": row["kind"]}
        rows.append(entry)
        camera_rows.append({**entry, "image": camera_image.as_posix()})
        camera_source_rows.append({"image": camera_image.as_posix(), "source_image": relative.as_posix(),
                                   "source_label": label.as_posix(), "geometry": vars(geometry),
                                   "source_image_sha256": row["sha256"], "source_label_sha256": digest(args.final_dir/label)})
        provenance.append({**row, "split": split, "relative_image": relative.as_posix(), "relative_label": label.as_posix(),
                           "camera_image": camera_image.as_posix(), "camera_geometry": vars(geometry),
                           "annotation_ids": [a["id"] for a in row["annotations"]], "source_person_annotations_preserved": len(boxes)})
        preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for cx, cy, bw, bh in transformed:
            cv2.rectangle(preview, (round((cx-bw/2)*320), round((cy-bh/2)*240)), (round((cx+bw/2)*320), round((cy+bh/2)*240)), (0, 180, 0), 1)
        preview_name = f"{index:03d}_{info['id']:012d}.png"
        cv2.imwrite(str(previews/preview_name), preview)
        cards.append(f"<article><h3>{split} — COCO {info['id']}</h3><p>{len(boxes)} personne(s) ; {html.escape(row['kind'])}</p>"
                     f"<a href='{relative.as_posix()}'><img loading='lazy' src='apercus/{preview_name}' alt='Copie grise et annotations'></a></article>")
    for root, manifest_rows in ((args.final_dir, rows), (camera, camera_rows)):
        with (root/"manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0])); writer.writeheader(); writer.writerows(manifest_rows)
    metadata = {"dataset": "COCO2017 train2017, sous-ensemble local corps entiers", "source": "https://cocodataset.org/#download",
                "selection": "Keypoints + corps non coupés/grands + revue visuelle avant entraînement, aucun score de modèle utilisé.",
                "orientation_warning": "Les indications de profil sont des heuristiques, pas des annotations officielles d'orientation.",
                "local_split_warning": "Deux lots locaux tirés de train2017 ; pas un score COCO officiel ni un test aveugle sur une nouvelle porte.",
                "train_images": split_counts["train"], "validation_images": split_counts["validation"], "seed": source["seed"],
                "candidate_plan": str((args.output_dir/"plan.json").resolve()), "visual_decisions": decisions,
                "near_duplicates_removed": duplicates, "licenses": source["licenses"], "images": provenance,
                "training_performed": False, "aidge_validated": False, "hardware_validated": False,
                "license_note": "Photos : licences individuelles COCO conservées, pas de publication automatique. Vérifier les attributions avant redistribution."}
    dump(args.final_dir/"SOURCE.json", metadata)
    dump(camera/"SOURCE.json", {"source_dataset": str(args.final_dir.resolve()), "width": 320, "height": 240, "grayscale": True,
                              "source_manifest_sha256": digest(args.final_dir/"manifest.csv"), "resize": "INTER_AREA + bandes grises, sans déformation", "rows": camera_source_rows})
    with (args.final_dir/"ATTRIBUTIONS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("coco_id", "split", "image", "license_name", "license_url", "source_url", "flickr_url"))
        writer.writeheader()
        for row in provenance:
            writer.writerow({"coco_id": row["image"]["id"], "split": row["split"], "image": row["relative_image"],
                             "license_name": row["license"]["name"], "license_url": row["license"]["url"],
                             "source_url": row["url"], "flickr_url": row["image"].get("flickr_url", "")})
    (args.final_dir/"rapport.html").write_text("<!doctype html><html lang='fr'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>COCO — sélection corps entiers</title><style>body{font:17px/1.5 system-ui;max-width:1250px;margin:30px auto;padding:16px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}img{max-width:100%}h3{font-size:17px}.warning{background:#fff1d6;padding:16px}</style>"
        f"<h1>Premier lot COCO — personnes debout, corps entiers</h1><p>{split_counts['train']} images train, {split_counts['validation']} validation ; {len(positives)} images avec personnes, {len(negatives)} sans personne. Profils et trois-quarts privilégiés, autres vues debout conservées.</p>"
        "<p class='warning'>Sélection de données, pas modèle entraîné ni garantie de 90 %. Corps entiers contrôlés visuellement ; orientation seulement indicative. Les annotations vertes sont les rectangles COCO transformés, pas des prédictions. "
        "Les vraies images d'entraînement n'ont aucun rectangle dessiné. Cliquer sur un aperçu pour voir l'original.</p>"
        "<p>Copies caméra : 320 × 240 gris, sans étirement. Licences et provenance conservées localement. Les deux splits proviennent de train2017 : pas un benchmark COCO officiel. "
        "À compléter par des vidéos de votre caméra ; ces images seules n'enseignent pas les trajectoires entrée/sortie.</p><main>"+"".join(cards)+"</main></html>", encoding="utf-8")
    print(f"Lot final prêt : {args.final_dir}, {dict(Counter(r['view_hint'] for r in rows))}", flush=True)


def write_labels(path, boxes):
    path.write_text("".join("0 "+" ".join(f"{v:.8f}" for v in box)+"\n" for box in boxes), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "fetch", "finalize"))
    parser.add_argument("--annotations-zip", type=Path, default=ROOT/"dataset/coco_person_subset/annotations_trainval2017.zip")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"dataset/coco_profile_candidates_v1")
    parser.add_argument("--final-dir", type=Path)
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--positive-count", type=int, default=200)
    parser.add_argument("--negative-count", type=int, default=50)
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("Nombre de workers positif requis.")
    if any(value < 5 or value % 5 for value in (args.positive_count, args.negative_count)):
        raise SystemExit("Les effectifs doivent être des multiples positifs de cinq (répartition 80/20).")
    {"plan": build_plan, "fetch": fetch_and_preview, "finalize": finalize}[args.stage](args)


if __name__ == "__main__":
    main()
