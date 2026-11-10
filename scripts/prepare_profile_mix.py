"""Propose des annotations vidéo, puis assemble COCO + vidéos relues.

Les trois vidéos de test restent exclues. Les propositions YOLO ne sont pas
des annotations humaines : le statut de relecture par l'assistant est explicite.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from camera_preprocess import prepare_camera_frame
from person_detection_math import targets_from_boxes
from profile_training_data import WIDTH, HEIGHT, STRIDE


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def video_name(value):
    return str(value).replace("\\", "/").rsplit("/", 1)[-1]


def local_video(value):
    """Retrouve une vidéo partagée même si la provenance vient d'un autre OS."""
    path=Path(value)
    if path.is_file(): return path
    candidate=ROOT/"videotest2/Echantillon"/video_name(value)
    if candidate.is_file(): return candidate
    raise FileNotFoundError(f"Vidéo absente ; copier le dossier privé videotest2 : {video_name(value)}")


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_labels(path, boxes):
    path.write_text("".join("0 "+" ".join(f"{v:.8f}" for v in box)+"\n" for box in boxes), encoding="utf-8")


def candidates(args):
    if args.candidates.exists():
        raise SystemExit("Dossier candidat existant : choisir un nouveau nom.")
    diagnostic = json.loads(args.diagnostic.read_text())
    split_config = json.loads(args.split.read_text())
    assignments = {name: split for split in ("train", "validation", "test") for name in split_config[split]}
    if len(assignments) != sum(len(split_config[s]) for s in ("train", "validation", "test")):
        raise ValueError("Une prise apparaît dans plusieurs lots.")
    if set(assignments) != {video_name(c["source"]) for c in diagnostic["clips"]}:
        raise ValueError("La répartition ne couvre pas exactement toutes les vidéos.")
    by_hash, videos, rows = {}, [], []
    args.candidates.mkdir(parents=True)
    (args.candidates/"images").mkdir()
    (args.candidates/"labels").mkdir()
    for clip in diagnostic["clips"]:
        path = local_video(clip["source"])
        split = assignments[path.name]
        if digest(path) != clip["sha256"]:
            raise ValueError(f"Vidéo modifiée : {path}")
        if clip["sha256"] in by_hash:
            if by_hash[clip["sha256"]] != split:
                raise ValueError("Un doublon traverse les lots.")
            videos.append({"video": str(path), "sha256": clip["sha256"], "split": split, "duplicate_skipped": True})
            continue
        by_hash[clip["sha256"]] = split
        videos.append({"video": str(path), "sha256": clip["sha256"], "split": split, "duplicate_skipped": False})
        if split == "test":
            continue  # Ni extraction ni pseudo-étiquetage du lot de test.
        capture = cv2.VideoCapture(str(path))
        for prediction in clip["predictions"][::5]:  # 10 -> environ 2 images/s.
            capture.set(cv2.CAP_PROP_POS_FRAMES, prediction["frame"])
            ok, frame = capture.read()
            if not ok:
                raise ValueError("Image source indécodable.")
            gray, geometry = prepare_camera_frame(frame, WIDTH, HEIGHT)
            h, w = frame.shape[:2]
            boxes = []
            for det in prediction["yolo_original"]:
                x1, y1, x2, y2 = det["box"]
                boxes.append(geometry.transform_box(((x1+x2)/(2*w), (y1+y2)/(2*h), (x2-x1)/w, (y2-y1)/h)))
            review_id = f"{len(rows)+1:03d}"
            image_path, label_path = f"images/{review_id}.png", f"labels/{review_id}.txt"
            cv2.imwrite(str(args.candidates/image_path), gray)
            write_labels(args.candidates/label_path, boxes)
            try:
                targets_from_boxes(boxes, WIDTH, HEIGHT, STRIDE)
                geometry_error = None
            except ValueError as error:
                geometry_error = str(error)
            rows.append({"review_id": review_id, "split": split, "video": path.name,
                         "video_sha256": clip["sha256"], "frame": prediction["frame"], "time_seconds": prediction["seconds"],
                         "image": image_path, "label": label_path, "boxes": boxes, "geometry_error": geometry_error,
                         "image_sha256": digest(args.candidates/image_path), "label_sha256": digest(args.candidates/label_path)})
        capture.release()
    dump(args.candidates/"SOURCE.json", {"rows": rows, "videos": videos, "diagnostic_sha256": digest(args.diagnostic),
         "split_config_sha256": digest(args.split), "teacher_threshold": diagnostic["threshold"],
         "note": "Propositions de YOLO local, pas une vérité terrain humaine."})
    sheets = args.candidates/"planches"
    sheets.mkdir()
    for offset in range(0, len(rows), 20):
        canvas = np.full((4*220, 5*330, 3), 245, np.uint8)
        for index, row in enumerate(rows[offset:offset+20]):
            frame = cv2.imread(str(args.candidates/row["image"]))
            for j, (cx, cy, bw, bh) in enumerate(row["boxes"]):
                x1,y1,x2,y2=round((cx-bw/2)*WIDTH),round((cy-bh/2)*HEIGHT),round((cx+bw/2)*WIDTH),round((cy+bh/2)*HEIGHT)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(30,210,30),1)
                cv2.putText(frame,str(j),(x1,max(12,y1)),cv2.FONT_HERSHEY_SIMPLEX,.4,(30,210,30),1)
            resized=cv2.resize(frame,(320,185))
            top,left=(index//5)*220+30,(index%5)*330+5
            canvas[top:top+185,left:left+320]=resized
            label=f"{row['review_id']} {row['split'][:3]} {row['time_seconds']:.1f}s {len(row['boxes'])}p"
            cv2.putText(canvas,label,(left,top-8),cv2.FONT_HERSHEY_SIMPLEX,.49,(20,20,20),1)
        cv2.imwrite(str(sheets/f"{offset//20+1:02d}.jpg"),canvas)
    print(f"{len(rows)} images candidates. Répartition : {dict(Counter(r['split'] for r in rows))}")
    print(f"Géométrie incompatible : {[r['review_id'] for r in rows if r['geometry_error']]}")


def assemble(args):
    if args.output_dir.exists() or not args.decisions:
        raise SystemExit("Nouveau dossier et --decisions requis.")
    data=json.loads((args.candidates/"SOURCE.json").read_text())
    decisions=json.loads(args.decisions.read_text())
    kept,rejected=set(decisions["accepted_review_ids"]),set(decisions["rejected_review_ids"])
    if kept & rejected or kept | rejected != {r["review_id"] for r in data["rows"]}:
        raise ValueError("Chaque image doit avoir une décision unique.")
    args.output_dir.mkdir(parents=True)
    counts={}
    for source, source_root in (("coco",args.coco),("portes",args.candidates)):
        destination=args.output_dir/source
        destination.mkdir()
        rows=[]
        if source=="coco":
            originals=list(csv.DictReader((source_root/"manifest.csv").open()))
        else:
            originals=[{**r,"reviewed":"annotation_assistant"} for r in data["rows"] if r["review_id"] in kept]
        for original in originals:
            if original["split"] not in ("train","validation"):
                raise ValueError("Le test ne doit pas être intégré.")
            image, label=source_root/original["image"],source_root/original["label"]
            if source=="portes" and (digest(image)!=original["image_sha256"] or digest(label)!=original["label_sha256"]):
                raise ValueError("Candidat modifié après la revue.")
            boxes=[tuple(map(float,line.split()[1:])) for line in label.read_text().splitlines()]
            targets_from_boxes(boxes,WIDTH,HEIGHT,STRIDE)
            # Le miroir est testé maintenant pour ne pas découvrir une cible invalide pendant l'apprentissage.
            targets_from_boxes([(1-cx,cy,bw,bh) for cx,cy,bw,bh in boxes],WIDTH,HEIGHT,STRIDE)
            frame=cv2.imread(str(image),cv2.IMREAD_UNCHANGED)
            if frame is None or frame.shape!=(HEIGHT,WIDTH):
                raise ValueError("Image caméra grise attendue.")
            rel_image=Path(original["split"])/"images"/image.name
            rel_label=Path(original["split"])/"labels"/label.name
            for rel in (rel_image,rel_label): (destination/rel.parent).mkdir(parents=True,exist_ok=True)
            shutil.copy2(image,destination/rel_image)
            shutil.copy2(label,destination/rel_label)
            rows.append({"split":original["split"],"video":original["video"],"image":rel_image.as_posix(),
                         "label":rel_label.as_posix(),"reviewed":original["reviewed"],
                         "group_id":original.get("video_sha256",original["video"])})
        with (destination/"manifest.csv").open("w",newline="",encoding="utf-8") as target:
            writer=csv.DictWriter(target,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        counts[source]=dict(Counter(r["split"] for r in rows))
        dump(destination/"SOURCE.json",{"source_root":str(source_root.resolve()),"rows":originals,
                                       "review":decisions if source=="portes" else "Annotations COCO officielles"})
    dump(args.output_dir/"SOURCE.json",{"width":WIDTH,"height":HEIGHT,"stride":STRIDE,"datasets":counts,
        "videos":data["videos"],"test_used_for_training":False,"human_reviewed_local_labels":False,
        "validation_warning":"Validation locale proposée par YOLO et relue par l'assistant, pas une vérité terrain humaine indépendante. Même lieu dans toutes les prises.",
        "coco_source_root":str(args.coco.resolve()),"video_candidates_root":str(args.candidates.resolve())})
    print("Dataset mixte prêt :",counts)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("candidates","assemble"))
    parser.add_argument("--diagnostic",type=Path,default=ROOT/"runs/videotest2_diagnostic_v2/results.json")
    parser.add_argument("--split",type=Path,default=ROOT/"configs/profile_split_v1.json")
    parser.add_argument("--candidates",type=Path,default=ROOT/"dataset/profile_video_candidates_v1")
    parser.add_argument("--coco",type=Path,default=ROOT/"dataset/coco_profile_fullbody_v1/camera_320")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"dataset/profile_mix_v1")
    parser.add_argument("--decisions",type=Path)
    args=parser.parse_args()
    {"candidates":candidates,"assemble":assemble}[args.stage](args)


if __name__=="__main__": main()
