"""Compare des détecteurs sur des vidéos locales, sans entraînement ni comptage.

Notre candidat Aidge reste en 160x120 gris. YOLO sert de référence logicielle
en couleur originale et en 320x240 gris. Les pipelines diffèrent : ce test ne
permet pas d'isoler l'effet du modèle ou de la résolution à lui seul.
"""

import argparse
from collections import Counter
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
from imageio_ffmpeg import write_frames
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from camera_preprocess import prepare_camera_frame
from person_detection_math import decode_boxes


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, values):
    path.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")


def sampled_frames(path, every):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Vidéo illisible : {path}")
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % every == 0:
                yield index, frame
            index += 1
    finally:
        capture.release()


def yolo_predictions(result):
    return [{"score": float(confidence), "box": list(map(float, box))}
            for confidence, box in zip(result.boxes.conf.cpu().tolist(), result.boxes.xyxy.cpu().tolist())]


def panel(frame, predictions, label, subtitle):
    """Annotation sur une copie : pas de dessin dans les pixels du détecteur."""
    canvas = np.full((300, 400, 3), 24, np.uint8)
    image = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()
    h, w = image.shape[:2]
    factor = min(400/w, 240/h)
    rw, rh = round(w*factor), round(h*factor)
    left, top = (400-rw)//2, 60+(240-rh)//2
    canvas[top:top+rh, left:left+rw] = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_NEAREST)
    for prediction in predictions:
        x1, y1, x2, y2 = prediction["box"]
        first = (round(left+x1*factor), round(top+y1*factor))
        second = (round(left+x2*factor), round(top+y2*factor))
        cv2.rectangle(canvas, first, second, (40, 220, 40), 2)
        cv2.putText(canvas, f"{prediction['score']:.2f}", (first[0], max(72, first[1]-3)), cv2.FONT_HERSHEY_SIMPLEX, .45, (40, 220, 40), 1)
    cv2.putText(canvas, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (245, 245, 245), 1)
    cv2.putText(canvas, f"{subtitle} | {len(predictions)} rectangle(s)", (8, 44), cv2.FONT_HERSHEY_SIMPLEX, .41, (225, 225, 225), 1)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, default=ROOT/"videotest2/Echantillon")
    parser.add_argument("--run-dir", type=Path, default=ROOT/"runs/person_qqvga_hard_context_full_v3")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--yolo-model", type=Path, default=ROOT/"yolo26n.pt")
    parser.add_argument("--aidge-python", type=Path, default=ROOT/".venv-aidge-colibry"/("Scripts/python.exe" if os.name == "nt" else "bin/python"))
    parser.add_argument("--sample-fps", type=float, default=10.)
    parser.add_argument("--threshold", type=float, default=.5)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("Choisir un dossier de sortie nouveau pour ne rien écraser.")
    if not 0 < args.threshold < 1 or not 0 < args.sample_fps <= 60:
        raise SystemExit("Seuil entre 0 et 1 et cadence entre 0 et 60 requis.")
    model_path = args.run_dir/"best.onnx"
    for path in (model_path, args.yolo_model, args.aidge_python):
        if not path.is_file():
            raise FileNotFoundError(path)
    paths = sorted(p for p in args.video_dir.rglob("*") if p.is_file() and p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv"))
    if not paths:
        raise SystemExit("Aucune vidéo dans le dossier indiqué.")
    args.output_dir.mkdir(parents=True)
    (args.output_dir/"videos").mkdir()
    (args.output_dir/"apercus").mkdir()
    clips, arrays = [], []
    for clip_index, path in enumerate(paths, 1):
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"Vidéo illisible : {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        capture.release()
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"Cadence invalide : {path}")
        every = max(1, round(fps/args.sample_fps))
        indices, offset = [], len(arrays)
        for frame_index, frame in sampled_frames(path, every):
            gray, _ = prepare_camera_frame(frame, 160, 120)
            arrays.append(gray[None].astype(np.float32)/255)
            indices.append(frame_index)
        if not indices:
            raise ValueError(f"Vidéo sans image décodable : {path}")
        clips.append({"id": f"{clip_index:02d}", "source": str(path.resolve()), "sha256": sha256(path),
                      "source_fps": fps, "every": every, "sample_fps": fps/every,
                      "frame_indices": indices, "offset": offset, "samples": len(indices)})
        print(f"Préparation {clip_index}/{len(paths)} : {path.name}, {len(indices)} images", flush=True)
    inputs = np.stack(arrays)
    np.savez_compressed(args.output_dir/"inputs_160_gray.npz", images=inputs)
    environment = os.environ.copy()
    environment.update(MPLBACKEND="Agg", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    # Ne pas résoudre le lien symbolique de Python : cela ferait perdre le venv.
    subprocess.run([str(args.aidge_python.absolute()), str(ROOT/"scripts/infer_person_qqvga_aidge.py"),
                    "--model", str(model_path.resolve()), "--inputs", str((args.output_dir/"inputs_160_gray.npz").resolve()),
                    "--output", str((args.output_dir/"aidge_logits.npy").resolve()), "--batch-size", "8"],
                   cwd=ROOT, env=environment, check=True)
    logits = np.load(args.output_dir/"aidge_logits.npy")
    if logits.shape != (len(inputs), 5, 30, 40) or not np.isfinite(logits).all():
        raise ValueError("Sorties Aidge incorrectes.")
    from ultralytics import YOLO
    import torch
    torch.set_num_threads(4)
    yolo = YOLO(str(args.yolo_model.resolve()))
    cards, details = [], []
    for clip in clips:
        path = Path(clip["source"])
        output_name = clip["id"]+"_comparaison.mp4"
        writer = write_frames(str(args.output_dir/"videos"/output_name), (1200, 324), fps=clip["sample_fps"],
                              codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p", macro_block_size=2,
                              output_params=["-crf", "20", "-movflags", "+faststart"])
        writer.send(None)
        records = []
        try:
            for i, (frame_index, frame) in enumerate(sampled_frames(path, clip["every"])):
                if i >= clip["samples"] or frame_index != clip["frame_indices"][i]:
                    raise ValueError("Décodage source non reproductible.")
                qvga, _ = prepare_camera_frame(frame, 320, 240)
                gray3 = cv2.cvtColor(qvga, cv2.COLOR_GRAY2BGR)
                reference_color = yolo_predictions(yolo.predict(frame, classes=[0], conf=args.threshold, iou=.5,
                    imgsz=640, rect=False, device="cpu", verbose=False)[0])
                reference_gray = yolo_predictions(yolo.predict(gray3, classes=[0], conf=args.threshold, iou=.5,
                    imgsz=640, rect=False, device="cpu", verbose=False)[0])
                own = [{"score": score, "box": box.tolist()} for score, box in
                       decode_boxes(logits[clip["offset"]+i], args.threshold, local_maximum=True)]
                gray = np.round(inputs[clip["offset"]+i, 0]*255).astype(np.uint8)
                compare = np.concatenate([panel(frame, reference_color, "YOLO reference - original couleur", "Analyse interne 640"),
                    panel(qvga, reference_gray, "YOLO reference - 320x240 gris", "Analyse interne 640"),
                    panel(gray, own, "Notre modele - Aidge CPU", "Entree 160x120 gris")], axis=1)
                banner = np.full((24, 1200, 3), 24, np.uint8)
                cv2.putText(banner, f"Video {clip['id']} | t={frame_index/clip['source_fps']:.2f}s | Detection seulement - ni suivi ni comptage", (8, 17), cv2.FONT_HERSHEY_SIMPLEX, .45, (240, 240, 240), 1)
                compare = np.concatenate([compare, banner], axis=0)
                writer.send(cv2.cvtColor(compare, cv2.COLOR_BGR2RGB).tobytes())
                if i == clip["samples"]//2:
                    cv2.imwrite(str(args.output_dir/"apercus"/(clip["id"]+".jpg")), compare)
                records.append({"frame": frame_index, "seconds": frame_index/clip["source_fps"],
                                "yolo_original": reference_color, "yolo_qvga_gray": reference_gray, "aidge_qqvga_gray": own})
            if len(records) != clip["samples"] or sha256(path) != clip["sha256"]:
                raise ValueError("Source modifiée ou décodage incomplet.")
        finally:
            writer.close()
        histograms = {key: dict(Counter(len(record[key]) for record in records))
                      for key in ("yolo_original", "yolo_qvga_gray", "aidge_qqvga_gray")}
        details.append({**clip, "output_video": "videos/"+output_name, "predicted_box_count_histograms": histograms, "predictions": records})
        save_json(args.output_dir/"progress.json", {"completed": len(details), "total": len(clips), "clips": details})
        cards.append(f"<article><h2>{clip['id']} — {html.escape(path.name)}</h2>"
            f"<video controls preload='none' poster='apercus/{clip['id']}.jpg' src='videos/{output_name}'></video>"
            f"<p><a href='videos/{output_name}'>Ouvrir / télécharger la vidéo annotée</a> — {clip['samples']} images analysées, environ {clip['sample_fps']:.1f} images/s.</p></article>")
        print(f"Test {len(details)}/{len(clips)} terminé : {path.name}", flush=True)
    metadata = {"video_dir": str(args.video_dir.resolve()), "aidge_model": str(model_path.resolve()),
                "aidge_model_sha256": sha256(model_path), "yolo_model_sha256": sha256(args.yolo_model),
                "threshold": args.threshold, "nms_iou": .5, "yolo_internal_size": 640,
                "aidge_input": [1, 120, 160], "aidge_local_maximum": True,
                "coco_selection_used_for_new_training": False, "training_performed": False,
                "annotation_available": False, "accuracy_measured": False, "tracking_tested": False,
                "counting_tested": False, "hardware_tested": False, "clips": details}
    save_json(args.output_dir/"results.json", metadata)
    (args.output_dir/"rapport.html").write_text("<!doctype html><html lang='fr'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>FlowSense — nouvelles vidéos</title><style>body{font:17px/1.5 system-ui;max-width:1250px;margin:30px auto;padding:16px}video{width:100%;background:#111}article{margin:35px 0;border-top:1px solid #ccc}h2{font-size:20px}.warning{background:#fff1d6;padding:16px}</style>"
        f"<h1>Test des {len(clips)} nouvelles vidéos</h1><p>Aucun entraînement. Originaux conservés, modèles inchangés.</p>"
        "<p>De gauche à droite : YOLO sur la couleur originale ; YOLO sur gris 320 × 240 ; notre candidat Aidge CPU sur gris 160 × 120. Les rectangles sont des prédictions.</p>"
        f"<p>Seuil {args.threshold}, NMS 0,5. YOLO analyse en interne à 640 ; Aidge reçoit 160 × 120. Les pipelines diffèrent : les écarts ne mesurent pas uniquement la qualité du réseau.</p>"
        "<p class='warning'>Le nouveau lot COCO n'a pas encore servi à entraîner le modèle. Sans annotations humaines, aucun pourcentage de précision, rappel ou F1 n'est mesuré. "
        "Le nombre de rectangles à un instant n'est pas le nombre de personnes distinctes ni de passages. Ce test ne fait ni suivi ni comptage entrée/sortie. Aucun résultat ne valide la carte ou les 5 mW.</p>"
        "<p>Analyse à environ 10 images/s avec lecture à la cadence correspondante, sans accélérer volontairement les mouvements. Vidéos de résultat sans piste audio.</p>"
        +"".join(cards)+"</html>", encoding="utf-8")
    print(f"Rapport prêt : {args.output_dir/'rapport.html'}", flush=True)


if __name__ == "__main__":
    main()
