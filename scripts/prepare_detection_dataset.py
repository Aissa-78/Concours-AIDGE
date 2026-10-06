"""Extrait des images des vidéos et crée des propositions de rectangles YOLO.

Les étiquettes produites automatiquement ne sont PAS une vérité terrain.
Elles doivent être revues avant de mesurer la qualité d'un modèle entraîné.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
from ultralytics import YOLO


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
FIELDS = [
    "split", "video", "frame", "time_seconds", "image", "label",
    "proposals", "suspicious_overlap", "reviewed",
]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, default=Path("videos_test"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("dataset/detection_proposals")
    )
    parser.add_argument("--model", type=Path, default=Path("yolo26n.pt"))
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--max-per-video", type=int, default=16)
    parser.add_argument("--confidence", type=float, default=0.20)
    parser.add_argument("--image-size", type=int, default=640)
    return parser.parse_args()


def intersection_over_union(first: tuple, second: tuple) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union else 0.0


def draw_preview(frame, boxes, video_name: str, frame_index: int):
    preview = frame.copy()
    for x1, y1, x2, y2, confidence in boxes:
        cv2.rectangle(preview, (int(x1), int(y1)), (int(x2), int(y2)), (0, 190, 255), 2)
        cv2.putText(
            preview, f"{confidence:.2f}", (int(x1), max(20, int(y1) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 190, 255), 2,
        )
    cv2.rectangle(preview, (0, 0), (preview.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        preview, f"{video_name} | frame {frame_index} | YOLO: {len(boxes)}",
        (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
    )
    return preview


def main() -> None:
    args = arguments()
    if args.sample_fps <= 0 or args.max_per_video < 1:
        raise SystemExit("--sample-fps et --max-per-video doivent être positifs.")
    if not 0 < args.confidence < 1:
        raise SystemExit("--confidence doit être entre 0 et 1.")
    if args.output_dir.exists():
        raise SystemExit(
            f"{args.output_dir} existe déjà. Choisissez un autre --output-dir "
            "pour ne pas écraser des corrections humaines."
        )
    videos = sorted(
        path for path in args.video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if len(videos) < 2:
        raise SystemExit("Il faut au moins deux vidéos pour séparer train et validation.")
    if not args.model.is_file():
        raise SystemExit(f"Modèle YOLO absent : {args.model}")

    validation_count = max(1, math.ceil(len(videos) * 0.2))
    validation_videos = set(videos[-validation_count:])
    model = YOLO(str(args.model))
    rows: list[dict[str, str | int | float]] = []
    preview_dir = args.output_dir / "apercus"
    preview_dir.mkdir(parents=True)

    for video in videos:
        split = "validation" if video in validation_videos else "train"
        capture = cv2.VideoCapture(str(video))
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or frame_count <= 0:
            print(f"Ignorée, métadonnées illisibles : {video.name}")
            capture.release()
            continue
        step = max(1, round(fps / args.sample_fps))
        indices = list(range(0, frame_count, step))[: args.max_per_video]
        video_previews = []
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            result = model.predict(
                source=frame, classes=[0], conf=args.confidence,
                imgsz=args.image_size, verbose=False,
            )[0]
            xyxy = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            boxes = [(*map(float, rectangle), float(confidence))
                     for rectangle, confidence in zip(xyxy, confidences)]
            height, width = frame.shape[:2]
            stem = f"{video.stem}_f{frame_index:05d}"
            relative_image = Path(split) / "images" / f"{stem}.jpg"
            relative_label = Path(split) / "labels" / f"{stem}.txt"
            image_path = args.output_dir / relative_image
            label_path = args.output_dir / relative_label
            image_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise RuntimeError(f"Écriture impossible : {image_path}")
            labels = []
            for x1, y1, x2, y2, _ in boxes:
                center_x = (x1 + x2) / (2 * width)
                center_y = (y1 + y2) / (2 * height)
                box_width = (x2 - x1) / width
                box_height = (y2 - y1) / height
                labels.append(
                    f"0 {center_x:.6f} {center_y:.6f} "
                    f"{box_width:.6f} {box_height:.6f}"
                )
            label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
            suspicious = any(
                intersection_over_union(first, second) > 0.30
                for index, first in enumerate(boxes)
                for second in boxes[index + 1:]
            )
            rows.append({
                "split": split, "video": video.name, "frame": frame_index,
                "time_seconds": round(frame_index / fps, 3),
                "image": relative_image.as_posix(),
                "label": relative_label.as_posix(),
                "proposals": len(boxes),
                "suspicious_overlap": "oui" if suspicious else "non",
                "reviewed": "non",
            })
            if len(video_previews) < 6:
                preview = draw_preview(frame, boxes, video.name, frame_index)
                scale = min(320 / width, 240 / height)
                video_previews.append(cv2.resize(
                    preview, (round(width * scale), round(height * scale))
                ))
        capture.release()
        if video_previews:
            cell_width = 340
            cell_height = 260
            import numpy as np
            grid = np.full((2 * cell_height, 3 * cell_width, 3), 235, dtype=np.uint8)
            for index, preview in enumerate(video_previews):
                top = (index // 3) * cell_height
                left = (index % 3) * cell_width
                grid[top:top + preview.shape[0], left:left + preview.shape[1]] = preview
            cv2.imwrite(str(preview_dir / f"{video.stem}.jpg"), grid)
        print(f"{video.name}: {len(indices)} images prévues ({split})", flush=True)

    manifest = args.output_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Images extraites : {len(rows)}")
    print(f"Propositions de rectangles : {sum(int(row['proposals']) for row in rows)}")
    print(f"Cas avec rectangles superposés : {sum(row['suspicious_overlap'] == 'oui' for row in rows)}")
    print(f"Manifest : {manifest}")
    print("ATTENTION : toutes les propositions restent à vérifier, surtout en validation.")


if __name__ == "__main__":
    main()
