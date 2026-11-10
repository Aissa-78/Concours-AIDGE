"""Prépare 20 images TRAIN uniquement pour contrôler la capacité d'apprentissage.

10 images locales et 10 publiques, dont deux images vides par source. Ni la
validation ni la porte B ne sont utilisées. Ce test n'évalue pas la généralisation.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from heatmap_v2_data import read_manifest
from person_detector_qqvga import PersonDetectorQQVGA
from person_detection_math import targets_from_boxes, detection_loss


def diverse_pick(candidates, count, rng):
    groups = {}
    for sample in candidates:
        groups.setdefault(sample["row"]["video"], []).append(sample)
    names = list(groups)
    rng.shuffle(names)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < count:
        progressed = False
        for name in names:
            if groups[name] and len(selected) < count:
                selected.append(groups[name].pop())
                progressed = True
        if not progressed:
            raise ValueError("Pas assez d'images admissibles dans le train pour ce protocole.")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-data", type=Path, default=ROOT/"dataset/camera_qqvga_v1")
    parser.add_argument("--run-dir", type=Path, default=ROOT/"runs/person_qqvga_mem20_v1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=6)
    args = parser.parse_args()
    if args.run_dir.exists() or min(args.channels, args.blocks) < 1:
        raise SystemExit("Choisir un nouveau dossier d'essai et une architecture positive.")
    rng = np.random.default_rng(args.seed)
    selected = []
    for source in ("portes", "wepdtof"):
        root = (args.camera_data/source).resolve()
        positive, empty = [], []
        for row in read_manifest(root):
            if row["split"] != "train":
                continue
            boxes = []
            for line in (root/row["label"]).read_text().splitlines():
                fields = line.split()
                if len(fields) != 5 or fields[0] != "0":
                    raise ValueError("Les annotations doivent contenir uniquement la classe personne 0.")
                boxes.append(tuple(map(float, fields[1:])))
            if len(boxes) > 2 or any(bw*160 < 8 or bh*120 < 8 for _, _, bw, bh in boxes):
                continue
            try:
                targets_from_boxes(boxes)
            except ValueError:
                continue
            record = {"source": source, "root": str(root), "row": row, "boxes": boxes,
                      "image_sha256": hashlib.sha256((root/row["image"]).read_bytes()).hexdigest(),
                      "label_sha256": hashlib.sha256((root/row["label"]).read_bytes()).hexdigest()}
            (positive if boxes else empty).append(record)
        selected += diverse_pick(positive, 8, rng)+diverse_pick(empty, 2, rng)
    images, targets = [], []
    for sample in selected:
        image = cv2.imread(str(Path(sample["root"])/sample["row"]["image"]), cv2.IMREAD_UNCHANGED)
        if image is None or image.shape != (120, 160) or image.dtype != np.uint8:
            raise ValueError("Les images doivent déjà être grises 160x120 uint8.")
        images.append(image[None].astype(np.float32)/255.)
        targets.append(targets_from_boxes(sample["boxes"]))
    images = np.stack(images)
    heat, distances, masks = (np.stack([target[i] for target in targets]) for i in range(3))
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    compact = PersonDetectorQQVGA(args.channels, args.blocks).eval()
    dense = PersonDetectorQQVGA(args.channels, args.blocks, training_dense=True).eval()
    dense.copy_from_compact(compact)
    check_indices = np.array([0, 8])  # Une image avec personne, une image vide.
    prediction = dense(torch.from_numpy(images[check_indices]))
    _, gradient, _ = detection_loss(prediction.detach().numpy(), heat[check_indices], distances[check_indices], masks[check_indices])
    prediction.backward(torch.from_numpy(gradient))
    reference = {"indices": check_indices, "logits": prediction.detach().numpy(),
                 **{"grad_"+name.replace(".", "_"): value.grad.numpy() for name, value in dense.named_parameters()}}
    destination = args.run_dir
    destination.mkdir(parents=True)
    np.savez_compressed(destination/"dataset.npz", images=images, heat=heat, distances=distances, masks=masks)
    np.savez_compressed(destination/"gradient_reference.npz", **reference)
    np.savez_compressed(destination/"parameter_masks.npz", **{name.replace(".", "_"): mask.numpy() for name, mask in dense.parameter_masks().items()})
    torch.onnx.export(dense, torch.from_numpy(images[:2]), str(destination/"initial_dense.onnx"),
                      input_names=["image"], output_names=["detections"], opset_version=17, dynamo=False)
    torch.onnx.export(compact, torch.from_numpy(images[:1]), str(destination/"initial_compact.onnx"),
                      input_names=["image"], output_names=["detections"], opset_version=17, dynamo=False)
    info = {"seed": args.seed, "channels": args.channels, "blocks": args.blocks,
            "input_shape": [1, 120, 160], "output_shape": [5, 30, 40],
            "parameters_compact": sum(p.numel() for p in compact.parameters()),
            "parameters_training_graph": sum(p.numel() for p in dense.parameters()),
            "images": len(selected), "presences": sum(len(s["boxes"]) for s in selected),
            "rows": selected, "threshold": .5, "iou_threshold": .5,
            "note": "Mémorisation technique, mêmes images pour apprendre et mesurer ; aucun test indépendant, aucune augmentation."}
    (destination/"dataset.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Préparé : {len(selected)} images train, {info['presences']} rectangles, 4 images vides.")
    print(f"CNN compact : {info['parameters_compact']} paramètres, entrée gris 160x120, sortie 5x30x40.")
    print(f"Graphes et contrôle des gradients : {destination}")


if __name__ == "__main__":
    main()
