"""Prépare les 579 images train + 98 validation pour le détecteur Aidge QQVGA.

Aucun téléchargement, aucune porte B, aucune annotation écrasée. Le départ est
aléatoire : les poids du test sur 20 images ne servent pas d'initialisation.
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
from person_detector_qqvga import make_person_detector
from person_detection_math import targets_from_boxes, detection_loss
from person_training_data import assert_disjoint
from prepare_person_memorization import diverse_pick


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-data", type=Path, default=ROOT/"dataset/camera_qqvga_v1")
    parser.add_argument("--run-dir", type=Path, default=ROOT/"runs/person_qqvga_full_v1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--residual", action="store_true", help="Tester des connexions directes entre blocs.")
    parser.add_argument("--train-limit", type=int, help="Lot diagnostic pair, équilibré entre les deux sources ; validation inchangée.")
    parser.add_argument("--classification", choices=("focal", "balanced_bce", "hard_bce"), default="focal")
    parser.add_argument("--local-maximum", action="store_true", help="Décoder seulement les pics 3x3 des scores.")
    parser.add_argument("--architecture", choices=("separable", "regular", "regular_context"), default="separable")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if run.exists():
        raise SystemExit("Choisir un nouveau dossier d'essai : aucun résultat existant n'est écrasé.")
    if min(args.channels, args.blocks) < 1 or (args.train_limit is not None and (args.train_limit < 16 or args.train_limit % 2)):
        raise SystemExit("Architecture positive et limite train paire >= 16 requises.")
    records, arrays, all_targets, source_manifests = [], [], [], {}
    for source in ("portes", "wepdtof"):
        root = (args.camera_data/source).resolve()
        source_manifests[str(root/"manifest.csv")] = sha256(root/"manifest.csv")
        for row in read_manifest(root):
            boxes = []
            for line in (root/row["label"]).read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5 or parts[0] != "0":
                    raise ValueError(f"Annotation invalide : {row['label']}")
                boxes.append(tuple(map(float, parts[1:])))
            # Échec explicite si une annotation est incompatible ; jamais ignorée.
            targets = targets_from_boxes(boxes)
            image = cv2.imread(str(root/row["image"]), cv2.IMREAD_UNCHANGED)
            if image is None or image.dtype != np.uint8 or image.shape != (120, 160):
                raise ValueError(f"Image grise 160x120 uint8 requise : {row['image']}")
            records.append({"source": source, "root": str(root), "row": row, "boxes": boxes,
                            "image_sha256": sha256(root/row["image"]),
                            "label_sha256": sha256(root/row["label"])})
            arrays.append(image[None].astype(np.float32)/255.)
            all_targets.append(targets)
    assert_disjoint(records)
    if args.train_limit:
        rng = np.random.default_rng(args.seed)
        keep = {i for i, r in enumerate(records) if r["row"]["split"] == "validation"}
        per_source = args.train_limit//2
        negative_count = max(1, per_source//8)
        for source in ("portes", "wepdtof"):
            candidates = [{**r, "index": i} for i, r in enumerate(records)
                          if r["row"]["split"] == "train" and r["source"] == source]
            for empty, count in ((False, per_source-negative_count), (True, negative_count)):
                picked = diverse_pick([r for r in candidates if (len(r["boxes"]) == 0) == empty], count, rng)
                keep.update(r["index"] for r in picked)
        indices = sorted(keep)
        records, arrays, all_targets = ([values[i] for i in indices] for values in (records, arrays, all_targets))
    run.mkdir(parents=True)
    info = {"seed": args.seed, "channels": args.channels, "blocks": args.blocks,
            "residual": args.residual, "train_limit": args.train_limit, "rows": records,
            "classification": args.classification, "local_maximum": args.local_maximum,
            "architecture": args.architecture,
            "manifests": source_manifests, "input_shape": [1, 120, 160],
            "output_shape": [5, 30, 40], "initialization": "random, no pretrained weights",
            "threshold": .5, "iou_threshold": .5,
            "note": "Validation hors apprentissage, mais utilisée pour choisir le checkpoint ; pas un test final aveugle."}
    for split in ("train", "validation"):
        indices = [i for i, record in enumerate(records) if record["row"]["split"] == split]
        selected = [records[i] for i in indices]
        images = np.stack([arrays[i] for i in indices])
        heat, distances, masks = (np.stack([all_targets[i][j] for i in indices]) for j in range(3))
        np.savez_compressed(run/f"{split}.npz", images=images, heat=heat, distances=distances, masks=masks)
        (run/f"{split}.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
        info[split] = {source: {"images": sum(r["source"] == source for r in selected),
                                "presences": sum(len(r["boxes"]) for r in selected if r["source"] == source)}
                       for source in ("portes", "wepdtof")}
    # PyTorch fournit seulement la définition ONNX et le contrôle des gradients.
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    compact = make_person_detector(args.channels, args.blocks, residual=args.residual, architecture=args.architecture).eval()
    dense = make_person_detector(args.channels, args.blocks, training_dense=True, residual=args.residual, architecture=args.architecture).eval()
    dense.copy_from_compact(compact)
    train_records = json.loads((run/"train.json").read_text())
    check_indices = np.array([next(i for i, row in enumerate(train_records) if row["boxes"]),
                              next(i for i, row in enumerate(train_records) if not row["boxes"])])
    with np.load(run/"train.npz") as data:
        images = data["images"][check_indices]
        prediction = dense(torch.from_numpy(images))
        _, gradient, _ = detection_loss(prediction.detach().numpy(), data["heat"][check_indices],
                                       data["distances"][check_indices], data["masks"][check_indices], classification=args.classification)
    prediction.backward(torch.from_numpy(gradient))
    np.savez_compressed(run/"gradient_reference.npz", indices=check_indices, logits=prediction.detach().numpy(),
                        **{"grad_"+name.replace(".", "_"): p.grad.numpy() for name, p in dense.named_parameters()})
    np.savez_compressed(run/"parameter_masks.npz",
                        **{name.replace(".", "_"): mask.numpy() for name, mask in dense.parameter_masks().items()})
    torch.onnx.export(dense, torch.from_numpy(images), str(run/"initial_dense.onnx"),
                      input_names=["image"], output_names=["detections"], opset_version=17, dynamo=False)
    torch.onnx.export(compact, torch.from_numpy(images[:1]), str(run/"initial_compact.onnx"),
                      input_names=["image"], output_names=["detections"], opset_version=17, dynamo=False)
    info["parameters_compact"] = sum(p.numel() for p in compact.parameters())
    (run/"dataset.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: info[key] for key in ("train", "validation", "parameters_compact")}, indent=2))
    print(f"Préparation terminée : {run}. Porte B non utilisée.")


if __name__ == "__main__":
    main()
