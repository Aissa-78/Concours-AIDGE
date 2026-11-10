"""Audit et préparation V2 : portes + images publiques COCO ou WEPDTOF.

PyTorch initialise uniquement le graphe ONNX et fournit une référence de
gradients pour vérifier Aidge. Aucun entraînement PyTorch n'est effectué.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from heatmap_v2 import HeatmapV2
from heatmap_v2_data import read_manifest, read_sample, target_heatmap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-data", type=Path, default=ROOT / "dataset/detection_proposals")
    parser.add_argument("--coco-data", "--pretrain-data", dest="coco_data", type=Path,
                        default=ROOT / "dataset/coco_person_subset", help="Dataset public annoté pour le préentraînement.")
    parser.add_argument("--pretrain-name", choices=("coco", "wepdtof"), default="coco")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/heatmap_v2")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    output = args.output_dir
    if any((output / name).exists() for name in ("dataset.npz", "initial.onnx", "training_history.csv")):
        raise SystemExit("Ce dossier contient déjà un essai. Choisir un nouveau --output-dir pour le conserver.")
    output.mkdir(parents=True, exist_ok=True)
    arrays, info = {}, {"seed": args.seed, "pretrain_name": args.pretrain_name, "input_size": 128, "content_size": 80,
                        "grid_size": 20, "target": "centre du rectangle visible", "sources": {}}
    for source_name, root in (("local", args.local_data), ("coco", args.coco_data)):
        rows = read_manifest(root)
        train_videos = {r["video"] for r in rows if r["split"] == "train"}
        val_videos = {r["video"] for r in rows if r["split"] == "validation"}
        if train_videos & val_videos:
            raise ValueError(f"Vidéo/identifiant commun entre train et validation : {source_name}")
        info["sources"][source_name] = {"name": "portes" if source_name == "local" else args.pretrain_name,
                                       "root": str(root.resolve()), "splits": {}}
        hashes = {}
        for split in ("train", "validation"):
            selected = [r for r in rows if r["split"] == split]
            samples = [read_sample(root, row) for row in selected]
            if not samples:
                raise ValueError(f"Split vide : {source_name}/{split}")
            prefix = f"{source_name}_{split}"
            arrays[prefix + "_images"] = np.stack([sample[0] for sample in samples])
            arrays[prefix + "_targets"] = np.stack([target_heatmap(sample[1]) for sample in samples])
            hashes[split] = {hashlib.sha256(image.tobytes()).hexdigest()
                             for image in arrays[prefix + "_images"]}
            info["sources"][source_name]["splits"][split] = {
                "images": len(selected), "empty": sum(not sample[1] for sample in samples),
                "presences": sum(len(sample[1]) for sample in samples),
                "rows": selected, "centers": [sample[1] for sample in samples],
            }
        if hashes["train"] & hashes["validation"]:
            raise ValueError(f"Image préparée identique dans les deux splits : {source_name}")
    np.savez_compressed(output / "dataset.npz", **arrays)
    calibration_indices = np.linspace(0, len(arrays["local_train_images"]) - 1, 32, dtype=int)
    np.savez_compressed(output / "calibration.npz", images=arrays["local_train_images"][calibration_indices])
    (output / "dataset.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    model = HeatmapV2().eval()
    example = torch.from_numpy(arrays["local_train_images"][:2])
    target = torch.from_numpy(arrays["local_train_targets"][:2])
    prediction = model(example)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        prediction, target, pos_weight=torch.tensor([10.0])
    )
    loss.backward()
    reference = {"logits": prediction.detach().numpy()}
    reference.update({"grad_" + name.replace(".", "_"): param.grad.numpy()
                      for name, param in model.named_parameters()})
    np.savez_compressed(output / "gradient_reference.npz", **reference)
    model.zero_grad()
    torch.onnx.export(model, example, str(output / "initial.onnx"),
                      input_names=["image"], output_names=["heatmap_logits"],
                      opset_version=17, dynamo=False)
    print("Audit : annotations lisibles, coordonnées valides, splits séparés.")
    print("Les annotations d'origine n'ont pas été modifiées.")
    for source_name, source in info["sources"].items():
        for split, description in source["splits"].items():
            print(f"{source_name}/{split} : {description['images']} images, {description['presences']} présences")
    print(f"Architecture initialisée : {sum(p.numel() for p in model.parameters())} paramètres.")
    print(f"Dossier prêt pour l'apprentissage Aidge : {output}")


if __name__ == "__main__":
    main()
