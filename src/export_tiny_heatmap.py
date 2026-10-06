"""Exporte le prototype PyTorch vers ONNX avec un exemple de référence.

Commande : python src/export_tiny_heatmap.py
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from tiny_heatmap import TinyHeatmap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Poids entraînés à exporter.")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("runs/tiny_heatmap"),
        help="Dossier des fichiers ONNX et de référence.",
    )
    args = parser.parse_args()
    torch.manual_seed(7)
    model = TinyHeatmap().eval()
    if args.checkpoint:
        if not args.checkpoint.is_file():
            raise SystemExit(f"Checkpoint introuvable : {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
    image = torch.rand(1, 3, 96, 96)

    with torch.no_grad():
        expected = model(image).numpy()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "input.npy", image.numpy())
    np.save(output_dir / "expected_pytorch.npy", expected)

    onnx_path = output_dir / "tiny_heatmap.onnx"
    torch.onnx.export(
        model,
        image,
        str(onnx_path),
        input_names=["image"],
        output_names=["heatmap_logits"],
        opset_version=17,
        dynamo=False,
    )

    print(f"Modèle ONNX : {onnx_path}")
    print(f"Paramètres : {sum(p.numel() for p in model.parameters())}")
    print(f"Entrée : {tuple(image.shape)} ; sortie : {tuple(expected.shape)}")
    if args.checkpoint:
        print(f"Poids chargés : {args.checkpoint}")
        print("L'export ONNX ne mesure pas la qualité de détection : utilisez evaluate_tiny_heatmap.py.")
    else:
        print("Prototype NON entraîné : ces scores ne détectent pas encore les personnes.")


if __name__ == "__main__":
    main()
