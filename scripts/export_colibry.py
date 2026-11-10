"""Exporte un CNN ReLU vers un bitstream Colibry, sans carte connectée.

À lancer avec .venv-aidge-colibry, après export_tiny_heatmap.py et
prepare_colibry_calibration.py. Cet export ne valide ni la précision réelle,
ni le fonctionnement du firmware/capteur sur la carte.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

import aidge_core
import aidge_backend_cpu  # noqa: F401 : enregistre le backend CPU
import aidge_onnx
import aidge_quantization


def load_float_graph(onnx_path: Path, expand_padding: bool = True):
    """Charge le même graphe pour l'évaluation et pour l'export."""
    graph = aidge_onnx.load_onnx(str(onnx_path))
    aidge_core.remove_flatten(graph)
    aidge_core.fuse_batchnorm(graph)
    if expand_padding:
        aidge_core.expand_metaops(graph)
    aidge_core.matmul_to_fc(graph)
    aidge_core.remove_flatten(graph)
    graph.set_datatype(aidge_core.dtype.float32)
    graph.set_backend("cpu")
    return graph


def build_quantized_graph(onnx_path: Path, images: np.ndarray):
    """PTQ réelle : activations/poids INT8, biais et accumulation INT32.

    Les adaptateurs d'entrée et de sortie restent flottants. Le quantificateur
    d'entrée transforme les images RGB [0,1] ; ne pas multiplier par 127 pour
    exécuter ce graphe CPU. La sortie déquantifiée reste un score brut (logit).
    """
    if images.ndim != 4 or tuple(images.shape[1:]) not in ((3, 96, 96), (1, 128, 128), (1, 120, 160)) or len(images) < 1:
        raise ValueError("Calibration attendue : (N,3,96,96), (N,1,128,128) ou (N,1,120,160), N >= 1.")
    if images.dtype != np.float32 or not np.isfinite(images).all():
        raise ValueError("La calibration doit être en float32, avec des valeurs finies.")
    if images.min() < 0 or images.max() > 1:
        raise ValueError("Les images de calibration doivent être dans [0, 1].")
    # Garder le padding dans la convolution QQVGA : InputRescaling attend un
    # premier opérateur affine, pas un Pad isolé dépourvu d'accumulationprecision.
    graph = load_float_graph(onnx_path, expand_padding=tuple(images.shape[1:]) != (1, 120, 160))
    calibration_tensors = []
    for image in images:
        tensor = aidge_core.Tensor(image[None])
        tensor.to_backend("cpu")
        tensor.set_data_format(aidge_core.dformat.nchw)
        calibration_tensors.append(tensor)

    aidge_quantization.auto_assign_node_precision(
        graphview=graph,
        act_precision=aidge_core.dtype.int8,
        weight_precision=aidge_core.dtype.int8,
        bias_precision=aidge_core.dtype.int32,
    )
    print(f"Calibration INT8 avec {len(images)} images réelles...", flush=True)
    aidge_quantization.quantize_network(
        network=graph,
        rescale_inputs=True,
        calibration_set=calibration_tensors,
        clipping_mode=aidge_quantization.Clipping.MAX,
        equalize=True,
        channel_wise=False,
        no_quant=False,
        optimize_signs=False,
        single_shift=False,
        fold_graph=False,
        fake_quantize=False,
        dequantize=True,
        verbose=False,
    )
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/tiny_heatmap_relu"))
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--model", type=Path, help="ONNX explicite, par exemple runs/heatmap_v2/best.onnx.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/colibry_relu"))
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    calibration_path = (args.calibration or run_dir / "calibration.npz").resolve()
    onnx_path = args.model.resolve() if args.model else run_dir / "tiny_heatmap.onnx"
    if not onnx_path.is_file() or not calibration_path.is_file():
        raise SystemExit("ONNX ou calibration absents. Voir les commandes dans le README.")
    with np.load(calibration_path) as data:
        images = data["images"].astype(np.float32)
    graph = build_quantized_graph(onnx_path, images)

    if tuple(images.shape[1:]) == (1, 120, 160):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
        from aidge_export_compat import enable_padded_meta_lookup
        enable_padded_meta_lookup()
    import aidge_export_colibry

    graph.compile("cpu", aidge_core.dtype.int8, dims=[[1, *images.shape[1:]]])
    if tuple(images.shape[1:]) == (1, 120, 160):
        # La PTQ a besoin des convolutions paddées intactes ; l'exporteur attend
        # ensuite Pad->Conv->Quantizer pour fusionner le facteur de remise à l'échelle.
        for node in list(graph.get_nodes()):
            if node.type() in ("PaddedConv2D", "PaddedConvDepthWise2D"):
                aidge_core.expand_metaop(node)
    aidge_core.fuse_to_metaop(graph, "@NNPA")

    first_input = aidge_core.Tensor(np.round(images[0:1] * 127).astype(np.int8))
    first_input.to_backend("cpu")
    first_input.set_data_format(aidge_core.dformat.nchw)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    previous_dir = Path.cwd()
    try:
        os.chdir(output_dir)
        exported = Path(
            aidge_export_colibry.export(
                graph=graph, input_tensors=[first_input], dtype="int8"
            )
        ).resolve()
    finally:
        os.chdir(previous_dir)
    bitstream = exported / "dnn" / "aidge_model_bitstream.nnpa"
    if not bitstream.is_file():
        raise SystemExit(f"Export incomplet : bitstream absent dans {exported}")
    print(f"Export Colibry : {exported}")
    print(f"Bitstream : {bitstream}")
    print("Export logiciel validé ; carte, capteur, firmware et qualité non testés.")


if __name__ == "__main__":
    main()
