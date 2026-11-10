"""Exécute les versions Aidge FP32 et INT8 pour compare_tiny_heatmap_int8.py.

Ce worker utilise l'environnement Aidge, sans importer PyTorch ni OpenCV.
Il teste le graphe quantifié sur CPU ; il n'exécute pas le bitstream Colibry.
"""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import aidge_core
import aidge_backend_cpu  # noqa: F401

from export_colibry import build_quantized_graph, load_float_graph


def forward_all(graph, images: np.ndarray) -> np.ndarray:
    scheduler = aidge_core.SequentialScheduler(graph)
    outputs = []
    for image in images:
        tensor = aidge_core.Tensor(np.ascontiguousarray(image[None]))
        tensor.to_backend("cpu")
        tensor.set_data_format(aidge_core.dformat.nchw)
        scheduler.forward(data=[tensor])
        port = graph.get_ordered_outputs()[0]
        output = np.array(port.node.get_operator().get_output(port.index), copy=True)
        expected = (1, 5, 30, 40) if tuple(images.shape[1:]) == (1, 120, 160) else (1, 1, 20, 20)
        if output.shape != expected or not np.isfinite(output).all():
            raise RuntimeError(f"Sortie Aidge invalide : {output.shape}")
        outputs.append(output)
    return np.concatenate(outputs)


def audit_integer_convolutions(graph) -> list[dict]:
    """Contrôle les tenseurs réellement calculés, après l'inférence."""
    audit = []
    for node in graph.get_nodes():
        if node.type() not in ("Conv2D", "ConvDepthWise2D", "PaddedConv2D", "PaddedConvDepthWise2D"):
            continue
        op = node.get_operator()
        row = {"name": node.name(), "type": node.type()}
        for field, index in (("activation", 0), ("weights", 1), ("bias", 2)):
            row[field] = str(np.array(op.get_input(index)).dtype)
        row["accumulation"] = str(np.array(op.get_output(0)).dtype)
        if any(row[key] != "int8" for key in ("activation", "weights")):
            raise RuntimeError(f"La convolution n'utilise pas INT8 : {row}")
        if any(row[key] != "int32" for key in ("bias", "accumulation")):
            raise RuntimeError(f"Biais/accumulation inattendus : {row}")
        audit.append(row)
    if not audit:
        raise RuntimeError("Aucune convolution INT8 vérifiable dans le graphe.")
    return sorted(audit, key=lambda row: row["name"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.inputs) as data:
        images = data["images"]
    with np.load(args.calibration) as data:
        calibration = data["images"]

    graph = load_float_graph(args.onnx)
    graph.compile("cpu", aidge_core.dtype.float32, dims=[[1, *images.shape[1:]]])
    floating = forward_all(graph, images)
    graph_int8 = build_quantized_graph(args.onnx, calibration)
    # Pas de fausse quantification : les convolutions calculent réellement en
    # entiers. L'adaptateur InputRescaling reçoit les mêmes images float32 [0,1].
    integer = forward_all(graph_int8, images)
    audit = audit_integer_convolutions(graph_int8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, aidge_float=floating, aidge_int8=integer)
    args.output.with_suffix(".json").write_text(
        json.dumps({
            "backend": "Aidge CPU",
            "bitstream_executed": False,
            "fake_quantize": False,
            "input": f"NCHW float32 [0,1], forme {list(images.shape[1:])}, adaptateur InputRescaling interne",
            "output": "logits float32 déquantifiés, avant sigmoid",
            "convolutions": audit,
            "versions": {name: version(name) for name in (
                "aidge_core", "aidge_onnx", "aidge_backend_cpu", "aidge_quantization"
            )},
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Aidge : {len(images)} images exécutées en FP32 puis en INT8.", flush=True)
    print(f"Audit réussi : {len(audit)} convolutions INT8, accumulation INT32.", flush=True)


if __name__ == "__main__":
    main()
