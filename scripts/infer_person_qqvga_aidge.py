"""Inférence Aidge CPU d'un détecteur gris QQVGA ou QVGA."""

import argparse
from pathlib import Path

import numpy as np
import aidge_core as ac
import aidge_onnx
from train_heatmap_v2_aidge import tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    with np.load(args.inputs) as source:
        images = source["images"]
    if args.batch_size < 1 or images.ndim != 4 or images.shape[1] != 1 or not len(images) or not (1 <= images.shape[2] <= 240 and 1 <= images.shape[3] <= 320):
        raise ValueError("Images grises (N,1,H,W), au plus 320x240, batch positif.")
    graph = aidge_onnx.load_onnx(str(args.model))
    ac.expand_metaops(graph)
    graph.compile("cpu", ac.dtype.float32, dims=[[1, *images.shape[1:]]])
    scheduler, outputs = ac.SequentialScheduler(graph), []
    for offset in range(0, len(images), args.batch_size):
        output = scheduler.forward(data=[tensor(images[offset:offset+args.batch_size])])[0]
        outputs.append(np.array(output, copy=True))
    np.save(args.output, np.concatenate(outputs))
    print(f"Inférence Aidge {images.shape[3]}x{images.shape[2]} : {len(images)} images -> {args.output}")


if __name__ == "__main__":
    main()
