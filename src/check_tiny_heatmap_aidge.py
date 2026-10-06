"""Compare l'inférence Aidge CPU à la sortie PyTorch du prototype.

Commande : python src/check_tiny_heatmap_aidge.py
À lancer avec un environnement Python contenant les modules Aidge.
"""

import argparse
from pathlib import Path

import numpy as np

try:
    import aidge_core
    import aidge_backend_cpu  # noqa: F401 : enregistre les opérateurs CPU
    import aidge_onnx
except ImportError as exc:
    raise SystemExit(
        "Aidge n'est pas installé dans cet environnement Python. "
        "Lancez ce script avec l'environnement .venv-aidge."
    ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/tiny_heatmap"))
    args = parser.parse_args()
    output_dir = args.run_dir
    onnx_path = output_dir / "tiny_heatmap.onnx"
    input_path = output_dir / "input.npy"
    reference_path = output_dir / "expected_pytorch.npy"
    if not all(path.is_file() for path in (onnx_path, input_path, reference_path)):
        raise SystemExit("Fichiers absents : lancez d'abord src/export_tiny_heatmap.py.")

    input_array = np.load(input_path)
    expected = np.load(reference_path)
    graph = aidge_onnx.load_onnx(str(onnx_path))
    aidge_core.expand_metaops(graph)
    graph.compile("cpu", aidge_core.dtype.float32, dims=[[1, 3, 96, 96]])

    tensor = aidge_core.Tensor(input_array)
    tensor.to_backend("cpu")
    tensor.set_data_format(aidge_core.dformat.nchw)
    tensor.to_dtype(aidge_core.dtype.float32)
    aidge_core.SequentialScheduler(graph).forward(data=[tensor])

    port = graph.get_ordered_outputs()[0]
    actual = np.array(port.node.get_operator().get_output(port.index))
    if actual.shape != expected.shape:
        raise SystemExit(f"Forme différente : Aidge {actual.shape}, PyTorch {expected.shape}")

    max_difference = float(np.max(np.abs(actual - expected)))
    print(f"Sortie Aidge : {actual.shape}")
    print(f"Écart maximal PyTorch/Aidge : {max_difference:.2e}")
    if not np.allclose(actual, expected, rtol=1e-4, atol=1e-5):
        raise SystemExit("ÉCHEC : les sorties PyTorch et Aidge sont différentes.")
    print("OK : l'export et l'exécution CPU dans Aidge sont validés.")
    print("Ce test valide la compatibilité numérique, pas la qualité de détection.")


if __name__ == "__main__":
    main()
