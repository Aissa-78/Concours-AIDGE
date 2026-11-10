"""Teste l'export logiciel d'une architecture QQVGA avant son entraînement long.

Poids initiaux seulement, calibration train : ne valide ni précision ni carte.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnx
from onnx import numpy_helper
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from person_detector_qqvga import make_person_detector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--aidge-python", type=Path, default=ROOT/".venv-aidge-colibry"/("Scripts/python.exe" if os.name == "nt" else "bin/python"))
    args = parser.parse_args()
    run = args.run_dir.resolve()
    log_path = run/"architecture_export.log"
    if log_path.exists():
        raise SystemExit("Un contrôle existe déjà ; conserver son log et choisir un autre essai.")
    info = json.loads((run/"dataset.json").read_text())
    compact = make_person_detector(info["channels"], info["blocks"], residual=info.get("residual", False), architecture=info.get("architecture", "separable")).eval()
    model = onnx.load(run/"initial_dense.onnx")
    weights = {item.name.replace(".", "_"): numpy_helper.to_array(item) for item in model.graph.initializer}
    compact.load_state_dict(compact.compact_state(weights))
    torch.onnx.export(compact, torch.zeros(1, 1, 120, 160), str(run/"initial_compact.onnx"),
                      input_names=["image"], output_names=["detections"], opset_version=17, dynamo=False)
    rows = json.loads((run/"train.json").read_text())
    indices = []
    for source in ("portes", "wepdtof"):
        candidates = [i for i, row in enumerate(rows) if row["source"] == source]
        indices.extend(candidates[i] for i in np.linspace(0, len(candidates)-1, min(8, len(candidates)), dtype=int))
    with np.load(run/"train.npz") as data:
        np.savez_compressed(run/"architecture_calibration.npz", images=data["images"][indices])
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run([str(args.aidge_python.absolute()), str(ROOT/"scripts/export_colibry.py"),
                                  "--run-dir", str(run), "--model", str(run/"initial_compact.onnx"),
                                  "--calibration", str(run/"architecture_calibration.npz"),
                                  "--output-dir", str(run/"architecture_export")],
                                 cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    bitstreams = list((run/"architecture_export").rglob("aidge_model_bitstream.nnpa"))
    result = {"software_export_passed": process.returncode == 0 and bool(bitstreams),
              "parameters": info["parameters_compact"], "residual": info.get("residual", False),
              "calibration_train_indices": indices, "bitstreams": [str(path) for path in bitstreams],
              "initial_weights_only": True, "hardware_executed": False, "uint8_validated": False,
              "power_validated": False, "log": str(log_path), "returncode": process.returncode}
    (run/"architecture_export.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["software_export_passed"]:
        raise SystemExit("Export non validé : consulter le log, ne pas annoncer la compatibilité matérielle.")


if __name__ == "__main__":
    main()
