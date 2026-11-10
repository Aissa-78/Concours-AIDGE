"""Choisit le seuil sur la validation locale, jamais sur porte_B_test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from heatmap_metrics import evaluate_logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/heatmap_v2")
    parser.add_argument("--aidge-python", type=Path, default=ROOT / ".venv-aidge-colibry" /
                        ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
    args = parser.parse_args()
    run = args.run_dir.resolve()
    info = json.loads((run / "dataset.json").read_text(encoding="utf-8"))
    truth = info["sources"]["local"]["splits"]["validation"]["centers"]
    with np.load(run / "dataset.npz") as data:
        np.savez_compressed(run / "threshold_inputs.npz", images=data["local_validation_images"])
    subprocess.run([str(args.aidge_python.absolute()), str(ROOT / "scripts/train_heatmap_v2_aidge.py"),
                    "--run-dir", str(run), "--predict-inputs", str(run / "threshold_inputs.npz"),
                    "--predict-output", str(run / "threshold_logits.npy")], cwd=ROOT, check=True)
    logits = np.load(run / "threshold_logits.npy")
    candidates = [{"threshold": round(float(t), 2), "metrics": evaluate_logits(logits, truth, round(float(t), 2))[0]}
                  for t in np.arange(0.2, 0.81, 0.05)]
    selected = max(candidates, key=lambda row: (row["metrics"]["f1"], -abs(row["threshold"] - 0.5)))
    result = {"source": "local_validation_only", "images": len(truth),
              "model_sha256": hashlib.sha256((run / "best.onnx").read_bytes()).hexdigest(),
              "selected": selected, "candidates": candidates,
              "note": "Réglage sur validation, pas une mesure de généralisation. Porte B non utilisée."}
    (run / "threshold.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Seuil choisi sur validation uniquement : {selected['threshold']:.2f}, F1 validation={selected['metrics']['f1']:.1%}")


if __name__ == "__main__":
    main()
