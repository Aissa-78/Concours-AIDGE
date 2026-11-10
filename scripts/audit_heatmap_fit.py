"""Diagnostic train/validation d'un essai, sans mise à jour des poids.

Les scores train sont optimistes : ils ne sont pas des mesures de généralisation.
Les métriques restent calculées au seuil fixe 0,5, sans recherche sur un test.
"""

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
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--aidge-python", type=Path, default=ROOT / ".venv-aidge-colibry" /
                        ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
    args = parser.parse_args()
    run = args.run_dir.resolve()
    info = json.loads((run / "dataset.json").read_text(encoding="utf-8"))
    groups, arrays = [], []
    with np.load(run / "dataset.npz") as data:
        for source in ("local", "coco"):
            for split in ("train", "validation"):
                groups.append((source, split, len(data[f"{source}_{split}_images"])))
                arrays.append(data[f"{source}_{split}_images"])
    output = run / "diagnostic_fit"
    output.mkdir(exist_ok=True)
    np.savez_compressed(output / "inputs.npz", images=np.concatenate(arrays))
    env = os.environ.copy()
    env.update(MPLBACKEND="Agg", MPLCONFIGDIR=str(run / ".matplotlib"))
    subprocess.run([str(args.aidge_python.absolute()), str(ROOT / "scripts/train_heatmap_v2_aidge.py"),
                    "--run-dir", str(run), "--predict-inputs", str(output / "inputs.npz"),
                    "--predict-output", str(output / "logits.npy")], cwd=ROOT, env=env, check=True)
    logits = np.load(output / "logits.npy")
    result = {"threshold": .5, "radius_normalized": .125,
              "model_sha256": hashlib.sha256((run / "best.onnx").read_bytes()).hexdigest(),
              "note": "Diagnostic interne : train déjà appris, validations déjà utilisées pour sélectionner. Aucun test indépendant.",
              "sources": {}}
    offset = 0
    for source, split, count in groups:
        truth = info["sources"][source]["splits"][split]["centers"]
        metrics, _ = evaluate_logits(logits[offset:offset+count], truth)
        result["sources"].setdefault(source, {})[split] = {"images": count, **metrics}
        offset += count
        print(f"{source}/{split} : {metrics}", flush=True)
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
