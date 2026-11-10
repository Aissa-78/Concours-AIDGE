"""Test de mémorisation d'un détecteur QQVGA dans Aidge, pas un test de fiabilité.

Les gradients sont ceux d'Aidge. Les convolutions depthwise sont représentées
par des convolutions denses diagonales contraintes, car leur backward CPU manque
dans l'installation locale. Loss et Adam NumPy ; aucune mise à jour PyTorch.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
import numpy as np
import aidge_core as ac
import aidge_onnx
from train_heatmap_v2_aidge import tensor, save_weights
from person_detection_math import detection_loss, evaluate_boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT/"runs/person_qqvga_mem20_v1")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise SystemExit("Paramètres d'apprentissage invalides.")
    run = args.run_dir.resolve()
    if not args.check_only and (run/"training_history.csv").exists():
        raise SystemExit("Cet entraînement existe : choisir un nouvel essai pour le conserver.")
    info = json.loads((run/"dataset.json").read_text())
    data, reference, structural_masks = (dict(np.load(run/name)) for name in
                                        ("dataset.npz", "gradient_reference.npz", "parameter_masks.npz"))
    if any(row["row"]["split"] != "train" for row in info["rows"]):
        raise ValueError("Ce test technique ne doit utiliser que des images train.")
    graph = aidge_onnx.load_onnx(str(run/"initial_dense.onnx"))
    ac.expand_metaops(graph)
    graph.compile("cpu", ac.dtype.float32, dims=[[2, 1, 120, 160]])
    scheduler = ac.SequentialScheduler(graph)
    parameters = {node.name(): node.get_operator().get_output(0) for node in ac.producers(graph)}
    views = {name: np.array(value, copy=False) for name, value in parameters.items()}
    if {"grad_"+name for name in parameters} != {key for key in reference if key.startswith("grad_")}:
        raise RuntimeError("Paramètres partagés ou manquants après import ONNX ; apprentissage refusé.")
    indices = reference["indices"]
    prediction = scheduler.forward(data=[tensor(data["images"][indices])])[0]
    logits = np.array(prediction, copy=True)
    if not np.allclose(logits, reference["logits"], atol=2e-5, rtol=2e-4):
        raise RuntimeError(f"Écart forward Aidge/PyTorch : {np.max(np.abs(logits-reference['logits']))}")
    _, gradient, _ = detection_loss(logits, data["heat"][indices], data["distances"][indices], data["masks"][indices])
    prediction.grad = tensor(gradient)
    scheduler.backward()
    error = 0.
    for name, value in parameters.items():
        actual = np.array(value.grad, copy=True)
        expected = reference["grad_"+name]
        error = max(error, float(np.max(np.abs(actual-expected))))
        if not np.allclose(actual, expected, atol=2e-4, rtol=5e-3):
            raise RuntimeError(f"Écart gradient {name} : {np.max(np.abs(actual-expected))}")
        value.grad.zeros()
    audit = {"forward_max_error": float(np.max(np.abs(logits-reference["logits"]))),
             "gradient_max_error": error, "backend": "Aidge CPU FP32",
             "depthwise_backward": "Equivalent dense diagonal contraint ; pas de backward natif depthwise.",
             "hardware_validated": False}
    (run/"compatibility.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Aidge vérifié : sorties et gradients, écart gradient maximal {error:.2e}.", flush=True)
    if args.check_only:
        return
    protocol = {"epochs_requested": args.epochs, "batch_size": args.batch_size,
                "learning_rate": args.learning_rate, "lr_schedule": "0.3 x LR à partir de l'époque 151",
                "threshold": .5, "box_match_iou": .5, "augmentations": False,
                "stop_rule": "precision et rappel >= 0.95 sur les 20 mêmes images, deux évaluations consécutives",
                "evaluate_every": 10, "seed": info["seed"],
                "dataset_sha256": hashlib.sha256((run/"dataset.npz").read_bytes()).hexdigest(),
                "note": "Test de mémorisation uniquement, pas une évaluation de généralisation."}
    (run/"protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    rng = np.random.default_rng(info["seed"])
    moments = {name: (np.zeros_like(view), np.zeros_like(view)) for name, view in views.items()}
    truths = [row["boxes"] for row in info["rows"]]
    step, best_key, best_epoch, successes = 0, None, 0, 0
    start = time.monotonic()
    fields = ["epoch", "loss", "focal", "box", "tp", "fp", "fn", "precision", "recall", "f1", "seconds"]
    with (run/"training_history.csv").open("w", newline="", encoding="utf-8") as history:
        writer = csv.DictWriter(history, fieldnames=fields)
        writer.writeheader()
        for epoch in range(args.epochs+1):
            if epoch:
                order = rng.permutation(len(truths))
                lr = args.learning_rate*(.3 if epoch > 150 else 1.)
                for offset in range(0, len(order), args.batch_size):
                    batch = order[offset:offset+args.batch_size]
                    prediction = scheduler.forward(data=[tensor(data["images"][batch])])[0]
                    loss, gradient, _ = detection_loss(np.array(prediction, copy=True), data["heat"][batch], data["distances"][batch], data["masks"][batch])
                    if not np.isfinite(loss):
                        raise RuntimeError("Loss non finie.")
                    prediction.grad = tensor(gradient)
                    scheduler.backward()
                    step += 1
                    for name, value in parameters.items():
                        grad = np.array(value.grad, copy=True)
                        if not np.isfinite(grad).all():
                            raise RuntimeError(f"Gradient non fini : {name}")
                        if name in structural_masks:
                            grad *= structural_masks[name]
                        first, second = moments[name]
                        first *= .9
                        first += .1*grad
                        second *= .999
                        second += .001*grad*grad
                        views[name] -= lr*(first/(1-.9**step))/(np.sqrt(second/(1-.999**step))+1e-8)
                        if name in structural_masks:
                            views[name] *= structural_masks[name]
                        value.grad.zeros()
            if epoch % 10 and epoch != args.epochs:
                continue
            outputs = []
            for offset in range(0, len(truths), args.batch_size):
                pred = scheduler.forward(data=[tensor(data["images"][offset:offset+args.batch_size])])[0]
                outputs.append(np.array(pred, copy=True))
            outputs = np.concatenate(outputs)
            loss, _, components = detection_loss(outputs, data["heat"], data["distances"], data["masks"])
            metrics, _ = evaluate_boxes(outputs, truths)
            key = (metrics["f1"], -loss)
            if best_key is None or key > best_key:
                best_key, best_epoch = key, epoch
                save_weights(run/"initial_dense.onnx", views, run/"best_dense.onnx")
                np.save(run/"best_logits.npy", outputs)
            successes = successes+1 if metrics["precision"] >= .95 and metrics["recall"] >= .95 else 0
            writer.writerow({"epoch": epoch, "loss": loss, **components, **metrics, "seconds": time.monotonic()-start})
            history.flush()
            print(f"Mémo {epoch:03d}/{args.epochs} : loss={loss:.3f}, P={metrics['precision']:.1%}, R={metrics['recall']:.1%}, F1={metrics['f1']:.1%} (TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']})", flush=True)
            if successes >= 2:
                print("Critère de mémorisation atteint deux fois ; arrêt du test.", flush=True)
                break
    result = {**protocol, **audit, "forward_backward": "Aidge CPU", "optimizer": "Adam NumPy",
              "pytorch_training": False, "epochs_completed": epoch, "selected_epoch": best_epoch,
              "memorization_passed": successes >= 2, "elapsed_seconds": time.monotonic()-start}
    (run/"training.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Test terminé ; meilleur état à l'époque {best_epoch}. Ne pas annoncer ce score comme une fiabilité terrain.", flush=True)


if __name__ == "__main__":
    main()
