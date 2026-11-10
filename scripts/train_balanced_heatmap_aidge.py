"""Adaptation Aidge 50/50, sélection sur le plus faible des deux F1 de validation.

Pas de préentraînement supplémentaire : repartir d'un checkpoint public existant.
Ni la porte B ni un score de test ne participent à l'apprentissage/sélection.
"""

from __future__ import annotations

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
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import aidge_core as ac
import aidge_onnx
from train_heatmap_v2_aidge import tensor, save_weights  # importe aussi le backend CPU
from heatmap_metrics import weighted_bce, evaluate_logits
from heatmap_v2_data import augment
from heatmap_balanced_protocol import balanced_batches, selection_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--init-weights", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=.0002)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.min_epochs) < 1 or args.batch_size < 2 or args.batch_size % 2:
        raise SystemExit("Époques/patience positives ; batch pair >= 2 requis.")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise SystemExit("Taux d'apprentissage positif et fini requis.")
    run = args.run_dir.resolve()
    if any((run / name).exists() for name in ("training_history.csv", "best.onnx", "protocol.json")):
        raise SystemExit("Cet essai existe déjà : préparer un nouveau dossier pour le conserver.")
    info = json.loads((run / "dataset.json").read_text(encoding="utf-8"))
    with np.load(run / "dataset.npz") as source:
        data = dict(source)
    train_videos, val_videos = set(), set()
    for source in ("local", "coco"):
        for split, videos in (("train", train_videos), ("validation", val_videos)):
            videos.update(row["video"] for row in info["sources"][source]["splits"][split]["rows"])
            if not len(data[f"{source}_{split}_images"]):
                raise ValueError("Source d'entraînement ou de validation vide.")
    if train_videos & val_videos:
        raise ValueError("Une vidéo de validation est présente dans le train.")
    train_hashes = {hashlib.sha256(image.tobytes()).hexdigest() for source in ("local", "coco")
                    for image in data[source + "_train_images"]}
    if any(hashlib.sha256(image.tobytes()).hexdigest() in train_hashes for source in ("local", "coco")
           for image in data[source + "_validation_images"]):
        raise ValueError("Une image préparée de validation est présente dans le train.")

    graph = aidge_onnx.load_onnx(str(run / "initial.onnx"))
    ac.expand_metaops(graph)
    graph.compile("cpu", ac.dtype.float32, dims=[[2, 1, 128, 128]])
    scheduler = ac.SequentialScheduler(graph)
    params = {node.name(): node.get_operator().get_output(0) for node in ac.producers(graph)}
    views = {name: np.array(value, copy=False) for name, value in params.items()}

    # Contrôler le calcul Aidge avec les références initiales avant de charger les poids.
    output = scheduler.forward(data=[tensor(data["local_train_images"][:2])])[0]
    logits = np.array(output, copy=True)
    _, gradient = weighted_bce(logits, data["local_train_targets"][:2], 10.)
    output.grad = tensor(gradient)
    scheduler.backward()
    with np.load(run / "gradient_reference.npz") as reference:
        if not np.allclose(logits, reference["logits"], atol=1e-5, rtol=1e-4):
            raise RuntimeError("Les sorties Aidge diffèrent de la référence.")
        gradient_error = 0.
        for name, value in params.items():
            actual = np.array(value.grad, copy=True)
            expected = reference["grad_" + name]
            gradient_error = max(gradient_error, float(np.max(np.abs(actual - expected))))
            if not np.allclose(actual, expected, atol=1e-5, rtol=5e-3):
                raise RuntimeError(f"Gradient incorrect : {name}")
            value.grad.zeros()
    checkpoint_hash = hashlib.sha256(args.init_weights.read_bytes()).hexdigest()
    with np.load(args.init_weights) as weights:
        if set(weights.files) != set(views):
            raise ValueError("Noms de paramètres incompatibles.")
        for name, view in views.items():
            if weights[name].shape != view.shape or not np.isfinite(weights[name]).all():
                raise ValueError(f"Poids incompatible : {name}")
        for name, view in views.items():
            view[:] = weights[name]
    initial_values = {name: view.copy() for name, view in views.items()}

    selection = "maximiser min(F1 local, F1 public), puis moyenne F1, puis minimiser moyenne loss ; seuil fixe 0,5 ; baseline époque 0 éligible"
    protocol = {"initial_checkpoint": str(args.init_weights.resolve()), "initial_checkpoint_sha256": checkpoint_hash,
                "dataset_sha256": hashlib.sha256((run / "dataset.npz").read_bytes()).hexdigest(),
                "seed": info["seed"], "learning_rate": args.learning_rate, "batch_size": args.batch_size,
                "epochs_requested": args.epochs, "patience": args.patience, "min_epochs": args.min_epochs,
                "selection": selection, "public_fraction_per_batch": .5,
                "local_images_per_epoch": len(data["local_train_images"]),
                "public_images_per_epoch": len(data["local_train_images"]),
                "note": "Images publiques rééchantillonnées à chaque époque ; aucun test utilisé pour sélectionner."}
    (run / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")

    def validation(source):
        inputs, targets = data[source + "_validation_images"], data[source + "_validation_targets"]
        outputs = []
        for offset in range(0, len(inputs), args.batch_size):
            output = scheduler.forward(data=[tensor(inputs[offset:offset + args.batch_size])])[0]
            outputs.append(np.array(output, copy=True))
        logits = np.concatenate(outputs)
        loss, _ = weighted_bce(logits, targets, 10.)
        truth = info["sources"][source]["splits"]["validation"]["centers"]
        metrics, _ = evaluate_logits(logits, truth)
        return loss, metrics

    images = np.concatenate([data["local_train_images"], data["coco_train_images"]])
    targets = np.concatenate([data["local_train_targets"], data["coco_train_targets"]])
    local_count, public_count = len(data["local_train_images"]), len(data["coco_train_images"])
    rng = np.random.default_rng(info["seed"])
    moments = {name: (np.zeros_like(view), np.zeros_like(view)) for name, view in views.items()}
    step, stale, selected_epoch, completed_epochs = 0, 0, 0, 0
    best_key, selected_validation, baseline_validation = None, None, None
    start = time.monotonic()
    fields = ["epoch", "train_loss", "local_loss", "public_loss", "local_f1", "public_f1",
              "selection_min_f1", "selection_mean_f1", "selected", "elapsed_seconds"]
    with (run / "training_history.csv").open("w", newline="", encoding="utf-8") as history:
        writer = csv.DictWriter(history, fieldnames=fields)
        writer.writeheader()
        # L'époque 0 évite de perdre un bon checkpoint si aucune mise à jour n'aide.
        for epoch in range(args.epochs + 1):
            train_loss = None
            if epoch:
                losses, seen = 0., 0
                for indices in balanced_batches(local_count, public_count, args.batch_size, rng):
                    batch = [augment(images[i], targets[i], rng) for i in indices]
                    bx, by = np.stack([x for x, _ in batch]), np.stack([y for _, y in batch])
                    prediction = scheduler.forward(data=[tensor(bx)])[0]
                    loss, gradient = weighted_bce(np.array(prediction, copy=True), by, 10.)
                    if not np.isfinite(loss):
                        raise RuntimeError("Loss non finie.")
                    prediction.grad = tensor(gradient)
                    scheduler.backward()
                    step += 1
                    for name, value in params.items():
                        grad = np.array(value.grad, copy=True)
                        if not np.isfinite(grad).all():
                            raise RuntimeError(f"Gradient non fini : {name}")
                        first, second = moments[name]
                        first *= .9
                        first += .1 * grad
                        second *= .999
                        second += .001 * grad * grad
                        views[name] -= args.learning_rate * (first / (1 - .9 ** step)) / (
                            np.sqrt(second / (1 - .999 ** step)) + 1e-8)
                        value.grad.zeros()
                    losses += loss * len(indices)
                    seen += len(indices)
                train_loss = losses / seen
                completed_epochs = epoch
            local_loss, local = validation("local")
            public_loss, public = validation("coco")
            key = selection_key(local["f1"], public["f1"], local_loss, public_loss)
            improved = best_key is None or key > best_key
            if epoch == 0:
                baseline_validation = {"local": local, "public": public}
            if improved:
                best_key, stale, selected_epoch = key, 0, epoch
                selected_validation = {"local": local, "public": public}
                save_weights(run / "initial.onnx", views, run / "best.onnx")
            else:
                stale += 1
            writer.writerow({"epoch": epoch, "train_loss": train_loss, "local_loss": local_loss,
                             "public_loss": public_loss, "local_f1": local["f1"], "public_f1": public["f1"],
                             "selection_min_f1": key[0], "selection_mean_f1": key[1], "selected": improved,
                             "elapsed_seconds": time.monotonic() - start})
            history.flush()
            print(f"Mixte {epoch:02d}/{args.epochs} : F1 portes={local['f1']:.1%}, "
                  f"public={public['f1']:.1%}, minimum={key[0]:.1%}"
                  + (" [retenu]" if improved else ""), flush=True)
            if epoch >= args.min_epochs and stale >= args.patience:
                print("Arrêt anticipé : aucun progrès du critère commun.", flush=True)
                break

    with np.load(run / "best.npz") as weights:
        changed = max(float(np.max(np.abs(weights[name] - initial_values[name]))) for name in views)
    result = {**protocol, "forward_backward": "Aidge CPU", "optimizer": "Adam NumPy, BCE analytique pondérée 10",
              "pytorch_training": False, "gradient_reference_max_error": gradient_error,
              "epochs_completed": completed_epochs, "selected_epoch": selected_epoch,
              "baseline_validation": baseline_validation, "selected_validation": selected_validation,
              "max_weight_change": changed, "threshold": .5, "radius_normalized": .125,
              "input_size": 128, "content_size": 80, "grid_size": 20,
              "augmentations": "rotations 0/90/180/270°, miroir, contraste/luminosité, bruit léger",
              "elapsed_seconds": time.monotonic() - start}
    (run / "training.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Modèle retenu : époque {selected_epoch}, {run / 'best.onnx'}", flush=True)


if __name__ == "__main__":
    main()
