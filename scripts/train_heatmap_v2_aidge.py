"""Apprentissage Aidge FP32, préentraînement public puis adaptation aux portes.

Forward et backward utilisent Aidge CPU. La BCE pondérée et l'optimiseur Adam
sont calculés en NumPy, pour contourner une erreur locale bad any_cast dans
aidge_learning.Adam. Leurs gradients sont vérifiés contre PyTorch au démarrage.
PyTorch n'est pas importé dans ce processus d'entraînement.
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
import onnx
from onnx import numpy_helper
import aidge_core as ac
import aidge_backend_cpu  # noqa: F401
import aidge_onnx

from heatmap_metrics import weighted_bce, evaluate_logits
from heatmap_v2_data import augment


def tensor(array):
    value = ac.Tensor(np.ascontiguousarray(array, dtype=np.float32))
    value.to_backend("cpu")
    value.set_data_format(ac.dformat.nchw)
    return value


def save_weights(initial, parameters, destination):
    """Garde le graphe ONNX initial vérifié, remplace seulement les poids appris.

    Évite le plantage de l'exporteur ONNX Aidge sur les bindings ReLU locaux.
    """
    arrays = {name: np.array(value, copy=True) for name, value in parameters.items()}
    np.savez_compressed(destination.with_suffix(".npz"), **arrays)
    model = onnx.load(initial)
    for index, initializer in enumerate(model.graph.initializer):
        name = initializer.name.replace(".", "_")
        if name not in arrays:
            raise ValueError(f"Paramètre appris manquant : {name}")
        model.graph.initializer[index].CopyFrom(numpy_helper.from_array(arrays[name], initializer.name))
    onnx.checker.check_model(model)
    onnx.save(model, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/heatmap_v2")
    parser.add_argument("--pretrain-epochs", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--init-weights", type=Path, help="Repartir d'un checkpoint NPZ compatible après le contrôle des gradients.")
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--overwrite", action="store_true", help="Autoriser explicitement le remplacement des résultats de cet essai.")
    parser.add_argument("--predict-inputs", type=Path, help="Mode inférence seule : NPZ contenant images.")
    parser.add_argument("--predict-model", type=Path)
    parser.add_argument("--predict-output", type=Path)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience) < 1 or args.pretrain_epochs < 0:
        raise SystemExit("Paramètres d'entraînement invalides.")
    if not np.isfinite([args.pretrain_lr, args.learning_rate]).all() or min(args.pretrain_lr, args.learning_rate) <= 0:
        raise SystemExit("Les taux d'apprentissage doivent être positifs et finis.")
    run = args.run_dir
    if args.predict_inputs:
        model_path = args.predict_model or run / "best.onnx"
        output_path = args.predict_output or run / "predictions.npy"
        with np.load(args.predict_inputs) as source:
            images = source["images"]
        graph = aidge_onnx.load_onnx(str(model_path))
        ac.expand_metaops(graph)
        graph.compile("cpu", ac.dtype.float32, dims=[[1, 1, 128, 128]])
        scheduler = ac.SequentialScheduler(graph)
        predictions = []
        for offset in range(0, len(images), args.batch_size):
            value = scheduler.forward(data=[tensor(images[offset:offset + args.batch_size])])[0]
            predictions.append(np.array(value, copy=True))
        np.save(output_path, np.concatenate(predictions))
        print(f"Inférence Aidge : {len(images)} images -> {output_path}", flush=True)
        return
    if (run / "training_history.csv").exists() and not args.overwrite:
        raise SystemExit("Un entraînement existe déjà ici. Choisir un nouvel essai, ou --overwrite pour le remplacer.")
    info = json.loads((run / "dataset.json").read_text(encoding="utf-8"))
    data = dict(np.load(run / "dataset.npz"))
    rng = np.random.default_rng(info["seed"])
    graph = aidge_onnx.load_onnx(str(run / "initial.onnx"))
    ac.expand_metaops(graph)
    graph.compile("cpu", ac.dtype.float32, dims=[[2, 1, 128, 128]])
    scheduler = ac.SequentialScheduler(graph)
    params = {node.name(): node.get_operator().get_output(0) for node in ac.producers(graph)}
    views = {name: np.array(value, copy=False) for name, value in params.items()}

    # Vérifier les gradients réels avant de lancer un long entraînement.
    pred = scheduler.forward(data=[tensor(data["local_train_images"][:2])])[0]
    logits = np.array(pred, copy=True)
    _, gradient = weighted_bce(logits, data["local_train_targets"][:2], 10.0)
    pred.grad = tensor(gradient)
    scheduler.backward()
    reference = dict(np.load(run / "gradient_reference.npz"))
    if not np.allclose(logits, reference["logits"], atol=1e-5, rtol=1e-4):
        raise RuntimeError("Les sorties Aidge et la référence PyTorch diffèrent.")
    gradient_error = 0.0
    for name, value in params.items():
        actual = np.array(value.grad, copy=True)
        expected = reference["grad_" + name]
        gradient_error = max(gradient_error, float(np.max(np.abs(actual - expected))))
        if not np.allclose(actual, expected, atol=1e-5, rtol=5e-3):
            raise RuntimeError(f"Gradient Aidge incorrect : {name}")
        value.grad.zeros()
    print(f"Aidge forward/backward vérifiés ; écart maximal des gradients : {gradient_error:.2e}", flush=True)
    initial_checkpoint_hash = None
    if args.init_weights:
        initial_checkpoint_hash = hashlib.sha256(args.init_weights.read_bytes()).hexdigest()
        with np.load(args.init_weights) as weights:
            if set(weights.files) != set(views):
                raise ValueError("Les noms des poids initiaux ne correspondent pas au réseau.")
            for name, view in views.items():
                if weights[name].shape != view.shape or not np.isfinite(weights[name]).all():
                    raise ValueError(f"Poids initial incompatible : {name}")
            for name, view in views.items():
                view[:] = weights[name]
        print(f"Reprise des poids : {args.init_weights}", flush=True)
    initial_values = {name: view.copy() for name, view in views.items()}

    def validation(source):
        inputs, targets = data[source + "_validation_images"], data[source + "_validation_targets"]
        outputs = []
        for offset in range(0, len(inputs), args.batch_size):
            pred = scheduler.forward(data=[tensor(inputs[offset:offset + args.batch_size])])[0]
            outputs.append(np.array(pred, copy=True))
        logits = np.concatenate(outputs)
        loss, _ = weighted_bce(logits, targets, 10.0)
        truth = info["sources"][source]["splits"]["validation"]["centers"]
        metrics, _ = evaluate_logits(logits, truth)
        return loss, metrics

    history_path = run / "training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as target:
        fields = ["stage", "epoch", "train_loss", "validation_loss", "tp", "fp", "fn", "precision", "recall", "f1", "elapsed_seconds"]
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        start = time.monotonic()
        for stage, epochs, source, learning_rate in (
            (info.get("pretrain_name", "coco"), args.pretrain_epochs, "coco", args.pretrain_lr),
            ("portes", args.epochs, "local", args.learning_rate),
        ):
            if not epochs:
                continue
            moments = {name: (np.zeros_like(view), np.zeros_like(view)) for name, view in views.items()}
            step, best_f1, best_loss, stale = 0, -1.0, float("inf"), 0
            best_path = run / f"best_{stage}.onnx"
            for epoch in range(1, epochs + 1):
                images, targets = data[source + "_train_images"], data[source + "_train_targets"]
                if source == "local":
                    # Garder des exemples publics pendant l'adaptation aux portes.
                    public_count = len(data["coco_train_images"])
                    indices = rng.choice(public_count, min(64, public_count), replace=False)
                    images = np.concatenate([images, data["coco_train_images"][indices]])
                    targets = np.concatenate([targets, data["coco_train_targets"][indices]])
                losses = []
                order = rng.permutation(len(images))
                for offset in range(0, len(order), args.batch_size):
                    indices = order[offset:offset + args.batch_size]
                    batch = [augment(images[i], targets[i], rng) for i in indices]
                    bx, by = np.stack([sample[0] for sample in batch]), np.stack([sample[1] for sample in batch])
                    prediction = scheduler.forward(data=[tensor(bx)])[0]
                    raw = np.array(prediction, copy=True)
                    loss, gradient = weighted_bce(raw, by, 10.0)
                    if not np.isfinite(loss):
                        raise RuntimeError("Loss non finie ; entraînement arrêté.")
                    prediction.grad = tensor(gradient)
                    scheduler.backward()
                    step += 1
                    for name, value in params.items():
                        grad = np.array(value.grad, copy=True)
                        if not np.isfinite(grad).all():
                            raise RuntimeError(f"Gradient non fini : {name}")
                        first, second = moments[name]
                        first *= 0.9
                        first += 0.1 * grad
                        second *= 0.999
                        second += 0.001 * grad * grad
                        views[name] -= learning_rate * (first / (1 - 0.9 ** step)) / (
                            np.sqrt(second / (1 - 0.999 ** step)) + 1e-8)
                        value.grad.zeros()
                    losses.append(loss * len(indices))
                val_loss, metrics = validation(source)
                train_loss = sum(losses) / len(images)
                writer.writerow({"stage": stage, "epoch": epoch, "train_loss": train_loss,
                                 "validation_loss": val_loss, **metrics, "elapsed_seconds": time.monotonic() - start})
                target.flush()
                print(f"{stage} {epoch:02d}/{epochs} : loss={train_loss:.4f}, val={val_loss:.4f}, "
                      f"F1={metrics['f1']:.1%} (TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']})", flush=True)
                improved = metrics["f1"] > best_f1 or (metrics["f1"] == best_f1 and val_loss < best_loss)
                if improved:
                    best_f1, best_loss, stale = metrics["f1"], val_loss, 0
                    save_weights(run / "initial.onnx", views, best_path)
                else:
                    stale += 1
                if stage == "portes" and epoch >= 15 and stale >= args.patience:
                    print("Arrêt anticipé : pas d'amélioration de la validation.", flush=True)
                    break
            with np.load(best_path.with_suffix(".npz")) as weights:
                for name, view in views.items():
                    view[:] = weights[name]
        changed = max(float(np.max(np.abs(views[name] - initial_values[name]))) for name in views)
        save_weights(run / "initial.onnx", views, run / "best.onnx")
        (run / "training.json").write_text(json.dumps({
            "forward_backward": "Aidge CPU", "optimizer": "Adam NumPy, BCE analytique pondérée 10",
            "pytorch_training": False, "gradient_reference_max_error": gradient_error,
            "pretraining_dataset": info.get("pretrain_name", "coco"),
            "initial_checkpoint": str(args.init_weights.resolve()) if args.init_weights else None,
            "initial_checkpoint_sha256": initial_checkpoint_hash,
            "pretrain_learning_rate": args.pretrain_lr, "learning_rate": args.learning_rate,
            "max_weight_change": changed, "threshold": 0.5, "radius_normalized": 0.125,
            "grid_size": 20, "input_size": 128, "content_size": 80,
            "selection": "F1 validation locale, puis loss en cas d'égalité ; porte B non utilisée",
            "augmentations": "rotations 0/90/180/270°, miroir, contraste/luminosité, bruit léger",
            "pretrain_epochs_requested": args.pretrain_epochs, "epochs_requested": args.epochs,
            "elapsed_seconds": time.monotonic() - start,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Modèle entraîné dans Aidge : {run / 'best.onnx'}", flush=True)


if __name__ == "__main__":
    main()
