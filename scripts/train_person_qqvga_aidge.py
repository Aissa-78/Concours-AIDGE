"""Entraînement QQVGA ou QVGA : Aidge FP32 forward/backward, loss/Adam NumPy.

Le checkpoint est choisi sur la validation préparée, jamais sur le test réservé.
Les images train préparées sont toutes utilisées à chaque époque, avec
répétition de la petite source pour un équilibre 50/50. La géométrie et les
sources sont décrites dans dataset.json ; les anciens essais gardent leurs
valeurs par défaut QQVGA. Pas d'import PyTorch ici.
"""

import argparse
import csv
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import subprocess
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
from person_training_data import assert_disjoint, augment_sample, balanced_order, selection_key


def predict(scheduler, images, batch_size):
    outputs = []
    for offset in range(0, len(images), batch_size):
        output = scheduler.forward(data=[tensor(images[offset:offset+batch_size])])[0]
        outputs.append(np.array(output, copy=True))
    return np.concatenate(outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT/"runs/person_qqvga_full_v1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--patience", type=int, default=6, help="Nombre d'évaluations sans amélioration.")
    parser.add_argument("--evaluate-every", type=int, default=2)
    parser.add_argument("--init-weights", type=Path, help="Poids denses d'un essai précédent ; Adam repart de zéro.")
    parser.add_argument("--report-python", type=Path, help="Générer le rapport en fin de calcul avec cet environnement Python.")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--no-augmentation", action="store_true", help="Diagnostic : apprendre les images non modifiées.")
    parser.add_argument("--lr-decay-every", type=int, default=10, help="Diviser le LR par deux après ce nombre d'époques.")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience, args.evaluate_every, args.lr_decay_every) < 1 or args.batch_size % 2:
        raise SystemExit("Époques/patience positives et batch pair requis pour le mélange 50/50.")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise SystemExit("Taux d'apprentissage positif et fini requis.")
    run = args.run_dir.resolve()
    if not args.check_only and (run/"training_history.csv").exists():
        raise SystemExit("Entraînement existant : préparer un nouveau dossier pour le conserver.")
    info = json.loads((run/"dataset.json").read_text())
    geometry = info.get("geometry", {})
    source_names = sorted(info["train"])
    if len(source_names) != 2:
        raise ValueError("Le mélange équilibré attend exactement deux sources.")
    input_shape = tuple(info.get("input_shape", [1, 120, 160]))
    loss_function = partial(detection_loss, classification=info.get("classification", "focal"))
    metrics_function = partial(evaluate_boxes, local_maximum=info.get("local_maximum", False), **geometry)
    records = {split: json.loads((run/f"{split}.json").read_text()) for split in ("train", "validation")}
    assert_disjoint(records["train"]+records["validation"])
    for split, rows in records.items():
        if any(row["row"]["split"] != split for row in rows):
            raise RuntimeError("Mélange de splits dans les fichiers préparés.")
    for path, digest in info["manifests"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Manifest modifié depuis la préparation : préparer un nouvel essai.")
    for row in info["rows"]:
        for key in ("image", "label"):
            path = Path(row["root"])/row["row"][key]
            if hashlib.sha256(path.read_bytes()).hexdigest() != row[key+"_sha256"]:
                raise RuntimeError(f"Source modifiée depuis la préparation : {path}")
    data = {split: dict(np.load(run/f"{split}.npz")) for split in records}
    for split, values in data.items():
        if values["images"].shape != (len(records[split]), *input_shape):
            raise ValueError("Nombre ou forme des images incorrect.")
        if any(not np.isfinite(value).all() for value in values.values()):
            raise ValueError("Données non finies.")
    reference = dict(np.load(run/"gradient_reference.npz"))
    structural_masks = dict(np.load(run/"parameter_masks.npz"))
    graph = aidge_onnx.load_onnx(str(run/"initial_dense.onnx"))
    ac.expand_metaops(graph)
    graph.compile("cpu", ac.dtype.float32, dims=[[2, *input_shape]])
    scheduler = ac.SequentialScheduler(graph)
    parameters = {node.name(): node.get_operator().get_output(0) for node in ac.producers(graph)}
    views = {name: np.array(value, copy=False) for name, value in parameters.items()}
    if {"grad_"+name for name in parameters} != {key for key in reference if key.startswith("grad_")}:
        raise RuntimeError("Paramètres partagés/manquants après import ONNX.")
    indices = reference["indices"]
    output = scheduler.forward(data=[tensor(data["train"]["images"][indices])])[0]
    raw = np.array(output, copy=True)
    if not np.allclose(raw, reference["logits"], atol=2e-5, rtol=2e-4):
        raise RuntimeError("Le forward Aidge ne correspond pas à la référence.")
    _, gradient, _ = loss_function(raw, *(data["train"][key][indices] for key in ("heat", "distances", "masks")))
    output.grad = tensor(gradient)
    scheduler.backward()
    gradient_error, gradient_relative = 0., {}
    for name, value in parameters.items():
        actual, expected = np.array(value.grad, copy=True), reference["grad_"+name]
        gradient_error = max(gradient_error, float(np.max(np.abs(actual-expected))))
        gradient_relative[name] = float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected), 1e-12))
        if not np.allclose(actual, expected, atol=2e-4, rtol=5e-3):
            raise RuntimeError(f"Écart de gradient : {name}")
        value.grad.zeros()
    audit = {"backend": "Aidge CPU FP32", "forward_max_error": float(np.max(np.abs(raw-reference["logits"]))),
             "gradient_max_error": gradient_error,
             "gradient_relative_l2_by_parameter": gradient_relative,
             "depthwise_backward": ("Non concerné : convolutions ordinaires" if info.get("architecture") in ("regular", "regular_context", "profile_context")
                                    else "Dense diagonal contraint, pas le backward natif depthwise"),
             "pytorch_training": False, "optimizer": "Adam NumPy", "hardware_tested": False}
    (run/"compatibility.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Sorties et gradients Aidge vérifiés (écart max {gradient_error:.2e}).", flush=True)
    if args.check_only:
        return
    initialization = {"initialization": info["initialization"]}
    if args.init_weights:
        with np.load(args.init_weights) as checkpoint:
            if set(checkpoint.files) != set(views):
                raise ValueError("Le checkpoint ne possède pas les paramètres de cette architecture.")
            for name, view in views.items():
                value = checkpoint[name]
                if value.shape != view.shape or not np.isfinite(value).all():
                    raise ValueError(f"Poids incompatibles : {name}")
                if name in structural_masks and np.any(value*(1-structural_masks[name]) != 0):
                    raise ValueError("Poids hors diagonale dans le checkpoint : conversion refusée.")
            for name, view in views.items():
                view[...] = checkpoint[name]
        initialization = {"initialization": "checkpoint, poids conservés mais Adam réinitialisé",
                          "initial_checkpoint": str(args.init_weights.resolve()),
                          "initial_checkpoint_sha256": hashlib.sha256(args.init_weights.read_bytes()).hexdigest(),
                          "optimizer_resumed": False}
    protocol = {"epochs_requested": args.epochs, "batch_size": args.batch_size,
                "learning_rate": args.learning_rate, "lr_schedule": f"LR x 0.5 tous les {args.lr_decay_every} epochs",
                "patience_evaluations": args.patience, "evaluate_every": args.evaluate_every,
                "threshold": .5, "box_match_iou": .5, "nms_iou": .5, "seed": info["seed"],
                "classification": info.get("classification", "focal"), "local_maximum": info.get("local_maximum", False),
                **initialization, "all_train_images_per_epoch": True,
                "sampling": f"50/50 {' / '.join(source_names)} ; toute la petite source puis répétitions aléatoires",
                "augmentations": False if args.no_augmentation else ("Train uniquement : miroir horizontal synchronisé boîtes, contraste/luminosité, flou 3x3 et bruit légers" if info.get("profile_run") else "Train uniquement : miroirs H/V synchronisés boîtes, contraste/luminosité, flou 3x3 et bruit légers"),
                "architecture": {"type": info.get("architecture", "separable"), **{key: info.get(key, False) for key in ("channels", "blocks", "residual")}},
                "selection": f"max minimum(F1 validation {' / '.join(source_names)}), puis moyenne, puis loss",
                "reserved_door_B_used": False,
                "data_sha256": {split: hashlib.sha256((run/f"{split}.npz").read_bytes()).hexdigest() for split in data},
                "geometry": geometry,
                "note": info.get("validation_warning", "Validation déjà utilisée pour le développement ; ne remplace pas une nouvelle porte aveugle.")}
    (run/"protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    moments = {name: (np.zeros_like(view), np.zeros_like(view)) for name, view in views.items()}
    rng = np.random.default_rng(info["seed"])
    truths = {split: [record["boxes"] for record in rows] for split, rows in records.items()}
    sources = {split: np.array([record["source"] for record in rows]) for split, rows in records.items()}
    step, best_key, best_epoch, stale = 0, None, 0, 0
    best_metrics = None
    start = time.monotonic()
    fields = ["epoch", "train_loss_augmented", "train_loss", "train_f1", "validation_loss"]
    fields += [source+"_"+key for source in source_names for key in ("f1", "precision", "recall", "tp", "fp", "fn")]
    fields.append("seconds")
    with (run/"training_history.csv").open("w", newline="", encoding="utf-8") as history:
        writer = csv.DictWriter(history, fieldnames=fields)
        writer.writeheader()
        for epoch in range(args.epochs+1):
            training_losses = []
            if epoch:
                order = balanced_order(sources["train"], rng)
                lr = args.learning_rate*.5**((epoch-1)//args.lr_decay_every)
                epoch_start = time.monotonic()
                for offset in range(0, len(order), args.batch_size):
                    batch = order[offset:offset+args.batch_size]
                    if info.get("profile_run"):
                        from profile_training_data import augment_profile
                        augmented = [augment_profile(data["train"]["images"][i], truths["train"][i], rng, enabled=not args.no_augmentation) for i in batch]
                    else:
                        augmented = [augment_sample(data["train"]["images"][i], truths["train"][i], rng,
                                      geometric=not args.no_augmentation, photometric=not args.no_augmentation) for i in batch]
                    images = np.stack([sample[0] for sample in augmented])
                    targets = [np.stack([sample[1][j] for sample in augmented]) for j in range(3)]
                    prediction = scheduler.forward(data=[tensor(images)])[0]
                    loss, gradient, _ = loss_function(np.array(prediction, copy=True), *targets)
                    if not np.isfinite(loss):
                        raise RuntimeError("Loss non finie.")
                    training_losses.append(loss)
                    prediction.grad = tensor(gradient)
                    scheduler.backward()
                    gradients = {name: np.array(value.grad, copy=True) for name, value in parameters.items()}
                    for name, grad in gradients.items():
                        if not np.isfinite(grad).all():
                            raise RuntimeError(f"Gradient non fini : {name}")
                        if name in structural_masks:
                            grad *= structural_masks[name]
                    norm = np.sqrt(sum(float(np.square(grad, dtype=np.float64).sum()) for grad in gradients.values()))
                    scale = min(1., 10./max(norm, 1e-12))
                    step += 1
                    for name, value in parameters.items():
                        grad = gradients[name]*scale
                        first, second = moments[name]
                        first *= .9
                        first += .1*grad
                        second *= .999
                        second += .001*grad*grad
                        views[name] -= lr*(first/(1-.9**step))/(np.sqrt(second/(1-.999**step))+1e-8)
                        if name in structural_masks:
                            views[name] *= structural_masks[name]
                        value.grad.zeros()
                print(f"Époque {epoch:02d}/{args.epochs} : loss de batch={np.mean(training_losses):.3f}, "
                      f"{len(order)} présentations ({len(records['train'])} images distinctes), "
                      f"{time.monotonic()-epoch_start:.0f}s.", flush=True)
                # Sauvegarde atomique à chaque époque : un arrêt ne détruit pas
                # le dernier état complet (poids, Adam, tirages aléatoires).
                state = {"epoch": np.array(epoch), "step": np.array(step),
                         "rng_json": np.array(json.dumps(rng.bit_generator.state)),
                         **{"weight_"+name: view for name, view in views.items()},
                         **{"adam_first_"+name: pair[0] for name, pair in moments.items()},
                         **{"adam_second_"+name: pair[1] for name, pair in moments.items()}}
                temporary = run/"last_training_state.tmp.npz"
                np.savez_compressed(temporary, **state)
                temporary.replace(run/"last_training_state.npz")
            if epoch % args.evaluate_every and epoch != args.epochs:
                continue
            outputs = {split: predict(scheduler, data[split]["images"], args.batch_size) for split in data}
            losses = {split: loss_function(outputs[split], *(data[split][key] for key in ("heat", "distances", "masks")))[0] for split in data}
            training_metrics, _ = metrics_function(outputs["train"], truths["train"])
            metrics = {}
            for source in source_names:
                indices = np.flatnonzero(sources["validation"] == source)
                metrics[source], _ = metrics_function(outputs["validation"][indices], [truths["validation"][i] for i in indices])
            key = selection_key(metrics, losses["validation"])
            improved = best_key is None or key > best_key
            if improved:
                best_key, best_epoch, best_metrics, stale = key, epoch, metrics, 0
                save_weights(run/"initial_dense.onnx", views, run/"best_dense.onnx")
                for split in data:
                    np.save(run/f"best_{split}_logits.npy", outputs[split])
                (run/"best_selection.json").write_text(json.dumps({"epoch": epoch, "metrics": metrics, "key": key}, indent=2), encoding="utf-8")
            else:
                stale += 1
            row = {"epoch": epoch, "train_loss_augmented": float(np.mean(training_losses)) if training_losses else "",
                   "train_loss": losses["train"], "train_f1": training_metrics["f1"],
                   "validation_loss": losses["validation"], "seconds": time.monotonic()-start}
            row.update({source+"_"+key: value for source, values in metrics.items() for key, value in values.items()})
            writer.writerow(row)
            history.flush()
            (run/"progress.json").write_text(json.dumps({"status": "running", "epoch": epoch,
                "epochs_requested": args.epochs, "best_epoch": best_epoch,
                "best_validation_metrics": best_metrics, "seconds": time.monotonic()-start}, indent=2), encoding="utf-8")
            print(f"Validation {epoch:02d} : "+", ".join(f"{name} F1={metrics[name]['f1']:.1%}" for name in source_names)+
                  f", train F1={training_metrics['f1']:.1%}"
                  f"{' — meilleur checkpoint' if improved else ''}.", flush=True)
            if stale >= args.patience:
                print("Arrêt anticipé : le critère de validation ne progresse plus.", flush=True)
                break
    result = {**protocol, **audit, "epochs_completed": epoch, "selected_epoch": best_epoch,
              "validation_metrics": best_metrics, "elapsed_seconds": time.monotonic()-start,
              "stopped_early": epoch < args.epochs, "train_images": len(records["train"]),
              "validation_images": len(records["validation"])}
    (run/"training.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (run/"progress.json").write_text(json.dumps({"status": "completed", "epoch": epoch,
        "epochs_requested": args.epochs, "best_epoch": best_epoch,
        "best_validation_metrics": best_metrics, "seconds": time.monotonic()-start}, indent=2), encoding="utf-8")
    print(f"Terminé : checkpoint de l'époque {best_epoch}. Ni comptage vidéo ni matériel validés.", flush=True)
    if args.report_python:
        environment = os.environ.copy()
        environment.update(MPLBACKEND="Agg", MPLCONFIGDIR=str(run/".matplotlib"))
        report_script = "report_profile_training.py" if info.get("profile_run") else "report_person_training.py"
        subprocess.run([str(args.report_python.absolute()), str(ROOT/"scripts"/report_script),
                        "--run-dir", str(run)], cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
