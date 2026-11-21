"""Test de mémorisation Aidge sur 8 images train, sans validation ni test.

Réussir ce diagnostic ne mesure PAS la généralisation du détecteur.
Les modèles de production et les données originales restent inchangés.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from person_detection_math import detection_loss, evaluate_boxes, targets_from_boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=ROOT / 'runs/person_camera_coco2000_v1')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--evaluate-every', type=int, default=20)
    args = parser.parse_args()
    if min(args.steps, args.evaluate_every) < 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('Paramètres positifs et finis requis.')
    source, run = args.source_run.absolute(), args.run_dir.absolute()
    if run.exists():
        raise FileExistsError('Conserver les anciens diagnostics : choisir un nouveau dossier.')
    all_rows = json.loads((source / 'train.json').read_text())
    frames = ['1', '5', '15', '44', '45', '47', '31', '50']
    indices = []
    for frame in frames:
        matching = [i for i, row in enumerate(all_rows)
                    if row['source'] == 'camera' and row['row']['split'] == 'train'
                    and str(row['row']['frame']) == frame]
        if len(matching) != 1:
            raise ValueError(f'Image train caméra {frame} manquante ou ambiguë.')
        indices.append(matching[0])
    rows = [all_rows[i] for i in indices]
    for row in rows:
        if row['row'].get('reviewed') != 'oui':
            raise ValueError('Utiliser uniquement les annotations caméra confirmées.')
        for kind in ('image', 'label'):
            actual = hashlib.sha256((Path(row['root']) / row['row'][kind]).read_bytes()).hexdigest()
            if actual != row[kind + '_sha256']:
                raise ValueError('Les sources ont changé depuis la préparation.')
    with np.load(source / 'train.npz') as archive:
        data = {name: archive[name][indices] for name in archive.files}
    truths = [row['boxes'] for row in rows]
    for i, boxes in enumerate(truths):
        for key, target in zip(('heat', 'distances', 'masks'), targets_from_boxes(boxes, 160, 120, 8)):
            np.testing.assert_array_equal(data[key][i], target)
    run.mkdir(parents=True)
    np.savez_compressed(run / 'memorization_inputs.npz', **data)
    protocol = {'diagnostic_only': True, 'evaluation_on_training_images': True,
                'generalization_measured': False, 'test_or_validation_used': False,
                'images': len(rows), 'persons': sum(map(len, truths)), 'frames': frames,
                'augmentations': False, 'initialization': 'Même ONNX initial que COCO 2000 v1',
                'initial_onnx_sha256': hashlib.sha256((source / 'initial_dense.onnx').read_bytes()).hexdigest(),
                'source_run': str(source), 'rows': rows, 'steps_max': args.steps,
                'learning_rate': args.learning_rate, 'batch_size': len(rows),
                'success_criterion': 'TP = personnes annotées et FP = FN = 0 au seuil 0,5 / IoU 0,5, sur le lot mémorisé seulement.'}
    (run / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    import aidge_core as ac
    import aidge_onnx
    from train_heatmap_v2_aidge import tensor, save_weights
    graph = aidge_onnx.load_onnx(str(source / 'initial_dense.onnx'))
    ac.expand_metaops(graph)
    graph.compile('cpu', ac.dtype.float32, dims=[[len(rows), 1, 120, 160]])
    scheduler = ac.SequentialScheduler(graph)
    parameters = {node.name(): node.get_operator().get_output(0) for node in ac.producers(graph)}
    views = {name: np.array(value, copy=False) for name, value in parameters.items()}
    moments = {name: (np.zeros_like(view), np.zeros_like(view)) for name, view in views.items()}
    targets = [data[key] for key in ('heat', 'distances', 'masks')]
    started = time.monotonic()
    best_key = None
    succeeded = False
    fields = ['step', 'loss', 'classification_loss', 'box_loss', 'tp', 'fp', 'fn', 'f1', 'seconds']
    with (run / 'history.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for step in range(args.steps + 1):
            output = scheduler.forward(data=[tensor(data['images'])])[0]
            raw = np.array(output, copy=True)
            loss, gradient, parts = detection_loss(raw, *targets, classification='focal')
            if not np.isfinite(loss) or not np.isfinite(gradient).all():
                raise RuntimeError('Loss ou gradient non fini.')
            if step % args.evaluate_every == 0 or step == args.steps:
                metrics, details = evaluate_boxes(raw, truths, threshold=.5, local_maximum=True,
                                                 width=160, height=120, stride=8)
                summary = {'step': step, 'loss': loss, 'classification_loss': parts['focal'],
                           'box_loss': parts['box'], **metrics, 'seconds': time.monotonic()-started}
                writer.writerow({key: summary[key] for key in fields})
                handle.flush()
                print(f"Étape {step}/{args.steps} : loss={loss:.4f}, F1 mémorisation={metrics['f1']:.1%}, TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}", flush=True)
                key = (metrics['f1'], -loss)
                if best_key is None or key > best_key:
                    best_key = key
                    save_weights(source / 'initial_dense.onnx', views, run / 'best_diagnostic.onnx')
                    np.save(run / 'best_logits.npy', raw)
                    (run / 'best_metrics.json').write_text(json.dumps({**summary, 'details': details}, indent=2))
                succeeded = metrics['fn'] == 0 and metrics['fp'] == 0
                progress = {'status': 'completed' if succeeded or step == args.steps else 'running',
                            'memorization_succeeded': succeeded, 'generalization_measured': False, **summary}
                (run / 'progress.json').write_text(json.dumps(progress, indent=2))
                if succeeded or step == args.steps:
                    break
            output.grad = tensor(gradient)
            scheduler.backward()
            grads = {name: np.array(value.grad, copy=True) for name, value in parameters.items()}
            if any(not np.isfinite(grad).all() for grad in grads.values()):
                raise RuntimeError('Gradient paramètre non fini.')
            norm = np.sqrt(sum(float(np.square(grad, dtype=np.float64).sum()) for grad in grads.values()))
            clip = min(1., 10./max(norm, 1e-12))
            for name, value in parameters.items():
                grad = grads[name]*clip
                first, second = moments[name]
                first *= .9
                first += .1*grad
                second *= .999
                second += .001*grad*grad
                views[name] -= args.learning_rate*(first/(1-.9**(step+1)))/(np.sqrt(second/(1-.999**(step+1)))+1e-8)
                value.grad.zeros()
    print('Diagnostic terminé. Mémorisation réussie :' + str(succeeded) + '. Cela ne mesure pas la généralisation.', flush=True)


if __name__ == '__main__':
    main()
