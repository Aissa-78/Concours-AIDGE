"""Évalue le modèle QQVGA sélectionné, sans réutiliser le test pour le régler."""
import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np

ROOT = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from person_detection_math import evaluate_boxes


def infer(model, images, destination, prefix, aidge_python=None):
    inputs = destination / (prefix + '_inputs.npz')
    outputs = destination / (prefix + '_logits.npy')
    np.savez_compressed(inputs, images=images)
    python = aidge_python or ROOT / '.venv-aidge-colibry' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    subprocess.run([str(python), str(ROOT / 'scripts/infer_person_qqvga_aidge.py'),
                    '--model', str(model), '--inputs', str(inputs), '--output', str(outputs),
                    '--batch-size', '4'], env=env, cwd=ROOT, check=True)
    return np.load(outputs)


def preview(image, detail, title):
    canvas = cv2.cvtColor(cv2.resize(image, (320, 240), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    # Rectangles vrais blancs, prédictions correctes vertes, autres oranges.
    for box in detail['truth_boxes_pixels']:
        x1, y1, x2, y2 = (round(float(v) * 2) for v in box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 1)
    for row in detail['predictions']:
        x1, y1, x2, y2 = (round(float(v) * 2) for v in row['box'])
        color = (0, 190, 0) if row['matched_truth'] is not None else (0, 150, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, f"{row['score']:.2f}", (x1, max(12, y1)), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
    banner = np.zeros((28, 320, 3), dtype=np.uint8)
    cv2.putText(banner, title, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, .46, (255, 255, 255), 1)
    return np.concatenate([banner, canvas])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--aidge-python', type=Path, help='Python Aidge à utiliser, notamment .venv-aidge-linux/bin/python.')
    args = parser.parse_args()
    run = args.run_dir.absolute()
    if json.loads((run / 'progress.json').read_text())['status'] != 'completed':
        raise RuntimeError('Attendre la fin de la sélection du modèle.')
    training = json.loads((run / 'training.json').read_text())
    info = json.loads((run / 'dataset.json').read_text())
    source = ROOT / 'dataset/camera_reelle_20261117_v1'
    manifest = list(csv.DictReader((source / 'manifest.csv').open()))
    assert all(row['reviewed'] == 'oui' for row in manifest)
    selected_model = run / 'best_dense.onnx'
    shutil.copy2(selected_model, run / 'best.onnx')
    previous_model = ROOT / 'runs/person_profile_mix_qvga_v1/best.onnx'
    camera_rows = [row for row in manifest if row['split'] in ('validation', 'test')]
    images, truths = [], []
    for row in camera_rows:
        image = cv2.imread(str(source / row['image']), cv2.IMREAD_GRAYSCALE)
        assert image.shape == (120, 160)
        images.append(image[None].astype(np.float32) / 255)
        truths.append([tuple(map(float, line.split()[1:])) for line in (source / row['label']).read_text().splitlines()])
    images = np.stack(images)
    new = infer(selected_model, images, run, 'camera_selected', args.aidge_python)
    old_images = np.stack([cv2.resize(image[0], (320, 240), interpolation=cv2.INTER_LINEAR)[None] for image in images])
    old = infer(previous_model, old_images, run, 'camera_previous', args.aidge_python)
    summary = {'threshold': .5, 'match_iou': .5, 'selected_epoch': training['selected_epoch'],
               'model_sha256': hashlib.sha256(selected_model.read_bytes()).hexdigest(),
               'previous_model_sha256': hashlib.sha256(previous_model.read_bytes()).hexdigest(),
               'geometry_changed': True, 'comparison_changes_training_and_stride': True,
               'hardware_execution_validated': False, 'counting_accuracy_measured': False,
               'same_camera_session': True, 'test_used_for_checkpoint_selection': False,
               'historical_validation': training['validation_metrics']['historique']}
    previews = run / 'apercus'
    previews.mkdir(exist_ok=True)
    cards, result_rows = [], []
    for split in ('validation', 'test'):
        indices = [i for i, row in enumerate(camera_rows) if row['split'] == split]
        expected = [truths[i] for i in indices]
        before, before_details = evaluate_boxes(old[indices], expected, width=320, height=240, stride=16, local_maximum=True)
        after, after_details = evaluate_boxes(new[indices], expected, width=160, height=120, stride=8, local_maximum=True)
        summary[split] = {'images': len(indices), 'persons': sum(map(len, expected)), 'before': before, 'after': after}
        result_rows.append(f'<tr><td>{split}</td><td>{len(indices)}</td><td>{sum(map(len, expected))}</td><td>{before["f1"]:.1%}</td><td>{after["f1"]:.1%}</td><td>{after["tp"]}</td><td>{after["fp"]}</td><td>{after["fn"]}</td></tr>')
        for index, b, a in zip(indices, before_details, after_details):
            # Normaliser les deux aperçus à la géométrie caméra native.
            b = {**b, 'predictions': [{**row, 'box': [v / 2 for v in row['box']]} for row in b['predictions']]}
            for detail in (b, a):
                detail['truth_boxes_pixels'] = [[(cx-bw/2)*160, (cy-bh/2)*120, (cx+bw/2)*160, (cy+bh/2)*120] for cx, cy, bw, bh in truths[index]]
            image = np.round(images[index, 0] * 255).astype(np.uint8)
            comparison = np.concatenate([preview(image, b, 'Avant : V1 320x240'), preview(image, a, 'Apres : modele 160x120')], axis=1)
            filename = f'{split}_{camera_rows[index]["frame"]}.png'
            cv2.imwrite(str(previews / filename), comparison)
            cards.append(f'<article><h3>{split} — image {camera_rows[index]["frame"]}</h3><p>Après : {a["tp"]} retrouvée(s), {a["fp"]} fausse(s), {a["fn"]} manquée(s).</p><img src="apercus/{html.escape(filename)}"></article>')
    (run / 'camera_metrics.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (run / 'rapport.html').write_text('''<!doctype html><html lang="fr"><meta charset="utf-8"><title>FlowSense : entraînement caméra réelle</title><style>body{font:18px system-ui;max-width:1100px;margin:35px auto;padding:15px}p{line-height:1.5}table{border-collapse:collapse;width:100%}td,th{padding:12px;border:1px solid #ddd}img{width:100%;max-width:960px}article{margin-top:30px;border-top:1px solid #aaa}.note{background:#fff1d6;padding:18px}</style><h1>Notre modèle adapté à la caméra réelle</h1><p>Entrée 160 × 120 en gris. Les effectifs propres à cet essai sont décrits dans dataset.json : captures réelles, images COCO et anciennes images de profil. Mélange équilibré caméra/historique, augmentations uniquement en entraînement. Apprentissage Aidge CPU FP32, sans poids YOLO.</p>'''+f'<p>Meilleur checkpoint : époque {training["selected_epoch"]}, après {training["epochs_completed"]} époques. Seuil fixé à 0,50, correspondance rectangle à IoU ≥ 0,50.</p>'+'''<p class="note">F1 de détection, pas pourcentage de passages correctement comptés. Validation caméra : seulement 8 images. Test : 10 images et 3 présences annotées, même salle et même session. Ce lot n’a pas servi à sélectionner les poids. Les images et annotations ont déjà été vues pour préparer les données : ce n’est pas un test final aveugle dans un nouveau lieu.</p><table><tr><th>Lot</th><th>Images</th><th>Présences</th><th>F1 avant</th><th>F1 après</th><th>Retrouvées</th><th>Fausses</th><th>Manquées</th></tr>'''+''.join(result_rows)+'''</table><p>Avant : V1 originale sur images caméra agrandies vers 320 × 240. Après : modèle adapté, quatrième convolution à stride 1 et grille de sortie à stride 8. La comparaison inclut le changement de géométrie et l’entraînement. Elle ne mesure pas le seul effet des nouvelles photos.</p><p>Blanc : annotation humaine. Vert : prédiction associée à une annotation. Orange : prédiction sans correspondance. Les images train restent hors de ces aperçus. Le modèle n’est pas encore quantifié ni déployé sur Colibry. La fenêtre caméra ne fait pas encore tourner ce détecteur.</p>'''+''.join(cards), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
