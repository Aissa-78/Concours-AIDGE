"""Explique les ratés sans changer les poids ni appliquer un nouveau seuil."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from person_detection_math import box_iou, decode_boxes, evaluate_boxes


def metrics(outputs, truths, threshold):
    return evaluate_boxes(outputs, truths, threshold=threshold,
                          local_maximum=True, width=160, height=120, stride=8)[0]


def panel(image, title, boxes=()):
    canvas = cv2.cvtColor(cv2.resize(image, (480, 360), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    for box, text, color in boxes:
        x1, y1, x2, y2 = [round(float(v) * 3) for v in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, text, (max(0, x1), max(16, y1)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
    banner = np.zeros((30, 480, 3), np.uint8)
    cv2.putText(banner, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    return np.concatenate([banner, canvas])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.absolute()
    metadata = json.loads((run / 'dataset.json').read_text())
    source = Path(metadata['test_manifest']).parent
    outputs = np.load(run / 'camera_selected_logits.npy')
    rows = [r for r in csv.DictReader((source / 'manifest.csv').open())
            if r['split'] in ('validation', 'test')]
    if outputs.shape != (len(rows), 5, 15, 20):
        raise ValueError('Sorties incompatibles ; générer le rapport caméra avant ce diagnostic.')
    truths = [[tuple(map(float, line.split()[1:])) for line in (source / r['label']).read_text().splitlines()]
              for r in rows]
    val_rows = json.loads((run / 'validation.json').read_text())
    val_outputs = np.load(run / 'best_validation_logits.npy')
    if len(val_rows) != len(val_outputs):
        raise ValueError('Validation incohérente.')
    # Choisir le seuil sur les deux sources de validation, jamais sur le test.
    scan = []
    for threshold in np.arange(.05, .901, .025):
        threshold = round(float(threshold), 3)
        item = {'threshold': threshold}
        for name in ('camera', 'historique'):
            indices = [i for i, row in enumerate(val_rows) if row['source'] == name]
            item[name] = metrics(val_outputs[indices], [val_rows[i]['boxes'] for i in indices], threshold)
        scan.append(item)
    def selection_key(item):
        scores = [item[name]['f1'] for name in ('camera', 'historique')]
        return min(scores), sum(scores) / 2, item['threshold']
    selected = max(scan, key=selection_key)
    test_indices = [i for i, row in enumerate(rows) if row['split'] == 'test']
    test_outputs = outputs[test_indices]
    test_truths = [truths[i] for i in test_indices]
    result = {'model_sha256': hashlib.sha256((run / 'best_dense.onnx').read_bytes()).hexdigest(),
              'weights_changed': False, 'threshold_applied': False,
              'threshold_selection': 'Validation seulement : maximiser le minimum des F1 caméra/historique, puis leur moyenne.',
              'validation_scan': scan, 'validation_selected': selected,
              'test_fixed_050': metrics(test_outputs, test_truths, .5),
              'test_validation_threshold': metrics(test_outputs, test_truths, selected['threshold']),
              'warning': 'Petit test déjà examiné. Diagnostic exploratoire, pas une preuve indépendante ni un réglage pour la production.',
              'cases': []}
    destination = run / 'diagnostic_rates'
    destination.mkdir(exist_ok=True)
    cards = []
    for index in test_indices:
        row, output = rows[index], outputs[index]
        if not truths[index]:
            continue
        image = cv2.imread(str(source / row['image']), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(row['image'])
        candidates = decode_boxes(output, threshold=.001, local_maximum=True, width=160, height=120, stride=8)
        expected = []
        for box_index, (cx, cy, bw, bh) in enumerate(truths[index]):
            gt = [(cx-bw/2)*160, (cy-bh/2)*120, (cx+bw/2)*160, (cy+bh/2)*120]
            expected.append((gt, 'Annotation', (255, 255, 255)))
            # Meilleur IoU choisi à l'aide de l'annotation : diagnostic uniquement.
            candidate = max(candidates, key=lambda item: box_iou(item[1], gt)) if candidates else None
            result['cases'].append({'frame': row['frame'], 'image': row['image'], 'person_index': box_index,
                                    'truth_box_pixels': gt,
                                    'max_image_score': float(1 / (1 + np.exp(-np.clip(output[4].max(), -80, 80)))),
                                    'candidate_with_best_iou': None if candidate is None else {
                                        'score': candidate[0], 'box_pixels': candidate[1].tolist(),
                                        'iou': float(box_iou(candidate[1], gt))},
                                    'candidate_warning': 'Choisi avec la vérité terrain ; ne constitue pas une prédiction sélectionnable en production.'})
        predictions = [(box, f'Score {score:.3f}', (0, 180, 255)) for score, box in candidates[:3]]
        heat = 1 / (1 + np.exp(-np.clip(output[4], -80, 80)))
        heat = cv2.applyColorMap(np.round(cv2.resize(heat, (480, 360), interpolation=cv2.INTER_NEAREST)*255).astype(np.uint8), cv2.COLORMAP_JET)
        banner = np.zeros((30, 480, 3), np.uint8)
        cv2.putText(banner, 'Scores : bleu=0, rouge=1 (echelle fixe)', (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        montage = np.concatenate([panel(image, 'Annotation humaine', expected),
                                  panel(image, '3 meilleures propositions, meme sous 0.5', predictions),
                                  np.concatenate([banner, heat])], axis=1)
        name = f"image_{row['frame']}.png"
        if not cv2.imwrite(str(destination / name), montage):
            raise IOError(name)
        cards.append(f'<h2>Image {html.escape(row["frame"])}</h2><img src="{name}" style="width:100%">')
    (destination / 'diagnostic.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    (destination / 'rapport.html').write_text('<!doctype html><meta charset="utf-8"><title>Diagnostic des ratés</title><body style="font:18px system-ui;margin:30px"><h1>Pourquoi les trois personnes sont manquées</h1><p>Blanc : annotation humaine. Orange : trois propositions de plus fort score, même rejetées au seuil 0,50. Carte de scores à échelle fixe 0–1. Aucune modification des poids ni du seuil actif.</p>' + ''.join(cards) + '<p>Les causes chiffrées et le bilan du seuil sélectionné uniquement sur validation sont dans diagnostic.json.</p>', encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key != 'validation_scan'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
