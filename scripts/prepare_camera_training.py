"""Prépare un mélange caméra réelle / historique (COCO et profils), sans test."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from person_detection_math import targets_from_boxes, detection_loss
from person_training_data import assert_disjoint
from profile_detector import ProfileDetector


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def detection_record(root, row, source, **metadata):
    image = root / row['image']
    label = root / row['label']
    boxes = []
    for line in label.read_text().splitlines():
        values = line.split()
        if len(values) != 5 or values[0] != '0':
            raise ValueError(f'Une seule classe attendue (0 = personne) dans {label}.')
        boxes.append(tuple(map(float, values[1:])))
    return {'source': source, 'root': str(root), 'row': row, 'boxes': boxes,
            'image_sha256': digest(image), 'label_sha256': digest(label), **metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--coco-dir', type=Path, help='Lot COCO supplémentaire avec manifest.csv et labels.')
    parser.add_argument('--extra-camera-dir', type=Path, action='append', default=[],
                        help='Lot caméra annoté avec manifest.csv et splits train/validation.')
    parser.add_argument('--spatial-augmentation', action='store_true', help='Activer zooms et recadrages pendant le train uniquement.')
    parser.add_argument('--camera-only', action='store_true', help='Diagnostic : apprendre les seules photos caméra ; conserver la validation commune.')
    args = parser.parse_args()
    run = args.run_dir.absolute()
    if run.exists():
        raise FileExistsError('Choisir un nouveau dossier.')
    camera = ROOT / 'dataset/camera_reelle_20261117_v1'
    camera_rows = list(csv.DictReader((camera / 'manifest.csv').open()))
    if len(camera_rows) != 54 or any(row['reviewed'] != 'oui' for row in camera_rows):
        raise ValueError('Les 54 annotations caméra doivent être confirmées.')
    test_hashes = {row['source_sha256'] for row in camera_rows if row['split'] == 'test'}
    records = []
    manifests = {str(camera / 'manifest.csv'): digest(camera / 'manifest.csv')}
    for split in ('train', 'validation'):
        historical = json.loads((ROOT / f'runs/person_profile_mix_qvga_v2/{split}.json').read_text())
        for row in historical:
            if args.camera_only and split == 'train':
                continue
            item = {**row, 'source': 'historique', 'previous_source': row['source']}
            # Retrouver les données partagées même si le dossier vient d'un Mac.
            old_root = str(row['root']).replace('\\', '/')
            if '/dataset/' in old_root:
                item['root'] = str(ROOT / 'dataset' / old_root.split('/dataset/', 1)[1])
            records.append(item)
            manifest = Path(item['root']) / 'manifest.csv'
            manifests[str(manifest)] = digest(manifest)
        for row in camera_rows:
            if row['split'] != split:
                continue
            records.append(detection_record(camera, row, 'camera'))
    for root in args.extra_camera_dir:
        root = root.absolute()
        manifest = root / 'manifest.csv'
        manifests[str(manifest)] = digest(manifest)
        with manifest.open() as handle:
            for row in csv.DictReader(handle):
                if row['split'] not in ('train', 'validation'):
                    raise ValueError(f'Split caméra inattendu dans {manifest}.')
                if row['reviewed'] not in ('oui', 'assistant_visual'):
                    raise ValueError(f'Annotation à vérifier pour {row["image"]}.')
                records.append(detection_record(root, row, 'camera'))
    if args.coco_dir:
        coco = args.coco_dir.absolute()
        manifest = coco / 'manifest.csv'
        manifests[str(manifest)] = digest(manifest)
        for row in csv.DictReader(manifest.open()):
            if row['split'] not in ('train', 'validation'):
                raise ValueError('Split COCO inattendu.')
            if args.camera_only and row['split'] == 'train':
                continue
            records.append(detection_record(coco, row, 'historique', previous_source='coco_2000'))
    assert_disjoint(records)
    if test_hashes & {row['image_sha256'] for row in records}:
        raise ValueError('Image de test présente dans le mélange.')
    run.mkdir(parents=True)
    info = {'profile_run': True, 'seed': 20261117, 'channels': 24, 'blocks': 3,
            'profile_spatial_augmentation': args.spatial_augmentation,
            'camera_only_training': args.camera_only,
            'architecture': 'profile_context', 'residual': False,
            'geometry': {'width': 160, 'height': 120, 'stride': 8},
            'input_shape': [1, 120, 160], 'output_shape': [5, 15, 20],
            'classification': 'focal', 'local_maximum': True, 'threshold': .5,
            'initialization': 'Poids v1, quatrième convolution à stride 1 pour QQVGA. Adam neuf.',
            'manifests': manifests, 'rows': records,
            'validation_warning': 'Les images caméra de validation peuvent venir de la même session de capture que le train. Vérifier la généralisation sur des vidéos réservées.',
            'reserved_camera_test_sha256': sorted(test_hashes),
            'test_manifest': str(camera / 'manifest.csv')}
    for split in ('train', 'validation'):
        selected = [row for row in records if row['row']['split'] == split]
        arrays, targets = [], []
        for row in selected:
            image_path = Path(row['root']) / row['row']['image']
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(str(image_path))
            if image.shape != (120, 160):
                image = cv2.resize(image, (160, 120), interpolation=cv2.INTER_AREA)
            arrays.append(image[None].astype(np.float32) / 255)
            targets.append(targets_from_boxes(row['boxes'], 160, 120, 8))
        np.savez_compressed(run / f'{split}.npz', images=np.stack(arrays),
                            heat=np.stack([v[0] for v in targets]),
                            distances=np.stack([v[1] for v in targets]),
                            masks=np.stack([v[2] for v in targets]))
        (run / f'{split}.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2))
        info[split] = {name: {'images': sum(row['source'] == name for row in selected),
                             'presences': sum(len(row['boxes']) for row in selected if row['source'] == name)}
                       for name in ('camera', 'historique') if any(row['source'] == name for row in selected)}
    torch.set_num_threads(4)
    torch.manual_seed(info['seed'])
    model = ProfileDetector(output_stride=8).eval()
    checkpoint = ROOT / 'runs/person_profile_mix_qvga_v1/best_dense.npz'
    with np.load(checkpoint) as weights, torch.no_grad():
        for name, value in model.named_parameters():
            value.copy_(torch.from_numpy(weights[name.replace('.', '_')]))
    info['initial_checkpoint_sha256'] = digest(checkpoint)
    train = json.loads((run / 'train.json').read_text())
    indices = np.array([next(i for i, row in enumerate(train) if row['boxes']),
                        next(i for i, row in enumerate(train) if not row['boxes'])])
    with np.load(run / 'train.npz') as values:
        images = values['images'][indices]
        output = model(torch.from_numpy(images))
        if tuple(output.shape) != (2, 5, 15, 20):
            raise ValueError('Géométrie incorrecte.')
        _, gradient, _ = detection_loss(output.detach().numpy(), *(values[key][indices] for key in ('heat', 'distances', 'masks')))
    output.backward(torch.from_numpy(gradient))
    np.savez_compressed(run / 'gradient_reference.npz', indices=indices,
                        logits=output.detach().numpy(),
                        **{'grad_' + name.replace('.', '_'): value.grad.numpy() for name, value in model.named_parameters()})
    np.savez_compressed(run / 'parameter_masks.npz')
    torch.onnx.export(model, torch.from_numpy(images), str(run / 'initial_dense.onnx'),
                      input_names=['image'], output_names=['detections'], opset_version=17, dynamo=False)
    shutil.copy2(run / 'initial_dense.onnx', run / 'baseline_dense.onnx')
    info['parameters_compact'] = sum(v.numel() for v in model.parameters())
    (run / 'dataset.json').write_text(json.dumps(info, ensure_ascii=False, indent=2))
    print(json.dumps({key: info[key] for key in ('geometry', 'train', 'validation', 'parameters_compact')}, indent=2))


if __name__ == '__main__':
    main()
