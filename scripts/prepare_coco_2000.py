"""Prépare 2 000 images COCO annotées en gris QQVGA, sans entraîner de modèle."""
import argparse
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import zipfile

import cv2

ROOT = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from camera_preprocess import prepare_camera_frame, CameraGeometry
from coco_sideview_selection import ALLOWED_LICENSES, SPORTS, WALKING, PLACE
from person_detection_math import targets_from_boxes
from download_coco_person_subset import BASE, download


def plan(root, archive_path, seed):
    with zipfile.ZipFile(archive_path) as archive:
        data = json.loads(archive.read('annotations/instances_train2017.json'))
        keypoints = json.loads(archive.read('annotations/person_keypoints_train2017.json'))
        captions = json.loads(archive.read('annotations/captions_train2017.json'))
    people, points, texts = defaultdict(list), {}, defaultdict(list)
    for ann in data['annotations']:
        if ann['category_id'] == 1:
            people[ann['image_id']].append(ann)
    for ann in keypoints['annotations']:
        points[ann['id']] = ann
    for ann in captions['annotations']:
        texts[ann['image_id']].append(ann['caption'])
    excluded = set()
    for name in ['dataset/coco_person_subset/SOURCE.json', 'dataset/coco_profile_fullbody_v1/SOURCE.json']:
        source = ROOT / name
        if source.exists():
            old = json.loads(source.read_text())
            for image in old.get('images', []):
                excluded.add(image.get('id', image.get('image', {}).get('id')))
    excluded.discard(None)
    old_manifest = ROOT / 'dataset/profile_mix_v1/coco/manifest.csv'
    if old_manifest.exists():
        for row in csv.DictReader(old_manifest.open()):
            excluded.add(int(Path(row['image']).stem))
    positive, negative = [], []
    for info in data['images']:
        if info['id'] in excluded or info['license'] not in ALLOWED_LICENSES:
            continue
        anns = people[info['id']]
        caption = ' '.join(texts[info['id']])
        if SPORTS.search(caption):
            continue
        w, h = info['width'], info['height']
        scale = min(160 / w, 120 / h)
        rw, rh = round(w * scale), round(h * scale)
        geometry = CameraGeometry(160, 120, rw, rh, (160-rw)//2, (120-rh)//2)
        boxes = []
        if anns:
            if len(anns) > 3 or any(a.get('iscrowd') for a in anns):
                continue
            valid, upright = True, False
            for ann in anns:
                x, y, bw, bh = ann['bbox']
                x1, y1, x2, y2 = max(0, x), max(0, y), min(w, x+bw), min(h, y+bh)
                if (x2-x1)*scale < 8 or (y2-y1)*scale < 22:
                    valid = False
                    break
                boxes.append(geometry.transform_box(((x1+x2)/(2*w), (y1+y2)/(2*h), (x2-x1)/w, (y2-y1)/h)))
                kp = points.get(ann['id'], {}).get('keypoints', [])
                if len(kp) == 51:
                    visible = sum(kp[i+2] > 0 for i in range(0, 51, 3))
                    ankles = [kp[i*3+1] for i in (15, 16) if kp[i*3+2] > 0]
                    shoulders = [kp[i*3+1] for i in (5, 6) if kp[i*3+2] > 0]
                    if visible >= 8 and ankles and shoulders and min(ankles) > max(shoulders) + .4*bh and .12 <= bw/bh <= .85:
                        upright = True
            if not valid or not upright:
                continue
        elif not PLACE.search(caption):
            continue
        try:
            targets_from_boxes(boxes, 160, 120, 8)
        except ValueError:
            continue  # Écarter l'image entière, jamais effacer une personne.
        item = {'image': info, 'boxes': boxes, 'annotations': anns,
                'captions': texts[info['id']], 'upright_keypoint_heuristic': bool(anns),
                'walking_caption': bool(WALKING.search(caption))}
        (positive if anns else negative).append(item)
    rng = random.Random(seed)
    rng.shuffle(positive)
    rng.shuffle(negative)
    positive.sort(key=lambda row: row['walking_caption'], reverse=True)
    if len(positive) < 1300 or len(negative) < 700:
        raise ValueError(f'Lot insuffisant : {len(positive)} positifs, {len(negative)} négatifs.')
    selected = positive[:1300] + negative[:700]
    positives, negatives = selected[:1300], selected[1300:]
    rng.shuffle(positives)
    rng.shuffle(negatives)
    rows = []
    for split, items in [('train', positives[:1040] + negatives[:560]),
                         ('validation', positives[1040:] + negatives[560:])]:
        rng.shuffle(items)
        for item in items:
            rows.append({**item, 'split': split})
    result = {'seed': seed, 'available_positive': len(positive), 'available_negative': len(negative),
              'excluded_previous_ids': sorted(excluded), 'licenses': data['licenses'], 'rows': rows}
    (root / 'plan.json').write_text(json.dumps(result, ensure_ascii=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'dataset/coco_qqvga_2000_v1')
    parser.add_argument('--annotations-zip', type=Path, default=ROOT / 'dataset/coco_person_subset/annotations_trainval2017.zip')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20261117)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers doit être positif.')
    root = args.output_dir.absolute()
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / 'plan.json'
    if not plan_path.exists():
        download(BASE + '/annotations/annotations_trainval2017.zip', args.annotations_zip)
    selection = json.loads(plan_path.read_text()) if plan_path.exists() else plan(root, args.annotations_zip, args.seed)
    print('Candidats disponibles :', selection['available_positive'], selection['available_negative'], flush=True)
    if args.plan_only:
        return
    def fetch(item):
        info, split = item['image'], item['split']
        original = root / 'originals' / info['file_name']
        download(BASE + '/train2017/' + info['file_name'], original)
        frame = cv2.imread(str(original))
        if frame is None or frame.shape[:2] != (info['height'], info['width']):
            raise ValueError(f'Image illisible : {original}')
        gray, _ = prepare_camera_frame(frame, 160, 120)
        image = Path(split) / 'images' / f"{info['id']:012d}.png"
        label = Path(split) / 'labels' / f"{info['id']:012d}.txt"
        (root / image.parent).mkdir(parents=True, exist_ok=True)
        (root / label.parent).mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(root / image), gray):
            raise IOError('Écriture image impossible.')
        (root / label).write_text(''.join('0 ' + ' '.join(f'{v:.8f}' for v in box) + '\n' for box in item['boxes']))
        return {'split': split, 'video': f"COCO2017-{info['id']}", 'frame': 0,
                'image': image.as_posix(), 'label': label.as_posix(), 'reviewed': 'annotation_coco',
                'group_id': f"COCO2017-{info['id']}", 'coco_id': info['id'],
                'license_id': info['license'], 'original_sha256': hashlib.sha256(original.read_bytes()).hexdigest()}
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(fetch, selection['rows']), 1):
            rows.append(row)
            if i % 50 == 0:
                print(f'COCO vérifié : {i}/2000', flush=True)
    hashes = defaultdict(set)
    for row in rows:
        hashes[row['original_sha256']].add(row['split'])
    if any(len(value) > 1 for value in hashes.values()):
        raise ValueError('Doublon exact entre train et validation.')
    with (root / 'manifest.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    licenses = {value['id']: value for value in selection['licenses']}
    with (root / 'ATTRIBUTIONS.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['coco_id', 'source', 'flickr_url', 'license', 'license_url'])
        for item in selection['rows']:
            info = item['image']; lic = licenses[info['license']]
            writer.writerow([info['id'], info.get('coco_url'), info.get('flickr_url'), lic['name'], lic['url']])
    metadata = {'dataset': 'COCO train2017, lot QQVGA de 2000 images', 'source': 'https://cocodataset.org/#download',
                'split_counts': dict(Counter(row['split'] for row in rows)),
                'geometry': [160, 120], 'grayscale': True, 'all_person_annotations_preserved': True,
                'selection': '1 à 3 personnes non-crowd, taille minimale et ancres valides ; au moins un corps debout suggéré par keypoints. 35 % sans personne annotée. Proportions conservées avec bandes grises.',
                'review_warning': 'Présélection automatique, annotations officielles. Pas de revue visuelle individuelle ni garantie de vue de profil pour les 2000 images.',
                'split_warning': '1600 train, 400 validation issus de train2017, pas un score COCO officiel. Les anciens IDs COCO sont exclus.',
                'training_performed': False, 'licenses': selection['licenses'], 'seed': selection['seed']}
    (root / 'SOURCE.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print('Prêt :', root, metadata['split_counts'], flush=True)


if __name__ == '__main__':
    main()
