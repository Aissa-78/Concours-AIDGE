"""Présélection COCO corps entiers : heuristique, pas vérité d'orientation."""

import re

import numpy as np

ALLOWED_LICENSES = {4, 5, 7, 8}
SPORTS = re.compile(r"\b(riding|skiing|skis|snowboarding|surfing|skateboarding|playing tennis|playing baseball|playing soccer|playing football|baseball game|tennis court|baseball field|surfboard|skateboard|snowboard|game controller|video game|wii|bathroom|toilet|bathtub|bikini|swimsuit|swimming|naked|nude|bedroom)\b", re.I)
WALKING = re.compile(r"\b(walk|walks|walking|pedestrian|standing|stands)\b", re.I)
PLACE = re.compile(r"\b(street|sidewalk|crosswalk|walkway|hallway|door|entrance|room|building|station|platform)\b", re.I)


def inspect_person(annotation, width, height):
    if width <= 0 or height <= 0:
        return None
    if annotation.get("iscrowd", 0) or len(annotation.get("keypoints", [])) != 51:
        return None
    x, y, w, h = annotation["bbox"]
    if not np.isfinite([x, y, w, h]).all() or min(w, h) <= 0:
        return None
    if min(x/width, y/height, (width-x-w)/width, (height-y-h)/height) < .015:
        return None
    scale = min(320/width, 240/height)
    if h*scale < 65 or w*scale < 16 or not .12 <= w/h <= .7:
        return None
    points = np.asarray(annotation["keypoints"], dtype=np.float64).reshape(17, 3)
    if not np.isfinite(points).all() or not np.isin(points[:, 2], [0, 1, 2]).all():
        return None
    labeled, visible = points[:, 2] > 0, points[:, 2] == 2
    if labeled.sum() < 13 or visible.sum() < 9:
        return None
    if not labeled[[5, 6, 11, 12, 13, 14, 15, 16]].all() or not visible[[15, 16]].any():
        return None
    head = points[:5][labeled[:5]]
    if not len(head):
        return None
    head_y = float(head[:, 1].min())
    shoulder_y, hip_y, ankle_y = (float(points[indices, 1].mean()) for indices in ([5, 6], [11, 12], [15, 16]))
    if not (head_y < shoulder_y < hip_y < ankle_y):
        return None
    if head_y > y+.25*h or ankle_y < y+.8*h or ankle_y-head_y < .65*h or ankle_y-hip_y < .3*h:
        return None
    eye_count, ear_count = int(visible[1:3].sum()), int(visible[3:5].sum())
    # Une seule oreille/un seul œil visible et des épaules resserrées suggèrent
    # un profil, mais peuvent aussi traduire une occlusion ou une annotation.
    profile = .4*(ear_count == 1)+.4*(eye_count == 1)+.2*(abs(points[5, 0]-points[6, 0])/h < .18)
    return {"profile_proxy": float(profile), "visible_keypoints": int(visible.sum()),
            "qvga_height_pixels": float(h*scale), "qvga_width_pixels": float(w*scale),
            "orientation": "profil_probable" if profile >= .6 else "autre_vue_debout"}


def inspect_image(image, annotations, captions):
    if image["license"] not in ALLOWED_LICENSES or not 1 <= len(annotations) <= 2:
        return None
    text = " ".join(captions)
    if SPORTS.search(text):
        return None
    people = [inspect_person(a, image["width"], image["height"]) for a in annotations]
    # Ne jamais effacer un petit personnage, un corps coupé ou une foule pour
    # transformer ces pixels humains en fond : on écarte alors toute l'image.
    if any(person is None for person in people):
        return None
    for i, a in enumerate(annotations):
        for b in annotations[i+1:]:
            x, y, w, h = a["bbox"]
            bx, by, bw, bh = b["bbox"]
            intersection = max(0, min(x+w, bx+bw)-max(x, bx))*max(0, min(y+h, by+bh)-max(y, by))
            if intersection/max(min(w*h, bw*bh), 1) > .15:
                return None
    context = 1.0*bool(WALKING.search(text))+.4*bool(PLACE.search(text))
    profile = max(p["profile_proxy"] for p in people)
    return {"people": people, "profile_proxy": profile, "context_score": context,
            "selection_score": context+profile+min(p["visible_keypoints"] for p in people)/34,
            "orientation": "profil_probable" if profile >= .6 else "autre_vue_debout"}
