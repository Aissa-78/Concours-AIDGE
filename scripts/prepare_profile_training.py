"""Prépare un essai Aidge QVGA COCO + prises de profil, sans utiliser le test."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from person_detection_math import targets_from_boxes, detection_loss
from person_training_data import assert_disjoint
from profile_detector import ProfileDetector
from profile_training_data import WIDTH, HEIGHT, STRIDE


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,default=ROOT/"dataset/profile_mix_v1")
    parser.add_argument("--run-dir",type=Path,default=ROOT/"runs/person_profile_mix_qvga_v1")
    parser.add_argument("--seed",type=int,default=20261103)
    args=parser.parse_args()
    run=args.run_dir
    if run.exists(): raise SystemExit("Choisir un nouveau dossier pour conserver les essais existants.")
    source=json.loads((args.data_dir/"SOURCE.json").read_text())
    test_hashes={v["sha256"] for v in source["videos"] if v["split"]=="test"}
    rows,arrays,targets,manifests=[],[],[],{}
    split_groups={"train":set(),"validation":set()}
    for name in ("coco","portes"):
        root=(args.data_dir/name).resolve()
        manifests[str(root/"manifest.csv")]=digest(root/"manifest.csv")
        for row in csv.DictReader((root/"manifest.csv").open()):
            if row["split"] not in split_groups or row["reviewed"] not in ("annotation_coco","annotation_assistant","oui"):
                raise ValueError("Image non relue ou split non autorisé.")
            if row["group_id"] in test_hashes: raise ValueError("Le test est interdit dans l'apprentissage / sélection.")
            split_groups[row["split"]].add((name,row["group_id"]))
            paths={key:(root/row[key]).resolve() for key in ("image","label")}
            if any(not path.is_relative_to(root) or not path.is_file() for path in paths.values()):
                raise ValueError("Fichier hors dataset ou absent.")
            boxes=[]
            for line in paths["label"].read_text().splitlines():
                parts=line.split()
                if len(parts)!=5 or parts[0]!="0": raise ValueError("Label de détection invalide.")
                boxes.append(tuple(map(float,parts[1:])))
            frame=cv2.imread(str(paths["image"]),cv2.IMREAD_UNCHANGED)
            if frame is None or frame.shape!=(HEIGHT,WIDTH) or frame.dtype!=np.uint8:
                raise ValueError("Image grise uint8 320x240 attendue.")
            rows.append({"source":name,"root":str(root),"row":row,"boxes":boxes,
                         "image_sha256":digest(paths["image"]),"label_sha256":digest(paths["label"])})
            arrays.append(frame[None].astype(np.float32)/255)
            targets.append(targets_from_boxes(boxes,WIDTH,HEIGHT,STRIDE))
    assert_disjoint(rows)
    if split_groups["train"] & split_groups["validation"]: raise ValueError("Une prise traverse les lots.")
    local_review = {split: dict(Counter(r["row"]["reviewed"] for r in rows
                                     if r["source"] == "portes" and r["row"]["split"] == split))
                    for split in ("train", "validation")}
    local_validation = [r for r in rows if r["source"] == "portes" and r["row"]["split"] == "validation"]
    human_validation = bool(local_validation) and all(r["row"]["reviewed"] == "oui" for r in local_validation)
    run.mkdir(parents=True)
    info={"profile_run":True,"seed":args.seed,"channels":24,"blocks":3,"architecture":"profile_context","residual":False,
          "geometry":{"width":WIDTH,"height":HEIGHT,"stride":STRIDE},"input_shape":[1,HEIGHT,WIDTH],
          "output_shape":[5,HEIGHT//STRIDE,WIDTH//STRIDE],"classification":"focal","local_maximum":True,
          "initialization":"Poids aléatoires, entraînement conjoint COCO + vidéos, sans poids YOLO",
          "manifests":manifests,"rows":rows,"threshold":.5,"iou_threshold":.5,
          "local_annotation_review":local_review,"human_reviewed_local_validation":human_validation,
          "validation_warning":("Annotations locales proposées par YOLO et validées par l'utilisateur ; même lieu dans toutes les prises, validation de développement non indépendante." if human_validation else source["validation_warning"]),
          "reserved_test_videos":sorted(test_hashes)}
    for split in ("train","validation"):
        indices=[i for i,r in enumerate(rows) if r["row"]["split"]==split]
        selected=[rows[i] for i in indices]
        np.savez_compressed(run/f"{split}.npz",images=np.stack([arrays[i] for i in indices]),
            heat=np.stack([targets[i][0] for i in indices]),distances=np.stack([targets[i][1] for i in indices]),
            masks=np.stack([targets[i][2] for i in indices]))
        (run/f"{split}.json").write_text(json.dumps(selected,ensure_ascii=False,indent=2),encoding="utf-8")
        info[split]={name:{"images":sum(r["source"]==name for r in selected),
                          "presences":sum(len(r["boxes"]) for r in selected if r["source"]==name)} for name in ("coco","portes")}
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    model=ProfileDetector().eval()
    train=json.loads((run/"train.json").read_text())
    check_indices=np.array([next(i for i,r in enumerate(train) if r["boxes"]),next(i for i,r in enumerate(train) if not r["boxes"])])
    with np.load(run/"train.npz") as values:
        images=values["images"][check_indices]
        output=model(torch.from_numpy(images))
        if tuple(output.shape)!=(2,5,15,20): raise ValueError("Sortie réseau incorrecte.")
        _,grad,_=detection_loss(output.detach().numpy(),*(values[key][check_indices] for key in ("heat","distances","masks")))
    output.backward(torch.from_numpy(grad))
    np.savez_compressed(run/"gradient_reference.npz",indices=check_indices,logits=output.detach().numpy(),
        **{"grad_"+name.replace(".","_"):value.grad.numpy() for name,value in model.named_parameters()})
    np.savez_compressed(run/"parameter_masks.npz")
    torch.onnx.export(model,torch.from_numpy(images),str(run/"initial_dense.onnx"),input_names=["image"],output_names=["detections"],opset_version=17,dynamo=False)
    info["parameters_compact"]=sum(p.numel() for p in model.parameters())
    # Estimation illustrative des tenseurs UINT8, pas allocation certifiée du compilateur.
    activations=[WIDTH*HEIGHT]
    with torch.no_grad():
        value=torch.zeros(1,1,HEIGHT,WIDTH)
        for layer in model.layers:
            value=layer(value)
            activations.append(value.numel())
    info["memory_estimate_uint8"]={"weights_bytes_without_metadata":info["parameters_compact"],
        "largest_activation_bytes":max(activations),"sum_all_activation_bytes":sum(activations),
        "warning":"Somme des tenseurs théoriques, hors padding, buffers, alignement, firmware et plan mémoire Colibry. Budget matériel non validé."}
    (run/"dataset.json").write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({key:info[key] for key in ("train","validation","parameters_compact","memory_estimate_uint8")},indent=2))


if __name__=="__main__": main()
