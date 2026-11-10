"""Compare l'ancien candidat et le nouvel essai mixte, puis filme le test réservé.

Les poids sont déjà sélectionnés avant l'ouverture des trois vidéos de test.
Le statut de relecture des annotations locales est repris du dataset préparé.
"""

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
from imageio_ffmpeg import write_frames
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from camera_preprocess import prepare_camera_frame
from person_detection_math import evaluate_boxes,decode_boxes
from test_video_detectors import panel,sampled_frames
from prepare_profile_mix import local_video


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def infer(python,model,inputs,output):
    environment=os.environ.copy()
    environment.update(MPLBACKEND="Agg",OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4")
    subprocess.run([str(python.absolute()),str(ROOT/"scripts/infer_person_qqvga_aidge.py"),
        "--model",str(model.resolve()),"--inputs",str(inputs.resolve()),"--output",str(output.resolve()),"--batch-size","8"],
        cwd=ROOT,env=environment,check=True)
    return np.load(output)


def preview(image,boxes,detail,title):
    value=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    for i,(cx,cy,bw,bh) in enumerate(boxes):
        color=(20,20,230) if i in detail["missed"] else (240,240,240)
        cv2.rectangle(value,(round((cx-bw/2)*320),round((cy-bh/2)*240)),
                      (round((cx+bw/2)*320),round((cy+bh/2)*240)),color,1)
    for row in detail["predictions"]:
        x1,y1,x2,y2=map(round,row["box"])
        color=(30,200,30) if row["matched_truth"] is not None else (0,145,255)
        cv2.rectangle(value,(x1,y1),(x2,y2),color,2)
    banner=np.full((25,320,3),20,np.uint8)
    cv2.putText(banner,title,(5,18),cv2.FONT_HERSHEY_SIMPLEX,.43,(245,245,245),1)
    return np.concatenate([banner,value],axis=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,default=ROOT/"runs/person_profile_mix_qvga_v1")
    parser.add_argument("--legacy-run",type=Path,default=ROOT/"runs/person_qqvga_hard_context_full_v3")
    parser.add_argument("--aidge-python",type=Path,default=ROOT/".venv-aidge-colibry"/("Scripts/python.exe" if os.name=="nt" else "bin/python"))
    parser.add_argument("--diagnostic",type=Path,default=ROOT/"runs/videotest2_diagnostic_v2/results.json")
    args=parser.parse_args()
    run=args.run_dir
    info=json.loads((run/"dataset.json").read_text())
    trained=json.loads((run/"training.json").read_text())
    legacy_info=json.loads((args.legacy_run/"dataset.json").read_text())
    legacy_geometry=legacy_info.get("geometry",{"width":160,"height":120,"stride":4})
    old_width,old_height=legacy_geometry["width"],legacy_geometry["height"]
    old_label=f"Avant Aidge {old_width}x{old_height}"
    human_validation=info.get("human_reviewed_local_validation",False)
    if json.loads((run/"progress.json").read_text())["status"]!="completed":
        raise SystemExit("Attendre la fin de l'entraînement avant de figer le test.")
    shutil.copy2(run/"best_dense.onnx",run/"best.onnx")
    selected_hash=digest(run/"best.onnx")
    rows=json.loads((run/"validation.json").read_text())
    with np.load(run/"validation.npz") as data: images=data["images"]
    current=infer(args.aidge_python,run/"best.onnx",run/"validation.npz",run/"verified_validation_logits.npy")
    np.testing.assert_allclose(current,np.load(run/"best_validation_logits.npy"),atol=5e-5,rtol=5e-4)
    legacy_inputs=np.stack([cv2.resize(image[0],(old_width,old_height),interpolation=cv2.INTER_AREA)[None] for image in images])
    np.savez_compressed(run/"legacy_validation_inputs.npz",images=legacy_inputs)
    previous=infer(args.aidge_python,args.legacy_run/"best.onnx",run/"legacy_validation_inputs.npz",run/"legacy_validation_logits.npy")
    truths=[row["boxes"] for row in rows]
    new_metrics,new_details=evaluate_boxes(current,truths,width=320,height=240,stride=16,local_maximum=True)
    old_metrics,old_details=evaluate_boxes(previous,truths,local_maximum=True,**legacy_geometry)
    by_source={}
    for source in ("coco","portes"):
        indices=[i for i,row in enumerate(rows) if row["source"]==source]
        new,_=evaluate_boxes(current[indices],[truths[i] for i in indices],width=320,height=240,stride=16,local_maximum=True)
        old,_=evaluate_boxes(previous[indices],[truths[i] for i in indices],local_maximum=True,**legacy_geometry)
        if new!=trained["validation_metrics"][source]: raise RuntimeError("Le checkpoint ne correspond pas au bilan de sélection.")
        by_source[source]={"images":len(indices),"old":old,"new":new}
    values={"threshold":.5,"iou_match":.5,"selected_epoch":trained["selected_epoch"],
        "train_images":trained["train_images"],"validation_images":len(rows),"validation_person_instances":sum(map(len,truths)),
        "old_model":str((args.legacy_run/"best.onnx").resolve()),"old_model_sha256":digest(args.legacy_run/"best.onnx"),
        "new_model_sha256":selected_hash,"old":old_metrics,"new":new_metrics,"by_source":by_source,
        "validation_used_for_selection":True,"human_reviewed_local_validation":human_validation,
        "local_annotation_review":info.get("local_annotation_review"),
        "comparison_changes_resolution":legacy_geometry!=info["geometry"],
        "comparison_changes_architecture":any(legacy_info.get(k)!=info.get(k) for k in ("architecture","channels","blocks")),
        "comparison_changes_training_data":[(r["image_sha256"],r["label_sha256"]) for r in json.loads((args.legacy_run/"train.json").read_text())]!=[(r["image_sha256"],r["label_sha256"]) for r in json.loads((run/"train.json").read_text())],
        "hardware_validated":False}
    (run/"metrics.json").write_text(json.dumps(values,ensure_ascii=False,indent=2),encoding="utf-8")
    previews=run/"apercus"
    previews.mkdir(exist_ok=True)
    cards=[]
    for i,(row,old,new) in enumerate(zip(rows,old_details,new_details)):
        old_scaled={**old,"predictions":[{**p,"box":[v*(320/old_width if j%2==0 else 240/old_height) for j,v in enumerate(p["box"])]} for p in old["predictions"]]}
        frame=np.round(images[i,0]*255).astype(np.uint8)
        comparison=np.concatenate([preview(frame,row["boxes"],old_scaled,old_label),
                                   preview(frame,row["boxes"],new,"Nouveau Aidge 320x240")],axis=1)
        cv2.imwrite(str(previews/f"{i+1:03d}.jpg"),comparison)
        cards.append(f"<article><h3>{i+1:03d} — {html.escape(row['source'])}</h3><p>Retrouvées {new['tp']}, supplémentaires {new['fp']}, manquées {new['fn']}</p><img loading='lazy' src='apercus/{i+1:03d}.jpg'></article>")
    # Le lot de test ne sert ni aux gradients ni au choix du meilleur checkpoint.
    diagnostic=json.loads(args.diagnostic.read_text())
    test_hashes=set(info["reserved_test_videos"])
    clips=[clip for clip in diagnostic["clips"] if clip["sha256"] in test_hashes]
    if {clip["sha256"] for clip in clips}!=test_hashes: raise ValueError("Test réservé incomplet.")
    test_images,test_info=[],[]
    for clip in clips:
        path=local_video(clip["source"])
        if digest(path)!=clip["sha256"]: raise ValueError("Vidéo test modifiée.")
        offset=len(test_images)
        for frame_index,frame in sampled_frames(path,clip["every"]):
            if frame_index!=clip["frame_indices"][len(test_images)-offset]: raise ValueError("Cadence test incohérente.")
            gray,_=prepare_camera_frame(frame,320,240)
            test_images.append(gray[None].astype(np.float32)/255)
        if len(test_images)-offset!=clip["samples"]: raise ValueError("Vidéo test incomplète.")
        test_info.append({**clip,"source":str(path.resolve()),"offset":offset})
    np.savez_compressed(run/"test_inputs.npz",images=np.stack(test_images))
    test_outputs=infer(args.aidge_python,run/"best.onnx",run/"test_inputs.npz",run/"test_logits.npy")
    legacy_test_images=np.stack([cv2.resize(image[0],(old_width,old_height),interpolation=cv2.INTER_AREA)[None] for image in test_images])
    np.savez_compressed(run/"legacy_test_inputs.npz",images=legacy_test_images)
    legacy_test_outputs=infer(args.aidge_python,args.legacy_run/"best.onnx",run/"legacy_test_inputs.npz",run/"legacy_test_logits.npy")
    videos=run/"videos_test_reservees"
    videos.mkdir(exist_ok=True)
    video_cards,test_predictions=[],[]
    for clip in test_info:
        output_name=clip["id"]+"_avant_apres.mp4"
        writer=write_frames(str(videos/output_name),(1200,324),fps=clip["sample_fps"],codec="libx264",
            pix_fmt_in="rgb24",pix_fmt_out="yuv420p",macro_block_size=2,output_params=["-crf","20","-movflags","+faststart"])
        writer.send(None)
        predictions=[]
        try:
            for i,(frame_index,frame) in enumerate(sampled_frames(Path(clip["source"]),clip["every"])):
                gray,_=prepare_camera_frame(frame,320,240)
                old_gray=np.round(legacy_test_images[clip["offset"]+i,0]*255).astype(np.uint8)
                reference=clip["predictions"][i]
                predicted=[{"score":s,"box":b.tolist()} for s,b in decode_boxes(test_outputs[clip["offset"]+i],
                           local_maximum=True,width=320,height=240,stride=16)]
                old_predicted=[{"score":s,"box":b.tolist()} for s,b in decode_boxes(legacy_test_outputs[clip["offset"]+i],
                               local_maximum=True,**legacy_geometry)]
                comparison=np.concatenate([panel(gray,reference["yolo_qvga_gray"],"YOLO reference","320x240 gris, interne 640"),
                    panel(old_gray,old_predicted,"Modele Aidge avant",f"{old_width}x{old_height} gris"),
                    panel(gray,predicted,"Nouveau modele Aidge","320x240 gris")],axis=1)
                banner=np.full((24,1200,3),20,np.uint8)
                cv2.putText(banner,f"TEST RESERVE {clip['id']} | t={reference['seconds']:.2f}s | Detection seule, sans comptage",(8,17),cv2.FONT_HERSHEY_SIMPLEX,.45,(245,245,245),1)
                writer.send(cv2.cvtColor(np.concatenate([comparison,banner],axis=0),cv2.COLOR_BGR2RGB).tobytes())
                if i==clip["samples"]//2: cv2.imwrite(str(videos/(clip["id"]+".jpg")),comparison)
                predictions.append({"frame":frame_index,"seconds":reference["seconds"],"predictions":predicted})
        finally: writer.close()
        test_predictions.append({"video":clip["source"],"sha256":clip["sha256"],"predictions":predictions})
        video_cards.append(f"<h3>{html.escape(Path(clip['source']).name)}</h3><video controls preload='none' poster='videos_test_reservees/{clip['id']}.jpg' src='videos_test_reservees/{output_name}'></video>")
    if digest(run/"best.onnx")!=selected_hash: raise RuntimeError("Poids changés pendant le test.")
    (run/"test_results.json").write_text(json.dumps({"model_sha256":selected_hash,"threshold":.5,
        "legacy_model_sha256":values["old_model_sha256"],"test_previously_observed_in_development":True,
        "used_for_training_or_checkpoint_selection":False,"no_human_reference_labels":True,
        "same_scene_as_training":True,"accuracy_measured":False,"counting_tested":False,"videos":test_predictions},ensure_ascii=False,indent=2),encoding="utf-8")
    table=[]
    initialization_note=("Poursuite des poids du checkpoint précédent ; Adam réinitialisé. Les époques affichées concernent cette poursuite uniquement." if trained.get("initial_checkpoint") else "Départ avec des poids aléatoires.")
    annotation_note=("Les annotations de validation locale ont été proposées par YOLO puis validées par l'utilisateur. Les images train locales restent au statut indiqué dans le manifest ; elles ne sont pas automatiquement validées par cette relecture." if human_validation else "Les labels locaux ont été proposés par YOLO puis relus par l'assistant, pas par une personne indépendante.")
    comparison_note=("Même architecture, résolution et images d'entraînement pour les deux checkpoints ; comparaison de la poursuite d'apprentissage." if not any(values[k] for k in ("comparison_changes_resolution","comparison_changes_architecture","comparison_changes_training_data")) else "L'ancien et le nouveau système peuvent différer par la résolution, l'architecture et les données : on ne mesure pas le seul effet de COCO.")
    for source,item in by_source.items():
        for version in ("old","new"):
            m=item[version]
            table.append(f"<tr><td>{source}</td><td>{'Ancien' if version=='old' else 'Nouveau'}</td><td>{item['images']}</td><td>{m['precision']:.1%}</td><td>{m['recall']:.1%}</td><td>{m['f1']:.1%}</td><td>{m['tp']}</td><td>{m['fp']}</td><td>{m['fn']}</td></tr>")
    (run/"rapport.html").write_text("<!doctype html><html lang='fr'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>FlowSense — COCO + profils</title><style>body{font:17px/1.5 system-ui;max-width:1250px;margin:30px auto;padding:16px}table{border-collapse:collapse}th,td{padding:7px;border:1px solid #ccc}.scroll{overflow:auto}.warning{background:#fff1d6;padding:16px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(500px,1fr));gap:20px}img,video{max-width:100%;width:100%}h3{font-size:18px}</style>"
        f"<h1>Entraînement COCO + vidéos de profil</h1><p>{trained['train_images']} images train, {len(rows)} validation. 320 × 240 gris, {info['parameters_compact']} paramètres, {trained['epochs_completed']} époques ; checkpoint retenu : {trained['selected_epoch']}.</p><p>{initialization_note}</p>"
        "<p>Poids appris dans Aidge CPU FP32 (forward/backward), loss et Adam NumPy. PyTorch sert à définir le modèle et vérifier les gradients, pas à l'entraîner. Mélange équilibré COCO / vidéos à chaque époque, pas deux phases de préentraînement et adaptation.</p>"
        f"<p class='warning'>Résultats de développement provisoires. {annotation_note} La validation choisit le checkpoint. Toutes les vidéos partagent le même décor. Ce n'est pas une précision garantie en conditions réelles. {comparison_note}</p>"
        f"<h2>Ancien / nouveau sur les mêmes {len(rows)} images de validation</h2><div class='scroll'><table><tr><th>Source</th><th>Modèle</th><th>Images</th><th>Précision</th><th>Rappel</th><th>F1</th><th>Retrouvées</th><th>En trop</th><th>Manquées</th></tr>"
        +"".join(table)+"</table></div><p>Seuil 0,5, correspondance rectangle IoU ≥ 0,5, pics 3×3 et NMS 0,5. Une présence est une personne sur une image, pas un individu distinct. Rouge : annotation manquée ; blanc : annotation retrouvée ; vert : détection appariée ; orange : détection supplémentaire.</p>"
        "<h2>Trois vidéos hors apprentissage</h2><p>Poids figés avant ce test. De gauche à droite : YOLO de référence, Aidge avant, Aidge après. Ces vidéos avaient déjà été observées dans les essais précédents : ce n'est plus un test final aveugle. Environ 10 images/s, sans audio. Pas de métrique sur ces vidéos sans annotations humaines. Ni suivi ni comptage.</p>"
        +"".join(video_cards)+"<p class='warning'>Import et inférence Aidge CPU vérifiés, pas le déploiement ColibryNPU, l'UINT8 matériel ou les 5 mW. Ne pas déduire la mémoire matérielle du seul nombre de paramètres. Aucun modèle de l'interface n'a été remplacé.</p>"
        "<h2>Images de validation : avant / après</h2><main>"+"".join(cards)+"</main></html>",encoding="utf-8")
    print(json.dumps(values,ensure_ascii=False,indent=2))
    print(f"Rapport prêt : {run/'rapport.html'}")


if __name__=="__main__": main()
