# Caméra réelle et COCO : point du 17 novembre 2026

## Ce qui est prêt

Le dossier local `dataset/coco_qqvga_2000_v1` contient 2 000 images COCO :

| Ensemble | Avec personnes | Sans personne annotée | Total |
|---|---:|---:|---:|
| train | 1 040 | 560 | 1 600 |
| validation | 260 | 140 | 400 |

Les images sont converties en niveaux de gris et en 160 × 120 pixels.
Les proportions sont conservées avec des bandes grises. Les rectangles
proviennent des annotations officielles COCO et suivent le redimensionnement.
La classe des rectangles est `0 = personne` ; elle ne reprend pas les classes
du premier classifieur vide/personne.

La sélection automatique privilégie les personnes debout et des silhouettes
assez grandes. Ce n'est pas une vérification humaine des 2 000 images, ni une
garantie que toutes les personnes soient vues de profil. Les images sans
personne annotée peuvent aussi contenir des oublis de COCO.
Les anciens identifiants COCO présents localement sont exclus. Les splits
sont internes à train2017 : leurs résultats ne sont pas un score COCO officiel.
Les licences et liens des photographies figurent dans `ATTRIBUTIONS.csv`.

## Télécharger sur un autre ordinateur

Ouvrir un terminal dans le dossier du projet, après installation des dépendances.

Mac :

```bash
./.venv/bin/python scripts/prepare_coco_2000.py
```

Windows PowerShell :

```powershell
.\.venv\Scripts\python.exe scripts/prepare_coco_2000.py
```

Le script télécharge les annotations officielles si nécessaire, puis les
photographies. Le dossier peut prendre plusieurs centaines de Mo. Une relance
réutilise le plan et les originaux déjà téléchargés. Pour récupérer exactement
le même lot, partager aussi `plan.json` : une première sélection peut différer
si les anciens jeux COCO exclus ne sont pas présents sur cet ordinateur.

Les données restent locales, hors de GitHub. Pour relire les rectangles :

```bash
python scripts/review_detection_dataset.py --dataset-dir dataset/coco_qqvga_2000_v1 --split all
```

Utiliser le Python de `.venv` comme ci-dessus. Si des annotations sont corrigées,
ne pas relancer le téléchargement dans le même dossier : il réécrit les labels.

## Préparer un nouvel entraînement avec les 2 000 images

Il faut aussi récupérer par le partage privé de l'équipe les 54 photos annotées
de `dataset/camera_reelle_20261117_v1`, les anciens dossiers de données référencés
par les manifests, les manifests `runs/person_profile_mix_qvga_v2/train.json` et
`validation.json`, et `runs/person_profile_mix_qvga_v1/best_dense.npz`.
Ces données et poids ne sont pas inclus dans un clone GitHub.

Mac :

```bash
./.venv/bin/python scripts/prepare_camera_training.py --coco-dir dataset/coco_qqvga_2000_v1 --run-dir runs/person_camera_coco2000_v1
./.venv-aidge-colibry/bin/python scripts/train_person_qqvga_aidge.py --run-dir runs/person_camera_coco2000_v1 --epochs 20 --batch-size 8 --learning-rate 0.0003 --evaluate-every 2 --patience 5 --lr-decay-every 8
./.venv/bin/python scripts/report_camera_training.py --run-dir runs/person_camera_coco2000_v1
```

Windows : la préparation utilise la même commande, en remplaçant le premier
exécutable par `.\.venv\Scripts\python.exe`. L'installateur Windows du dépôt
**n'installe pas Aidge**. L'entraînement Aidge doit se faire dans un environnement
Aidge opérationnel (par exemple celui du Mac de l'équipe ou un environnement
Linux installé séparément). Nous n'avons pas validé l'entraînement Aidge natif
ni la communication matérielle sur Windows.

Choisir un nouveau nom de dossier pour chaque essai. Les 10 images de test de
la caméra sont exclues de l'apprentissage. Le mélange utilise toutes les images
train et répète les photos caméra pour équilibrer caméra et historique.

## Fichiers ajoutés et changements

- `prepare_coco_2000.py` : téléchargement, sélection, conversion et annotations.
- `prepare_camera_training.py` : mélange les données, vérifie la séparation des
  ensembles, prépare les cibles et un modèle initial ONNX.
- `train_person_qqvga_aidge.py` : entraînement Aidge CPU FP32 et sélection du
  meilleur modèle par validation ; géométrie transmise aux augmentations.
- `profile_detector.py` : option stride 8 pour une sortie de 20 × 15 cellules
  à partir d'une image 160 × 120 ; les anciens modèles gardent leur défaut.
- `profile_training_data.py` : augmentations adaptées à la taille choisie.
- `report_camera_training.py` : comparaison des rectangles et bilan séparé
  validation/test, sans modifier les poids.
- `Connexion_Colibry_Mac.md`, les trois scripts de connexion/UART/caméra,
  `requirements-colibry-hardware.txt` et `Ouvrir_camera_Mac.command` : accès à la
  caméra et diagnostic sur Mac. Ils nécessitent le SDK privé, absent de GitHub.

## Résultat déjà obtenu, avant le lot COCO 2000

L'essai `person_camera_mix_qqvga_v1` a utilisé 36 photos caméra d'entraînement
et 204 images historiques, dont 132 COCO. Le modèle sélectionné à l'époque 18
sur 20 donne, au seuil 0,5 et à IoU 0,5 :

| Ensemble caméra | Personnes correctes | Fausses détections | Personnes manquées | F1 |
|---|---:|---:|---:|---:|
| Validation : 8 images | 10 | 0 | 2 | 90,9 % |
| Test : 10 images | 0 | 0 | 3 | 0 % |

Les photographies viennent de la même session : ces résultats ne prouvent pas
la généralisation à d'autres portes. Le test montre encore un échec important.
Le prochain essai devra être comparé sans choisir les paramètres sur ces
10 images ; prévoir ensuite une nouvelle vidéo indépendante.

Le lot de 2 000 images n'a pas encore servi à un entraînement terminé.
Ni le nouveau détecteur ni son comptage n'ont été validés sur ColibryNPU.
La lecture caméra utilise pour l'instant un programme de capture distinct.
