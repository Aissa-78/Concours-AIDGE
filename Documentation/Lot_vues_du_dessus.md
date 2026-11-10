# Petit lot d'images vues du dessus — 20 octobre 2026

Ce document conserve la description du **premier lot de 120 images**.
Un second lot ajoute 320 images d'apprentissage et garde les mêmes validations :
voir le [guide du lot augmenté et de la poursuite d'apprentissage](Entrainement_WEPDTOF_augmente.md).

## Ce qui est prêt

120 images recadrées issues de **WEPDTOF**, avec leurs annotations originales converties. Il ne s'agit pas d'images filmées par notre équipe ni de prédictions de notre modèle.

| Groupe | Images | Présences annotées | Images sans personne |
|---|---:|---:|---:|
| Apprentissage | 80 | 86 | 16 |
| Validation publique | 40 | 54 | 8 |

Les mêmes personnes peuvent apparaître sur plusieurs images. Les quatre lieux d'apprentissage sont `jewelry_store`, `tech_store`, `warehouse` et `exhibition_setup`. Les deux lieux de validation sont `empty_store` et `convenience_store`. Aucun de ces lieux ne se retrouve dans les deux groupes. Les clips suffixés `_2` sont considérés comme le même lieu que leur clip d'origine.

Le script a récupéré environ 82,7 Mo pour ce lot, plus 12,9 Mo pour l'inspection des 16 scènes. Il n'a pas téléchargé les 5,43 Go de l'archive complète. Les images, annotations sources et aperçus restent locaux, ignorés par Git.

## Regarder les images

Depuis la racine du projet, ouvrir dans un navigateur :

```text
dataset/wepdtof_subset/rapport.html
```

Le rectangle vert est une **annotation**, pas le résultat de notre IA. Le point jaune représente le centre que le modèle doit apprendre à retrouver. Notre CNN actuel continue à prédire des centres, pas des rectangles.

`planche.jpg` montre un exemple de chaque lieu. `negatives_a_verifier.jpg` rassemble les 24 images étiquetées sans personne. Les images complètes sont dans `train/images/` et `validation/images/` ; les rectangles sont dans les dossiers `labels/` correspondants.

## Précautions prises

Les images viennent de caméras au plafond avec un objectif fisheye. Des fenêtres carrées fixes réduisent le champ, mais **le recadrage ne corrige pas la déformation** et ne reproduit pas exactement notre caméra. L'intérêt pour notre position doit être confirmé visuellement, puis par un essai mesuré.

Seules les fenêtres entièrement situées dans la zone annotée de la scène sont retenues. Une fenêtre est rejetée si une personne annotée y devient trop petite, trop coupée, ou si elle contient plus de trois personnes. On ne supprime pas une petite personne de ses annotations pour en faire artificiellement du fond. Au moins cinq indices de frame séparent les images choisies d'un clip ; l'espacement temporel exact dépend de la source.

Les rectangles orientés de WEPDTOF sont convertis en enveloppes horizontales après recadrage. Cela change leur géométrie : ce n'est pas le format d'évaluation officiel WEPDTOF. Les coordonnées d'origine sont conservées dans les JSON d'inspection et les polygones recadrés sont conservés dans `SOURCE.json`.

Les annotations portent la provenance `annotation_wepdtof`, **pas la marque de vérification manuelle `oui`**. Le contrôle automatique valide les coordonnées et la séparation, pas la présence réelle de chaque personne. Toute image ambiguë doit être revue avant de servir au prochain entraînement.

Pour corriger une annotation avec la fenêtre existante, depuis VS Code sur Mac :

```bash
./.venv/bin/python scripts/review_detection_dataset.py --dataset-dir dataset/wepdtof_subset --split all
```

Sur Windows :

```powershell
.\.venv\Scripts\python.exe scripts/review_detection_dataset.py --dataset-dir dataset/wepdtof_subset --split all
```

Glisser avec le clic gauche pour ajouter un rectangle, clic droit pour supprimer, `S` pour enregistrer et avancer, `P` pour revenir, `N` pour passer sans valider et `Q` pour quitter. Une image sans personne garde zéro rectangle.

## Préparer ensuite un essai contrôlé

Le téléchargement et la sélection sont terminés. Après confirmation du point de vue de l'exemple 001 par l'équipe, **un entraînement distinct a été terminé dans `runs/heatmap_wepdtof_v1/`** : 43,5 % de F1 sur la porte B, contre 35,3 % pour la V2 COCO, mais les anciens modèles restent meilleurs sur cette porte. La V2 WEPDTOF manque encore presque toutes les présences des autres lieux publics. Voir le [bilan complet](Bilan_Aidge_WEPDTOF.md). Les anciens modèles et la porte B restent inchangés. Cette confirmation ne valide pas manuellement toutes les annotations.

Les commandes suivantes décrivent l'essai terminé, avec la même architecture V2, en remplaçant le préentraînement COCO par ce lot de vues du dessus. Pour refaire un essai, remplacer `heatmap_wepdtof_v1` par un nouveau nom :

```bash
./.venv/bin/python scripts/prepare_heatmap_v2.py --pretrain-data dataset/wepdtof_subset --pretrain-name wepdtof --output-dir runs/heatmap_wepdtof_v1
./.venv-aidge-colibry/bin/python scripts/train_heatmap_v2_aidge.py --run-dir runs/heatmap_wepdtof_v1 --pretrain-epochs 8 --epochs 35 --batch-size 16
./.venv/bin/python scripts/calibrate_heatmap_v2.py --run-dir runs/heatmap_wepdtof_v1
./.venv/bin/python scripts/evaluate_heatmap_v2.py --run-dir runs/heatmap_wepdtof_v1 --data-dir dataset/detection_proposals --calibrated
./.venv/bin/python scripts/evaluate_heatmap_v2.py --run-dir runs/heatmap_wepdtof_v1 --data-dir dataset/wepdtof_subset --calibrated
```

Sur Windows, avec les deux environnements compatibles déjà installés :

```powershell
.\.venv\Scripts\python.exe scripts/prepare_heatmap_v2.py --pretrain-data dataset/wepdtof_subset --pretrain-name wepdtof --output-dir runs/heatmap_wepdtof_v1
.\.venv-aidge-colibry\Scripts\python.exe scripts/train_heatmap_v2_aidge.py --run-dir runs/heatmap_wepdtof_v1 --pretrain-epochs 8 --epochs 35 --batch-size 16
.\.venv\Scripts\python.exe scripts/calibrate_heatmap_v2.py --run-dir runs/heatmap_wepdtof_v1
.\.venv\Scripts\python.exe scripts/evaluate_heatmap_v2.py --run-dir runs/heatmap_wepdtof_v1 --data-dir dataset/detection_proposals --calibrated
.\.venv\Scripts\python.exe scripts/evaluate_heatmap_v2.py --run-dir runs/heatmap_wepdtof_v1 --data-dir dataset/wepdtof_subset --calibrated
```

Un dossier d'essai existant n'est pas écrasé automatiquement. Si des annotations sont corrigées après la préparation, refaire la préparation dans **un nouveau dossier**, pour que les corrections soient prises en compte. Copier un environnement Mac sur Windows ne fonctionne pas ; l'exécution de ces nouveaux scripts sur Windows reste à tester sur un poste équipé.

On compare d'abord sur la validation locale et publique ; la porte B reste hors apprentissage. Pour évaluer ensuite cette porte, remplacer le chemin après `--data-dir` par `dataset/porte_B_test`. Elle a déjà servi au diagnostic : une porte encore jamais examinée restera nécessaire pour confirmer la généralisation finale. Ce lot ne prouve aucune amélioration, aucune précision UINT8 ni aucune consommation sur carte.

## Refaire la sélection

```bash
./.venv/bin/python scripts/inspect_wepdtof.py
./.venv/bin/python scripts/prepare_wepdtof_subset.py --output-dir dataset/wepdtof_mon_lot
./.venv/bin/python scripts/audit_wepdtof_subset.py --dataset-dir dataset/wepdtof_mon_lot
```

Le lecteur utilise HTTP Range, impose une limite de transfert et refuse un serveur qui renverrait l'archive complète. Il n'extrait que les fichiers utiles, sans installer ni exécuter le code fourni par le dataset.

## Source et conditions

[Présentation WEPDTOF — Boston University](https://vip.bu.edu/projects/vsns/cossy/datasets/wepdtof/) · [Page officielle de téléchargement et conditions](https://vip.bu.edu/projects/vsns/cossy/datasets/wepdtof/download/).

La page de téléchargement indique un usage de recherche non commercial et demande de citer M. Ozan Tezcan, Zhihao Duan, Mertcan Cokbas, Prakash Ishwar et Janusz Konrad, « WEPDTOF: A Dataset and Benchmark Algorithms for In-the-Wild People Detection and Tracking from Overhead Fisheye Cameras », WACV 2022. Vérifier ces conditions avec les organisateurs avant un autre usage ou une redistribution. Il ne faut pas publier ces images automatiquement sur GitHub.
