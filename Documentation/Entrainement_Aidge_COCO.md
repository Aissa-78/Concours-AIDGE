# Entraîner le modèle V2 avec COCO et nos images de portes

Cette expérience du 20 octobre 2026 est distincte des anciens modèles et de la démonstration YOLO. Elle vise à améliorer la localisation des personnes, pas encore le comptage des entrées et sorties.

## Les images utilisées

Le carnet de mission conseille COCO Person. Comme il ne donne pas de lien vers une archive précise, le script extrait directement la catégorie `person` de [COCO 2017 officiel](https://cocodataset.org/#download). Il ne s'agit pas d'une base privée fournie par ASYGN.

| Source | Apprentissage | Validation |
|---|---:|---:|
| Sous-ensemble COCO | 500 images | 125 images |
| Nos portes déjà annotées | 179 images | 58 images |

Les 27 images de `porte_B_test` restent séparées. Elles ne servent ni à modifier les poids ni à choisir la meilleure époque. Cette porte a cependant déjà servi à diagnostiquer les limites des anciens modèles : il faudra une autre porte, encore jamais examinée, pour une évaluation finale vraiment indépendante.

COCO apporte des personnes, des lieux et des objets différents. Les scènes retenues contiennent une à trois personnes suffisamment grandes, ou aucune personne. Les rectangles sont ceux de COCO, pas des propositions YOLO. Les 625 identifiants, les licences originales et les critères sont enregistrés dans `dataset/coco_person_subset/SOURCE.json`. Toutes ces images viennent de **val2017**, repartitionné localement en deux groupes distincts : nos scores ne sont donc pas des scores du benchmark officiel COCO. Les fichiers ne sont pas publiés sur GitHub.

## Ce que fait le modèle

Une image est mise en niveaux de gris, redimensionnée sans la déformer dans un carré de 80 × 80 pixels, puis entourée d'une marge pour obtenir une entrée 128 × 128. Le réseau produit une grille de 20 × 20 scores. Un maximum indique un centre possible de personne.

Le CNN utilise quatre convolutions, des ReLU et deux MaxPool. Ses 17 885 paramètres voient davantage de contexte que l'ancien réseau : un score dépend d'une zone de 52 × 52 pixels. Le modèle ne prédit pas encore des rectangles. Les centres visés sont ceux des rectangles annotés, pas systématiquement les têtes.

L'apprentissage a deux phases : d'abord COCO, ensuite les images de portes avec quelques exemples COCO conservés à chaque époque. Les rotations, miroirs et variations de lumière augmentent la diversité. Les positions des annotations subissent exactement les mêmes transformations que les images.

Les calculs avant/arrière du réseau sont exécutés par **Aidge CPU en FP32**. La loss BCE pondérée et la mise à jour Adam sont calculées en NumPy. C'est un contournement d'une erreur `bad any_cast` de l'optimiseur Aidge installé. Une comparaison de tous les gradients avec PyTorch est obligatoire avant l'apprentissage. PyTorch sert à initialiser le graphe ONNX et à vérifier les calculs ; il n'entraîne pas les poids de cette V2.

## Les fichiers

- `scripts/download_coco_person_subset.py` : télécharge les images et transforme les rectangles COCO en labels locaux.
- `scripts/prepare_heatmap_v2.py` : vérifie les annotations et les séparations, prépare les images, les cibles, la calibration et le graphe initial.
- `src/heatmap_v2.py` : architecture du CNN.
- `src/heatmap_v2_data.py` : niveaux de gris, redimensionnement, cibles et augmentations synchronisées.
- `src/heatmap_metrics.py` : loss et comparaison des centres prédits/annotés.
- `scripts/train_heatmap_v2_aidge.py` : vérification des gradients, apprentissage Aidge, sélection sur la validation et sauvegardes.
- `scripts/evaluate_heatmap_v2.py` : comparaison avec les anciens modèles et rapport illustré ; option `--int8` pour vérifier aussi le graphe quantifié sur CPU.
- `tests/test_heatmap_v2.py` : contrôles des coordonnées, transformations, gradients et suppression des doublons.

## Refaire l'expérience sur Mac

Dans le terminal de VS Code, depuis la racine du projet. Les deux environnements doivent déjà être installés : `.venv` pour les images/PyTorch, `.venv-aidge-colibry` pour Aidge. L'installation Aidge compatible est décrite dans [Aidge_Colibry_Mac.md](Aidge_Colibry_Mac.md).

```bash
./.venv/bin/python scripts/download_coco_person_subset.py
./.venv/bin/python scripts/prepare_heatmap_v2.py --output-dir runs/heatmap_v2_mon_essai
./.venv-aidge-colibry/bin/python scripts/train_heatmap_v2_aidge.py --run-dir runs/heatmap_v2_mon_essai --pretrain-epochs 8 --epochs 35
./.venv/bin/python scripts/calibrate_heatmap_v2.py --run-dir runs/heatmap_v2_mon_essai
./.venv/bin/python scripts/evaluate_heatmap_v2.py --run-dir runs/heatmap_v2_mon_essai --int8 --calibrated
```

Le téléchargement comprend environ 241 Mo d'annotations COCO, en plus des images sélectionnées. Ne pas relancer la préparation pendant l'entraînement. Utiliser un **nouveau nom de dossier** pour chaque essai afin de conserver les résultats précédents.

## Commandes Windows

Même dossier et mêmes données, mais chaque camarade doit installer ses propres environnements Windows. Copier les environnements du Mac ne fonctionne pas. Ces commandes supposent **un Aidge compatible déjà installé** dans `.venv-aidge-colibry` ; l'installateur habituel de la démonstration YOLO ne le fournit pas.

```powershell
.\.venv\Scripts\python.exe scripts/download_coco_person_subset.py
.\.venv\Scripts\python.exe scripts/prepare_heatmap_v2.py --output-dir runs/heatmap_v2_mon_essai
.\.venv-aidge-colibry\Scripts\python.exe scripts/train_heatmap_v2_aidge.py --run-dir runs/heatmap_v2_mon_essai --pretrain-epochs 8 --epochs 35
.\.venv\Scripts\python.exe scripts/calibrate_heatmap_v2.py --run-dir runs/heatmap_v2_mon_essai
.\.venv\Scripts\python.exe scripts/evaluate_heatmap_v2.py --run-dir runs/heatmap_v2_mon_essai --int8 --calibrated
```

Les scripts ne lancent ni `open` ni une commande spécifique à macOS. Les exécuter sur Windows reste à vérifier sur un poste Windows équipé d'Aidge.

## Lire le résultat

Ouvrir dans un navigateur `runs/heatmap_v2_mon_essai/evaluation_porte_B_test/rapport.html`. Le dossier contient aussi les détails par image, les métriques et les aperçus.

- `best.onnx` et `best.npz` contiennent les poids retenus **sur la validation locale**. `best_coco` et `best_portes` gardent les étapes intermédiaires.
- `training_history.csv` permet de suivre la loss et le F1 de chaque époque.
- `training.json` indique la méthode, le contrôle des gradients et le temps de calcul.
- `dataset.json` indique exactement quelles images et annotations ont été utilisées.

La **loss** mesure l'écart entre la carte prédite et les cibles ; elle sert à calculer les corrections des poids. Le **rappel** mesure la proportion de centres annotés retrouvés. La **précision** mesure la proportion de prédictions qui correspondent à une annotation. Le **F1** combine précision et rappel : ce n'est pas une accuracy de classification ni un taux de passages correctement comptés.

La comparaison montre les résultats au seuil fixe de 0,5 et, séparément, au seuil V2 choisi sur les 58 images de validation locale. La porte B ne sert pas à choisir ce seuil. La tolérance de localisation reste de 12,5 % du côté de l'image préparée. Un point sur une tête trop loin du centre annoté peut compter comme point hors tolérance et comme centre manqué. Regarder les aperçus avant de conclure.

## Ce qui restera à vérifier

Un bon score FP32 ne prouve ni un bon comptage vidéo, ni la précision UINT8 finale, ni la consommation de 5 mW. L'option INT8 teste un graphe CPU signé intermédiaire, pas le bitstream matériel ni la cible UINT8 asymétrique. Après l'évaluation viennent l'étude de quantification, le suivi vidéo, puis les essais avec la carte et la caméra.
