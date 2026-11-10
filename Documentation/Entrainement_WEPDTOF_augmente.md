# Poursuivre l'entraînement avec davantage de vues du dessus

Ce guide décrit l'essai `heatmap_wepdtof_v2`, préparé le 20 octobre 2026. Il ne remplace ni le premier classificateur personne/vide, ni la démonstration vidéo YOLO. Le petit CNN Aidge apprend à localiser des **centres de personnes**, pas à dessiner des rectangles ni à compter seul les passages.

L'essai est terminé : **45,5 % de F1 sur la porte B**, contre 43,5 % au premier
essai WEPDTOF, mais avec une forte perte de détection publique pendant
l'adaptation. Les anciens modèles n'ont pas été remplacés. Voir le
[bilan complet et la priorité du prochain essai](Bilan_Aidge_WEPDTOF_v2.md).

## Les données utilisées

| Lot | Apprentissage | Validation |
|---|---:|---:|
| Nos portes annotées | 179 images, 168 présences | 58 images, 84 présences |
| Vues du dessus WEPDTOF augmentées | 400 images, 582 présences | 40 images, 54 présences |
| Total | **579 images, 750 présences** | **98 images, 138 présences** |

Une « présence » est une personne annotée dans une image : la même personne
peut apparaître dans plusieurs images. Ce ne sont pas 750 personnes différentes.

Les 320 nouvelles images d'apprentissage sont extraites de vidéos publiques déjà annotées. Quatre lieux supplémentaires sont ajoutés : `printing_store`, `repair_store`, `it_office` et `large_office`. Les huit lieux d'apprentissage sont distincts des deux lieux de validation publique. Les 120 images du lot précédent et leurs annotations sont conservées à l'identique. Il y a 80 images publiques d'apprentissage sans personne, utiles pour apprendre le fond.

Les 27 images de `porte_B_test` restent à part. Elles ne servent pas à apprendre ni à sélectionner les poids ou le seuil. Cette porte a toutefois déjà servi au diagnostic : elle ne constitue plus une évaluation finale totalement nouvelle.

Les annotations publiques proviennent de WEPDTOF, converties lors des recadrages ; elles ne sont pas toutes vérifiées manuellement. Le recadrage ne corrige pas la déformation fisheye. Ne pas publier les images sur GitHub ; voir les [conditions et précautions du lot](Lot_vues_du_dessus.md).

## Avant de lancer les commandes

Ouvrir **le dossier FlowSense-AIDGE dans VS Code**, puis son terminal. Les chemins ci-dessous partent de la racine du projet.

Les deux environnements doivent déjà être installés sur cet ordinateur : `.venv` pour préparer les images et évaluer, `.venv-aidge-colibry` pour exécuter Aidge. Un environnement Mac ne se copie pas sur Windows. Ces commandes utilisent des chemins adaptés aux deux systèmes, mais le nouvel essai a été exécuté sur Mac uniquement.

Pour refaire l'essai, choisir un **nouveau nom de dossier** à la place de `heatmap_wepdtof_v2` dans toutes les commandes. Ne pas ajouter `--overwrite` pour remplacer involontairement un ancien résultat. Après une correction d'annotation, refaire aussi la préparation : l'entraînement lit une copie préparée des données.

## 1. Retrouver ou préparer le lot augmenté

Le lot déjà téléchargé est dans `dataset/wepdtof_expanded_v2/`. Ouvrir son `rapport.html` pour voir les annotations, et `nouveaux_exemples.jpg` pour un aperçu des ajouts. Les quatre pages `negatives_a_verifier_01.jpg` à `04.jpg` montrent les images sans personne.

Ne pas télécharger le lot une seconde fois s'il existe. Pour le reconstruire sur un autre poste disposant du lot initial et des métadonnées d'inspection :

Mac :

```bash
./.venv/bin/python scripts/expand_wepdtof_dataset.py
./.venv/bin/python scripts/audit_wepdtof_subset.py --dataset-dir dataset/wepdtof_expanded_v2
```

Windows :

```powershell
.\.venv\Scripts\python.exe scripts/expand_wepdtof_dataset.py
.\.venv\Scripts\python.exe scripts/audit_wepdtof_subset.py --dataset-dir dataset/wepdtof_expanded_v2
```

Le téléchargement est limité aux images utiles, pas à l'archive complète. Le script refuse un dossier terminé existant. `--resume` sert uniquement à reprendre un téléchargement incomplet avec exactement la même sélection. Les étapes pour obtenir le lot initial sont dans le [guide WEPDTOF](Lot_vues_du_dessus.md).

## 2. Préparer un nouvel essai

Mac :

```bash
./.venv/bin/python scripts/prepare_heatmap_v2.py --pretrain-data dataset/wepdtof_expanded_v2 --pretrain-name wepdtof --output-dir runs/heatmap_wepdtof_v2
```

Windows :

```powershell
.\.venv\Scripts\python.exe scripts/prepare_heatmap_v2.py --pretrain-data dataset/wepdtof_expanded_v2 --pretrain-name wepdtof --output-dir runs/heatmap_wepdtof_v2
```

Cette étape prépare les images en niveaux de gris, les cibles et le réseau initial. `dataset.json` garde la liste exacte des images ; `dataset.npz` contient les tableaux utilisés pendant l'apprentissage. Elle n'entraîne pas encore le réseau.

## 3. Poursuivre l'apprentissage

Le checkpoint `runs/heatmap_wepdtof_v1/best.npz` doit être présent. Les images et les poids ne sont pas inclus automatiquement dans GitHub. Si ce checkpoint manque, retirer `--init-weights` et son chemin pour un nouvel apprentissage depuis l'initialisation ; ce ne sera pas la même expérience.

Mac :

```bash
./.venv-aidge-colibry/bin/python scripts/train_heatmap_v2_aidge.py --run-dir runs/heatmap_wepdtof_v2 --init-weights runs/heatmap_wepdtof_v1/best.npz --pretrain-epochs 16 --epochs 40 --batch-size 16 --patience 10 --pretrain-lr 0.0005 --learning-rate 0.0002
```

Windows :

```powershell
.\.venv-aidge-colibry\Scripts\python.exe scripts/train_heatmap_v2_aidge.py --run-dir runs/heatmap_wepdtof_v2 --init-weights runs/heatmap_wepdtof_v1/best.npz --pretrain-epochs 16 --epochs 40 --batch-size 16 --patience 10 --pretrain-lr 0.0005 --learning-rate 0.0002
```

Les calculs avant/arrière sont exécutés dans Aidge CPU ; la loss et Adam utilisent NumPy pour contourner un problème de l'optimiseur installé. PyTorch sert de référence pour contrôler les gradients au démarrage, sans effectuer cet apprentissage.

- `--init-weights` : reprendre les poids appris, après vérification de leur compatibilité. L'état d'Adam repart à zéro ; ce n'est pas une reprise exacte de l'ancien optimiseur.
- `--pretrain-epochs 16` : 16 passages sur les 400 images publiques, avec augmentations.
- `--epochs 40` : au maximum 40 passages d'adaptation aux portes. Chaque passage mélange les 179 images locales avec 64 images publiques rééchantillonnées.
- `--batch-size 16` : calculer les gradients sur des groupes de 16 images.
- Les deux taux d'apprentissage limitent la taille des mises à jour des poids.
- `--patience 10` : arrêter l'adaptation si la validation ne s'améliore plus pendant 10 époques, au plus tôt après 15 époques.

Le meilleur modèle de chaque étape est restauré, pas forcément celui de la dernière époque. `best.onnx` et `best.npz` sont le résultat final ; `training_history.csv` garde l'évolution de la loss et du F1. Une loss qui baisse ne garantit pas une meilleure détection sur une nouvelle porte.

## 4. Régler le seuil et comparer

Mac :

```bash
./.venv/bin/python scripts/calibrate_heatmap_v2.py --run-dir runs/heatmap_wepdtof_v2
./.venv/bin/python scripts/evaluate_heatmap_v2.py --run-dir runs/heatmap_wepdtof_v2 --data-dir dataset/detection_proposals --calibrated --reference-run-dir runs/heatmap_wepdtof_v1
./.venv/bin/python scripts/evaluate_heatmap_v2.py --run-dir runs/heatmap_wepdtof_v2 --data-dir dataset/wepdtof_expanded_v2 --calibrated --reference-run-dir runs/heatmap_wepdtof_v1
./.venv/bin/python scripts/evaluate_heatmap_v2.py --run-dir runs/heatmap_wepdtof_v2 --data-dir dataset/porte_B_test --calibrated --int8 --reference-run-dir runs/heatmap_wepdtof_v1
```

Windows : mêmes arguments, avec `.\.venv\Scripts\python.exe` à la place de `./.venv/bin/python`.

Le seuil est choisi uniquement sur les 58 images de validation locale. Les comparaisons conservent aussi les résultats au seuil fixe 0,5, pour ne pas confondre un réglage de seuil avec un meilleur apprentissage. L'option `--int8` teste un graphe quantifié signé sur CPU, pas encore l'exécution sur carte ou l'UINT8 final.

Dans `runs/heatmap_wepdtof_v2/`, ouvrir les `rapport.html` des dossiers `evaluation_detection_proposals`, `evaluation_wepdtof_expanded_v2` et `evaluation_porte_B_test`. Vert : centre correctement localisé ; orange : point supplémentaire ou trop éloigné ; croix rouge : centre manqué.

## L'objectif de 90 %

On vise à terme un comptage fiable, pas simplement un bon score pendant l'entraînement. Pour la détection, suivre séparément **précision, rappel et F1**. Pour le système complet, vérifier sur de nouvelles vidéos chaque passage réel, son sens et les comptages supplémentaires : des entrées et sorties erronées peuvent se compenser dans un total final.

La règle d'évaluation et les vidéos finales doivent être fixées avant de tester. Viser au moins 90 % de précision et de rappel sur les événements de passage est un objectif de travail, **pas une performance déjà obtenue ni une garantie**. Davantage d'images peut aider, mais si les erreurs persistent, il faudra aussi examiner les annotations, la résolution et l'architecture.
