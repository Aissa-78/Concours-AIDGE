# Essai avec zooms et recadrages — 17 novembre 2026

## Pourquoi cet essai

Le modèle COCO 2000 v1 manque trois présences dans les images de test caméra.
Le diagnostic montre des scores faibles et des rectangles trop petits pour
certaines personnes proches ou partiellement hors champ.

Nous testons une seule modification : des augmentations géométriques
pendant l'entraînement. Ce n'est pas encore une amélioration démontrée.

## Ce qui change

Dans `src/profile_training_data.py`, l'option `spatial=True` peut appliquer :

- un zoom entre 1 et 1,8, sans déformer les proportions ;
- un décalage horizontal et vertical jusqu'à 15 % de la dimension de l'image,
  en plus du décalage nécessaire pour centrer le zoom ;
- un recadrage à 160 × 120, avec fond gris pour les zones hors image.

Cette transformation est tentée sur environ la moitié des présentations.
Les rectangles suivent exactement le zoom et le déplacement et sont limités
à leur partie visible. Un recadrage est rejeté s'il laisse une personne visible
trop petite, un point d'ancrage invalide ou deux centres dans une même cellule.
Une image positive conserve au moins une personne. Une personne totalement
hors champ n'a plus de rectangle. Après huit tentatives invalides, l'image
reste sans ce zoom. Les miroirs et variations de lumière précédents restent actifs.

Les originaux, annotations enregistrées, images de validation et images de
test ne sont pas modifiés. L'augmentation se fait en mémoire pendant le train.

## Comparaison avec la v1

Le nouvel essai `runs/person_camera_coco2000_crop_v2` reprend exactement :

- les 1 840 images d'entraînement et les 475 images de validation ;
- les mêmes rectangles et la même séparation des ensembles ;
- le même réseau et le même fichier ONNX initial, vérifiés identiques ;
- le même protocole de 20 époques maximum, batch 8, LR initial 0,0003,
  décroissance toutes les 8 époques et sélection sur validation au seuil 0,5.

Il repart des mêmes poids initiaux que la v1, pas de ses poids finaux.
Cela permet de tester l'effet des recadrages sans ajouter aussi 20 époques
à un modèle déjà entraîné. Les tirages aléatoires changent avec l'ajout de
transformations ; un seul essai ne démontre pas un gain reproductible.

## Vérifications avant lancement

96 tests unitaires réussis. Contrôle des augmentations sur les 1 840 images
train : sorties finies, dimensions conservées, cibles valides. Validation des
sorties et gradients Aidge : écart maximal de référence 2,67 × 10⁻⁵.
Le contrôle des tableaux et des poids initiaux est enregistré dans
`runs/person_camera_coco2000_crop_v2/augmentation_audit.json`.

Correction après relecture en plus grand : le signalement de silhouettes
humaines sans rectangle sur l'image COCO 140724 était erroné. Les formes
à côté de l'homme sont des chevaux. L'annotation d'une personne est cohérente
et n'a pas été modifiée. Une vérification des comptes officiels ne suffit
néanmoins pas à garantir la qualité de toutes les annotations du grand lot.

## Reproduire

Les données et poids initiaux privés doivent déjà être présents.
Pour un autre essai, choisir un nouveau nom de dossier.

```bash
./.venv/bin/python scripts/prepare_camera_training.py --coco-dir dataset/coco_qqvga_2000_v1 --spatial-augmentation --run-dir runs/person_camera_coco2000_crop_v2
./.venv-aidge-colibry/bin/python scripts/train_person_qqvga_aidge.py --run-dir runs/person_camera_coco2000_crop_v2 --epochs 20 --batch-size 8 --learning-rate 0.0003 --evaluate-every 2 --patience 5 --lr-decay-every 8
./.venv/bin/python scripts/report_camera_training.py --run-dir runs/person_camera_coco2000_crop_v2
```

La préparation fonctionne avec le Python Windows de `.venv\Scripts\python.exe`.
L'entraînement nécessite séparément une installation Aidge fonctionnelle ;
l'installation native Windows n'est pas validée sur les PC de l'équipe.

## Résultats obtenus

L'entraînement est terminé après environ 25 minutes. Il s'est arrêté à
l'époque 12 : cinq évaluations successives n'amélioraient plus le critère
commun caméra/historique. Le modèle retenu est celui de l'époque 2.

| Mesure du modèle retenu, au seuil 0,5 | V1 sans recadrages | V2 avec recadrages |
|---|---:|---:|
| F1 validation caméra | 90,9 % | 87,0 % |
| F1 validation historique | 24,0 % | 22,0 % |
| Détections correctes sur les 3 présences du test caméra | 0 | 0 |
| Fausses détections sur les 10 images de test caméra | 0 | 0 |

Cet essai ne démontre pas d'amélioration et ne remplace pas la v1.
Certaines époques atteignent 100 % sur les huit images de validation caméra,
mais échouent davantage sur l'historique ; ce chiffre isolé ne permet pas
de choisir un modèle fiable. Le modèle retenu échoue encore sur le test.

La prochaine priorité est un audit des données et de l'apprentissage
(annotations manquantes, cohérence des cibles, capacité à apprendre un petit
lot), plutôt qu'un ajout aveugle d'images ou un autre entraînement long.

Rapport : `runs/person_camera_coco2000_crop_v2/rapport.html`.
Mesures : `runs/person_camera_coco2000_crop_v2/camera_metrics.json`.

## Fichiers de suivi

Consulter `progress.json` pour l'état, `training_history.csv` pour les mesures
et `training.json` lorsque l'apprentissage est terminé. Le journal de cette
exécution est `training_console.log` dans le dossier de l'essai.
Le rapport caméra sera généré après l'entraînement si celui-ci réussit.

Les trois images déjà examinées restent un diagnostic de développement,
pas un test final aveugle. Aucune exécution du modèle de personnes sur
ColibryNPU et aucune mesure énergétique ne sont validées par cet essai.
