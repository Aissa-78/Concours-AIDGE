# Caméra seule ou mélange COCO : diagnostic du 17 novembre

## Ce qui a été fait

Un entraînement supplémentaire utilise uniquement les **36 images caméra
du train**, avec leurs 34 rectangles confirmés. Les 475 images de validation
restent strictement identiques à l'essai COCO 2000, ainsi que le fichier ONNX
initial : ces égalités ont été vérifiées avant lancement.

Le critère de sélection reste le minimum des F1 caméra/historique, puis leur
moyenne, puis la loss. Les dix images de test ne servent pas à cette sélection.

Pour ce diagnostic court, les augmentations sont désactivées. Il y a au
maximum 80 époques de 36 images, sans répétition de source, un batch de huit,
un LR initial de 0,0003, divisé par deux après 40 époques. L'évaluation est
faite toutes les dix époques. L'arrêt anticipé intervient à l'époque 70 et
le modèle retenu est celui de l'époque 20, après environ deux minutes de calcul.

Ce n'est pas une comparaison causale à budget identique : les sources,
augmentations, nombres de mises à jour et calendrier d'évaluation diffèrent.
Elle sert à vérifier si retirer COCO suffit à résoudre le problème.

## Résultats au seuil 0,5 et IoU 0,5

| Modèle retenu | F1 validation caméra, 8 images | F1 validation historique, 467 images | Présences du test retrouvées |
|---|---:|---:|---:|
| Mélange COCO 2000 v1 | 90,9 % | 24,0 % | 0 / 3 |
| Mélange avec recadrages v2 | 87,0 % | 22,0 % | 0 / 3 |
| Caméra seule, diagnostic | 85,7 % | 8,0 % | 0 / 3 |

Le modèle caméra seule retrouve 9 des 12 personnes de validation caméra,
sans fausse détection. À l'époque sélectionnée, le F1 sur ses images apprises
est de 98,5 %. Il apprend donc très bien ses exemples, mais réussit moins bien
sur d'autres images de la même session. Les modèles précédents sont conservés.

**Retirer COCO n'a pas suffi.** Cela ne prouve pas que COCO est inutile ou
que toutes ses annotations sont correctes. Le problème dépasse le simple
ajout ou retrait d'images publiques : l'apprentissage ne généralise pas
encore correctement, même dans ce lieu.

Les trois présences de test sont dans trois images, pas nécessairement trois
personnes différentes. Ce test a déjà été examiné plusieurs fois ; il ne
constitue plus un test final indépendant.

## Relecture COCO

Un échantillon de **16 images train** a été relu visuellement : quatre anciennes
images COCO, huit nouvelles images positives et quatre nouvelles images sans
personne annotée. Aucune erreur d'annotation n'a été confirmée sur cet échantillon.
Ce petit échantillon n'est pas une certification des 2 000 images.

Le nombre de rectangles des 1 600 nouvelles images train correspond à celui
du plan issu des annotations officielles. Ce contrôle vérifie la conservation
des annotations, pas leur exhaustivité sur la photographie.

Correction du signalement antérieur : les formes à côté de l'homme sur
l'image 140724 sont des chevaux, pas des personnes oubliées. Le rectangle
unique est cohérent et n'a pas été modifié. La relecture des images réduites
peut induire en erreur : revenir aux originaux avant toute correction ambiguë.

Plusieurs scènes publiques restent éloignées du scénario de porte : animaux,
transports, lieux extérieurs, positions assises ou silhouettes très petites.
Cela justifie une sélection plus proche de la caméra, mais ne démontre pas
que ces images sont la cause principale de l'échec.

## Suite à privilégier

Le test de mémorisation sur huit images a déjà montré que la chaîne sait
apprendre. Le diagnostic caméra seule montre les limites de généralisation
avec seulement 36 images très proches. Avant un autre entraînement long,
préparer un petit ensemble caméra plus varié et une évaluation indépendante :
positions et distances différentes, personnes différentes, caméra fixe,
passages dans les deux sens et périodes sans personne. Conserver les vidéos
entières dans un même split pour éviter les doublons temporels.

Sélectionner aussi des images COCO dont les personnes restent lisibles à
160 × 120. Une éventuelle stratégie de préentraînement COCO puis adaptation
caméra serait un nouvel essai à valider, pas une amélioration garantie.

## Reproduire le diagnostic caméra seule

```bash
./.venv/bin/python scripts/prepare_camera_training.py --coco-dir dataset/coco_qqvga_2000_v1 --camera-only --run-dir runs/person_camera_only_diagnostic_v1
./.venv-aidge-colibry/bin/python scripts/train_person_qqvga_aidge.py --run-dir runs/person_camera_only_diagnostic_v1 --epochs 80 --batch-size 8 --learning-rate 0.0003 --evaluate-every 10 --patience 5 --lr-decay-every 40 --no-augmentation
./.venv/bin/python scripts/report_camera_training.py --run-dir runs/person_camera_only_diagnostic_v1
```

Choisir un nouveau dossier si l'essai existe. Les données et poids initiaux
privés sont nécessaires. La préparation peut utiliser le Python Windows
du venv ; l'installation Aidge native Windows reste à vérifier séparément.

Rapports locaux :

- `runs/person_camera_only_diagnostic_v1/rapport.html` : prédictions caméra.
- `runs/person_camera_only_diagnostic_v1/comparison.json` : comparaison chiffrée.
- `runs/audit_coco_annotations_v1/rapport.html` : les 16 images avec leurs rectangles.

Aucun déploiement du détecteur sur ColibryNPU ni résultat de comptage vidéo
n'est validé par ces diagnostics.
