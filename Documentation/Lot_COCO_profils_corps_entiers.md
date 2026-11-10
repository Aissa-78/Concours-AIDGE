# Premier lot COCO : personnes debout et corps entiers

Préparé le 3 novembre 2026. Ce travail prépare des données : aucun nouvel entraînement n'a été lancé et aucune amélioration de précision n'est encore mesurée.

## Ce qui a été choisi

Sur 350 candidats téléchargés et examinés sur des planches, 165 images ont été retenues pour un premier lot propre :

| Lot | Avec personnes | Sans personne | Total |
| --- | ---: | ---: | ---: |
| Entraînement | 100 | 32 | 132 |
| Validation | 25 | 8 | 33 |
| Total | 125 | 40 | 165 |

Les 125 images positives contiennent 139 rectangles de personnes au total. Profils et trois-quarts sont privilégiés ; quelques vues de face et de dos apportent de la diversité. Des enfants, adultes, vêtements, éclairages et décors différents sont présents. Ce ne sont pas exclusivement des photos de portes.

Le premier objectif de 250 images a été réduit pour ne pas compléter avec des corps masqués, des poses sportives ou des cas difficiles à lire. Tous les candidats ont une décision explicite dans `Documentation/coco_fullbody_review_20261103.json`. Les candidats non retenus restent dans leur dossier, aucune ancienne donnée n'a été supprimée.

COCO ne fournit pas une étiquette fiable « profil ». Le script utilise les points du corps pour présélectionner des profils probables, puis une revue visuelle écarte les cas peu adaptés. Le champ `profil_probable` reste une indication automatique, pas une orientation certifiée pour chaque personne. La revue sur planches n'est pas une réannotation exhaustive à haute résolution.

## Où sont les fichiers ?

Dans `dataset/coco_profile_fullbody_v1/` :

- `train/images/` et `validation/images/` : les JPEG originaux, conservés sans modification.
- `train/labels/` et `validation/labels/` : les rectangles de toutes les personnes annotées dans chaque image.
- `camera_320/` : les mêmes lots et rectangles, avec des PNG en niveaux de gris de 320 × 240 pixels, sans étirement. Des bandes grises préservent les proportions.
- `manifest.csv` : liste des images, labels et lots.
- `SOURCE.json` : provenance, annotations originales, paramètres, décisions et empreintes des fichiers.
- `ATTRIBUTIONS.csv` : liens d'origine et licences individuelles.
- `rapport.html` et `apercus/` : galerie des copies grises avec rectangles pour contrôle. Ouvrir `rapport.html` par double-clic.

Les rectangles verts sont des annotations COCO, **pas des prédictions du modèle**. Ils sont dessinés uniquement sur les aperçus : les pixels d'entraînement n'ont pas de rectangles.

Une ligne de label est `0 centre_x centre_y largeur hauteur`, coordonnées normalisées entre 0 et 1. Ici **0 signifie la classe de détection « personne »**, pas l'ancienne classe binaire « vide ». Une image sans personne a un fichier de labels vide.

## Séparation et limites

Les 132 et 33 images sont distinctes par identifiant et empreinte. Aucun doublon proche n'a été trouvé avec le seuil de hash perceptuel utilisé. Cela ne prouve pas que toutes les scènes ou photographes sont indépendants. Les 165 identifiants sont différents des 625 images de l'ancien sous-ensemble COCO.

Les deux lots proviennent du **train2017 officiel**, réparti localement en entraînement et validation. Ce n'est ni le benchmark COCO officiel ni un test indépendant sur une nouvelle porte. Les 33 images de validation doivent rester hors entraînement ; un test final sur des vidéos distinctes reste nécessaire.

Ces images peuvent enrichir un détecteur de personnes. Elles n'apprennent pas à elles seules les déplacements, le suivi ou le franchissement d'une ligne. Quelques vidéos correspondant à la future position de caméra resteront indispensables. Si la caméra finale voit seulement les crânes d'en haut, un lot de corps entiers de profil ne correspondra pas à cette vue.

La préparation en gris 320 × 240 respecte la limite de résolution annoncée du projet. Cela ne démontre pas qu'un modèle donné tient en mémoire, fonctionne sur Aidge/Colibry ou respecte 5 mW. Aucune précision de 90 % n'est garantie.

## Refaire la préparation

Ouvrir un terminal à la racine du projet. Le script nécessite OpenCV et NumPy dans l'environnement Python existant. Les images et l'archive officielle sont déjà en cache sur ce Mac.

Étapes de sélection : `plan`, puis `fetch`, revue visuelle, puis `finalize`. Les dossiers de sortie finaux doivent être nouveaux pour éviter tout écrasement. Pour reproduire le lot actuel depuis les candidats conservés :

macOS :

```sh
./.venv/bin/python scripts/select_coco_fullbody.py finalize --final-dir dataset/coco_profile_fullbody_copie --decisions Documentation/coco_fullbody_review_20261103.json --positive-count 125 --negative-count 40
```

Windows PowerShell, avec un environnement Python local installé et les candidats téléchargés sur ce PC :

```powershell
.\.venv\Scripts\python.exe scripts/select_coco_fullbody.py finalize --final-dir dataset/coco_profile_fullbody_copie --decisions Documentation/coco_fullbody_review_20261103.json --positive-count 125 --negative-count 40
```

Il ne faut pas copier l'environnement `.venv` du Mac vers Windows. Les jeux de données restent locaux et sont ignorés par Git : envoyer seulement les scripts sur GitHub ne transmet pas les photos. Les licences 4, 5 et 7 sont présentes dans le lot ; vérifier les conditions et compléter les attributions nécessaires avant redistribution ou publication. Le CSV garde les liens, mais ne constitue pas à lui seul une autorisation générale.

## Prochaine étape

1. Vérifier quelques images dans la galerie pour confirmer la vue visée.
2. Préparer un essai d'entraînement séparé, avec un détecteur et un prétraitement compatibles. Le modèle QQVGA actuel attend 160 × 120 : ne pas lui donner directement les PNG 320 × 240 sans adapter ou refaire le prétraitement et les rectangles ensemble.
3. Comparer le modèle avant/après sur les mêmes données de validation, sans les intégrer à l'apprentissage.
4. Mesurer ensuite la détection et le comptage sur de nouvelles vidéos de la vraie installation.

Source officielle : [COCO](https://cocodataset.org/#download).
