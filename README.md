# Concours AIDGE 2026

## FlowSense

FlowSense est notre projet pour le Challenge AIDGE 2026. Nous développons un capteur vidéo embarqué capable de détecter et compter les passages de personnes à une porte ou dans un couloir, avec une consommation cible inférieure à 5 mW sur ColibryNPU.

Le traitement doit rester local. Le système ne transmet ni ne stocke les images ; il ne produit que des données agrégées telles que le nombre d'entrées, de sorties et l'occupation estimée.

Cette phrase décrit la cible embarquée. Le prototype de développement sur PC
conserve localement des images et vidéos pour l'entraînement et les essais ;
elles ne sont pas publiées sur GitHub.

## Pour les camarades sur Windows

1. Récupérer le dépôt à jour (`git pull --ff-only` ou **Code → Download ZIP**, puis extraire le ZIP).
2. Ouvrir le dossier extrait dans VS Code, avec Python 3.11 ou 3.12 installé.
3. Double-cliquer sur **Installer_Windows.bat**, puis sur **FlowSense_Windows.bat** pour l'interface YOLO.
4. Pour vérifier les annotations du jeu partagé : **Relecture_Windows.bat**, puis choisir validation ou entraînement.

Le [guide Windows complet](Documentation/Installation_Windows.md) explique
l'installation, les commandes PowerShell, les fichiers privés à récupérer et
la différence entre l'interface YOLO et l'entraînement de notre modèle Aidge.
L'installateur Windows installe l'interface et PyTorch CPU, **pas Aidge**.
Les tests automatiques Windows/Linux sont définis dans
`.github/workflows/tests.yml` ; ils ne valident ni l'affichage interactif ni
l'exécution Aidge/Colibry sur Windows.

## Documents du dépôt

- Vidéo explicative du projet : https://youtu.be/_fUpL8oQr6g
- [Fiche projet FlowSense](Documentation/Fiche_projet_FlowSense.docx)
- [Guide pour entraîner le détecteur](Documentation/Guide_entrainement_FlowSense.md)
- [Installation et utilisation sous Windows](Documentation/Installation_Windows.md)
- [Corriger les images de profil](Documentation/Corriger_les_images_profils.md)
- [Plaquette du Challenge AIDGE](Documentation/Plaquette_ChallengeAidge.pdf)
- [Règlement du Challenge AIDGE](Documentation/Reglement-ChallengeAidge.pdf)
-  Site inernet du Challenge : https://new.express.adobe.com/webpage/7GXZmONRFoJte
-  Eclipse Project AIDGE AI : https://gitlab.eclipse.org/eclipse/aidge/aidge
-  Depot Git AIDGE AI : https://eclipse.dev/aidge/index.html
  
## État actuel

**Phase : prototypes logiciels et préparation du modèle embarqué**

Le dépôt contient un premier classifieur `personne/vide`, une démonstration
vidéo avec YOLO et un prototype de carte de positions testé dans Aidge sur CPU.
Le comptage existe dans la démonstration, mais le petit modèle Aidge n'a pas
encore une qualité de détection validée et n'est pas déployé sur ColibryNPU.

Les essais du détecteur gris 160 × 120, les corrections d'annotations et les
commandes correspondantes sont dans le [bilan de correction QQVGA](Documentation/Bilan_correction_detecteur_QQVGA.md).
La comparaison des images originales, QVGA et QQVGA est dans le
[diagnostic de résolution caméra](Documentation/Diagnostic_resolution_camera.md).

Un nouvel essai **COCO + vidéos de profil en gris 320 × 240**, entraîné dans
Aidge CPU, est décrit dans le [bilan et guide de reproduction](Documentation/Entrainement_COCO_profils_QVGA.md).
Il est conservé séparément et ne remplace pas automatiquement le modèle de
l'interface. Les résultats restent provisoires, sans validation sur carte.
La [poursuite v2](Documentation/Poursuite_profils_QVGA_v2.md), après validation
humaine des annotations locales de validation, réduit les fausses détections
mais manque davantage de personnes : elle ne remplace pas automatiquement la v1.

## Étape 1 — première IA PyTorch : `personne / vide`

Avant de travailler sur Aidge ou le matériel, nous créons une base simple : un réseau de neurones qui reçoit une image et répond soit :

```text
0 = vide
1 = personne
```

Le modèle utilise des images RGB redimensionnées en **64 × 64 pixels**. Il ne compte pas encore les entrées, les sorties ou plusieurs personnes : il répond seulement à la question « une personne est-elle présente dans cette image ? ».

### 1. Placer les images

Ajoutez vos images JPG ou PNG dans les quatre dossiers suivants :

```text
dataset/
├── train/
│   ├── personne/      # images contenant au moins une personne
│   └── vide/          # images ne contenant personne
└── validation/
    ├── personne/      # mêmes catégories, images différentes de train
    └── vide/
```

Les images de données restent locales et ne sont pas publiées sur GitHub. Pour démarrer, essayez d'avoir des exemples variés : différentes lumières, distances, positions et arrière-plans, dans les deux classes.

### Base de départ automatique

Le script suivant télécharge une petite base locale équilibrée issue de **Wake Vision**, un dataset public destiné à la détection binaire de personnes en TinyML. Il récupère par défaut 100 images `vide` et 100 images `personne` pour `train`, puis 25 images de chaque classe pour `validation` :

```bash
python scripts/download_wake_vision_subset.py
```

Les images téléchargées restent sur l'ordinateur et ne sont pas envoyées sur GitHub. Consultez [dataset/DATASET.md](dataset/DATASET.md) pour la source, la licence et les limites de cette base de départ.

### 2. Installer les dépendances Python

Sur Mac, depuis le dossier du projet, créez un environnement pour cette IA puis installez les dépendances :

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Entraîner le modèle

Après avoir placé les images dans les dossiers ci-dessus :

```bash
python src/train.py --epochs 15 --batch-size 32
```

Le script affiche la `loss` et l'`accuracy` à chaque époque. Le meilleur modèle est automatiquement sauvegardé dans :

```text
models/best_personne_vide_cnn.pt
```

### 4. Tester une image

Une fois l'entraînement terminé, testez une nouvelle image avec :

```bash
python src/test.py --image chemin/vers/une_image.jpg
```

Le résultat affichera `PERSONNE` ou `VIDE`, puis le niveau de confiance.

## Prototype Aidge — carte de positions (non entraîné)

Ce prototype est séparé du classifieur `personne/vide` et de la démo YOLO. Il
reçoit une image RGB de **96 × 96** pixels et produit une grille de **20 × 20**
scores. À terme, une case à score élevé pourra indiquer une personne, mais
**pour l'instant les poids sont aléatoires** : la grille ne détecte rien de
fiable. L'objectif de cette étape est de vérifier le chemin PyTorch → ONNX →
Aidge avant de préparer des annotations et d'entraîner le modèle.

Depuis la racine du projet, après installation de `requirements.txt` dans
`.venv`, générez le modèle ONNX et une entrée de référence :

```bash
./.venv/bin/python src/export_tiny_heatmap.py
```

Sur ce Mac, Aidge est dans un environnement séparé. Comparez sa sortie CPU à
celle de PyTorch avec :

```bash
./.venv-aidge/bin/python src/check_tiny_heatmap_aidge.py
```

Sous Windows, les scripts Python sont les mêmes ; le chemin de l'interpréteur
devient `.venv\Scripts\python.exe` (et équivalent pour l'environnement Aidge,
si Aidge y est installé). Les fichiers générés sont dans `runs/tiny_heatmap/`
et ne sont pas publiés sur GitHub. Le réseau contient **3 729 paramètres**.
Le prototype initial utilise `Hardtanh`. Une variante `ReLU` est testée plus
bas pour l'export Colibry ; l'erreur `std::bad_cast` observée avec l'importeur
ONNX d'origine a été corrigée dans l'environnement Aidge local dédié.

### Préparer et corriger les images de détection

Les vidéos locales de `videos_test/` ont servi à créer
`dataset/detection_proposals/` : **237 images**, dont 179 pour l'entraînement
et 58 issues de quatre autres vidéos pour la validation. YOLO a proposé 240
rectangles. Ces fichiers restent locaux (ignorés par Git). Ils ne sont **pas**
des annotations fiables tant qu'une personne ne les a pas vérifiés.
Sur un autre ordinateur, après avoir copié les vidéos **en privé** dans
`videos_test/`, on peut recréer ces propositions avec
`python scripts/prepare_detection_dataset.py`. Le script refuse d'écraser un
jeu existant pour protéger les corrections humaines.

Les **237 images** (179 entraînement, 58 validation) ont été vérifiées. Pour
rouvrir les images de validation sur Mac :

```bash
./.venv/bin/python scripts/review_detection_dataset.py --split validation --include-reviewed
```

Dans la fenêtre, glissez avec le clic gauche pour ajouter un rectangle, faites
un clic droit sur un rectangle erroné pour l'enlever, puis appuyez sur `S` pour
enregistrer et passer à l'image suivante. `N` passe sans valider ; `Q` quitte.
Sur Windows, utilisez `.venv\Scripts\python.exe` à la place de
`./.venv/bin/python`. Le fichier `manifest.csv` indique `reviewed=oui` pour les
images corrigées. Pour revoir ensuite l'entraînement, utilisez `--split train`.

Les essais précédents sont conservés dans `models/tiny_heatmap_pseudo.pt` et
`models/tiny_heatmap_val_corrigee.pt`. Après correction de toutes les images,
un nouvel entraînement de 50 époques a produit `models/tiny_heatmap_corrige.pt`.
Pour reproduire l'entraînement et vérifier l'export :

```bash
./.venv/bin/python src/train_tiny_heatmap.py --epochs 50 --checkpoint models/tiny_heatmap_corrige.pt
./.venv/bin/python src/evaluate_tiny_heatmap.py --checkpoint models/tiny_heatmap_corrige.pt --threshold 0.5
./.venv/bin/python src/export_tiny_heatmap.py --checkpoint models/tiny_heatmap_corrige.pt --output-dir runs/tiny_heatmap_corrige
./.venv-aidge/bin/python src/check_tiny_heatmap_aidge.py --run-dir runs/tiny_heatmap_corrige
```

Au seuil fixe `0.5`, l'évaluation *provisoire* des centres sur les 58 images
de validation donne **29 % de précision** et **65 % de rappel**. Mais un
contrôle visuel a révélé que certaines images marquées « sans personne »
contiennent en réalité une personne partiellement visible (par exemple la
13e image de validation). Les chiffres de fausses alertes et la précision
ne sont donc pas fiables tant que la règle d'annotation des personnes
partielles n'est pas appliquée de façon cohérente. La loss de validation du
meilleur modèle est `0.4516`. La compatibilité Aidge CPU est vérifiée ; la
qualité du détecteur et la compatibilité ColibryNPU restent à établir.

Ces commandes servent à **fabriquer et tester** le modèle. Une fois la caméra
installée, elle n'aura pas besoin de ces annotations pour traiter une nouvelle
vidéo. L'emplacement de la porte et le sens entrée/sortie restent un réglage
séparé du détecteur.

### Essai d'une version ReLU (12 octobre 2026)

L'ancien modèle `Hardtanh` est conservé. Pour entraîner une variante dont le
fichier ONNX contient `Relu` à la place de `Clip` :

```bash
./.venv/bin/python src/train_tiny_heatmap.py --data-dir dataset/detection_proposals --activation relu --epochs 50 --checkpoint models/tiny_heatmap_relu.pt
./.venv/bin/python src/evaluate_tiny_heatmap.py --checkpoint models/tiny_heatmap_relu.pt --data-dir dataset/porte_B_test --threshold 0.5
./.venv/bin/python src/export_tiny_heatmap.py --checkpoint models/tiny_heatmap_relu.pt --output-dir runs/tiny_heatmap_relu
```

Les scripts d'évaluation et d'export lisent l'activation enregistrée dans le
checkpoint ; les anciens checkpoints sans cette information restent traités
comme des modèles `Hardtanh`.

À seuil et méthode de comptage des pics identiques, sur les **27 images de
porte B** tenues à part (24 personnes annotées), l'ancien modèle obtient
21 correspondances, 16 faux positifs et 3 oublis (F1 indicatif : 68,9 %).
La variante ReLU obtient 14 correspondances, 17 faux positifs et 10 oublis
(F1 indicatif : 50,9 %). Sur les 58 images de validation initiales, les F1
sont respectivement 42,1 % et 41,8 %. La variante ReLU n'est donc **pas une
amélioration de détection** à ce stade.

L'ONNX ReLU donne numériquement la même sortie que PyTorch sur l'image de
référence. Avec l'environnement local `.venv-aidge-colibry` (Aidge
0.10.0.post3 et correctifs de compatibilité), l'exécution Aidge CPU réussit :
écart maximal `1,19 × 10⁻⁶` sur une entrée de référence. Les scores ci-dessus
mesurent des centres sur des images, pas le suivi ni le comptage vidéo.

### Export logiciel Colibry (12 octobre 2026)

Sur le Mac du projet, l'environnement `.venv-aidge-colibry` et les sources
locales `.aidge-colibry-source` sont installés. Ces dossiers, volumineux et
propres à la machine, sont ignorés par Git : les camarades sous Windows
doivent installer leur propre environnement Aidge. Depuis la racine du projet :

```bash
./.venv/bin/python src/export_tiny_heatmap.py --checkpoint models/tiny_heatmap_relu.pt --output-dir runs/tiny_heatmap_relu
./.venv-aidge-colibry/bin/python src/check_tiny_heatmap_aidge.py --run-dir runs/tiny_heatmap_relu
./.venv/bin/python scripts/prepare_colibry_calibration.py --max-images 32
./.venv-aidge-colibry/bin/python scripts/export_colibry.py
```

Le dernier script utilise **32 images réelles de l'entraînement** pour régler
les échelles INT8, puis génère
`runs/colibry_relu/export_graph_int8_colibry/dnn/aidge_model_bitstream.nnpa`.
L'outil d'export indique que le modèle tient dans les contraintes mémoire
Colibry (4 443 mots de données, 4 128 octets de configuration) et estime
20,1 µJ par inférence. Cette énergie estimée ne démontre **pas** une puissance
moyenne de 5 mW. La comparaison CPU après quantification est décrite ci-dessous ;
l'exécution sur la carte et le comptage entrée/sortie ne sont pas validés. La variante ReLU est pour
l'instant moins précise que le modèle Hardtanh sur la porte B ; cet export
prouve seulement que la chaîne logicielle est possible.

### Comparer FP32 et INT8 sur la porte B (20 octobre 2026)

Une commande prépare les 27 images de test, calibre uniquement sur 32 images
du train, exécute Aidge FP32 et INT8, puis crée un rapport avec les images
comparées. Elle ne réentraîne pas le modèle et ne règle pas le seuil sur le test.

```bash
./.venv/bin/python scripts/compare_tiny_heatmap_int8.py
```

Sous Windows, si l'environnement Aidge compatible est installé :

```powershell
.venv\Scripts\python.exe scripts/compare_tiny_heatmap_int8.py
```

Le chemin de l'interpréteur Aidge peut être précisé avec `--aidge-python`.
Ouvrir ensuite `runs/comparaison_int8_porte_B/rapport.html` dans un navigateur.
Le dossier contient aussi `details.csv`, `metrics.json`, les sorties numériques
et 27 aperçus. Les images restent locales, dans `runs/` ignoré par Git.

| Mesure (24 présences annotées dans 27 images) | FP32 | INT8 CPU |
|---|---:|---:|
| Présences correctement localisées | 14 | 13 |
| Centres annotés manqués | 10 | 11 |
| Points supplémentaires ou hors tolérance | 17 | 19 |
| F1 de localisation | 50,9 % | 46,4 % |

Les quatre convolutions utilisent effectivement des activations et poids INT8,
avec biais et accumulation INT32 ; les adaptateurs d'entrée/sortie sont flottants.
Aidge FP32 reproduit PyTorch sur ces 27 images (écart maximal `3,81 × 10⁻⁶`).
Ce test ne lance pas le bitstream Colibry. Un point sur la tête mais trop éloigné
du centre du rectangle annoté peut compter à la fois comme point hors tolérance
et comme centre manqué. Voir le [bilan et les exemples](Documentation/Bilan_INT8_porte_B.md)
avant d'interpréter ces nombres comme des personnes entièrement non détectées.

### Modèle V2 : apprentissage Aidge avec COCO et les portes (20 octobre 2026)

Une expérience séparée utilise un CNN en niveaux de gris, 17 885 paramètres,
une grille de centres 20×20 et un contexte élargi. Ses calculs avant/arrière
sont exécutés dans Aidge CPU ; la BCE et Adam utilisent NumPy, avec contrôle
préalable des gradients. PyTorch initialise le graphe et sert de référence,
sans effectuer cet entraînement.

Les données combinent un sous-ensemble officiel COCO Person (500 images de train,
125 de validation) et nos portes annotées (179 de train, 58 de validation).
Les 27 images porte B ne participent pas à l'apprentissage ou au choix de l'époque.
Les anciens modèles sont conservés. Voir le [guide avec commandes Mac et Windows](Documentation/Entrainement_Aidge_COCO.md)
pour télécharger les données, entraîner et générer la comparaison illustrée.
Les poids, les images et les résultats de cette expérience restent locaux dans
`dataset/coco_person_subset/` et `runs/heatmap_v2/`. Le modèle n'est pas encore
intégré au comptage vidéo ; la cible UINT8 finale et la consommation sur carte
restent à vérifier.

Bilan du premier essai : sur la porte B, le F1 est de **35,3 % en FP32** et
**33,9 % en INT8 CPU**, contre **50,9 % pour l'ancien ReLU** et **68,9 % pour
l'ancien Hardtanh**. Le nouvel entraînement n'est donc pas retenu pour remplacer
les anciens modèles. La validation locale progresse, mais cette amélioration
ne se généralise pas à cette nouvelle porte. Voir le [bilan de l'essai COCO](Documentation/Bilan_Aidge_COCO_V2.md).

### Images publiques vues du dessus

Un lot local WEPDTOF de 120 recadrages annotés a été préparé : 80 images
d'apprentissage issues de quatre lieux et 40 de validation issues de deux
autres lieux. Les zones non annotées sont exclues. Ce sont des annotations
sources converties, pas des détections de notre réseau ni une validation humaine
complète. Un essai distinct, `runs/heatmap_wepdtof_v1/`, a été entraîné après
confirmation du point de vue de l'exemple 001 par l'équipe. Sur la porte B,
il obtient **43,5 % de F1 en FP32** et **42,6 % en INT8 CPU**, contre 35,3 %
en FP32 pour la V2 COCO. Il reste moins bon que les anciens modèles sur cette
porte et ne retrouve que 2 centres sur 54 sur les autres lieux publics : il ne
remplace donc pas les anciens modèles. Voir le [bilan de l'essai WEPDTOF](Documentation/Bilan_Aidge_WEPDTOF.md).
Les images restent locales dans `dataset/wepdtof_subset/` ; voir le
[rapport et les commandes de l'essai](Documentation/Lot_vues_du_dessus.md).

Un lot augmenté conserve ces validations et porte le public d'apprentissage à
400 images issues de huit lieux, soit **579 images d'apprentissage** avec nos
portes. Les 320 ajouts proviennent de vidéos publiques annotées ; aucune image
de la porte B n'est ajoutée au train. Le [guide de poursuite de l'entraînement](Documentation/Entrainement_WEPDTOF_augmente.md)
donne les commandes Mac/Windows, la reprise des poids du premier essai et les
contrôles à effectuer. L'objectif de 90 % concerne une future évaluation de
passages correctement détectés dans le bon sens, pas le score d'apprentissage.

Résultat du lot augmenté : **45,5 % de F1 sur la porte B**, contre 43,5 % pour
le premier essai WEPDTOF ; toujours 10 centres retrouvés sur 24, avec deux points
hors tolérance en moins. L'INT8 CPU donne 44,4 %. Le modèle final perd cependant
une grande partie de sa détection publique pendant l'adaptation aux portes :
le checkpoint public donne 55,4 % sur les 40 images publiques de validation,
contre 7,0 % après adaptation. La priorité est de corriger cette étape, pas de
considérer les 90 % comme acquis. Les anciens modèles restent conservés.
Voir le [bilan détaillé du lot augmenté](Documentation/Bilan_Aidge_WEPDTOF_v2.md).

Essai suivant : le mélange 50/50 et la sélection sur les deux validations
remontent le F1 public à **42,1 %**, mais dégradent la porte B à **29,9 %**
(29,4 % en INT8 CPU). Cette V3 n'est pas retenue pour remplacer les anciens
modèles. Voir le [bilan de l'essai équilibré](Documentation/Bilan_Aidge_equilibre_v3.md)
et ses [commandes reproductibles](Documentation/Entrainement_equilibre.md).

Le [prétraitement caméra commun](Documentation/Images_camera_QQVGA.md) prépare
aussi 704 copies en **160×120 gris**, sources et annotations conservées, dans
`dataset/camera_qqvga_v1/`. Ce lot est destiné à un futur réseau QQVGA : les
scripts V2 actuels restent à 80×80 utiles et les poids existants ne sont pas
réentraînés par cette préparation. Les originaux ne sont pas écrasés.

### Nouveau détecteur QQVGA : test technique de mémorisation (29 octobre 2026)

Un détecteur séparable de **4 645 paramètres** utilise maintenant réellement
l'entrée grise 160×120 et produit des rectangles. Son apprentissage Aidge sur
20 images de train (10 locales, 10 publiques, dont 4 vides) retrouve les **17
rectangles sans détection supplémentaire**, en FP32 puis en INT8 CPU. Le test
est arrêté à l'époque 70, avec les poids de l'époque 60 retenus.

**Ce sont les images apprises, pas une évaluation sur des images inconnues.**
Le score ne remplace pas celui de la porte B, et aucun modèle n'a été remplacé
dans l'interface vidéo. L'export logiciel produit un bitstream et le compilateur
valide ses contraintes mémoire, mais la carte, l'UINT8 final et les 5 mW ne sont
pas validés. Le [bilan et les commandes Mac/Windows](Documentation/Detecteur_QQVGA_test20.md)
expliquent le réseau, le contournement du backward depthwise et les limites.
Résultats locaux : `runs/person_qqvga_mem20_v1/rapport.html`.

### Apprentissage QQVGA sur le lot complet

Le [guide du détecteur QQVGA complet](Documentation/Entrainement_detecteur_QQVGA.md)
décrit maintenant un entraînement séparé sur **579 images** (179 de nos portes
et 400 de WEPDTOF), avec **98 images de validation** hors apprentissage. Chaque
époque utilise toutes les images, avec un mélange équilibré et des augmentations
réservées au train. La porte B reste exclue. Le checkpoint est choisi sur les
deux validations, pas sur les 20 images du test de mémorisation. Les commandes
Mac/Windows et les limites du protocole figurent dans le guide. L'interface et
les modèles précédents ne sont pas remplacés automatiquement.

La poursuite du 30 octobre est terminée : **F1 2,35 % sur nos portes de validation
et 0 % sur WEPDTOF**, soit une seule présence retrouvée sur 138. Cet essai n'est
pas retenu pour remplacer le compteur. La conversion compacte Aidge est vérifiée,
mais la qualité est insuffisante. Voir le [bilan du lot complet](Documentation/Bilan_detecteur_QQVGA_lot_complet.md)
et le rapport local `runs/person_qqvga_full_v2/rapport.html`.

## Étape suivante - démonstration YOLO : rectangles autour des personnes

Cette démonstration est séparée de la première IA `vide/personne`. Elle utilise
un petit modèle YOLO déjà entraîné pour repérer la position de chaque personne
sur une image. Elle ne fait pas encore de suivi vidéo ni de comptage.

```bash
python src/detect_image.py --image chemin/vers/une_image.jpg
```

Au premier lancement, le modèle nano est téléchargé. Une copie de l'image avec
les rectangles est enregistrée dans `runs/detect_image/`. Le dossier `runs/`
est généré localement et n'est pas envoyé sur GitHub.

### Calibrer une porte et compter des passages

Sur Mac, double-cliquez sur `FlowSense.command` dans le Finder pour ouvrir
l'interface à boutons. Choisissez une vidéo, placez les deux traits et le point
du côté intérieur, enregistrez la porte, puis cliquez sur « Lancer l'analyse ».
Les totaux et un bouton pour ouvrir la vidéo résultat apparaissent dans la
fenêtre. Si la caméra reste au même endroit, le bouton « Réutiliser un réglage »
évite de redessiner la porte.

Sur Windows, après avoir récupéré le dépôt, installez Python 3.11 ou 3.12
(64 bits) depuis https://www.python.org/downloads/windows/ en cochant
« Add python.exe to PATH ». Double-cliquez une seule fois sur
`Installer_Windows.bat` : ce fichier crée l'environnement `.venv` et installe les
dépendances sur ce PC. Ensuite, double-cliquez sur `FlowSense_Windows.bat`
pour ouvrir la même interface. Aucune version Pro de GitHub ou de VS Code
n'est nécessaire. Chaque ordinateur possède son propre `.venv` ; ne copiez
pas celui du Mac vers Windows.

Dans l'interface, sélectionnez la vidéo, placez les cinq points, enregistrez
le réglage, puis lancez l'analyse. La vidéo résultat est dans
`runs/count_video/`, sur Mac comme sur Windows. L'export MP4 utilise
H.264 avec le FFmpeg installé via les dépendances Python.

Les commandes ci-dessous utilisent la syntaxe du terminal Mac/Linux. Sur
Windows, utilisez les deux fichiers .bat et les boutons de l'interface.

La caméra ne doit pas voir de vrais traits au sol. Lors de l'installation, on
définit une fois la zone de porte dans l'image : deux limites et un point du
côté intérieur. Le réglage ne modifie pas l'IA et peut être réutilisé pour
toutes les vidéos filmées par la même caméra fixe.

```bash
python src/calibrate_gate.py \
  --video videos_test/ma_video.mp4 \
  --frame 50 \
  --output configs/porte_01.json
```

Cliquez les quatre extrémités des deux lignes, puis cliquez un point du côté
intérieur et appuyez sur `S`. Pour analyser ensuite une vidéo avec ce réglage :

```bash
python src/count_video.py \
  --video videos_test/ma_video.mp4 \
  --config configs/porte_01.json
```

Le résultat affiche les rectangles, les identifiants de suivi, la zone de
porte virtuelle et les totaux `Entrées` / `Sorties`. Une nouvelle caméra ou
une nouvelle porte demande simplement un nouveau fichier JSON de réglage ; le
code et le modèle restent les mêmes.

### Fichiers de cette étape

| Fichier | Rôle |
|---|---|
| `src/model.py` | Définit le petit réseau convolutif (CNN). |
| `src/preprocess.py` | Redimensionne les images en 64 × 64 et impose `0=vide`, `1=personne`. |
| `src/train.py` | Charge les données, entraîne le modèle, mesure ses résultats et sauvegarde le meilleur modèle. |
| `src/test.py` | Charge le modèle sauvegardé et prédit une seule image. |
| `src/calibrate_gate.py` | Petite interface locale pour régler une porte en cliquant dans l'image. |
| `src/count_video.py` | Détecte, suit et compte les passages à travers une porte calibrée. |
| `src/app.py`, `FlowSense.command` et `FlowSense_Windows.bat` | Interface à boutons et lancement par double-clic sur Mac ou Windows. |
| `Installer_Windows.bat` | Installation des dépendances sur un PC Windows. |
| `configs/` | Réglages de porte en JSON et mode d'emploi. |
| `requirements.txt` | Liste les dépendances Python nécessaires. |

## Cas d'usage choisi

Une caméra fixe est installée à une porte ou dans un couloir. Le système :

1. détecte les personnes dans une zone définie ;
2. suit leur déplacement sur quelques images ;
3. détecte le franchissement d'une ligne virtuelle ;
4. compte les entrées et les sorties ;
5. estime l'occupation d'un espace ;
6. conserve le traitement vidéo local.

## Architecture envisagée

```text
Caméra embarquée
        ↓
Réduction de la résolution de l'image
        ↓
Détection de mouvement légère
        ↓
Petit réseau neuronal quantifié INT8 sur ColibryNPU
        ↓
Détection ou estimation du nombre de personnes
        ↓
Suivi court et ligne virtuelle sur CPU
        ↓
Entrées, sorties et occupation estimée
```

## Décisions déjà prises

- Le nom de travail du projet est **FlowSense**.
- Le scénario principal est le comptage de passages à une porte.
- Le traitement doit être effectué localement, sans cloud.
- Le modèle doit être compact et quantifié en INT8.
- Aidge servira à importer le modèle ONNX, analyser son graphe, l'optimiser, le quantifier et l'exporter.
- Nous commencerons par une version simple et fonctionnelle avant les optimisations avancées.

## Plan de progression

| Version | Objectif |
|---|---|
| MVP | Détecter la présence d'une personne dans une zone définie. |
| Version 2 | Distinguer le sens de passage grâce à une ligne virtuelle. |
| Version 3 | Compter plusieurs passages, limiter les doubles comptes et mesurer la consommation réelle. |

## Mesures de réussite

| Aspect | Mesures prévues |
|---|---|
| Détection | Précision, rappel, F1-score et faux positifs |
| Comptage | Erreur absolue, doubles comptes et erreurs entrée/sortie |
| Temps réel | Latence par inférence et images par seconde |
| Mémoire | Taille du modèle et mémoire maximale utilisée |
| Énergie | Énergie par inférence et puissance moyenne |
| Confidentialité | Aucune image transmise ou stockée |

## Prochaines actions

- [ ] Tester l'environnement et les premiers tutoriels Aidge.
- [ ] Choisir une baseline légère de détection de personnes.
- [ ] Préparer un dataset et les scripts d'évaluation.
- [ ] Répartir les rôles dans l'équipe.
- [ ] Préparer les questions pour les organisateurs.
- [ ] Déployer et mesurer le modèle après réception du matériel.

## Points à confirmer avec les organisateurs

- Les 5 mW incluent-ils la caméra, l'acquisition et le mode veille ?
- Le seuil concerne-t-il une puissance moyenne, un pic ou une énergie par inférence ?
- Quel scénario de comptage et quelle fréquence d'images seront évalués ?
- Quels opérateurs sont accélérés par ColibryNPU ?
- Existe-t-il un exporteur Aidge vers ColibryNPU ?
- Quel outil permettra de mesurer la consommation ?

## Membre de l'equipe

- EL BAROUD Elias
- EL GAAMOUCH Aïssa
- MAHI Rayan
- MEBARKI Ilian
