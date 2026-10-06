# Concours AIDGE 2026

## FlowSense

FlowSense est notre projet pour le Challenge AIDGE 2026. Nous développons un capteur vidéo embarqué capable de détecter et compter les passages de personnes à une porte ou dans un couloir, avec une consommation cible inférieure à 5 mW sur ColibryNPU.

Le traitement doit rester local. Le système ne transmet ni ne stocke les images ; il ne produit que des données agrégées telles que le nombre d'entrées, de sorties et l'occupation estimée.

## Documents du dépôt

- Vidéo explicative du projet : https://youtu.be/_fUpL8oQr6g
- [Fiche projet FlowSense](Documentation/Fiche_projet_FlowSense.docx)
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
Une activation bornée (`Hardtanh`) remplace ici `ReLU`, car l'exécution CPU
Aidge 0.10.1 de `ReLU` échoue actuellement sur ce Mac avec `std::bad_cast` ;
la compatibilité matérielle Colibry n'est pas encore vérifiée.

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
