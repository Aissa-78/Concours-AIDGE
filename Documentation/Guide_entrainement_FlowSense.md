# Guide pour entraîner le détecteur FlowSense

**Nouvelle expérience :** pour le modèle V2 entraîné dans Aidge avec COCO et les images de portes, suivre [Entraînement Aidge + COCO](Entrainement_Aidge_COCO.md). Le présent guide reste celui de l'ancien modèle PyTorch.

Ce guide permet de refaire l'entraînement du **petit modèle PyTorch** sans utiliser Codex. Il reçoit une image de 96 × 96 pixels et produit une carte de positions possibles des personnes. Il ne suit pas encore les personnes et ne compte pas les entrées ou les sorties. La démonstration vidéo qui utilise YOLO est un programme distinct.

## 1 Ouvrir le projet et le terminal

Dans VS Code, ouvrir le dossier `FlowSense-AIDGE`, puis choisir **Terminal > Nouveau terminal**. Les commandes ci-dessous se tapent dans ce terminal, à la racine du projet : on doit y voir `src/`, `dataset/` et `requirements.txt`.

Sur le Mac actuel, si nécessaire :

```bash
cd "/Users/houdamebarki/Documents/FlowSense-AIDGE"
```

Sur Windows, ouvrir le dossier cloné dans VS Code avant de créer le terminal. Il n'est pas nécessaire d'avoir GitHub Pro ou Copilot.

## 2 Installer Python et vérifier les dépendances

Sur le Mac actuel, l'environnement `.venv` existe déjà. Pour le créer sur un autre Mac et installer les dépendances :

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -c "import torch, cv2; print('Installation OK')"
```

Sur Windows, double-cliquer une fois sur `Installer_Windows.bat` depuis l'Explorateur de fichiers. Il faut Python 3.11 ou 3.12. Puis, dans le terminal de VS Code :

```powershell
.\.venv\Scripts\python.exe -c "import torch, cv2; print('Installation OK')"
```

Dans la suite, **Mac** utilise `./.venv/bin/python` et **Windows** utilise `.\.venv\Scripts\python.exe`. Ce sont deux chemins vers le Python propre au projet ; ne pas taper les deux versions d'une même commande.

## 3 Comprendre les données avant de lancer le calcul

Le jeu déjà préparé est dans `dataset/detection_proposals/`. Son fichier `manifest.csv` liste les images, les vidéos d'origine et leur groupe. Les images sont dans `train/images/` et `validation/images/`. Les fichiers `labels/` portent les rectangles des personnes, un fichier texte par image.

Les **179 images de train** servent à modifier les poids du réseau. Les **58 images de validation**, issues d'autres vidéos, servent à choisir le meilleur modèle pendant l'entraînement. Les 237 images ont été revues manuellement, mais une erreur d'annotation reste toujours possible.

Les quatre vidéos `porte_B_...` sont réservées à un **test indépendant** dans `dataset/porte_B_test/`. Ne pas les ajouter à l'entraînement : sinon le test ne dira plus si le modèle fonctionne dans un nouvel endroit. Copier simplement des vidéos dans `videos_test/` ne les ajoute pas au jeu d'entraînement existant.

Les vidéos, images extraites, annotations et modèles entraînés restent sur l'ordinateur : ils sont ignorés par Git. Un camarade qui clone seulement GitHub doit recevoir ces données séparément ou recréer son propre jeu.

## 4 Corriger les rectangles si nécessaire

Avant d'entraîner, vérifier que chaque personne visible a un rectangle et qu'aucun objet n'en a. Une image sans personne doit garder **zéro rectangle**. Dans la fenêtre d'annotation : glisser avec le clic gauche pour ajouter un rectangle, clic droit pour retirer un faux rectangle, `S` pour enregistrer et avancer, `P` pour revenir, `Q` pour quitter.

Pour rouvrir les images d'entraînement déjà vérifiées :

```bash
./.venv/bin/python scripts/review_detection_dataset.py --split train --include-reviewed
```

Sur Windows :

```powershell
.\.venv\Scripts\python.exe scripts/review_detection_dataset.py --split train --include-reviewed
```

Remplacer `train` par `validation` pour revoir la validation. Pour un nouveau jeu dont les images ne sont pas encore vérifiées, **retirer** `--include-reviewed`.

## 5 Entraîner le modèle actuel

Lancer cette commande depuis le dossier du projet. Elle recommence un entraînement à partir de poids neufs ; elle ne continue pas automatiquement l'ancien modèle.

```bash
./.venv/bin/python src/train_tiny_heatmap.py --data-dir dataset/detection_proposals --epochs 50 --checkpoint models/tiny_heatmap_mon_essai.pt
```

Sur Windows :

```powershell
.\.venv\Scripts\python.exe src/train_tiny_heatmap.py --data-dir dataset/detection_proposals --epochs 50 --checkpoint models/tiny_heatmap_mon_essai.pt
```

À chaque époque, le terminal affiche `loss train` et `loss validation`. La *loss* mesure l'écart entre les positions prévues par le réseau et les annotations : plus elle est basse, mieux c'est pour cette mesure. Si la loss de train baisse mais que celle de validation monte, le modèle risque de mémoriser les exemples de train. Le programme conserve dans `models/tiny_heatmap_mon_essai.pt` les poids de l'époque qui a obtenu la meilleure loss de validation, pas forcément ceux de la dernière époque. Ne pas effacer ce fichier.

## 6 Mesurer le résultat sur la validation

Sur Mac :

```bash
./.venv/bin/python src/evaluate_tiny_heatmap.py --data-dir dataset/detection_proposals --checkpoint models/tiny_heatmap_mon_essai.pt --threshold 0.5
```

Sur Windows :

```powershell
.\.venv\Scripts\python.exe src/evaluate_tiny_heatmap.py --data-dir dataset/detection_proposals --checkpoint models/tiny_heatmap_mon_essai.pt --threshold 0.5
```

La **précision** indique quelle part des positions prédites correspond à une personne annotée. Le **rappel** indique quelle part des personnes annotées a été retrouvée. Les *faux positifs* sont des détections en trop ; les *oublis* sont des personnes manquées. Cette mesure compare des centres sur des images : elle ne mesure **ni le suivi vidéo ni le nombre d'entrées et de sorties**. Garder le même seuil `0.5` lorsque l'on compare deux entraînements.

## 7 Tester sur les nouvelles vidéos porte B

Le jeu de test `dataset/porte_B_test/` contient **27 images** extraites des quatre nouvelles vidéos. Les rectangles proposés automatiquement doivent d'abord être corrigés par une personne.

Sur Mac :

```bash
./.venv/bin/python scripts/review_detection_dataset.py --dataset-dir dataset/porte_B_test
./.venv/bin/python src/evaluate_tiny_heatmap.py --data-dir dataset/porte_B_test --checkpoint models/tiny_heatmap_mon_essai.pt --threshold 0.5
```

Sur Windows :

```powershell
.\.venv\Scripts\python.exe scripts/review_detection_dataset.py --dataset-dir dataset/porte_B_test
.\.venv\Scripts\python.exe src/evaluate_tiny_heatmap.py --data-dir dataset/porte_B_test --checkpoint models/tiny_heatmap_mon_essai.pt --threshold 0.5
```

La deuxième commande refusera de démarrer tant que toutes les images n'ont pas été enregistrées avec `S`. C'est voulu : un test sur des rectangles non corrigés ne serait pas fiable. Noter les résultats séparément de la validation initiale. La vidéo `porte_B_01_vide.mp4` est particulièrement utile pour repérer les fausses alertes.

## 8 Repartir avec vos propres vidéos d'entraînement

Si vous voulez créer **un nouveau jeu complet** plutôt que réutiliser les 237 images, mettez au moins deux vidéos d'entraînement dans un dossier séparé, par exemple `videos_entrainement_nouveau/`. N'y mettez pas les vidéos `porte_B_...` réservées au test. Variez les personnes, les lumières et les passages, puis lancez :

```bash
./.venv/bin/python scripts/prepare_detection_dataset.py --video-dir videos_entrainement_nouveau --output-dir dataset/detection_nouveau --sample-fps 2
./.venv/bin/python scripts/review_detection_dataset.py --dataset-dir dataset/detection_nouveau --split all
./.venv/bin/python src/train_tiny_heatmap.py --data-dir dataset/detection_nouveau --epochs 50 --checkpoint models/tiny_heatmap_nouveau.pt
```

Sur Windows, remplacer `./.venv/bin/python` par `.\.venv\Scripts\python.exe` dans ces trois lignes. Le premier script extrait les images et **propose** des rectangles avec YOLO ; ces rectangles ne sont pas une vérité. Le deuxième script doit être utilisé jusqu'à ce que toutes les images de train et de validation soient vérifiées. Le dernier entraîne un nouveau modèle **uniquement sur ce nouveau jeu** : il ne fusionne pas automatiquement les anciennes corrections avec les nouvelles. Le script de préparation refuse d'écraser un dossier de sortie existant pour protéger vos annotations.

## 9 Vérifications rapides en cas de blocage

- `No module named torch` : utiliser le Python de `.venv`, puis refaire l'installation de l'étape 2.
- `Manifest introuvable` : vérifier le chemin après `--data-dir` ou `--dataset-dir` et travailler depuis la racine du projet.
- `images de validation restent à vérifier` : reprendre l'étape 4 et enregistrer chaque image avec `S`.
- Peu de précision ou de rappel : examiner les annotations, surtout les personnes coupées par le bord de l'image, puis comparer sur le même jeu et au même seuil.
- Le fichier `.pt` ou les images n'apparaissent pas sur GitHub : c'est normal ; ces données restent locales.

L'export ONNX, Aidge et la carte Colibry sont des étapes ultérieures. Réussir cet entraînement sur ordinateur ne prouve pas encore que le modèle convient à la carte ni à la limite d'énergie du concours.
