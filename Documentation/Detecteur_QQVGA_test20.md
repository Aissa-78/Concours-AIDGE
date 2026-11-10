# Détecteur QQVGA : premier contrôle d'apprentissage réussi

29 octobre 2026

Le nouveau réseau reçoit réellement des images **160×120 en gris** et prédit des rectangles. Il a appris les **17 rectangles présents dans 20 images**, sans détection supplémentaire, en FP32 puis en INT8 CPU.

**Ce résultat est un test de mémorisation, pas une précision de 100 % sur des images inconnues.** Les mêmes 20 images servent à apprendre et à mesurer. Le modèle n'a pas été évalué sur la validation ni sur la porte B, et n'a pas remplacé l'interface vidéo ou les modèles précédents.

## Les images du test

Les copies QQVGA déjà préparées sont utilisées directement, sans les réduire à nouveau en 80×80. L'entrée est un canal gris normalisé dans `[0,1]`, forme `(N,1,120,160)`.

| Source | Images avec personne | Images vides | Total |
|---|---:|---:|---:|
| Nos portes, train uniquement | 8 | 2 | 10 |
| WEPDTOF, train uniquement | 8 | 2 | 10 |

Les 16 images positives contiennent 17 rectangles annotés. Il ne s'agit pas nécessairement de 17 personnes différentes. La sélection est reproductible avec la graine 7, répartie entre vidéos/lieux, sans utiliser de validation ni de porte B. Pour ce premier contrôle, on retient au plus deux personnes, des rectangles d'au moins 8 pixels de côté et sans collision de centres dans une cellule de grille. Ces choix facilitent le diagnostic ; ils ne couvrent pas tous les cas futurs.

Il n'y a **aucune augmentation** pendant ce test. On cherche d'abord à vérifier que la chaîne peut apprendre des exemples fixes.

## Le réseau et ses sorties

La conception reprend le principe des convolutions séparables de l'exemple ASYGN BikeCounter, avec une architecture plus petite et sans réutiliser ses poids appris sur les vélos. Ce n'est pas le modèle Ultralytics YOLO ni une reprise de son score.

- Une convolution initiale extrait des formes simples et réduit l'image par deux.
- Une convolution depthwise traite chaque canal séparément et réduit encore par deux.
- Les convolutions pointwise `1×1` mélangent les informations entre canaux.
- Six blocs depthwise/pointwise, avec ReLU, élargissent le contexte observé.
- La dernière convolution produit **5 valeurs par cellule** sur une grille **30×40** : quatre distances aux bords du rectangle et un score brut de présence. Ce ne sont pas « cinq personnes ».

Le modèle compact compte **4 645 paramètres** et 16 convolutions, dont 7 depthwise. La conversion du score et le filtrage des rectangles redondants (NMS) sont du post-traitement hors réseau, à porter ensuite dans la chaîne embarquée.

## Comment il apprend dans Aidge

Le backend CPU local n'implémente pas le backward natif de `ConvDepthWise2D`. Pour apprendre sans modifier l'installation, un graphe équivalent utilise des convolutions ordinaires dont les connexions entre canaux restent bloquées à zéro. Les gradients hors diagonale sont annulés avant Adam et le masque est réappliqué après chaque mise à jour.

Ce graphe de calcul contient 42 045 paramètres stockés, dont seuls 4 645 correspondent aux paramètres utiles du réseau séparable. **Le graphe dense n'est pas celui destiné au déploiement.** Les coefficients utiles sont ensuite reconstruits dans de vraies couches depthwise.

Les calculs avant/arrière sont exécutés dans **Aidge CPU FP32**. La loss analytique et Adam sont calculés en NumPy. PyTorch sert à définir/initialiser les graphes, fournir une référence de gradients et exporter ONNX ; aucun optimiseur PyTorch n'effectue cet apprentissage.

Avant entraînement, toutes les sorties et tous les gradients du graphe Aidge sont comparés à la référence : écart maximal de sortie `9,54×10⁻⁷`, écart maximal de gradient `8,34×10⁻⁷`. Les tests vérifient aussi l'équivalence du modèle séparable et du graphe dense contraint.

La loss combine une focal loss pour les centres et une SmoothL1 pour les quatre distances du rectangle. Ses dérivées NumPy sont vérifiées par autograd dans les tests. Adam ajuste les coefficients pour réduire cette erreur.

## Règles fixées avant l'essai

| Réglage | Valeur |
|---|---|
| Maximum | 300 époques |
| Batch | 4 images |
| Learning rate | 0,001 ; réduit à 0,0003 après l'époque 150 si nécessaire |
| Augmentations | Aucune |
| Score minimal pour afficher un rectangle | 0,5 |
| Correspondance avec l'annotation | IoU ≥ 0,5 |
| Filtrage des doublons | NMS, IoU 0,5 |
| Évaluation | Toutes les 10 époques et à la dernière |
| Arrêt du test | Précision ET rappel ≥ 95 %, deux évaluations consécutives |
| Meilleurs poids | F1 des 20 images, puis loss en cas d'égalité |

Avec seulement 17 annotations, le critère de rappel exige ici de toutes les retrouver. Les seuils n'ont pas été modifiés après observation du résultat.

À l'époque 40, le réseau retrouve 11 rectangles avec 2 supplémentaires ; à l'époque 50, 15 avec aucun supplémentaire. Aux époques 60 et 70, il retrouve les 17 sans détection supplémentaire. L'arrêt se produit à 70 et les poids de l'époque **60** sont retenus, car leur loss est plus faible. L'apprentissage dure **2 min 27 s** environ sur ce Mac.

Les scores précédents des modèles V2/V3 utilisaient surtout une distance entre centres, pas l'IoU des rectangles. De plus, les lots sont différents. **On ne doit donc pas présenter ce 100 % comme un passage de 29,9 % à 100 % sur la porte B.**

## Contrôles après apprentissage

Le graphe compact a été reconstruit avec les coefficients appris. Sur ces 20 images, ses sorties Aidge sont identiques à celles du graphe dense contraint dans le contrôle effectué.

| Exécution sur le lot mémorisé | Retrouvés | Supplémentaires | Manqués |
|---|---:|---:|---:|
| Aidge FP32 compact | 17 | 0 | 0 |
| Aidge INT8 signé CPU | 17 | 0 | 0 |

La calibration INT8 utilise ces mêmes 20 images. Les 16 convolutions ont été contrôlées : activations et poids INT8, biais et accumulation INT32. Ce résultat est optimiste et n'est pas une mesure de généralisation ou une validation de la cible finale UINT8 asymétrique.

### Export logiciel Colibry

Un bitstream a été généré dans :

```text
runs/person_qqvga_mem20_v1/export_colibry_final/export_graph_int8_colibry/dnn/aidge_model_bitstream.nnpa
```

Le compilateur 0.1.2 indique : 2 461 mots de données (3 buffers), 5 584 octets de configuration (349 mots, 1 buffer), 361 392 cycles et une compatibilité avec ses contraintes mémoire. L'énergie estimée est de **19,6 µJ par inférence**. Ce n'est pas une mesure de puissance totale du système ni une preuve des 5 mW : caméra, MCU, BLE et fréquence de fonctionnement restent à prendre en compte et à mesurer.

Deux points d'intégration ont été corrigés dans les scripts, sans modifier les bibliothèques installées :

1. Pour la PTQ QQVGA, conserver le padding dans la convolution afin que le premier opérateur porte les informations de précision nécessaires à `InputRescaling`.
2. Après PTQ, développer ces convolutions avant les fusions attendues par l'exporteur. Un adaptateur de recherche de méta-opérateurs, limité au processus d'export, couvre aussi les convolutions paddées et leurs fusions, sans introspecter certaines feuilles fragiles comme `Round`.

Les dossiers `export_colibry` et `export_colibry_padded_fix` sont des tentatives incomplètes conservées pour diagnostic ; l'export abouti est **`export_colibry_final`**. Aucun bitstream n'a été exécuté sur carte. Les messages génériques du compilateur mentionnant VWW ne constituent pas une évaluation de notre modèle sur ce dataset.

Les **43 tests automatisés** passent. Un contrôle supplémentaire reproduit exactement
les sorties FP32 de l'ancien modèle V2 sur ses 58 images de validation locale
(écart maximal 0), et son ancien chemin INT8 reste exécutable. Les empreintes des
20 images/labels sources sont inchangées. Le bitstream produit fait 5 584 octets ;
son SHA256 est `f6a3aeac6dc85aa526cff3961256acfa5c7def047a4bf868847020c188f47fc5`.

## Rôle des fichiers Python

| Fichier | Rôle |
|---|---|
| `src/person_detector_qqvga.py` | Architecture compacte et représentation dense contrainte équivalente |
| `src/person_detection_math.py` | Cibles, loss et gradient, décodage, NMS et correspondance des rectangles |
| `scripts/prepare_person_memorization.py` | Sélection des 20 images train, initialisation et références de gradients |
| `scripts/train_person_memorization_aidge.py` | Apprentissage Aidge et historique du test technique |
| `scripts/infer_person_qqvga_aidge.py` | Inférence Aidge sur les images QQVGA |
| `scripts/report_person_memorization.py` | Reconstruction compacte, contrôle d'équivalence et rapport illustré |
| `src/aidge_export_compat.py` | Adaptateur local au processus pour parcourir les fusions à l'export |
| `scripts/run_aidge_quantization.py` | Comparaison FP32 / INT8 CPU, maintenant compatible avec la sortie QQVGA |
| `scripts/export_colibry.py` | Calibration, adaptations d'export et génération logicielle du bitstream |

## Refaire le test soi-même

Depuis la racine du projet dans le terminal de VS Code. Les environnements Python locaux et le lot `dataset/camera_qqvga_v1/` doivent déjà exister.

L'essai terminé est conservé. Pour le refaire, utiliser un **nouveau nom** à la place de `person_qqvga_mem20_v1` dans toutes les commandes :

```bash
./.venv/bin/python scripts/prepare_person_memorization.py --run-dir runs/person_qqvga_mem20_v1
./.venv-aidge-colibry/bin/python scripts/train_person_memorization_aidge.py --run-dir runs/person_qqvga_mem20_v1 --check-only
./.venv-aidge-colibry/bin/python scripts/train_person_memorization_aidge.py --run-dir runs/person_qqvga_mem20_v1 --epochs 300 --batch-size 4 --learning-rate 0.001
./.venv/bin/python scripts/report_person_memorization.py --run-dir runs/person_qqvga_mem20_v1
```

Puis, pour les contrôles INT8 et l'export :

```bash
./.venv-aidge-colibry/bin/python scripts/run_aidge_quantization.py --onnx runs/person_qqvga_mem20_v1/best.onnx --inputs runs/person_qqvga_mem20_v1/dataset.npz --calibration runs/person_qqvga_mem20_v1/dataset.npz --output runs/person_qqvga_mem20_v1/quantization.npz
./.venv-aidge-colibry/bin/python scripts/export_colibry.py --run-dir runs/person_qqvga_mem20_v1 --model runs/person_qqvga_mem20_v1/best.onnx --calibration runs/person_qqvga_mem20_v1/dataset.npz --output-dir runs/person_qqvga_mem20_v1/export_colibry_final
./.venv/bin/python scripts/report_person_memorization.py --run-dir runs/person_qqvga_mem20_v1
```

Sous Windows, remplacer `./.venv/bin/python` par `.\.venv\Scripts\python.exe` et `./.venv-aidge-colibry/bin/python` par `.\.venv-aidge-colibry\Scripts\python.exe`, avec des environnements compatibles installés sur le PC. Ne pas copier ceux du Mac. Si Aidge est utilisé via WSL/Linux, employer les chemins de cet environnement. Ces nouveaux scripts n'ont pas été exécutés sur Windows ici.

Ouvrir ensuite `runs/person_qqvga_mem20_v1/rapport.html`. Les fichiers `dataset.json`, `protocol.json`, `compatibility.json`, `training_history.csv`, `training.json`, `metrics.json` et `quantization.json` gardent la trace de l'essai. Les images et les poids restent locaux, ignorés par Git.

## Étape suivante, distincte de ce test

Préparer un véritable apprentissage sur le lot complet QQVGA, avec des augmentations compatibles avec des images rectangulaires et leurs rectangles. Ne pas réutiliser telle quelle la rotation 90° du prétraitement V2 carré : elle échange largeur et hauteur.

Utiliser les validations séparées pour sélectionner les poids et le seuil, puis des images/vidéos inédites pour estimer la fiabilité. Il faudra mesurer les rectangles et, plus tard, chaque passage dans le bon sens. Le script de mémorisation ne fait pas encore cet entraînement complet. **L'objectif de 90 % sur le terrain reste à démontrer.**
