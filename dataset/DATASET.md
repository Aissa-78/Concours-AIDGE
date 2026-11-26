# Base d'images FlowSense

## Dataset équipe du 26 novembre 2026

Récupérez `FlowSense_dataset_equipe_2026-11-26.zip` dans le partage privé de
l’équipe. Extrayez l’archive **à la racine du dépôt** pour obtenir
`dataset/detection_flowsense_equipe_2026-11-26/`.

| Dossier | Images | Usage |
| --- | ---: | --- |
| `train` | 2 886 | Entraîner le modèle |
| `validation` | 154, dont 71 de la Colibry | Choisir l’époque et les réglages |
| `test` | 115 de la Colibry, issues de 24 vidéos | Évaluer sur le test réservé |

Les images sont en niveaux de gris, au format 160 × 120. Chaque image a une
annotation dans `labels/`. Une ligne d’annotation décrit un rectangle au format
`0 x_centre y_centre largeur hauteur` avec des coordonnées normalisées. Un
fichier vide indique une scène sans personne annotée. Le fichier
`manifest.tsv` donne la provenance, le groupe de capture et les chemins des
images et annotations.

Le train contient 827 captures originales de la Colibry, 311 images COCO,
72 images historiques de portes et 233 duos synthétiques validés. Chaque
image a une variante miroir et lumière. Les lots de validation et de test
restent à part. Le test de cette version ne contient aucune scène à deux
personnes.

### Relancer l’entraînement Aidge

Sur **Windows**, ouvrez Ubuntu/WSL2 et extrayez le ZIP dans votre copie
Linux du dépôt. Sur **Linux natif**, utilisez le même environnement
d’entraînement. Installez-le en suivant le
[guide Windows et Linux](../Documentation/Windows_Linux_complet.md).

À la racine du dépôt sous Linux ou WSL2 :

```bash
source .venv-aidge-linux/bin/activate
```

Sur le Mac déjà configuré :

```bash
source .venv-aidge-colibry/bin/activate
```

Vérifiez d’abord le lot et le calcul Aidge :

```bash
python scripts/train_person_qqvga_aidge.py --run-dir dataset/detection_flowsense_equipe_2026-11-26/aidge --check-only
```

Le contrôle passe sur le Mac avec Aidge Core et son backend CPU
`0.10.0.post3`. Le test équivalent sur Windows/WSL et Linux reste à
faire sur la machine qui va entraîner. Après réussite, lancez jusqu’à
20 époques avec les poids de départ du lot et la priorité caméra :

```bash
python scripts/train_person_qqvga_aidge.py --run-dir dataset/detection_flowsense_equipe_2026-11-26/aidge --init-weights dataset/detection_flowsense_equipe_2026-11-26/aidge/poids_depart.npz --epochs 20 --batch-size 8 --learning-rate 0.00012 --evaluate-every 1 --patience 7 --lr-decay-every 4 --prioritize-camera
```

Les résultats sont écrits dans `aidge/`. Pour refaire un essai, gardez
vos résultats et repartez d’une extraction propre.

Pour tester le détecteur **sur une image dans Windows natif**, suivez
[le test OpenCV](../Documentation/Windows_Linux_complet.md#tester-notre-détecteur-onnx).
Ce mode fonctionne à partir du fichier ONNX livré dans le dépôt.

## Sous-ensemble initial : Wake Vision

La première base locale est construite avec **Wake Vision**, un dataset public de classification binaire spécialement conçu pour la présence de personnes en TinyML.

- Source : [Harvard-Edge/Wake-Vision](https://huggingface.co/datasets/Harvard-Edge/Wake-Vision)
- Labels du projet : `0 = vide` et `1 = personne`
- Licence annoncée par le dataset : CC BY 4.0
- Téléchargement : `python scripts/download_wake_vision_subset.py`

Le téléchargeur récupère par défaut 100 images `vide` et 100 images `personne` pour l'entraînement, puis 25 images de chaque classe pour la validation. Elles restent exclusivement sur le poste local : les fichiers image sont ignorés par Git.

## Limite importante

Wake Vision fournit une bonne base pour apprendre le fonctionnement de l'IA, mais ses images ne ressemblent pas forcément à la future caméra de porte FlowSense. Quand la caméra sera disponible, il faudra compléter la base avec des images prises dans le scénario réel : porte ou couloir, différents éclairages, personnes proches ou éloignées, et scènes réellement vides.

Ne placez jamais une même image dans `train` et dans `validation`.
