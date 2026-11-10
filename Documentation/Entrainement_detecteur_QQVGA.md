# Entraîner le détecteur QQVGA sur le lot complet

Ce guide concerne le **nouveau petit détecteur à rectangles**, pas le premier
classifieur personne/vide, ni YOLO, ni l'ancien modèle heatmap.

**Historique :** les commandes par défaut ci-dessous reproduisent le premier
détecteur séparable, dont l'essai complet a échoué. Elles ne constituent pas la
recette d'un modèle fiable. Les essais de correction, les deux annotations
confirmées et la variante à convolutions ordinaires sont décrits dans
[le bilan de correction du 30 octobre](Bilan_correction_detecteur_QQVGA.md).

## Les images utilisées

Les copies préparées sont dans `dataset/camera_qqvga_v1/` :

| Source | Apprentissage | Validation | Personnes annotées train / validation |
|---|---:|---:|---:|
| Nos portes | 179 | 58 | 168 / 84 |
| WEPDTOF, vues du dessus | 400 | 40 | 582 / 54 |
| Total | **579** | **98** | **750 / 138** |

Les 20 images du test de mémorisation appartenaient déjà à ce train : on ne les
rajoute pas une deuxième fois. Les 579 images ne sont donc pas toutes nouvelles.
Les 27 images de la porte B ne sont utilisées ni pour entraîner, ni pour choisir
le modèle, ni pour calibrer l'INT8 de cet essai.

Les images sont en **160 × 120, un canal gris**, avec des bandes grises si le
format de départ ne correspond pas. Les annotations restent dans des fichiers
texte séparés : on ne dessine pas les rectangles sur les images d'apprentissage.
Les annotations locales sont marquées revues ; celles de WEPDTOF proviennent de
la base publique, converties et contrôlées géométriquement, sans prétendre que
l'équipe a vérifié manuellement chaque rectangle.

Les vidéos d'apprentissage et de validation sont différentes. Les doublons
exacts et les vidéos communes sont refusés. Cela ne prouve pas que tous les
décors sont nouveaux : la validation locale reste une mesure de développement,
pas une garantie sur une porte encore jamais vue.

## Ce que fait l'entraînement

Le réseau conserve ses **4 645 paramètres compacts**, ses convolutions ReLU et
séparables, et sa sortie de cinq valeurs par cellule : quatre distances aux
bords d'un rectangle et un score de présence. Il démarre avec des poids
aléatoires, sans reprendre ceux de YOLO ni ceux du test sur 20 images.

Une époque utilise toutes les images. Pour que les 400 images publiques
n'écrasent pas les 179 locales, les images locales sont répétées aléatoirement :
**800 présentations par époque, mais seulement 579 images distinctes**.
Avec un batch de huit : quatre images locales et quatre publiques.

Seulement pendant l'apprentissage, le code applique des miroirs horizontaux ou
verticaux, puis parfois de légères variations de contraste, luminosité, flou
et bruit. Les rectangles et leurs cibles changent en même temps que l'image.
Il n'y a pas de rotation de 90° qui échangerait largeur et hauteur.

La **loss** additionne une pénalité de présence/absence (focal loss) et une
pénalité d'erreur sur les dimensions des rectangles (SmoothL1). Les gradients
indiquent comment modifier les poids pour réduire ces erreurs.

Les calculs du réseau et les gradients sont exécutés dans **Aidge CPU FP32**.
La loss et l'optimiseur Adam sont en NumPy. PyTorch sert à définir/exporter le
graphe et à vérifier les gradients avant l'apprentissage, pas à entraîner les
poids. Le backward depthwise natif manque dans l'installation locale : le
graphe d'apprentissage emploie des convolutions denses contraintes diagonales,
puis le rapport reconstruit le modèle compact et vérifie son équivalence.

## Comment le meilleur modèle est choisi

Toutes les deux époques, les 98 images de validation sont évaluées **sans
augmentation ni mise à jour des poids**. Les résultats des portes et de WEPDTOF
sont affichés séparément.

- Précision : parmi les rectangles proposés, quelle proportion est correcte ?
- Rappel : parmi les personnes annotées, quelle proportion est retrouvée ?
- F1 : un équilibre entre précision et rappel.

Un rectangle est correct si son recouvrement avec une annotation atteint
**IoU ≥ 0,5**, avec une seule correspondance par personne. Le seuil de détection
est fixé à **0,5** avant l'essai. Ce n'est pas une accuracy personne/vide.

On retient le checkpoint qui améliore le **plus faible F1 des deux sources**,
puis leur moyenne en cas d'égalité, puis la loss. Cela empêche un bon résultat
public de masquer l'échec sur nos portes, ou l'inverse. Un arrêt anticipé est
prévu après six évaluations sans progrès de ce critère.

Ces validations servent à choisir les poids : ce n'est **pas un test final
aveugle**. Un bon score ne valide pas encore le suivi, le comptage, la carte
ColibryNPU, l'UINT8 final ni la consommation de l'ensemble du système.

## Commandes sur Mac

Dans le terminal de VS Code, ouvert dans le dossier du projet. Les environnements
`.venv` et `.venv-aidge-colibry` doivent déjà être installés.

Pour **un nouvel essai**, utiliser le même nouveau nom dans les trois commandes :

```bash
./.venv/bin/python scripts/prepare_person_training.py --run-dir runs/person_qqvga_mon_essai_01
./.venv-aidge-colibry/bin/python scripts/train_person_qqvga_aidge.py --run-dir runs/person_qqvga_mon_essai_01 --epochs 30 --batch-size 8 --learning-rate 0.001
./.venv/bin/python scripts/report_person_training.py --run-dir runs/person_qqvga_mon_essai_01
```

Puis ouvrir le rapport :

```bash
open runs/person_qqvga_mon_essai_01/rapport.html
```

Ne pas relancer la préparation sur un nom existant : le refus protège les
résultats. Un entraînement interrompu n'a pas de reprise automatique de l'état
Adam ; créer un nouvel essai plutôt que supprimer le contrôle de protection.
La première tentative du lot complet s'appelle `person_qqvga_full_v1`.

### Repartir de poids existants après une interruption

Le 30 octobre, la première tentative ne tournait plus et son dernier checkpoint
évalué était celui de l'époque 14/30, sans `training.json` final. Ce n'était pas
un entraînement terminé. La poursuite est enregistrée séparément dans
`runs/person_qqvga_full_v2/`, avec les mêmes images, 16 époques supplémentaires
maximum et un taux d'apprentissage de 0,0005. **Les poids sont conservés, mais
l'état Adam de la première tentative n'existait pas : Adam repart donc de zéro.**
Ce n'est pas une reprise exactement identique à une exécution ininterrompue.

Pour reproduire cette poursuite, préparer un autre nom de dossier libre puis :

```bash
./.venv/bin/python scripts/prepare_person_training.py --run-dir runs/person_qqvga_reprise_v3
./.venv-aidge-colibry/bin/python scripts/train_person_qqvga_aidge.py --run-dir runs/person_qqvga_reprise_v3 --init-weights runs/person_qqvga_full_v1/best_dense.npz --epochs 16 --batch-size 8 --learning-rate 0.0005 --report-python .venv/bin/python
```

`--report-python` génère le rapport automatiquement à la fin avec l'environnement
habituel. Sous Windows, remplacer les deux chemins d'interpréteurs par leurs
équivalents `Scripts\python.exe`. Ne pas exécuter les commandes d'un autre essai
dans le dossier d'un entraînement en cours.

Le script sauvegarde désormais un `last_training_state.npz` après chaque époque
complète (poids, moments Adam, pas et état du générateur aléatoire), ainsi qu'un
`progress.json` lors des évaluations. Ces fichiers protègent le dernier état ;
leur chargement pour une reprise exacte n'est pas encore une commande intégrée.
`--init-weights` attend le fichier plat `best_dense.npz`, pas ce fichier d'état.

## Commandes sur Windows

Même dossier de projet et mêmes scripts, avec **des environnements installés sur
Windows** : ne pas copier les environnements Mac. Si Aidge est installé dans
WSL, employer les commandes Linux depuis WSL plutôt que les chemins ci-dessous.
Le nouveau pipeline n'a pas été exécuté sur un poste Windows.

```powershell
.\.venv\Scripts\python.exe scripts/prepare_person_training.py --run-dir runs/person_qqvga_mon_essai_01
.\.venv-aidge-colibry\Scripts\python.exe scripts/train_person_qqvga_aidge.py --run-dir runs/person_qqvga_mon_essai_01 --epochs 30 --batch-size 8 --learning-rate 0.001
.\.venv\Scripts\python.exe scripts/report_person_training.py --run-dir runs/person_qqvga_mon_essai_01
Start-Process "runs/person_qqvga_mon_essai_01/rapport.html"
```

## Fichiers et paramètres à connaître

- `scripts/prepare_person_training.py` : charge images/annotations, contrôle la
  séparation et prépare les tableaux ainsi que le graphe initial.
- `src/person_training_data.py` : mélange équilibré, augmentations et critère
  de sélection. NumPy uniquement pour fonctionner dans l'environnement Aidge.
- `scripts/train_person_qqvga_aidge.py` : contrôle des gradients, apprentissage,
  validation et sauvegarde. `--epochs` est un maximum, pas une garantie de progrès.
  `--learning-rate` règle la taille des corrections ; `--batch-size` doit être
  pair ; `--evaluate-every` et `--patience` règlent l'arrêt anticipé.
- `scripts/report_person_training.py` : reconstruit les convolutions depthwise
  compactes, vérifie les sorties Aidge sur les deux groupes et montre les erreurs
  sur les 98 images de validation.
- `runs/<nom>/training_history.csv` : résultats au fil des évaluations.
- `runs/<nom>/best_dense.onnx` : checkpoint du graphe d'apprentissage contraint.
- `runs/<nom>/best.onnx` : modèle compact à utiliser pour la suite, après rapport.
- `runs/<nom>/rapport.html` et `metrics.json` : erreurs et mesures du checkpoint
  retenu, pas les seules dernières valeurs affichées pendant l'apprentissage.

Les images et les runs restent locaux, sans publication automatique sur GitHub.
L'interface vidéo et les anciens modèles ne sont pas remplacés par ces commandes.
