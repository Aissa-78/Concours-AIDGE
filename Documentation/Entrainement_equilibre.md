# Relancer l'essai avec un apprentissage équilibré

Ce protocole répond à la perte de détection publique observée pendant le [deuxième essai](Bilan_Aidge_WEPDTOF_v2.md). Il utilise les **mêmes images et le même réseau** : pas de nouveau téléchargement, pas de changement de résolution ou de rectangles.

**Essai terminé, non retenu pour remplacer les anciens modèles.** Le F1 public
remonte de 7,0 % à 42,1 %, mais la porte B baisse de 45,5 % à 29,9 % à cause
de davantage de fausses détections. Voir le [bilan avec le diagnostic train/validation](Bilan_Aidge_equilibre_v3.md).

## Ce qui change

On repart du meilleur checkpoint public de l'essai précédent, **avant son adaptation aux portes** : `runs/heatmap_wepdtof_v2/best_wepdtof.npz`. Il ne s'agit pas du modèle final `best.npz`, qui avait perdu une grande partie de sa détection publique.

Chaque époque présente 179 images locales et 179 images publiques tirées parmi les 400 disponibles. Chaque groupe de 16 contient **8 images locales et 8 publiques** ; le dernier groupe est lui aussi équilibré. Les publiques sont rééchantillonnées à chaque époque. Une même image peut ainsi servir plusieurs fois au cours de l'apprentissage.

À chaque époque, on mesure les F1 sur les 58 images de validation locale et les 40 images de validation publique. Le critère principal est **le plus faible des deux F1** : un excellent score local ne peut pas masquer un effondrement public. En cas d'égalité, on compare la moyenne des F1, puis la moyenne des losses. L'état de départ, appelé époque 0, peut être conservé si aucune mise à jour ne fait mieux.

Le seuil reste **0,5**, fixé avant cet essai. On ne le règle pas ensuite uniquement sur les portes, ce qui pourrait annuler le compromis recherché. Les 27 images de la porte B ne servent ni à entraîner, ni à sélectionner les poids ou le seuil.

Il s'agit d'un changement de protocole complet (point de reprise, mélange et sélection), pas d'une preuve de l'effet isolé d'un seul réglage.

## Commandes Mac

Dans le terminal de VS Code, ouvert à la racine de **FlowSense-AIDGE**. Les deux environnements doivent déjà être installés. Le checkpoint public et le dataset augmenté doivent aussi être présents ; ils ne sont pas téléchargés automatiquement par GitHub.

Pour un nouvel essai, remplacer `heatmap_wepdtof_balanced_v3` par un nouveau nom dans **toutes** les commandes. Ne pas écraser un ancien dossier.

### 1. Préparer les mêmes données

```bash
./.venv/bin/python scripts/prepare_heatmap_v2.py --pretrain-data dataset/wepdtof_expanded_v2 --pretrain-name wepdtof --output-dir runs/heatmap_wepdtof_balanced_v3
```

### 2. Entraîner

```bash
./.venv-aidge-colibry/bin/python scripts/train_balanced_heatmap_aidge.py --run-dir runs/heatmap_wepdtof_balanced_v3 --init-weights runs/heatmap_wepdtof_v2/best_wepdtof.npz --epochs 30 --batch-size 16 --patience 8 --min-epochs 12 --learning-rate 0.0002
```

Le script exécute Aidge CPU pour les calculs avant/arrière et NumPy pour la loss et Adam, comme précédemment. Les gradients sont vérifiés contre la référence initiale avant de charger les poids publics. L'état d'Adam repart à zéro. Une époque contient 358 présentations d'images.

`--epochs 30` fixe le maximum. `--patience 8` arrête après huit époques sans amélioration du critère commun, au plus tôt à l'époque 12. `--learning-rate 0.0002` conserve le taux d'adaptation du précédent essai. `--batch-size` doit être pair pour permettre le mélange 50/50.

Le fichier `protocol.json` est enregistré **avant** l'apprentissage avec les réglages et les empreintes des données et du checkpoint. `training_history.csv` contient les deux F1 à chaque époque, dont l'époque 0. `training.json` indique l'époque retenue et les scores correspondants. `best.onnx` et `best.npz` contiennent uniquement le meilleur état selon cette règle, pas forcément le dernier calculé.

### 3. Comparer les résultats

Ne pas lancer `calibrate_heatmap_v2.py` pour cet essai : conserver le seuil fixé à 0,5.

```bash
./.venv/bin/python scripts/evaluate_heatmap_v2.py --run-dir runs/heatmap_wepdtof_balanced_v3 --data-dir dataset/detection_proposals --reference-run-dir runs/heatmap_wepdtof_v2 --model-label "V3 mixte 50/50" --reference-label "V2 adaptee"
./.venv/bin/python scripts/evaluate_heatmap_v2.py --run-dir runs/heatmap_wepdtof_balanced_v3 --data-dir dataset/wepdtof_expanded_v2 --reference-run-dir runs/heatmap_wepdtof_v2 --model-label "V3 mixte 50/50" --reference-label "V2 adaptee"
./.venv/bin/python scripts/evaluate_heatmap_v2.py --run-dir runs/heatmap_wepdtof_balanced_v3 --data-dir dataset/porte_B_test --reference-run-dir runs/heatmap_wepdtof_v2 --model-label "V3 mixte 50/50" --reference-label "V2 adaptee" --int8
```

Les noms affichés distinguent les deux essais, bien qu'ils utilisent tous les deux 400 images publiques. Ouvrir les `rapport.html` des trois dossiers `evaluation_*` créés dans le nouvel essai. Les fichiers `metrics.json` et `details.csv` permettent de vérifier chaque résultat.

## Windows

Les scripts utilisent des chemins multiplateformes, mais ils doivent être exécutés avec des environnements installés sur le PC Windows ; ne pas copier les environnements du Mac. Le fonctionnement de ce nouvel essai sur Windows n'a pas été testé ici.

Pour les commandes précédentes :

- remplacer `./.venv/bin/python` par `.\.venv\Scripts\python.exe` ;
- remplacer `./.venv-aidge-colibry/bin/python` par `.\.venv-aidge-colibry\Scripts\python.exe` ;
- garder les arguments et les chemins de données identiques.

Ces substitutions ne constituent pas une installation d'Aidge sur Windows. Si Aidge n'est pas disponible dans l'environnement du poste, préparer d'abord un environnement compatible avec l'équipe ; les chemins diffèrent si l'on utilise Linux/WSL.

## Interpréter le résultat

Le score concerne la localisation de centres, pas encore le comptage vidéo. Regarder les personnes manquées et les points supplémentaires, pas seulement un pourcentage. Un compromis entre deux validations peut ne pas améliorer une nouvelle porte. Les validations utilisées pour sélectionner le modèle ne constituent pas un test final indépendant.

La porte B sert encore au diagnostic ; une porte et des vidéos encore jamais examinées seront nécessaires pour confirmer un objectif de 90 % sur les passages et leur sens. L'INT8 CPU n'est pas une mesure sur carte ni une validation de l'UINT8 final ou des 5 mW.
