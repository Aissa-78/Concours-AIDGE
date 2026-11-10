# Poursuite du modèle Aidge — version 2

## Résultats : ne pas remplacer la v1 pour le moment

Les 20 époques ont été exécutées. Le critère de sélection a retenu l'époque 8 de la poursuite, pas la dernière. La comparaison a été recalculée avec les mêmes 67 images et les mêmes annotations de validation vérifiées.

| Validation | F1 v1 | F1 v2 | Retrouvées v1 → v2 | Fausses détections v1 → v2 |
| --- | ---: | ---: | ---: | ---: |
| Porte locale, 32 présences | 75,0 % | 66,7 % | 21 → 17 | 3 → 2 |
| COCO, 27 présences | 40,8 % | 45,0 % | 10 → 9 | 12 → 4 |
| Ensemble, 59 présences | 59,0 % | 57,1 % | 31 → 26 | 15 → 6 |

La v2 propose moins de faux rectangles, mais rate davantage de personnes. Le gain de F1 COCO ne signifie donc pas qu'elle retrouve plus de personnes COCO : son rappel baisse aussi. Le critère du moins bon domaine favorise ce gain COCO malgré la régression sur notre porte et le F1 global. Ce comportement respecte le protocole annoncé ; ce n'est pas une amélioration générale.

**Recommandation : conserver la v1 comme référence pour la suite**, et ne pas remplacer automatiquement le modèle de l'interface par la v2. Les deux essais et leurs poids sont préservés. La prochaine étape utile est la relecture des 72 images d'entraînement, puis le ciblage de cas nouveaux et variés ; pas une répétition indéfinie du même entraînement.

Le score F1 du train passe de 77,4 % au départ à 88,6 % à la fin, alors que la validation ne suit pas. C'est compatible avec un surapprentissage, sans prouver à lui seul que c'est l'unique cause. Les époques les plus tardives n'ont pas été retenues.

## Ce qui a été lancé

On poursuit **notre détecteur Aidge**, pas YOLO, depuis les poids de l'époque 30 de `person_profile_mix_qvga_v1`.

Les 34 images de validation locale portent maintenant le statut `reviewed=oui`. La comparaison des labels ne révèle aucun changement du nombre ou de la géométrie des rectangles au-delà de 10⁻⁵ en coordonnées normalisées ; les différences de fichiers viennent des arrondis à l'enregistrement. C'est donc une validation humaine des annotations existantes, pas de nouveaux exemples appris.

Les 72 images locales d'entraînement restent marquées `annotation_assistant`. Elles n'ont pas été validées par l'utilisateur lors de cette relecture. COCO conserve ses annotations officielles.

Les 204 images d'entraînement et leurs cibles sont identiques, bit à bit, à celles de la première version. Les 67 images de validation restent hors gradients ; les trois vidéos réservées restent hors apprentissage. Revoir la validation ne transmet pas ses corrections au réseau.

## Réglages

- Entrée inchangée : 320 × 240 en niveaux de gris.
- Architecture inchangée : 49 577 paramètres.
- Initialisation avec `runs/person_profile_mix_qvga_v1/best_dense.npz`.
- Adam réinitialisé : ce n'est pas une reprise complète de son état précédent.
- Maximum 20 époques supplémentaires, batch 8.
- Learning rate initial 0,0002, puis 0,0001 après dix époques.
- Même mélange 50/50 COCO et vidéos, mêmes augmentations train.
- Même sélection : maximiser le plus faible F1 entre COCO et la porte, puis la moyenne, puis la loss.
- Les poids de départ sont évalués à l'époque 0 et peuvent rester le meilleur checkpoint si la poursuite n'améliore pas ce critère.

Une amélioration du critère équilibré peut s'accompagner d'une régression sur notre porte : il faut regarder les deux sources séparément, pas seulement annoncer un meilleur score global. Aucun remplacement automatique du modèle de l'interface n'est effectué.

## Où regarder ?

`runs/person_profile_mix_qvga_v2/rapport.html` contient la comparaison **v1 / v2** recalculée sur la même validation vérifiée. Les deux modèles utilisent maintenant la même résolution : la comparaison n'utilise plus comme référence l'ancien réseau QQVGA des premiers essais.

Les métriques détaillées sont dans `metrics.json`, les poids retenus dans `best.onnx`, la progression dans `training_history.csv` et les conditions dans `protocol.json`. Le numéro d'époque de cette version désigne uniquement la poursuite, pas le nombre total de passages de la première version.

Les trois vidéos de comparaison reprennent les prises réservées déjà observées dans les essais précédents : ce ne sont plus des tests finaux aveugles. Leurs rectangles n'ont pas de référence humaine complète et ne donnent pas un score de précision de comptage.

## Refaire cet essai sur Mac

Dans le terminal à la racine du projet, choisir un nouveau nom de dossier :

```sh
./.venv/bin/python scripts/prepare_profile_training.py --data-dir dataset/profile_mix_human_v1 --run-dir runs/person_profile_mix_qvga_v2_reproduction
```

Puis :

```sh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 ./.venv-aidge-colibry/bin/python scripts/train_person_qqvga_aidge.py --run-dir runs/person_profile_mix_qvga_v2_reproduction --epochs 20 --batch-size 8 --learning-rate 0.0002 --patience 6 --evaluate-every 2 --init-weights runs/person_profile_mix_qvga_v1/best_dense.npz
```

Après la fin du calcul :

```sh
./.venv/bin/python scripts/report_profile_training.py --run-dir runs/person_profile_mix_qvga_v2_reproduction --legacy-run runs/person_profile_mix_qvga_v1
```

Sous Windows, utiliser `.\.venv\Scripts\python.exe` et `.\.venv-aidge-colibry\Scripts\python.exe`, avec des environnements installés sur Windows. Définir les deux variables par `$env:OMP_NUM_THREADS="4"` et `$env:OPENBLAS_NUM_THREADS="4"` avant l'entraînement, au lieu du préfixe de commande Mac. Les fichiers de données et modèles restent à partager séparément du GitHub. Cette exécution a été vérifiée sur Mac, pas sur Windows.

## Limites inchangées

La validation locale a été marquée vérifiée par l'utilisateur, mais les prises partagent toujours le même décor et la même session. Le modèle reste un détecteur, sans suivi ou comptage intégré ici. L'exécution Aidge CPU ne valide pas ColibryNPU, la mémoire réelle, l'UINT8 matériel ou la contrainte de 5 mW.
