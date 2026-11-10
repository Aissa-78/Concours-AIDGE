# Essai COCO + apprentissage Aidge — 20 octobre 2026

Le nouveau modèle a bien été entraîné, mais **il ne remplace pas les anciens** : il fait moins bien sur la nouvelle porte B. Ajouter COCO n'a pas suffi dans cet essai.

## Ce qui a été fait

Un sous-ensemble de [COCO Person officiel](https://cocodataset.org/#download) a été téléchargé : 500 images d'apprentissage et 125 de validation, avec leurs annotations et les informations de licence. Les scènes choisies contiennent une à trois personnes suffisamment grandes, ou aucune personne. Ce sous-ensemble local vient de val2017 ; il ne constitue pas une évaluation officielle COCO.

Le CNN V2 reçoit une image en niveaux de gris et prédit une grille de centres. Il possède 17 885 paramètres, contre 3 729 pour l'ancien modèle. Le contexte de chaque prédiction est plus large. L'apprentissage comporte huit époques sur COCO, puis une adaptation sur les 179 images de portes avec 64 images COCO rééchantillonnées à chaque époque. Les rotations, miroirs et variations de lumière sont appliqués aux images et, lorsque nécessaire, aux annotations.

Les calculs avant/arrière sont exécutés avec Aidge CPU FP32 ; la BCE pondérée et Adam sont calculés en NumPy pour contourner une erreur locale de l'optimiseur Aidge. Les gradients de tous les paramètres ont été vérifiés avant l'apprentissage : écart maximal avec la référence PyTorch `1,12 × 10⁻⁸`. PyTorch n'a pas entraîné ces poids.

L'adaptation a été arrêtée à l'époque 22 faute de progrès sur la validation. Les poids de l'époque 10 ont été conservés. L'ensemble a pris environ 10 min 52 s. Le seuil a été choisi uniquement sur les 58 images de validation locale, entre 0,20 et 0,80 : le meilleur reste 0,50.

## Résultats sur la nouvelle porte B

Les 27 images représentent 24 présences annotées, pas 24 personnes différentes. Elles ne participent ni à l'apprentissage, ni au choix de l'époque, ni au choix du seuil. Le critère de comparaison est la proximité avec le centre du rectangle annoté.

| Modèle | Centres retrouvés | Points hors tolérance | Centres manqués | F1 |
|---|---:|---:|---:|---:|
| Ancien Hardtanh FP32 | 21 | 16 | 3 | 68,9 % |
| Ancien ReLU FP32 | 14 | 17 | 10 | 50,9 % |
| V2 Aidge FP32 | 9 | 18 | 15 | 35,3 % |
| V2 INT8 CPU | 10 | 25 | 14 | 33,9 % |

Le F1 combine précision et rappel : ce n'est ni une accuracy de classification ni un pourcentage d'entrées/sorties correctement comptées. Un point sur une tête trop éloignée du centre annoté peut être compté hors tolérance et laisser un centre manqué. Les aperçus permettent de vérifier ces situations.

La comparaison des anciens modèles utilise l'inférence image par image et le sigmoid de l'évaluateur historique. C'est important pour Hardtanh, dont les plateaux de scores rendent les maxima sensibles aux très petits arrondis du calcul par lots.

Sur la validation locale, la V2 obtient un F1 de 49,6 %, contre 41,8 % pour l'ancien ReLU. Elle donne moins de points hors tolérance (13 contre 94), mais manque davantage de centres (52 contre 37). Cette amélioration de validation **ne se retrouve pas sur la nouvelle porte**. Les scores de validation ont servi à sélectionner le modèle ; ils ne constituent pas un test indépendant.

Le résultat ne permet pas de dire que COCO est inutile : architecture, résolution, niveaux de gris, données et augmentations ont changé en même temps. Il faut les comparer séparément pour identifier ce qui aide ou pénalise le modèle.

## Quantification et export

La sortie FP32 Aidge reproduit celle de la référence PyTorch portant les mêmes poids appris : écart maximal `2,15 × 10⁻⁶` sur les 27 images. Les quatre convolutions quantifiées ont été contrôlées : activations et poids INT8, biais et accumulation INT32. La calibration utilise 32 images de l'apprentissage, pas celles de la porte B.

L'exporteur a également généré un bitstream pour les poids appris. Il indique 7 231 mots de données, 20 880 octets de configuration et une compatibilité avec ses contraintes mémoire. Son estimation est de 140,2 µJ par inférence, contre 20,1 µJ pour l'ancienne structure ReLU : la V2 coûte donc davantage selon cet estimateur.

Cela ne vérifie **ni la précision du bitstream sur la carte, ni le format UINT8 asymétrique final, ni la puissance totale de 5 mW**. La comparaison INT8 présentée ici concerne un graphe signé intermédiaire exécuté sur CPU. Le capteur, le firmware et le comptage restent séparés de cet essai.

## Où retrouver les fichiers

Depuis la racine du projet :

- Images et licences : `dataset/coco_person_subset/`.
- Poids V2 : `runs/heatmap_v2/best.onnx` et `best.npz`.
- Déroulement : `runs/heatmap_v2/training_history.csv` et `training.json`.
- Données exactes et seuil : `runs/heatmap_v2/dataset.json` et `threshold.json`.
- Rapport illustré : `runs/heatmap_v2/evaluation_porte_B_test/rapport.html`.
- Détails : `runs/heatmap_v2/evaluation_porte_B_test/details.csv` et `metrics.json`.
- Audit INT8 : `runs/heatmap_v2/evaluation_porte_B_test/quantization.json`.
- Export : `runs/heatmap_v2/export_colibry/export_graph_int8_colibry/dnn/aidge_model_bitstream.nnpa`.

Les images, poids et résultats restent locaux, ignorés par Git. Les scripts et cette note peuvent être partagés. Le [guide d'entraînement](Entrainement_Aidge_COCO.md) donne les commandes Mac et Windows, sous réserve d'une installation Aidge compatible sur chaque poste.

## Suite proposée

Conserver les anciens modèles et examiner les erreurs de la V2. Faire ensuite un essai contrôlé en ne changeant qu'un facteur à la fois, notamment l'ajout de COCO à architecture et prétraitement constants. Pour le domaine cible, rechercher des données réellement vues du dessus et plusieurs portes ; COCO n'apporte pas à lui seul cette géométrie. Garder une autre porte jamais examinée pour le test final, car la porte B a déjà servi à analyser les limites du projet.
