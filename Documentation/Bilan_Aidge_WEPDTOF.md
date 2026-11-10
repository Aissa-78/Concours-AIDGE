# Essai avec les vues du dessus — 20 octobre 2026

L'essai WEPDTOF progresse par rapport à l'essai V2 COCO sur la porte B : **43,5 % de F1 contre 35,3 %**. Il reste moins bon que les anciens modèles sur cette porte et très faible sur les autres lieux publics. Il n'a donc pas remplacé les anciens modèles ni la démonstration vidéo YOLO.

## Données et apprentissage

L'équipe a confirmé que le point de vue de l'exemple 001 ressemblait suffisamment à celui de la caméra prévue. Cette confirmation concerne l'angle de vue, pas l'ensemble des annotations. Les labels restent des annotations WEPDTOF originales converties, non revues intégralement par l'équipe.

Le jeu public contient 80 images d'apprentissage issues de quatre lieux, et 40 images de validation issues de deux autres lieux. Il comporte 24 images sans personne. Les 179 images de portes d'apprentissage et les 58 de validation locale ont été conservées sans modification. Les 27 images de la porte B n'ont servi ni à modifier les poids, ni à choisir l'époque, ni à régler le seuil.

Le réseau V2, ses 17 885 paramètres, les prétraitements, les augmentations et la graine 7 sont identiques à l'essai COCO. Les fichiers ONNX d'initialisation sont identiques octet par octet. Le préentraînement public est remplacé par WEPDTOF ; on ne part pas des poids COCO déjà entraînés.

Le calcul avant/arrière est exécuté dans Aidge CPU en FP32. La BCE pondérée et Adam sont calculés en NumPy pour contourner le problème de l'optimiseur Aidge installé. Le contrôle des gradients avant apprentissage donne un écart maximal de `1,12 × 10⁻⁸` avec la référence PyTorch ; PyTorch n'effectue pas la mise à jour des poids.

| Réglage | Valeur |
|---|---|
| Préentraînement public | 8 époques, batch de 16, taux d'apprentissage 0,001 |
| Adaptation aux portes | Jusqu'à 35 époques, batch de 16, taux 0,0005 |
| Images publiques conservées pendant l'adaptation | 64 rééchantillonnées par époque |
| Poids positif de la BCE | 10 |
| Arrêt | Après 12 époques sans amélioration, au plus tôt à l'époque 15 |
| Sélection finale | Meilleur F1 de validation locale, puis loss en cas d'égalité |

L'apprentissage s'est arrêté à l'époque 26 de l'adaptation et a restauré l'époque 14. Il a duré environ **8 min 9 s**. Le seuil a été choisi sur les 58 images de validation locale parmi les valeurs 0,20 à 0,80 ; le meilleur reste 0,50.

Les tailles de données diffèrent : huit époques publiques représentent 640 présentations d'images avec WEPDTOF, contre 4 000 avec COCO. L'arrêt anticipé diffère également entre les deux essais. Il s'agit donc d'une comparaison pratique de deux entraînements, pas d'une preuve que l'angle de vue est l'unique cause de l'écart.

## Résultats au même seuil 0,5

Les scores ci-dessous mesurent les centres de rectangles annotés, pas le comptage des entrées et sorties. Le F1 combine précision et rappel ; ce n'est pas une accuracy de classification.

| Modèle FP32 | Validation locale : 58 images | Validation publique : 40 images | Porte B : 27 images |
|---|---:|---:|---:|
| Ancien Hardtanh | 42,1 % | 30,9 % | 68,9 % |
| Ancien ReLU | 41,8 % | 25,7 % | 50,9 % |
| V2 préentraînée COCO | 49,6 % | 0,0 % | 35,3 % |
| V2 préentraînée WEPDTOF | 48,8 % | 7,1 % | 43,5 % |

Sur la porte B, les deux essais V2 donnent :

| Mesure sur 24 présences annotées | V2 COCO FP32 | V2 WEPDTOF FP32 | V2 WEPDTOF INT8 CPU |
|---|---:|---:|---:|
| Centres retrouvés | 9 | 10 | 10 |
| Points supplémentaires ou hors tolérance | 18 | 12 | 13 |
| Centres manqués | 15 | 14 | 14 |
| F1 | 35,3 % | 43,5 % | 42,6 % |

Le gain FP32 par rapport à COCO est d'environ 8,2 points de F1. Il repose surtout sur une diminution des points hors tolérance : le nombre de centres retrouvés ne passe que de 9 à 10. Il ne faut pas présenter cela comme une grande amélioration du rappel.

Sur les 40 images des autres lieux publics, la V2 WEPDTOF ne retrouve que **2 centres sur 54**, sans point supplémentaire au seuil 0,5. Elle manque donc encore presque toutes les présences de ces lieux. Un changement de données seul n'a pas résolu la généralisation.

Les résultats de validation servent à sélectionner les poids ou le seuil : ils ne sont pas des tests indépendants. La porte B est hors apprentissage, mais elle a déjà servi à diagnostiquer le projet et à orienter les essais. Une porte jamais examinée restera nécessaire pour une évaluation finale. Les mêmes personnes peuvent apparaître sur plusieurs images.

Un point sur une tête trop éloigné du centre du rectangle peut compter à la fois comme point hors tolérance et comme centre manqué. Le rapport illustré permet de contrôler ces cas. Les anciens modèles sont évalués image par image pour garder la même méthode que leur évaluateur historique.

## Aidge, quantification et export

Sur la porte B, l'écart maximal de sortie entre Aidge FP32 et la référence PyTorch portant les mêmes poids appris est de `2,86 × 10⁻⁶`. Les quatre convolutions du graphe quantifié ont été contrôlées : activations et poids INT8, biais et accumulation INT32. La calibration utilise 32 images du train local, sans les images de la porte B.

Le graphe INT8 CPU obtient 42,6 % de F1, contre 43,5 % avant quantification. Ce test n'exécute pas le bitstream matériel et ne valide pas la cible finale UINT8 asymétrique.

L'export logiciel Colibry a produit un bitstream avec les nouveaux poids. Le compilateur indique 7 231 mots de données, 20 880 octets de configuration et une compatibilité avec ses contraintes mémoire. Son estimation reste de 140,2 µJ par inférence, l'architecture étant inchangée. Cette estimation n'est pas une mesure de puissance totale : **les 5 mW, la caméra, le firmware et l'exécution sur carte ne sont pas validés**.

## Retrouver et refaire l'essai

Tous les résultats sont dans `runs/heatmap_wepdtof_v1/`, ignoré par Git :

- `best.onnx` et `best.npz` : poids retenus à l'époque 14 ;
- `training_history.csv`, `training.json`, `dataset.json` : historique et données exactes ;
- `threshold.json` : choix du seuil sur validation locale ;
- `evaluation_porte_B_test/rapport.html` : comparaison illustrée ReLU / V2 COCO / V2 WEPDTOF / INT8 ;
- `evaluation_porte_B_test/details.csv` et `metrics.json` : résultats par image et synthèse ;
- `evaluation_wepdtof_subset/rapport.html` : erreurs sur les autres lieux publics ;
- `evaluation_detection_proposals/rapport.html` : validation locale ;
- `export_colibry/export_graph_int8_colibry/dnn/aidge_model_bitstream.nnpa` : export logiciel.

La commande de comparaison avec COCO, après entraînement et calibration du seuil, est :

```bash
./.venv/bin/python scripts/evaluate_heatmap_v2.py --run-dir runs/heatmap_wepdtof_v1 --data-dir dataset/porte_B_test --reference-run-dir runs/heatmap_v2 --calibrated --int8
```

Sous Windows, remplacer le Python du début par `.\.venv\Scripts\python.exe` ; un environnement Aidge Windows compatible doit être installé séparément. Les nouveaux scripts ont été vérifiés sur le Mac, pas exécutés sur Windows.

Le [guide du lot de vues du dessus](Lot_vues_du_dessus.md) donne les commandes de préparation et d'entraînement. Le comparateur accepte maintenant `--reference-run-dir` pour confronter deux essais V2 au même prétraitement, avec contrôle de l'absence de vidéos/images d'entraînement parmi les images évaluées.

## Prochaine expérience raisonnable

Avant de demander de nouvelles vidéos, tester un préentraînement plus long sur les images publiques existantes, en vérifiant la validation publique avant l'adaptation aux portes. Par exemple, 50 époques sur les 80 images WEPDTOF représenteraient environ 4 000 présentations d'images, comparables au budget public du précédent essai COCO. Cela ne garantit pas une amélioration et doit rester un nouvel essai, sans écraser celui-ci. Il faudra ensuite identifier séparément les limites de résolution, d'architecture et de cible, plutôt que modifier plusieurs éléments à la fois ou régler le modèle sur la porte B.
