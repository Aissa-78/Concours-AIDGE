# Essai équilibré : résultat et limites

20 octobre 2026. Essai conservé dans `runs/heatmap_wepdtof_balanced_v3/`.

Le mélange 50/50 préserve mieux les performances publiques, **mais dégrade la porte B**. Il n'est pas retenu pour remplacer le programme ou les anciens modèles.

## Protocole fixé avant l'apprentissage

Même architecture de 17 885 paramètres et mêmes tableaux d'images/cibles que l'essai précédent : 179 images locales et 400 publiques disponibles en train, 58 locales et 40 publiques en validation. Une comparaison des tableaux préparés a confirmé leur égalité. La porte B reste hors apprentissage et sélection.

L'essai reprend `runs/heatmap_wepdtof_v2/best_wepdtof.npz`, le checkpoint public **avant** adaptation aux portes, et non le modèle final qui obtenait 7 % sur les vues publiques. Adam est réinitialisé.

Chaque époque présente les 179 images locales et 179 publiques rééchantillonnées. Chaque batch est exactement 50/50, y compris le dernier. Le taux d'apprentissage est 0,0002, le batch maximum 16, les augmentations sont inchangées.

La règle de sélection maximise le **minimum des F1 local et public**, puis leur moyenne, puis minimise la loss moyenne. Le seuil reste fixé à 0,5, sans calibration après coup. Le point de départ (époque 0) est également éligible. Le protocole et les empreintes sont enregistrés avant l'entraînement dans `protocol.json`.

Les calculs avant/arrière sont exécutés dans Aidge CPU, avec BCE et Adam NumPy. Les gradients initiaux sont contrôlés contre la référence PyTorch : écart maximal `1,12 × 10⁻⁸`. PyTorch n'effectue pas cet apprentissage.

Le départ donne 34,1 % de F1 local et 55,4 % public. L'arrêt anticipé intervient après 14 époques, sans amélioration depuis huit époques ; l'époque 6 est retenue. Durée d'entraînement : **5 min 58 s**. Le maximum fixé était 30 époques, et l'arrêt autorisé à partir de 12.

Ce protocole change le point de reprise, le mélange et la sélection. Il ne permet pas d'isoler l'effet d'un seul de ces facteurs.

## Comparaison au seuil 0,5

| Lot | V2 adaptée aux portes | V3 équilibrée |
|---|---:|---:|
| Validation locale | 49,6 % | **41,0 %** |
| Validation publique | 7,0 % | **42,1 %** |
| Porte B, FP32 | 45,5 % | **29,9 %** |
| Porte B, INT8 signé CPU | 44,4 % | **29,4 %** |

Le pire des deux F1 de validation passe de 7,0 % à 41,0 %, ce qui correspond au compromis recherché. Mais le gain public se fait avec une baisse locale et ne se généralise pas à la porte B. Ces validations sont utilisées pour la sélection : ce ne sont pas des tests finaux indépendants.

Sur les 24 centres annotés de la porte B, les deux modèles retrouvent toujours **10 centres** et en manquent **14**. Les points supplémentaires ou trop éloignés passent de **10 à 33**. La précision baisse à 23,3 %, tandis que le rappel reste à 41,7 %. Sur plusieurs aperçus examinés, des objets fixes sont pris pour des personnes. Il ne s'agit pas uniquement d'un écart de quantification.

L'INT8 CPU retrouve aussi 10 centres, avec 34 points hors tolérance. Les quatre convolutions ont été contrôlées en INT8 avec accumulation INT32. L'écart FP32 Aidge/PyTorch sur la porte B est de `3,10 × 10⁻⁶`. L'UINT8 asymétrique final, le bitstream matériel et les 5 mW ne sont pas validés par ce test.

L'ancien Hardtanh reste à 68,9 % sur cette porte, mais sa validation publique n'est que de 30,9 %. Aucun de ces résultats ne démontre une fiabilité de 90 % sur des lieux inconnus.

## Diagnostic supplémentaire : même le train est mal localisé

Une inférence Aidge sur les 677 images de train et validation a été faite après sélection, sans modifier les poids. Les scores train sont optimistes et ne mesurent pas la généralisation.

| Diagnostic au seuil 0,5 | V2 adaptée | V3 équilibrée |
|---|---:|---:|
| Train local, 179 images | 51,8 % | **40,6 %** |
| Train public, 400 images | 42,3 % | **39,0 %** |

La V3 ne localise que 77 centres sur 168 dans le train local, avec 134 points hors tolérance. La faiblesse ne concerne donc pas seulement les portes nouvelles. Cela suggère aussi un problème d'apprentissage, de capacité, de résolution ou de cible ; ce diagnostic ne permet pas d'en désigner une cause unique.

La suite utile est un test de mémorisation sur 10 à 20 images annotées, sans augmentation, avant un autre long entraînement. Si le réseau ne parvient pas à les apprendre, vérifier la loss, la géométrie des cibles et la capacité du réseau. Le carnet de mission permet aussi de tester une image utile de 160×120 grise, au lieu des 80×80 utiles actuels ; cette modification exige une adaptation du réseau et de son évaluation, pas seulement un changement de taille dans un script.

## Retrouver et refaire

- [Guide de l'entraînement équilibré](Entrainement_equilibre.md) : commandes Mac, adaptation des chemins Windows et règle de sélection.
- `training.json` : point de départ, époque retenue, deux validations et empreintes.
- `training_history.csv` : toutes les époques, y compris la référence 0.
- `evaluation_porte_B_test/rapport.html` : comparaison illustrée avec la V2 et l'INT8.
- `evaluation_detection_proposals/rapport.html` et `evaluation_wepdtof_expanded_v2/rapport.html` : validations.
- `diagnostic_fit/metrics.json` : diagnostic train/validation, reproductible avec `scripts/audit_heatmap_fit.py --run-dir ...`.

Les poids et résultats restent locaux, ignorés par Git. Les nouveaux scripts d'entraînement ont passé les 30 tests alors présents ; cinq tests supplémentaires couvrent ensuite le prétraitement caméra. Aucun ancien modèle n'a été écrasé ni remplacé dans l'interface vidéo.
