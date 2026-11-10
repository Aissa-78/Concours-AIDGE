# Test avant/après quantification — 20 octobre 2026

Le petit modèle ReLU a été testé sur les mêmes 27 images de la porte B. Elles
contiennent 24 présences annotées : ce ne sont pas 24 personnes différentes,
puisque les mêmes personnes apparaissent sur plusieurs images.

| Mesure | Avant : FP32 | Après : INT8 dans Aidge CPU |
|---|---:|---:|
| Présences correctement localisées | 14 | 13 |
| Centres annotés manqués | 10 | 11 |
| Points supplémentaires ou hors tolérance | 17 | 19 |
| Précision de localisation | 45,2 % | 40,6 % |
| Rappel | 58,3 % | 54,2 % |
| F1 | 50,9 % | 46,4 % |
| Images vides déclenchant une alerte | 2 sur 7 | 2 sur 7 |

Le F1 baisse de 4,5 points après quantification. L'effet existe, mais le modèle
fait déjà beaucoup d'erreurs avant la conversion. Ce test ne permet pas de
conclure qu'il est prêt pour compter des entrées et des sorties.

## Ce qu'on observe dans les images

Le seul centre correctement localisé en FP32 qui devient manqué en INT8 est
dans `porte_B_04_2_personnes_ensemble2_f00180.jpg` (aperçu 15 du rapport).
Le modèle propose toujours deux points, proches des têtes. Pour une des personnes,
le point passe de `(7, 5)` à `(6, 5)` sur la grille 20×20. Sa distance au centre
annoté passe de 2,23 à 2,91 cases, et dépasse alors la tolérance fixée à 2,5 cases.
Ce changement compte dans la métrique comme un faux positif et un centre manqué ;
il ne signifie pas que toute prédiction de cette personne a disparu.

D'autres erreurs existaient déjà avant INT8 : sur l'aperçu 13, les deux personnes
partiellement visibles sont manquées et un point apparaît sur un objet du décor.
Les aperçus montrent pourquoi il faut distinguer les erreurs de localisation,
les objets confondus avec une personne et les personnes partiellement visibles.

## Comment la comparaison a été faite

Le checkpoint est `models/tiny_heatmap_relu.pt`. Les images sont préparées
exactement comme pour l'entraînement : RGB 96×96 avec conservation des proportions
et bandes de remplissage. Le seuil reste 0,5, la séparation des pics 2 cases et
la tolérance de correspondance 2,5 cases. Ces paramètres n'ont pas été choisis
pour améliorer le résultat sur la porte B.

La calibration INT8 utilise 32 images du train ; aucune vidéo de test ni image
prétraitée identique ne se trouve dans ce jeu de calibration. L'exécution Aidge
FP32 est comparée à PyTorch sur les 27 images (écart maximal `3,81 × 10⁻⁶`).
Les tenseurs des quatre convolutions ont été contrôlés après exécution :
activations et poids INT8, biais et accumulation INT32. Les scores de sortie
sont déquantifiés puis passent dans la même fonction de détection de pics.

Cela valide une comparaison du graphe Aidge sur CPU. Le bitstream Colibry,
le firmware, la caméra, la consommation et le comptage vidéo ne sont pas exécutés
par ce test.

## Pour refaire le test

Depuis le dossier du projet sur le Mac :

```bash
./.venv/bin/python scripts/compare_tiny_heatmap_int8.py
```

Sous Windows, après installation d'un environnement Aidge compatible :

```powershell
.venv\Scripts\python.exe scripts/compare_tiny_heatmap_int8.py
```

Le rapport illustré se trouve dans `runs/comparaison_int8_porte_B/rapport.html`.
`details.csv` donne les mesures par image ; `metrics.json` conserve les paramètres,
les empreintes du modèle et les types numériques vérifiés. Tout reste local.

La prochaine amélioration doit commencer par examiner ces erreurs et préciser
le point visé dans la vue de dessus (centre de la personne ou tête), en restant
cohérent entre entraînement et évaluation. On pourra ensuite réentraîner et
refaire exactement cette comparaison. Les 27 images de la porte B restent à part
de l'entraînement ; après s'en être servi pour orienter les changements, un autre
jeu indépendant sera nécessaire pour mesurer la généralisation finale.
