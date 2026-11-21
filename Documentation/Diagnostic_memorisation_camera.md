# Diagnostic de mémorisation sur huit images caméra

## Question vérifiée

La chaîne Aidge peut-elle apprendre un petit lot simple avant de chercher
à généraliser sur les milliers d'images du projet ?

## Protocole

Huit images sont prises uniquement dans le train existant, avec annotations
confirmées : images 1, 5, 15, 44, 45, 47, 31 et 50. Elles contiennent huit
présences annotées sur six images positives et deux images vides.
Le réseau, ses poids initiaux et les cibles sont ceux de l'essai COCO 2000 v1.
Les empreintes des images et labels sont vérifiées et les cibles recalculées
depuis les rectangles pour contrôler leur cohérence.

Le diagnostic apprend ET évalue sur ces mêmes huit images. Ce n'est pas une
validation : aucune image de validation ou de test n'est chargée. Il n'y a
aucune augmentation. Batch de huit, Adam NumPy, LR 0,001, forward et backward
Aidge CPU FP32. Les modèles existants ne sont pas remplacés.

## Résultat

| Mises à jour | Loss | Personnes correctement détectées | Fausses détections | Personnes manquées | F1 sur le lot appris |
|---|---:|---:|---:|---:|---:|
| 0 | 4,3154 | 0 | 6 | 8 | 0 % |
| 20 | 0,3546 | 7 | 0 | 1 | 93,3 % |
| 40 | 0,0195 | 8 | 0 | 0 | 100 % |

Critères : score ≥ 0,5, IoU ≥ 0,5. Le diagnostic s'arrête à l'étape 40,
après environ dix secondes de calcul dans sa boucle d'apprentissage.

## Interprétation

La chaîne sait mémoriser ce petit lot, y compris des personnes proches et
partiellement hors champ. Un blocage total des mises à jour n'explique donc
pas les échecs précédents. Cela ne certifie pas tout le code ni toutes les
annotations et ne mesure pas la généralisation à d'autres images.

**Le projet n'a pas atteint 100 % de fiabilité.** Les résultats de test des
modèles précédents restent inchangés : trois présences manquées.
Le modèle de ce diagnostic n'est pas un remplaçant du détecteur principal.

## Suite proposée

Auditer un petit échantillon du COCO et de l'historique : personnes visibles
sans annotation, cohérence des rectangles et diversité de cadrage. Comparer
ensuite, sur les mêmes ensembles, un apprentissage caméra seul et le mélange
actuel pour vérifier si ce dernier dégrade l'adaptation caméra. Les poids
restent sélectionnés sur validation, sans ajuster le protocole aux trois
images de test déjà examinées. Une nouvelle vidéo sera nécessaire pour une
évaluation indépendante.

## Reproduire

Depuis la racine, avec les données privées de l'essai source déjà présentes :

```bash
./.venv-aidge-colibry/bin/python scripts/diagnose_small_batch_aidge.py --run-dir runs/diagnostic_memorisation_camera_v1 --steps 300 --learning-rate 0.001
```

Choisir un nouveau dossier si ce diagnostic existe déjà. L'environnement
Aidge doit être installé séparément sur chaque machine ; cette commande
a été vérifiée sur le Mac de l'équipe.

Résultats locaux : `runs/diagnostic_memorisation_camera_v1/` : `history.csv`,
`protocol.json`, `best_metrics.json`, `best_logits.npy`, modèle de diagnostic
et `rapport.html`. Les images, poids et rapports ne sont pas publiés sur GitHub.
