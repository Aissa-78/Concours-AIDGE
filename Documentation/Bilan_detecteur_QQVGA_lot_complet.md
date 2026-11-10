# Détecteur QQVGA — bilan du lot complet, 30 octobre 2026

## Ce qui a été exécuté

Le nouveau détecteur à rectangles de **4 645 paramètres compacts**, en entrée
**160 × 120 gris**, a été entraîné sur les **579 images préparées** : 179 de nos
portes et 400 de WEPDTOF. Les 98 images de validation restent hors des calculs
de gradients ; elles servent au choix du checkpoint. La porte B reste exclue.
Les 20 images du test de mémorisation appartenaient déjà au train, sans ajout
en double au dataset.

La tentative `person_qqvga_full_v1` s'est interrompue ; le dernier état évalué
était celui de l'époque 14/30. L'absence de processus et de `training.json`
final a été constatée le 30 octobre ; la cause de l'arrêt n'a pas été établie.

La poursuite `person_qqvga_full_v2` repart de ces poids, avec **Adam réinitialisé**
(son état n'avait pas été enregistré dans la première tentative). Elle termine
16 époques supplémentaires en environ **24 min 43 s**. Ce n'est pas une reprise
exactement identique à une exécution continue. Son checkpoint retenu est celui
de sa dernière époque, soit 16, après 14 époques du premier essai.

Toutes les images train sont utilisées à chaque époque. Les répétitions de
la petite source équilibrent le mélange : 800 présentations, mais 579 images
distinctes. Les miroirs et petites variations de luminosité, contraste, flou et
bruit sont réservés au train. Forward/backward : **Aidge CPU FP32** ; loss et
Adam : NumPy. PyTorch n'effectue pas les mises à jour des poids.

## Résultats du checkpoint retenu

Seuil de détection **0,5**, correspondance d'un rectangle **IoU ≥ 0,5**.

| Groupe | Images | Retrouvées | Supplémentaires | Manquées | F1 |
|---|---:|---:|---:|---:|---:|
| Train, portes et WEPDTOF | 579 | 2 | 4 | 748 | 0,53 % |
| Validation de nos portes | 58 | 1 | 0 | 83 | 2,35 % |
| Validation WEPDTOF | 40 | 0 | 0 | 54 | 0 % |
| Validation globale | 98 | 1 | 0 | 137 | 1,44 % |

Il s'agit de **présences annotées sur des images**, pas d'autant de personnes
différentes ni de passages de porte. Une précision affichée à 100 % correspond
ici à une seule proposition correcte : le rappel global n'est que **0,72 %**.
Il ne faut surtout pas présenter ce 100 % comme la fiabilité de l'IA.

Le F1 des rectangles n'est pas directement comparable aux anciens scores de
centres du modèle heatmap. Les 100 % du test QQVGA sur 20 images étaient une
mesure de mémorisation, pas un score sur ces 98 images séparées.

## Ce que l'on peut conclure

**Cet essai est un échec de détection ; il ne doit pas remplacer le modèle de
l'interface.** Ajouter le lot public et prolonger l'entraînement ne suffisent
pas avec cette configuration. Les erreurs concernent aussi les images apprises :
le problème ne se limite donc pas à la reconnaissance d'une nouvelle porte.

La cause exacte n'est pas encore isolée. Avant de récolter encore plus d'images,
il faut contrôler la cohérence des rectangles entre les deux sources, puis
comparer un apprentissage contrôlé sans augmentation et une architecture plus
adaptée. Changer uniquement le seuil peut produire davantage de rectangles,
mais ne garantit pas de bonnes détections. Aucune de ces pistes n'a été validée
par ce bilan.

## Ce qui fonctionne techniquement

Les sorties du graphe compact depthwise dans Aidge sont **identiques** à celles
du graphe dense contraint appris, sur les 579 images train et les 98 images de
validation : écart maximal **0,0**. Cela valide le chemin logiciel, pas la qualité
du modèle. Le backward depthwise natif CPU manque toujours ; le contournement
par convolutions diagonales contraintes est conservé et contrôlé.

Le rapport contient 98 aperçus des erreurs. Aucun ancien modèle, aucune image
ni annotation n'a été remplacé. Les sauvegardes par époque sont désormais
enregistrées. Aucun nouvel export matériel, test de comptage, test UINT8 ni
mesure de puissance n'a été effectué pour ces poids.

## Où regarder

- `runs/person_qqvga_full_v2/rapport.html` : résultat lisible et images annotées.
- `runs/person_qqvga_full_v2/metrics.json` : toutes les mesures du checkpoint.
- `runs/person_qqvga_full_v2/training_history.csv` : évolution des évaluations.
- `runs/person_qqvga_full_v2/protocol.json` : paramètres et provenance des poids.
- `runs/person_qqvga_full_v2/best.onnx` : modèle compact, **non retenu pour le compteur**.

Les commandes de reproduction et les limites de reprise sont dans
[le guide du détecteur QQVGA](Entrainement_detecteur_QQVGA.md).
