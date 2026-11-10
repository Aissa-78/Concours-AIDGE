# Correction du détecteur QQVGA — 30 octobre 2026

## Ce qui a été vérifié

L'ancien modèle échoue aussi sur les images d'apprentissage : le problème ne
vient donc pas uniquement d'un manque de nouvelles portes. Les sorties et
gradients Aidge ont été comparés à une référence numérique. Abaisser seulement
le seuil fait apparaître trop de fausses détections.

Plusieurs petits essais utilisent les mêmes 64 images d'apprentissage et les
mêmes 98 images de validation. Une loss qui équilibre personne/fond retrouve
plus de personnes, mais produit trop de rectangles. Le dernier candidat limite
cette pénalité aux faux positifs les plus difficiles et utilise des convolutions
ordinaires plus expressives. Ce n'est pas encore un détecteur validé.

Les variantes n'utilisent pas toutes les mêmes réglages ; ce travail n'isole
pas chaque cause à lui seul. Les résultats sur des images déjà apprises ne
mesurent pas la capacité à reconnaître une nouvelle porte.

Sur la validation commune **corrigée**, avec seuil 0,5, pics 3 × 3 et IoU 0,5 :

| Essai | Images du lot train | Paramètres | Retrouvées | En trop | Manquées | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Ancien lot complet, focal | 579 | 4 645 | 1 | 0 | 136 | 1,45 % |
| Séparable, BCE équilibrée | 64 | 4 645 | 29 | 926 | 108 | 5,31 % |
| Convolutions ordinaires, BCE équilibrée | 64 | 22 757 | 31 | 251 | 106 | 14,80 % |
| Convolutions contextuelles, négatifs difficiles | 64 | 41 189 | 15 | 63 | 122 | 13,95 % |

Le F1 le plus élevé de ces quatre essais n'en fait pas un modèle acceptable :
il manque encore 106 présences sur 137 et propose 251 rectangles de trop. Le
candidat contextuel fait moins de fausses alertes, mais retrouve moins de
personnes. Les variantes séparables avec focal, avec ou sans connexions
résiduelles, n'ont retrouvé aucune présence dans leurs checkpoints sélectionnés.

## Les deux annotations confirmées

Dans `testvideo_jour33_f00120`, le troisième rectangle désignait la même
personne que le deuxième : il a été retiré. Dans `testvideo_jour33_f00210`, le
premier rectangle a été resserré sur la personne du bas ; celui du haut est
conservé. L'utilisateur a confirmé ces deux décisions.

Les anciennes annotations restent dans `dataset/detection_proposals/`. La
copie corrigée est dans `dataset/detection_corrected_20261030/`, avec un journal
`CORRECTIONS.json` et deux aperçus dans `rapport.html`. Aucune image n'est
effacée et le train n'a pas changé.

Les copies grises corrigées sont dans `dataset/camera_qqvga_corrige_v2/` :

| Groupe | Images | Présences annotées |
|---|---:|---:|
| Train : portes + WEPDTOF | 579 | 750 |
| Validation : portes | 58 | 83 |
| Validation : WEPDTOF | 40 | 54 |
| Validation totale | 98 | 137 |

Ce sont des présences sur des images, pas 137 personnes distinctes. Les 27
images de la porte B restent exclues de l'entraînement et de la sélection.

## Candidat complet

`runs/person_qqvga_hard_context_full_v3/` contient un essai de 12 époques maximum
sur les 579 images, en repartant du checkpoint sélectionné du petit essai
`person_probe64_hard_context_v1`. Adam est réinitialisé : ce n'est pas une
reprise exacte de l'ancien état d'optimisation.

**Résultat : arrêt anticipé après 10 époques, en 14 min 45 s environ.** Le
critère de validation n'a jamais dépassé celui du checkpoint de départ. L'état
retenu est donc l'époque **0**, avant les nouvelles mises à jour : 15 présences
retrouvées, 63 fausses détections, 122 manquées, F1 **13,95 %**. Ce n'est pas un
gain obtenu grâce aux 579 images. La dernière époque atteint 57,7 % de F1 sur
le train, mais seulement 12,8 % sur les portes de validation et 5,3 % sur le
public : ces nouveaux poids ne sont pas retenus.

Les **62 tests** automatisés passent. L'équivalence des graphes Aidge compact et d'apprentissage
est exacte sur les 579 images train et les 98 images de validation. Cela
confirme les calculs logiciels, pas la qualité de détection.

Le réseau reçoit une image grise 160 × 120. Il emploie 41 189 paramètres et
produit quatre distances de rectangle et un score par cellule. Les convolutions
ordinaires sont entraînées directement par Aidge, sans le contournement
depthwise des anciens essais. NumPy calcule la loss et Adam ; PyTorch définit
et exporte le graphe et sert de référence, sans mettre à jour les poids.

Réglages : batch 8, taux initial 0,0005, division par deux après six époques,
évaluation toutes les deux époques. Pas d'augmentation dans cet essai : on
vérifie d'abord l'apprentissage des images réelles. Le seuil reste à 0,5 et
une correspondance exige IoU ≥ 0,5. Les pics locaux 3 × 3 limitent les rectangles
issus de cellules voisines non entraînées pour leurs dimensions.

Le checkpoint maximise le plus faible F1 des deux sources, puis leur moyenne,
puis la loss en cas d'égalité. Le rapport ne présente donc pas nécessairement
la dernière époque. Ne pas confondre son F1 avec une accuracy personne/vide.

## Relancer le même type d'essai

Depuis le dossier du projet dans le terminal de VS Code, avec les deux
environnements déjà installés. Choisir un **nouveau nom** pour conserver les
résultats. Le dataset corrigé doit être présent localement.

Sur Mac ou Linux :

```bash
./.venv/bin/python scripts/prepare_person_training.py --camera-data dataset/camera_qqvga_corrige_v2 --run-dir runs/mon_essai_context_01 --architecture regular_context --channels 16 --blocks 3 --classification hard_bce --local-maximum
./.venv-aidge-colibry/bin/python scripts/train_person_qqvga_aidge.py --run-dir runs/mon_essai_context_01 --epochs 12 --batch-size 8 --learning-rate 0.0005 --evaluate-every 2 --patience 5 --lr-decay-every 6 --no-augmentation --init-weights runs/person_probe64_hard_context_v1/best_dense.npz --report-python .venv/bin/python
```

Sans `--init-weights`, le nouvel essai commence avec des poids aléatoires et
n'est pas la reproduction de cette poursuite.

Sur Windows PowerShell, après installation d'Aidge pour cet environnement :

```powershell
.\.venv\Scripts\python.exe scripts/prepare_person_training.py --camera-data dataset/camera_qqvga_corrige_v2 --run-dir runs/mon_essai_context_01 --architecture regular_context --channels 16 --blocks 3 --classification hard_bce --local-maximum
.\.venv-aidge-colibry\Scripts\python.exe scripts/train_person_qqvga_aidge.py --run-dir runs/mon_essai_context_01 --epochs 12 --batch-size 8 --learning-rate 0.0005 --evaluate-every 2 --patience 5 --lr-decay-every 6 --no-augmentation --init-weights runs/person_probe64_hard_context_v1/best_dense.npz --report-python .venv/Scripts/python.exe
Start-Process "runs/mon_essai_context_01/rapport.html"
```

Ne pas copier un environnement Mac sur Windows. Si Aidge est dans WSL, utiliser
ses environnements Linux et les commandes Linux depuis WSL. Ce pipeline n'a
pas été exécuté sur un PC Windows ; les chemins seuls ne garantissent pas son
installation. Les images, checkpoints et environnements ne sont pas dans Git.

## Comparer sans fausser les chiffres

`scripts/compare_person_detectors.py` refuse les validations différentes. Son
option `--validation-run` permet de réévaluer **tous** les modèles avec les
annotations corrigées d'un même essai, uniquement si les pixels et leur ordre
sont exactement identiques. Les prédictions sauvegardées sont réutilisées,
pas les anciens scores. Les rapports historiques restent inchangés.

Les rapports comparatifs sont dans `runs/person_comparison_corrected_v1/`
(petits essais) et, une fois produit, `runs/person_comparison_corrected_v2/`
(avec l'essai complet). Ils utilisent tous seuil 0,5, pics 3 × 3 et IoU 0,5.

Un diagnostic supplémentaire a utilisé le YOLO préentraîné déjà présent sur
les mêmes pixels gris : F1 de 1,45 % avec taille interne 160 ; 30,67 % en
agrandissant ces pixels à une taille interne 640 (25 présences retrouvées,
une fausse alerte, 112 manquées). L'agrandissement n'ajoute aucun détail et
augmente le calcul. Ce n'est ni un réseau Aidge embarqué, ni une solution
respectant les contraintes matérielles. Les propositions locales d'origine
venaient de YOLO avant leur revue : ce repère n'est pas indépendant de la
construction des annotations. Il ne prouve pas que 160 × 120 est impossible.

## Limites à garder en tête

La validation est réutilisée pour le développement et le choix des poids : ce
n'est pas une nouvelle porte aveugle. L'objectif de 90 % n'est pas garanti par
un entraînement plus long. Il faudra encore mesurer détection et comptage sur
des séquences indépendantes.

L'export logiciel de l'architecture contextuelle initiale a produit un fichier
NNPA dans le petit essai. Cela ne valide ni les poids appris sur carte, ni
l'UINT8 final, ni le firmware, ni la consommation totale de 5 mW. Le compteur
vidéo existant et ses modèles n'ont pas été remplacés.
