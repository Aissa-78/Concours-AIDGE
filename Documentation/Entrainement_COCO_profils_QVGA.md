# Entraînement COCO + vidéos de profil — 3 novembre 2026

## Résultat de ce premier essai

Un nouveau détecteur de personnes a été entraîné **30 époques dans Aidge CPU FP32**, avec les images COCO sélectionnées et une partie des nouvelles vidéos. Le checkpoint retenu est celui de l'époque 30. Les anciens essais, modèles et vidéos sont conservés. Le modèle de l'interface n'a pas été remplacé.

| Validation | Images | Présences annotées | Retrouvées avant → après | Fausses détections avant → après | F1 avant → après |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vos vidéos de profil | 34 | 32 | 0 → 21 | 37 → 3 | 0 → 75 % |
| COCO | 33 | 27 | 5 → 10 | 165 → 12 | 5,1 → 40,8 % |
| Ensemble | 67 | 59 | 5 → 31 | 202 → 15 | 3,8 → 59,0 % |

Sur vos images : précision 87,5 %, rappel 65,6 %. Cela signifie que 21 des 24 rectangles proposés sont appariés à une annotation, mais **11 présences sur 32 restent manquées**. Ce n'est pas « 87,5 % de passages bien comptés ». Le F1 combine précision et rappel.

Ces résultats sont **provisoires**. Les rectangles des vidéos ont été proposés par YOLO, puis contrôlés visuellement par l'assistant ; ils ne sont pas une vérité terrain humaine indépendante. La validation sert à choisir le checkpoint. Toutes les prises locales montrent le même lieu. Une relecture humaine et une autre séance / porte restent nécessaires pour une mesure plus solide.

Le nouveau système change les données, la résolution et l'architecture en même temps. La comparaison montre un progrès du système complet sur ce lot, pas le bénéfice isolé de COCO. L'ancien modèle est `runs/person_qqvga_hard_context_full_v3/best.onnx`.

## Quelles données ont servi ?

Les 165 images COCO retenues sont conservées avec leur répartition : 132 train et 33 validation. Les images des vidéos sont extraites environ deux fois par seconde, sans rectangle dessiné dans leurs pixels d'entraînement.

Parmi les 13 fichiers vidéo, deux sont des copies exactes : `2personnes_1m_gauchedroite.mp4` et `2personnes_gauchedroite.mp4`. Ils ont la même affectation et un seul est extrait. Les **12 prises uniques** sont réparties avant extraction :

- 6 pour l'apprentissage ;
- 3 pour la validation ;
- 3 réservées au test du checkpoint figé.

La répartition est dans `configs/profile_split_v1.json`. Les empreintes SHA256 sont contrôlées pour qu'un doublon ne traverse pas les lots. Les trois prises de test sont :

- `1personne_droitegauche_1personne_gauchedroite.mp4` ;
- `2 personne vers la DROITE.mp4` ;
- `2personnedroitegauche.mp4`.

142 images vidéo ont été proposées et examinées sur huit planches. 106 sont conservées : 72 train et 34 validation. Une image avec une personne non annotée au bord gauche est exclue entière ; les fonds vides redondants sont limités à deux par prise. Les décisions sont dans `Documentation/profile_video_review_20261103.json`. Ce n'est pas une validation humaine implicite.

Au total : **204 images d'entraînement et 67 de validation**. Les trois vidéos de test n'ont servi ni aux gradients ni au choix du meilleur checkpoint. Elles avaient déjà été vues lors du diagnostic précédent et partagent le décor de l'entraînement : ne pas présenter ce test comme un test aveugle de généralisation à n'importe quelle porte.

## Comment le modèle apprend

L'entrée est une image grise de **320 × 240**, avec des bandes grises si nécessaire pour conserver les proportions. La couleur est retirée, les images sont réduites sans couper les personnes et les rectangles suivent exactement la transformation.

Le modèle `src/profile_detector.py` contient 49 577 paramètres : quatre convolutions de réduction avec ReLU, trois convolutions 5 × 5 avec ReLU pour augmenter le contexte, puis une convolution de sortie. Son champ réceptif théorique est de 223 pixels. Cela permet de considérer davantage du corps que dans un petit réseau à contexte très local ; ce n'est pas une garantie de précision.

La sortie est une grille de **20 × 15**. Chaque case propose un score de présence et quatre distances vers les bords d'un rectangle. Ce n'est ni le premier classifieur binaire personne/vide, ni YOLO. Dans les fichiers de labels de détection, `0` signifie **personne** ; un label vide signifie qu'aucune personne n'est annotée.

L'entraînement est **conjoint**, et non deux phases successives. Chaque époque utilise toutes les images train ; la petite source est répétée pour présenter autant de COCO que de vidéos, soit 264 présentations pour 204 images distinctes. Les augmentations touchent uniquement le train : miroir horizontal, variations modérées de lumière, flou et bruit légers. Aucune rotation tête en bas. Les cibles sont recalculées avec les rectangles transformés.

La loss associe une focal loss pour les centres et une SmoothL1 pour les distances des rectangles. Les poids sont corrigés par Adam à partir des gradients. Taux initial : 0,001, divisé par deux tous les dix passages complets. Batch : 8 ; validation toutes les deux époques.

**Aidge calcule le forward et le backward.** La loss et Adam sont calculés en NumPy. PyTorch sert à initialiser / exporter le graphe et à fournir une référence de gradients, pas à entraîner les poids. Le contrôle initial donne un écart maximal de gradients de 3,81 × 10⁻⁶.

Le meilleur checkpoint maximise d'abord le moins bon F1 entre COCO et les vidéos, puis leur moyenne, puis minimise la loss. Cela évite de choisir seulement un modèle bon sur la salle locale. Le seuil de détection est fixé à 0,5 avant le test ; la correspondance rectangle exige IoU ≥ 0,5.

## Où regarder ?

Ouvrir `runs/person_profile_mix_qvga_v1/rapport.html` par double-clic. Il contient les scores, les trois vidéos de test avant/après et les 67 images de validation annotées.

- `dataset/profile_mix_v1/` : les deux sources COCO et portes, chacune séparée en train / validation.
- `runs/person_profile_mix_qvga_v1/best.onnx` : modèle entraîné, réimporté et vérifié dans Aidge CPU.
- `best_dense.npz` : poids d'apprentissage.
- `last_training_state.npz` : dernier état complet avec poids et Adam ; le script ne propose pas encore de reprise automatique complète depuis ce fichier.
- `training_history.csv`, `training.json`, `protocol.json` : progression et réglages.
- `metrics.json` : comparaison avec l'ancien modèle sur la même validation.
- `videos_test_reservees/` et `test_results.json` : trois comparaisons vidéo, à environ 10 images/s, sans son, avec les paramètres et prédictions.

Les trois vidéos produites ont été décodées entièrement : 251 images lisibles. Les originaux et le checkpoint sont inchangés pendant le test. Les 84 tests automatiques du dépôt passent, dont les contrôles de géométrie QVGA, de récupération des chemins Mac/Windows et de compatibilité des anciens comportements QQVGA.

## Refaire un entraînement soi-même

Depuis la racine du projet. Les dépendances, le dataset privé et les deux environnements doivent déjà être installés. Choisir un **nouveau nom d'essai** pour ne rien écraser. Les commandes ci-dessous refont un entraînement à partir des poids aléatoires ; ce n'est pas une reprise de l'époque 30.

macOS :

```sh
./.venv/bin/python scripts/prepare_profile_training.py --run-dir runs/person_profile_mix_qvga_v2
```

Puis :

```sh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 ./.venv-aidge-colibry/bin/python scripts/train_person_qqvga_aidge.py --run-dir runs/person_profile_mix_qvga_v2 --epochs 30 --batch-size 8 --learning-rate 0.001 --patience 8 --evaluate-every 2
```

Enfin :

```sh
./.venv/bin/python scripts/report_profile_training.py --run-dir runs/person_profile_mix_qvga_v2
```

Windows PowerShell, avec des environnements installés sous Windows :

```powershell
.\.venv\Scripts\python.exe scripts/prepare_profile_training.py --run-dir runs/person_profile_mix_qvga_v2
```

Puis :

```powershell
$env:OMP_NUM_THREADS="4"
$env:OPENBLAS_NUM_THREADS="4"
.\.venv-aidge-colibry\Scripts\python.exe scripts/train_person_qqvga_aidge.py --run-dir runs/person_profile_mix_qvga_v2 --epochs 30 --batch-size 8 --learning-rate 0.001 --patience 8 --evaluate-every 2
```

Enfin :

```powershell
.\.venv\Scripts\python.exe scripts/report_profile_training.py --run-dir runs/person_profile_mix_qvga_v2
```

Ne pas copier un `.venv` Mac sur Windows. Les scripts n'ont pas été exécutés sur Windows. Les chemins locaux des vidéos sont récupérés dans `videotest2/Echantillon/` si la provenance mentionne l'autre ordinateur. Les données et modèles ignorés par Git doivent être partagés séparément, en privé ; le dépôt Git seul ne suffit pas.

Pour recréer aussi les données, les étapes sont `prepare_profile_mix.py candidates`, revue puis `prepare_profile_mix.py assemble --decisions ...`, avec de nouveaux dossiers. Ne pas changer les images d'un essai déjà préparé : ses empreintes sont contrôlées avant apprentissage. Ne pas utiliser les mêmes vidéos pour apprendre et annoncer ensuite un test indépendant.

## Ce qui reste à faire

La priorité est de relire humainement les annotations et de comprendre les personnes manquées. Le résultat COCO reste nettement inférieur au résultat sur notre porte. Pour une autre caméra ou une autre porte, on ne peut pas annoncer 75 % ou 90 % à partir de ce bilan.

Ensuite : mesurer sur une autre séance / porte, puis intégrer le modèle choisi à l'interface et tester le suivi et le compteur. Le nouveau modèle n'a pas encore été branché dans l'interface existante.

L'import et l'inférence **Aidge CPU FP32** sont vérifiés. Le budget réel poids + activations + buffers, l'export et l'exécution ColibryNPU, la quantification UINT8 et la consommation maximale de 5 mW ne sont pas validés. La petite taille des poids ne suffit pas à prouver la conformité matérielle.
