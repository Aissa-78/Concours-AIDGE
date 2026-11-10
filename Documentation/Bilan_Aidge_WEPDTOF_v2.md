# Plus de données vues du dessus : résultat du deuxième essai

20 octobre 2026

L'apprentissage est terminé. Sur les 27 images de la porte B, le F1 passe de **43,5 % à 45,5 %** : le réseau retrouve toujours 10 centres sur 24, avec deux points hors tolérance en moins. C'est un petit gain, pas une détection fiable ni un résultat proche de 90 %. Les anciens modèles et le programme vidéo n'ont pas été remplacés.

## Ce qui a été ajouté

320 images annotées issues de vidéos publiques WEPDTOF ont été ajoutées au train. Le lot public passe de 80 à **400 images**, avec 582 présences annotées et 80 images sans personne. Il couvre huit lieux au lieu de quatre ; les nouveaux lieux sont `printing_store`, `repair_store`, `it_office` et `large_office`.

Avec les 179 images de nos portes, cela fait **579 images d'apprentissage et 750 présences annotées**. La même personne peut apparaître sur plusieurs images : ce ne sont pas 750 personnes différentes.

Les 40 images de validation publique et leurs annotations sont identiques à celles du premier lot. Les 58 images de validation locale et les 27 images de la porte B sont conservées hors apprentissage. Le seuil est choisi sur la validation locale, jamais sur la porte B. La porte B a cependant déjà servi au diagnostic ; une autre porte encore jamais examinée sera nécessaire pour l'évaluation finale.

Les 120 images et annotations du lot public initial ont été copiées sans modification. Les ajouts excluent les frames déjà sélectionnées et leurs voisines proches ; les lieux de validation sont exclus. Les fenêtres doivent être entièrement dans la zone annotée et ne doivent pas transformer une personne trop coupée ou trop petite en fond. Les contrôles automatiques ne remplacent pas une revue humaine complète. Les quatre pages d'images négatives et un aperçu des huit scènes ajoutées ont été inspectés visuellement, sans anomalie évidente ; cela ne valide pas toutes les annotations.

Le téléchargement supplémentaire a transféré environ 192,2 Mo, pas l'archive entière. Les conditions WEPDTOF et la provenance sont conservées dans les métadonnées et le [guide du lot initial](Lot_vues_du_dessus.md). Les images restent locales et ne sont pas publiées automatiquement sur GitHub.

## Comment l'apprentissage a été poursuivi

L'architecture est inchangée : CNN gris de 17 885 paramètres, entrée 128×128 dont une image utile de 80×80 et une bordure, sortie de centres sur une grille 20×20. Ce réseau ne prédit pas des rectangles et n'est pas encore intégré au comptage vidéo.

Cette fois, les poids de `runs/heatmap_wepdtof_v1/best.npz` sont repris. L'état de l'optimiseur repart à zéro. Le réseau fait ensuite :

1. **16 époques publiques**, batch 16, taux d'apprentissage 0,0005. Le meilleur checkpoint public est celui de l'époque 6.
2. Une **adaptation aux portes**, batch 16, taux 0,0002. Chaque époque combine les 179 images locales et 64 images publiques rééchantillonnées. Sur les 40 époques maximales, 39 ont été exécutées ; l'arrêt anticipé après 10 époques sans amélioration restaure l'époque 29.

La sélection utilise le F1 de validation, puis la loss en cas d'égalité. La phase publique est sélectionnée sur la validation publique ; la phase finale sur la validation locale. Le calcul avant/arrière est fait dans **Aidge CPU FP32**, avec la BCE pondérée et Adam en NumPy pour contourner le problème de l'optimiseur installé. Le contrôle préalable des gradients contre la référence PyTorch donne un écart maximal de `1,12 × 10⁻⁸`. PyTorch n'effectue pas cet apprentissage.

Durée : environ **18 min 25 s**. Le seuil retenu sur les 58 images de validation locale reste **0,50**, parmi les candidats 0,20 à 0,80. Les poids, le lot, la durée et les taux d'apprentissage changent par rapport au premier essai : ce n'est pas une expérience permettant d'attribuer le résultat à la seule quantité d'images.

## Résultats comparables

Même seuil 0,5 et même rayon de correspondance pour les deux essais V2. Le F1 combine précision et rappel des **centres annotés** : ce n'est ni une accuracy personne/vide ni un pourcentage d'entrées/sorties bien comptées.

| Lot évalué | Premier essai : 80 images publiques de train | Nouvel essai : 400 images publiques de train |
|---|---:|---:|
| Validation locale : 58 images, 84 présences | 48,8 % | **49,6 %** |
| Validation publique : 40 images, 54 présences | 7,1 % | **7,0 %** |
| Porte B : 27 images, 24 présences | 43,5 % | **45,5 %** |
| Porte B, INT8 signé sur CPU | 42,6 % | **44,4 %** |

Sur la porte B :

| Mesure FP32 | Premier essai | Nouvel essai |
|---|---:|---:|
| Centres retrouvés | 10 | 10 |
| Points supplémentaires ou trop éloignés | 12 | 10 |
| Centres manqués | 14 | 14 |
| Précision | 45,5 % | 50,0 % |
| Rappel | 41,7 % | 41,7 % |

Le gain de F1 est d'environ **2 points**, grâce à moins de points hors tolérance, pas à davantage de centres retrouvés. L'ancien ReLU reste à 50,9 % et l'ancien Hardtanh à 68,9 % sur cette porte. Aucun remplacement automatique n'est donc effectué.

Sur la validation locale, le nouveau réseau retrouve 30 centres sur 84, comme le précédent, avec 7 points hors tolérance au lieu de 9. Sa précision de 81,1 % peut sembler bonne, mais son rappel n'est que de 35,7 % : il manque beaucoup de présences. Il ne faut pas présenter ce 81,1 % comme la fiabilité globale du système.

Les validations servent à choisir les poids ou le seuil et ne sont pas des tests indépendants. Les 27 images de la porte B sont peu nombreuses et corrélées ; un faible gain ne prouve pas une meilleure généralisation. Un point sur une tête trop éloigné du centre du rectangle peut être compté comme point hors tolérance et centre manqué : regarder les exemples, pas seulement les totaux.

## Le problème repéré pendant cet essai

Sur **les mêmes 40 images publiques**, le checkpoint retenu avant adaptation aux portes donne :

| Étape | Centres retrouvés | Hors tolérance | Centres manqués | F1 |
|---|---:|---:|---:|---:|
| Meilleur checkpoint public, époque 6 | 28 | 19 | 26 | **55,4 %** |
| Modèle final adapté aux portes, époque 29 | 2 | 1 | 52 | **7,0 %** |

Cette comparaison a été refaite par inférence Aidge sur les mêmes entrées, sans régler le seuil sur ces images. Elle montre une forte perte de détection publique pendant l'adaptation. Elle est compatible avec un oubli lors de la spécialisation aux portes et montre une limite de la sélection uniquement locale. Elle ne démontre pas que c'est l'unique problème, ni que le checkpoint public serait meilleur sur toute nouvelle porte.

La priorité du prochain essai est donc de **mieux préserver ce qui a été appris sur les scènes publiques** : par exemple un mélange public/local plus équilibré et une règle de sélection tenant compte des deux validations, définie avant de lancer l'essai. Ne changer qu'un facteur à la fois pour savoir ce qui aide. Examiner ensuite les erreurs restantes et les limites de résolution/cible/architecture. Ajouter encore des centaines de vidéos n'est pas la première réponse à ce résultat.

## Contrôle INT8 et limites matérielles

Les quatre convolutions ont été contrôlées en activations et poids INT8 signés, avec accumulation INT32 sur CPU. La calibration utilise 32 images du train local. Le F1 sur la porte B passe de 45,5 % en FP32 à 44,4 % en INT8 CPU ; les centres retrouvés restent à 10, avec un point hors tolérance supplémentaire.

L'écart maximal entre Aidge FP32 et la référence PyTorch portant les mêmes poids appris est de `3,34 × 10⁻⁶` sur la porte B. Ce contrôle ne lance pas le bitstream matériel. Aucun nouveau bitstream n'a été généré pour cet essai ; la cible finale UINT8 asymétrique, le comptage complet, l'exécution sur carte et la consommation de 5 mW restent à valider.

## Retrouver les fichiers et refaire l'essai

Les résultats restent locaux dans `runs/heatmap_wepdtof_v2/`, ignoré par Git :

- `best.onnx` et `best.npz` : poids finaux, époque 29 de l'adaptation ;
- `best_wepdtof.onnx` et `best_wepdtof.npz` : meilleur checkpoint public, époque 6 ;
- `dataset.json`, `training.json`, `training_history.csv` : provenance, réglages, empreinte du checkpoint parent et historique ;
- `threshold.json` : choix du seuil uniquement sur validation locale ;
- `evaluation_detection_proposals/rapport.html` : comparaison locale ;
- `evaluation_wepdtof_expanded_v2/rapport.html` : comparaison publique après adaptation ;
- `evaluation_wepdtof_expanded_v2/pretrain_logits.npy` : sorties du checkpoint public avant adaptation sur les mêmes entrées ;
- `evaluation_porte_B_test/rapport.html` : comparaison porte B et INT8 ;
- `metrics.json` et `details.csv` dans chaque dossier d'évaluation : résultats chiffrés et par image.

Les nouveaux scripts et les modifications de sélection ont passé les **23 tests automatisés** du projet sur le Mac. Les [commandes de préparation, d'entraînement et de test, avec chemins Mac et Windows](Entrainement_WEPDTOF_augmente.md), sont enregistrées dans le projet. Il faut installer un environnement Aidge compatible sur chaque poste ; les commandes Windows de ce nouvel essai n'ont pas été exécutées ici.

L'objectif futur de 90 % doit être vérifié sur des vidéos inédites, en contrôlant chaque passage et son sens, ainsi que les fausses entrées/sorties. **Cet objectif n'est pas atteint avec ce nouvel essai.**
