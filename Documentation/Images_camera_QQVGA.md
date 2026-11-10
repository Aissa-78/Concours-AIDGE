# Préparer les images comme celles d'une petite caméra

Mise à jour du 29 octobre : un [détecteur QQVGA a passé le test technique sur 20 images apprises](Detecteur_QQVGA_test20.md).
Ce n'est pas encore un entraînement complet ou une mesure sur des images inconnues.

Le carnet de mission ASYGN du 8 octobre 2026, lu entièrement, recommande en page 12 une résolution au plus de 320×240 et un entraînement en niveaux de gris. Pour le prochain essai, on choisit **160×120 pixels utiles (QQVGA), un canal gris uint8**. Cela supprime de la couleur et du détail par rapport aux sources haute résolution, sans prétendre reproduire exactement la caméra encore absente.

## Ce qui est fait

Un prétraitement commun est dans `src/camera_preprocess.py`. Les deux opérations importantes, commentées dans le code, sont :

```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Un seul canal au lieu de trois.
resized = cv2.resize(gray, (rw, rh), interpolation=cv2.INTER_AREA)  # Réduire le nombre de pixels.
```

`rw` et `rh` sont calculés pour conserver les proportions. L'image est placée dans un cadre gris de 160×120 : on ne déforme pas les personnes et on ne les coupe pas. Il n'y a pas de rotation automatique, de flou arbitraire ni de compression JPEG supplémentaire. Les PNG sont grises ; seuls les aperçus ajoutent les cadres verts pour vérifier les annotations.

Le même traitement est appliqué aux images de nos vidéos et au lot public. Les rectangles sont transformés dans les fichiers `labels`, en tenant compte de la réduction et des bandes de remplissage. Ils ne sont **pas dessinés dans les véritables images d'apprentissage**.

Les sources restent intactes. Les splits, les identifiants vidéo et la provenance des annotations restent identiques ; les annotations publiques ne deviennent pas artificiellement « vérifiées par nous ».

## Les copies préparées

Dans `dataset/camera_qqvga_v1/` :

| Dossier | Train | Validation / test |
|---|---:|---:|
| `portes` | 179 images | 58 images de validation |
| `wepdtof` | 400 images | 40 images de validation |
| `porte_B_test` | **0** | 27 images hors apprentissage |

Soit **704 copies** : 579 images de train, 98 de validation et 27 de test déjà utilisé pour le diagnostic. Les trois jeux ne sont pas fusionnés automatiquement. `SUMMARY.json` garde les quantités et `SOURCE.json` garde les sources, leurs empreintes et les transformations.

Ouvrir `dataset/camera_qqvga_v1/rapport.html` pour les aperçus. Les images restent locales et sont ignorées par Git.

## Refaire les copies

À la racine de FlowSense-AIDGE, dans le terminal de VS Code :

Mac :

```bash
./.venv/bin/python scripts/prepare_camera_dataset.py --output-dir dataset/camera_qqvga_mon_essai
```

Windows, avec l'environnement installé sur ce PC :

```powershell
.\.venv\Scripts\python.exe scripts/prepare_camera_dataset.py --output-dir dataset/camera_qqvga_mon_essai
```

Le script refuse un dossier de sortie existant. Les arguments `--width` et `--height` sont configurables, avec 160 et 120 par défaut et un plafond à 320 et 240. Ne changer la résolution qu'avec un réseau conçu pour la recevoir. Les scripts ont été exécutés sur Mac ; leur exécution Windows reste à vérifier sur un poste équipé.

## Important : les poids n'ont pas changé

Les anciens essais Aidge V2/V3 étaient **déjà en gris**, avec 80×80 pixels utiles placés dans une entrée 128×128. Leur fonction de prétraitement appelle maintenant le helper commun, mais conserve exactement le même résultat, testé pixel par pixel. Le premier classificateur et les anciens modèles RGB ne sont pas modifiés.

**La préparation du lot, à elle seule, ne constitue pas un entraînement. Ne pas le passer à `prepare_heatmap_v2.py` en pensant entraîner en 160×120 : ce script réduit encore les images à 80×80 utiles.** Le nouveau `prepare_person_memorization.py` utilise directement ces images QQVGA, avec un détecteur, des cibles et un décodage adaptés. Son test sur 20 images est terminé, pas encore l'apprentissage sur le lot complet. L'inférence future devra utiliser le même prétraitement que l'entraînement.

Le contrôle sur un petit lot sans augmentation est désormais réussi ; voir le bilan lié en tête de cette page. La suite est un apprentissage complet et une évaluation séparée. La meilleure résolution n'est pas garantie : plus de pixels peut aider, mais augmente aussi les calculs et les buffers mémoire.

## Les autres contraintes à conserver

Le carnet recommande d'abord un scénario simple : personnes espacées en file indienne dans un sens, avant les groupes et les croisements. Les pages 11 à 13 imposent ensuite la compatibilité des opérateurs, la vérification de la mémoire poids + données sous 640 kB et une quantification **UINT8 asymétrique per-tensor**. La page 7 emploie aussi « UINT32 », en contradiction avec ces consignes détaillées ; ne pas transformer ce passage isolé en cible de déploiement sans clarification des organisateurs.

Les 19 200 octets d'une seule image grise 160×120 ne sont **pas** l'empreinte du réseau complet : les poids, cartes intermédiaires, buffers et leur disposition comptent aussi. L'exporteur devra vérifier le modèle adapté. Aucune consommation inférieure à 5 mW ni une performance de 90 % n'est prouvée par ces copies d'images.

## Documentation Aidge vérifiée

- [Accueil officiel et accès au workflow](https://eclipse.dev/aidge/index.html).
- [Index des tutoriels : entraînement, fournisseur de données, analyse statique, PTQ](https://eclipse.dev/aidge/source/Tutorial/index.html).
- [Dépôt bundle : modules, documentation et tutoriels](https://gitlab.eclipse.org/eclipse/aidge/aidge).

Ces ressources aident à vérifier la chaîne. Leur consultation n'a entraîné aucune réinstallation ou mise à jour des environnements locaux qui fonctionnent déjà. Elles ne garantissent pas à elles seules la fiabilité d'un détecteur de personnes.
