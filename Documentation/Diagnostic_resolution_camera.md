# Diagnostic de résolution — 30 octobre 2026

## Ce qui a été fait

Les **mêmes 98 images de validation et 137 présences annotées corrigées** ont
été analysées avec le YOLO local déjà entraîné. Aucun nouveau réseau n'a été
entraîné, aucune image/annotation source ni aucun modèle de l'interface n'a été
remplacé. Aucun téléchargement n'a été nécessaire.

Les sources ont été retrouvées via les fichiers de provenance et vérifiées par
SHA256. La copie grise 160 × 120 reconstruite est identique, pixel par pixel,
à celle du jeu QQVGA existant. Les rectangles sont recalculés à chaque format.

Huit conditions : originaux couleur/gris, format caméra 4:3 sans réduction
couleur/gris, 320 × 240 couleur/gris, 160 × 120 couleur/gris. La taille interne
YOLO reste à **640**, avec `rect=False`, pour tous les essais. Cette taille
interne est celle d'un diagnostic sur PC, pas celle d'un candidat embarqué.

Le format caméra conserve les proportions et ajoute des bandes grises plutôt
que de déformer ou couper l'image. Le contrôle « sans réduction » ajoute ces
bandes aux pixels d'origine sans les rééchantillonner. Les originaux et les
formats caméra n'ont donc pas exactement le même cadrage : on ne peut pas
attribuer tous leurs écarts à la seule réduction.

## Résultats principaux

Seuil fixé avant le test : **0,5**. NMS : 0,5. Une détection correspond à une
annotation si **IoU ≥ 0,5**, une seule fois par annotation.

| Images présentées au détecteur | Retrouvées | Fausses détections | Manquées | F1 |
|---|---:|---:|---:|---:|
| Original couleur | 50 | 1 | 87 | 53,2 % |
| Original gris | 44 | 0 | 93 | 48,6 % |
| Format caméra sans réduction, couleur | 43 | 1 | 94 | 47,5 % |
| Format caméra sans réduction, gris | 38 | 0 | 99 | 43,4 % |
| 320 × 240 couleur | 48 | 1 | 89 | 51,6 % |
| **320 × 240 gris** | **37** | **0** | **100** | **42,5 %** |
| 160 × 120 couleur | 29 | 1 | 108 | 34,7 % |
| **160 × 120 gris** | **22** | **0** | **115** | **27,7 %** |

Le gris 320 × 240 se rapproche du contrôle caméra gris sans réduction. La
réduction à 160 × 120 pénalise davantage ce détecteur sur ce lot. Le passage
en gris a aussi un effet au seuil 0,5, mais les images couleur originales
restent loin d'une détection fiable de toutes les présences.

**Cela justifie de tester un candidat 320 × 240 gris. Cela ne prouve pas que
notre petit modèle y atteindra 90 %.** Les échecs restants peuvent notamment
impliquer les vues du dessus, les personnes partielles, les annotations et
l'adaptation du modèle. Ce test ne sépare pas encore toutes ces causes.

## Diagnostic complémentaire du seuil

Les prédictions à faible confiance ont été conservées. Après la comparaison
principale, on a mesuré les seuils 0,25 et 0,1, sans modifier les poids.

En **320 × 240 gris**, au seuil **0,25** : **61 retrouvées, 0 fausse détection,
76 manquées, F1 61,6 %**. Au seuil 0,1 : 73 retrouvées, 10 fausses détections,
64 manquées, F1 66,4 %. Ce n'est pas une nouvelle amélioration de notre réseau
Aidge : c'est un autre réglage du YOLO de référence, sur la même validation.

Les 100 % de précision à 0,25 signifient seulement qu'aucun rectangle proposé
n'est faux sur ce lot ; **76 présences restent manquées**. Un seuil choisi à
partir de ce diagnostic devra être testé sur des séquences indépendantes.

## Ce qu'on fait ensuite

1. Préparer un candidat **320 × 240 gris**, avec une architecture dont les
   opérateurs et la mémoire peuvent être vérifiés dans Aidge/Colibry avant un
   entraînement long. Les scripts actuels du petit réseau sont en 160 × 120 :
   changer uniquement les images ne suffit pas, il faut adapter graphe,
   dimensions des cibles, inférence et calibration.
2. L'entraîner sur le train uniquement, puis comparer au QQVGA sur les mêmes
   annotations. Relire les erreurs ambiguës sans changer les labels pour
   favoriser le modèle. Ne pas lancer une collecte massive de vidéos avant
   d'avoir mesuré le bénéfice de cette adaptation.
3. Lorsque la détection est suffisamment fiable, mesurer le **comptage de
   passages** sur des vidéos indépendantes : les scores F1 des rectangles ne
   mesurent pas directement les erreurs entrée/sortie.

Avec deux mois restants, garder une étape vérifiable à la fois : un prototype
mesuré sur PC, puis la quantification/export et l'intégration matérielle, sans
repousser la validation de la carte et de la consommation à la toute fin.
La résolution supérieure accroît les besoins de calcul/mémoire selon le
réseau : elle reste un candidat à vérifier, pas une décision matérielle finale.

## Où regarder et comment refaire

- `runs/person_camera_resolution_v1/rapport.html` : tableau et exemples communs.
- `runs/person_camera_resolution_v1/comparison.json` : provenance et résultats.
- `scripts/benchmark_person_camera.py` : comparaison reproductible.

Depuis le terminal de VS Code dans le projet, avec les données locales et les
poids `yolo26n.pt` déjà présents. Choisir un dossier de sortie inexistant :

```bash
# Mac / Linux
./.venv/bin/python scripts/benchmark_person_camera.py --output-dir runs/diagnostic_resolution_02
```

```powershell
# Windows PowerShell, environnement installé sur Windows
.\.venv\Scripts\python.exe scripts/benchmark_person_camera.py --output-dir runs/diagnostic_resolution_02
Start-Process "runs/diagnostic_resolution_02/rapport.html"
```

Les dossiers `dataset/`, le run de validation corrigé et les poids sont locaux,
pas dans GitHub. Il faut les partager séparément entre camarades. Les chemins
Mac/Windows des fichiers de provenance sont récupérés dans le dossier local
du projet si nécessaire, puis leurs hashes sont contrôlés. Ne pas copier un
environnement Python Mac sur Windows. Le script n'a pas été exécuté sur un
poste Windows ; la récupération des chemins est couverte par un test.

## Limites

Cette validation a déjà servi au développement, ce n'est pas un test aveugle.
Les annotations locales avaient été proposées par YOLO avant leur revue : la
référence locale n'est donc pas totalement indépendante de leur construction.
Les scores concernent des présences sur des images, pas autant de personnes
distinctes. Aucun résultat de ce diagnostic ne valide Aidge, ColibryNPU,
l'UINT8, le suivi, le comptage ni les 5 mW de l'ensemble du système.
