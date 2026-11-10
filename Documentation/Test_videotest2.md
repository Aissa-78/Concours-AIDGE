# Test des nouvelles vidéos — 3 novembre 2026

## Ce qui a été testé

Les 13 fichiers MP4 de `videotest2/Echantillon/` ont été lus sans modification. Ils représentent **12 fichiers distincts** : `2personnes_1m_gauchedroite.mp4` et `2personnes_gauchedroite.mp4` sont des doublons exacts. Les deux ont été traités pour retrouver facilement chaque nom, mais ne doivent pas être comptés comme deux tests indépendants.

1 023 images ont été analysées, environ 10 par seconde, sur toute la durée des clips. Cela ne teste pas encore une exécution à 60 images/s ni la consommation matérielle. Les vidéos annotées sont à la cadence échantillonnée, sans accélération volontaire et sans piste audio.

Chaque vidéo comparative montre, de gauche à droite :

1. YOLO local `yolo26n.pt`, image couleur originale.
2. Le même YOLO, copie grise 320 × 240 avec proportions conservées.
3. Notre candidat `runs/person_qqvga_hard_context_full_v3/best.onnx`, réellement exécuté par **Aidge CPU**, sur une copie grise 160 × 120.

Seuil de détection fixé à 0,5 ; NMS 0,5 ; pics locaux 3 × 3 pour notre modèle. La taille interne de YOLO reste 640. Les pipelines ont des résolutions et réseaux différents : cette comparaison ne sépare pas à elle seule ces deux causes.

Le nouveau lot COCO **n'a pas encore été utilisé pour un entraînement**. Les modèles et annotations existants n'ont pas été remplacés.

## Ce qu'on observe

Sur plusieurs aperçus à une et deux personnes, YOLO dessine des rectangles sur les personnes, y compris en gris 320 × 240. Notre petit modèle manque des personnes et propose parfois des rectangles dans l'ouverture de porte / le décor. La nouvelle vue est donc exploitable pour poursuivre les essais de détection, mais notre candidat n'est pas encore fiable.

Certaines personnes sont partiellement hors champ pendant leur entrée ou leur sortie de l'image ; certains pieds sont coupés. Pour de prochains tournages destinés aux corps entiers, laisser davantage de marge en bas et au-dessus de la tête, avec une caméra immobile.

Il n'y a **pas de score de précision ou de F1** : ces vidéos n'ont pas de rectangles humains de référence. Un score de confiance affiché près d'un rectangle n'est pas un pourcentage de réussite mesuré. Le nombre de rectangles par image n'est pas le nombre de personnes distinctes ou de passages.

Ce test fait uniquement de la détection : ni association des personnes entre images, ni franchissement de ligne, ni compteur entrée/sortie. Il ne valide ni ColibryNPU, ni UINT8, ni la contrainte de 5 mW.

## Où voir les résultats ?

Ouvrir `runs/videotest2_diagnostic_v2/rapport.html` par double-clic. Les 13 comparaisons sont dans `runs/videotest2_diagnostic_v2/videos/`, les aperçus dans `apercus/`, et les rectangles / paramètres / empreintes dans `results.json`.

Les 13 vidéos générées ont été décodées entièrement après traitement : 1 023 images lisibles, dimensions 1200 × 324. Les empreintes des vidéos originales sont inchangées. Les 79 tests automatiques du dépôt passent, dont trois tests ajoutés pour la préparation des comparaisons.

Le dossier `runs/videotest2_diagnostic_v1/` ne contient que la préparation du premier lancement interrompu : aucun résultat final. Le lancement a été corrigé en gardant le chemin du Python de l'environnement virtuel au lieu de résoudre son lien symbolique vers le Python système. Les résultats terminés sont dans **v2**.

## Comment refaire ?

Depuis la racine du projet, choisir un dossier de sortie encore inexistant. Les deux environnements Python et les modèles doivent déjà être installés localement.

macOS :

```sh
./.venv/bin/python scripts/test_video_detectors.py --video-dir videotest2/Echantillon --output-dir runs/videotest2_diagnostic_v3
```

Windows PowerShell, avec Python, Aidge et les modèles disponibles sur ce PC :

```powershell
.\.venv\Scripts\python.exe scripts/test_video_detectors.py --video-dir videotest2/Echantillon --output-dir runs/videotest2_diagnostic_v3 --aidge-python .venv-aidge-colibry/Scripts/python.exe
```

L'environnement Aidge doit être installé sur Windows ; ne pas copier celui du Mac. Ce script n'a pas été exécuté sous Windows. Aucun téléchargement de modèle n'est lancé : les poids locaux doivent être présents. Les vidéos personnelles restent locales, ignorées par Git.

## Suite logique

Adapter l'entraînement du petit détecteur aux profils / corps entiers, puis réévaluer sans utiliser les mêmes clips pour apprendre et tester. Si une partie de ces vidéos sert à l'apprentissage, réserver d'abord un vrai lot de test distinct ; garder aussi ensemble les doublons et les prises liées. Pour une vraie mesure de généralisation, prévoir une autre séance ou une autre porte : tous ces clips montrent le même décor.

Quand la détection est utilisable, tester le suivi puis le comptage. Le côté intérieur et la position de la zone de passage doivent être confirmés pour la caméra choisie ; ne pas les déduire seulement du nom gauche/droite des fichiers.
