# FlowSense sur Windows — guide pour le groupe

Mise à jour du 10 novembre 2026. Le dépôt contient le code commun Mac/Windows, les lanceurs Windows et les bilans. Il ne contient pas les vidéos personnelles, datasets, poids locaux ou environnements Python. GitHub Pro et Copilot ne sont pas nécessaires pour utiliser le code.

## 1. Récupérer la dernière version

Si le projet a été cloné avec Git : dans le terminal de VS Code, à sa racine :

```powershell
git status
git pull --ff-only
```

S'il y a des modifications personnelles, les conserver et les sauvegarder avant de résoudre une éventuelle divergence. Ne pas faire de `reset --hard` pour forcer la mise à jour.

Sans Git : ouvrir [le dépôt](https://github.com/Aissa-78/Concours-AIDGE), choisir **Code → Download ZIP**, puis **Extraire tout** dans l'Explorateur Windows. Ouvrir le dossier extrait contenant `README.md` dans VS Code, pas le ZIP lui-même. Une ancienne copie téléchargée en ZIP ne se met pas à jour avec `git pull` : extraire la nouvelle version dans un dossier séparé et conserver ses données privées.

Installer les extensions gratuites **Python** et **Pylance** recommandées par VS Code. Aucun abonnement Copilot n'est requis.

## 2. Installer Python et les dépendances de l'interface

Installer **Python 3.11 ou 3.12, 64 bits**, depuis [python.org](https://www.python.org/downloads/windows/). Dans l'installateur, activer l'ajout de Python au PATH et le lanceur Python si ces options sont proposées. Fermer puis rouvrir VS Code.

Double-cliquer sur **Installer_Windows.bat**, à la racine du projet. Il :

1. Cherche Python 3.11/3.12 sur le PC.
2. Crée un environnement **Windows** dans `.venv/`.
3. Installe PyTorch CPU et les dépendances de `requirements.txt`.
4. Vérifie les imports, y compris OpenCV et Tkinter.

L'installation a besoin d'Internet et peut télécharger plusieurs centaines de Mo. Aucune carte NVIDIA n'est nécessaire pour cette configuration CPU. Le script ne demande aucun mot de passe GitHub et **n'installe pas Aidge**.

Ne jamais copier `.venv/`, `.venv-aidge/` ou `.venv-aidge-colibry/` depuis le Mac. Si un environnement incompatible est présent, le conserver sous un autre nom et créer un environnement local. Ne pas supprimer les données du projet.

Après l'installation, dans VS Code : `Ctrl+Maj+P` → **Python: Select Interpreter** → choisir `.venv\Scripts\python.exe`. Le réglage partagé vise le dossier `.venv`, sans imposer le chemin du Mac. Si un ancien interpréteur était déjà sélectionné, le changer manuellement.

Vérification manuelle :

```powershell
.\.venv\Scripts\python.exe scripts/check_setup.py
```

Cette commande ne crée pas de fenêtre et ne valide pas l'affichage graphique. Les lanceurs ne demandent pas d'activer PowerShell : on utilise directement l'exécutable du bon environnement.

## 3. Ouvrir l'interface de comptage YOLO

Double-cliquer sur **FlowSense_Windows.bat**. Puis : choisir une vidéo, tracer la zone de passage, indiquer le côté intérieur, enregistrer la porte et lancer l'analyse.

Cette interface utilise la démonstration **YOLO + suivi + comptage**, pas automatiquement notre nouveau modèle Aidge. Les modèles Aidge v1/v2 n'ont pas été branchés à cette interface.

Pour retrouver exactement les mêmes détections, récupérer le poids local `yolo26n.pt` dans le partage privé et le placer à la racine. Selon la version d'Ultralytics, un poids officiel peut aussi être téléchargé au premier lancement, mais les résultats ne sont reproductibles qu'avec les mêmes poids.

Les vidéos annotées sont produites localement dans `runs/count_video/`, avec un nom se terminant par `_count.mp4`. Utiliser le bouton **Ouvrir la vidéo résultat**. Les vidéos personnelles et les sorties ne sont pas envoyées sur GitHub.

## 4. Récupérer les fichiers privés nécessaires

Le ZIP GitHub ne suffit pas pour retrouver les images, modèles et rapports présents sur le Mac. Copier uniquement les dossiers utiles depuis le partage privé du groupe, en conservant leur organisation :

| Besoin | À récupérer séparément |
| --- | --- |
| Interface YOLO | `yolo26n.pt` et les vidéos souhaitées |
| Correction des profils | `dataset/profile_mix_v1/` ; aussi `dataset/profile_mix_human_v1/` pour reprendre les validations existantes |
| Entraînement de notre détecteur | `dataset/profile_mix_human_v1/`, ou le lot source `profile_mix_v1/` si la relecture n'est pas encore disponible |
| Attribution des photos COCO | `SOURCE.json` et `ATTRIBUTIONS.csv` de `dataset/coco_profile_fullbody_v1/`, à conserver avec les données partagées |
| Voir un bilan sans entraîner | Le `rapport.html`, `apercus/` et `videos_test_reservees/` du run souhaité, conservés ensemble |
| Reprendre l'entraînement v1 | `runs/person_profile_mix_qvga_v1/best_dense.npz` ; les rapports comparatifs nécessitent aussi les métadonnées de cet essai |
| Refaire les comparaisons vidéo | `videotest2/Echantillon/` et `runs/videotest2_diagnostic_v2/results.json` |

La copie de correction peut être reprise depuis un autre ordinateur : le programme contrôle les empreintes de la source avant d'autoriser un ancien chemin Mac à être remplacé par le dossier local. Il ne remplace pas silencieusement les annotations corrigées.

Ne pas publier les vidéos de camarades ni les images COCO sans vérifier les autorisations / licences. Les historiques de GitHub ne doivent pas servir de stockage de données personnelles.

Contrôle de présence du jeu source :

```powershell
.\.venv\Scripts\python.exe scripts/check_setup.py --data
```

## 5. Corriger les images sans taper une commande à chaque fois

Double-cliquer sur **Relecture_Windows.bat** et choisir :

1. La validation restante ;
2. Les 72 images d'entraînement local ;
3. Toutes les images, y compris les images déjà validées.

La fenêtre **FlowSense - annotation** apparaît hors du navigateur. Clic gauche glissé : ajouter un rectangle ; clic droit : supprimer ; **S** : sauver et continuer ; **P** : précédente ; **N** : passer sans sauver ; **Q** : quitter. Pour redessiner un cadre, supprimer l'ancien puis tracer le nouveau. Si l'image est vide, ne tracer aucun cadre et confirmer avec S.

Le programme crée ou réutilise `dataset/profile_mix_human_v1/`. Les originaux et les anciens bilans sont conservés. Les rectangles ici sont les annotations attendues, pas les prédictions du nouveau détecteur. Une annotation correcte ne doit pas être déplacée pour améliorer artificiellement un score.

Équivalent PowerShell, pour le train :

```powershell
.\.venv\Scripts\python.exe scripts/review_profile_dataset.py --split train
```

Pour la validation déjà vérifiée :

```powershell
.\.venv\Scripts\python.exe scripts/review_profile_dataset.py --include-reviewed
```

S ne réentraîne pas automatiquement le réseau. La validation reste hors apprentissage. Détails : [guide de correction](Corriger_les_images_profils.md).

## 6. Entraîner notre modèle Aidge : installation séparée

L'entraînement QVGA du projet utilise **Aidge CPU FP32**, avec loss et Adam NumPy. PyTorch définit le réseau et sert de référence de gradients. L'installation Windows de l'interface n'installe pas cette chaîne.

La [documentation officielle Aidge](https://eclipse.dev/aidge/source/GetStarted/install.html) distingue :

- Les paquets `pip` précompilés recommandés pour Linux ;
- La compilation depuis les sources sur Windows, avec `setup.ps1`, CMake et un compilateur C++ approprié.

Deux possibilités :

### A. Aidge déjà installé nativement sous Windows

Utiliser un environnement séparé `.venv-aidge-colibry/` installé sur **ce PC**. Les modules nécessaires à cet entraînement sont notamment `aidge_core`, `aidge_backend_cpu` et `aidge_onnx`. Leur installation et leurs dépendances natives doivent suivre la documentation officielle et être vérifiées avant apprentissage. Les correctifs locaux du Mac ne sont pas une procédure Windows certifiée.

```powershell
.\.venv-aidge-colibry\Scripts\python.exe scripts/check_setup.py --aidge-only
```

Puis préparer un essai neuf (le dossier de sortie doit être inexistant) :

```powershell
.\.venv\Scripts\python.exe scripts/prepare_profile_training.py --data-dir dataset/profile_mix_human_v1 --run-dir runs/profils_windows_v1
```

Contrôler d'abord les sorties / gradients :

```powershell
$env:OMP_NUM_THREADS="4"
$env:OPENBLAS_NUM_THREADS="4"
.\.venv-aidge-colibry\Scripts\python.exe scripts/train_person_qqvga_aidge.py --run-dir runs/profils_windows_v1 --check-only
```

Si le contrôle passe, lancer l'entraînement :

```powershell
.\.venv-aidge-colibry\Scripts\python.exe scripts/train_person_qqvga_aidge.py --run-dir runs/profils_windows_v1 --epochs 30 --batch-size 8 --learning-rate 0.001 --patience 8 --evaluate-every 2
```

Ces commandes ne démontrent pas que l'installation Aidge native fonctionne sur tous les PC. La première exécution sur le PC d'un camarade doit confirmer les imports puis le contrôle des gradients. Les versions locales Aidge du Mac et leurs correctifs ne sont pas transportés par Git.

### B. WSL2 / Ubuntu pour utiliser la procédure Linux

La [procédure Microsoft WSL](https://learn.microsoft.com/en-us/windows/wsl/install) demande, si WSL n'est pas déjà installé, un PowerShell administrateur avec `wsl --install`, puis un redémarrage. Cela modifie les composants Windows : à faire volontairement sur son PC, pas pour simplement ouvrir l'interface.

Une fois Ubuntu configuré, ouvrir le terminal Ubuntu et se placer à la racine du projet. Utiliser un Python Linux 3.11/3.12 et ses outils `venv` ; ne pas employer `.venv\Scripts\python.exe` dans les commandes Linux suivantes. Si Ubuntu ne fournit pas cette version, préparer la version de Python et le paquet `venv` correspondants avec l'aide de la documentation de la distribution.

Créer deux environnements **Linux** distincts, sans remplacer les environnements Windows :

```bash
python3 -m venv .venv-wsl
./.venv-wsl/bin/python -m pip install --upgrade pip
./.venv-wsl/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
./.venv-wsl/bin/python -m pip install -r requirements.txt
python3 -m venv .venv-aidge-wsl
./.venv-aidge-wsl/bin/python -m pip install aidge-core aidge-backend-cpu aidge-onnx numpy onnx
./.venv-aidge-wsl/bin/python scripts/check_setup.py --aidge-only
```

Préparer un **nouvel essai depuis Linux** pour que les chemins de ses manifests soient des chemins Linux, puis contrôler / entraîner :

```bash
./.venv-wsl/bin/python scripts/prepare_profile_training.py --data-dir dataset/profile_mix_human_v1 --run-dir runs/profils_wsl_v1
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 ./.venv-aidge-wsl/bin/python scripts/train_person_qqvga_aidge.py --run-dir runs/profils_wsl_v1 --check-only
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 ./.venv-aidge-wsl/bin/python scripts/train_person_qqvga_aidge.py --run-dir runs/profils_wsl_v1 --epochs 30 --batch-size 8 --learning-rate 0.001 --patience 8 --evaluate-every 2
```

Ne pas entraîner à partir de métadonnées préparées sur le Mac avec des chemins `/Users/...` non présents sur le PC. Si les versions Aidge téléchargées ont une API différente ou si le contrôle des gradients échoue, conserver le message et résoudre la compatibilité avant de continuer. Ce parcours WSL est documenté, mais n'a pas été exécuté sur le PC d'un camarade.

Pour les comparaisons avancées, [guide QVGA](Entrainement_COCO_profils_QVGA.md) et [bilan v2](Poursuite_profils_QVGA_v2.md). Les rapports ont besoin du modèle précédent et du diagnostic vidéo indiqués dans leurs options. Ils ne sont pas nécessaires pour simplement lancer l'entraînement.

## 7. Tests et limites de la compatibilité Windows

Tests locaux, sans données privées ni installation Aidge :

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Le workflow GitHub **Tests Windows et Linux** exécute la suite sur des machines Windows/Linux avec Python 3.12 et PyTorch CPU. Consulter son résultat dans l'onglet **Actions** : un workflow défini n'est pas une preuve de réussite tant qu'il n'est pas vert.

Ces tests couvrent les fonctions mathématiques, les transformations de rectangles, les chemins partagés et les copies de correction. Ils n'ouvrent pas de fenêtre graphique, ne nécessitent pas les images privées, n'entraînent pas le modèle complet et ne valident ni Aidge natif Windows, ni ColibryNPU, ni la consommation de 5 mW.

## 8. Résumé de l'état du modèle

La v1 demeure la référence recommandée pour la porte. La v2 réduit les fausses alertes mais manque davantage de personnes. Les 34 images locales de validation ont été marquées vérifiées ; la relecture humaine des 72 images train reste à faire. Le seuil 0,40 a amélioré le F1 local de la v1 sur la validation réutilisée, mais ce réglage n'a pas été déployé ni validé sur une nouvelle scène. Aucun taux de 90 % de passages correctement comptés n'est établi.
