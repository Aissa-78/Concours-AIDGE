# FlowSense : installation Windows et Linux

Préparé le 21 novembre 2026. Aucun abonnement GitHub ou Copilot nécessaire.

Version de référence du Mac : **Aidge 0.10.0, révision 3**, avec correctifs
locaux. Voir le [relevé des versions et les limites de reproduction](Versions_Aidge.md).
Le profil pip Linux n'impose pas cette révision ; ses contrôles doivent réussir
avant utilisation du réseau du projet.

## Quel parcours choisir ?

| Besoin | Windows | Linux Ubuntu 24.04, Python 3.12 |
|---|---|---|
| Interface YOLO, vidéos, annotation | Windows natif | Linux natif |
| Détecteur FlowSense ONNX sur une image | Windows natif avec OpenCV | Linux natif avec OpenCV |
| Entraînement Aidge CPU | Ubuntu 24.04 dans WSL2 | Linux natif |
| Caméra Colibry | Windows natif, pilote USB à configurer | Linux natif, permissions USB à configurer |

Les lanceurs et contrôles sont fournis. Les tests logiciels ne remplacent pas
un premier essai sur votre PC et votre carte. Le matériel a été vérifié sur
le Mac de l'équipe ; sa validation Windows/Linux reste à faire.
Pour la caméra, WSL nécessite une transmission USB supplémentaire : utiliser
Windows natif ou Linux natif dans un premier temps.

## 1. Récupérer le dossier

Télécharger le ZIP du dépôt puis **Extraire tout**, ou utiliser Git :

```text
git clone https://github.com/Aissa-78/Concours-AIDGE.git
```

Ouvrir dans VS Code le dossier contenant `README.md`, `src` et `scripts`.
Ne jamais copier un environnement `.venv` du Mac. Chaque ordinateur crée les
siens. Si ces nouveaux fichiers ne sont pas encore dans votre copie GitHub,
utiliser le ZIP de préparation fourni par l'équipe ou récupérer sa mise à jour.

## 2. Windows : interface et annotation

Installer Python **3.12 64 bits** depuis [python.org](https://www.python.org/downloads/windows/),
avec le lanceur Python et l'ajout au PATH lorsque proposés. Python 3.11 est
aussi accepté. Fermer et rouvrir VS Code après installation.

Dans le terminal PowerShell ouvert à la racine du projet :

```powershell
.\Installer_outils_Windows.bat app
.\FlowSense_Windows.bat
```

L'installateur télécharge PyTorch CPU et les bibliothèques du projet dans
`.venv`. Aucune carte NVIDIA nécessaire. Dans VS Code : **Ctrl+Maj+P →
Python: Select Interpreter → .venv\Scripts\python.exe**.

Contrôle et téléchargement COCO :

```powershell
.\.venv\Scripts\python.exe scripts/check_setup.py
.\.venv\Scripts\python.exe scripts/prepare_coco_2000.py
```

Ce téléchargement conserve les images et attributions dans
`dataset/coco_qqvga_2000_v1`. Le lot peut différer si les anciens jeux exclus
ne sont pas présents ; partager `plan.json` pour reproduire le même lot.
L'interface `FlowSense_Windows.bat` utilise **YOLO**, pas automatiquement notre
propre détecteur Aidge.

## 3. Linux : interface et annotation

Parcours prévu pour Ubuntu 24.04 avec Python 3.12, notamment dans WSL.
Dans un terminal Ubuntu :

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip python3-tk git libgl1 libglib2.0-0 libusb-1.0-0
cd ~/Concours-AIDGE
bash Installer_Linux.sh app
bash FlowSense_Linux.sh
```

Adapter `cd` à votre dossier réel. Sous Linux natif, une session graphique
est nécessaire pour les fenêtres. Sous WSL, vérifier que les applications
graphiques WSLg fonctionnent. Sans affichage, utiliser les scripts en terminal.

L'environnement est `.venv-linux`, séparé des environnements Mac/Windows.
Si votre distribution fournit Python 3.10 ou 3.13 par défaut, installer
séparément Python 3.11/3.12 ou utiliser Ubuntu 24.04. Puis, par exemple :

```bash
FLOWSENSE_PYTHON=python3.12 bash Installer_Linux.sh app
./.venv-linux/bin/python scripts/check_setup.py --no-gui
./.venv-linux/bin/python scripts/prepare_coco_2000.py
```

## Tester notre détecteur ONNX

Utilisez une photo dans le dossier du projet, par exemple `test.jpg`. Le
modèle `FlowSense_miroir_lumiere_v1.onnx` est livré dans `models/`.
Le script écrit une image PNG agrandie avec les rectangles dans
`runs/onnx/`. Les photos de votre caméra peuvent être utilisées directement.

Sur Windows, dans PowerShell à la racine du dépôt :

```powershell
.\Installer_outils_Windows.bat app
.\.venv\Scripts\python.exe scripts/predict_flowsense_onnx.py --image "test.jpg"
```

Sur Linux :

```bash
bash Installer_Linux.sh app
./.venv-linux/bin/python scripts/predict_flowsense_onnx.py --image "test.jpg"
```

Sur Mac, dans l’environnement local de l’interface :

```bash
./.venv/bin/python scripts/predict_flowsense_onnx.py --image "test.jpg"
```

Le seuil par défaut est `0.50`. Pour essayer `0.35`, ajouter
`--threshold 0.35`. L’option `--model "chemin/vers/modele.onnx"` permet de
tester un autre [modèle de l’équipe](https://github.com/Aissa-78/Concours-AIDGE/releases/tag/flowsense-modeles-2026-11-26). Le bouton de
comptage de l’interface vidéo utilise toujours YOLO.

## 4. Aidge sous Windows : Ubuntu/WSL

Cette étape concerne l'entraînement, pas l'ouverture de l'interface Windows.
Si WSL n'est pas déjà installé, ouvrir PowerShell **en administrateur** :

```powershell
wsl --install -d Ubuntu-24.04
```

Redémarrer si demandé, ouvrir Ubuntu et créer le compte Linux. Si WSL existe
déjà, vérifier les distributions avec `wsl -l -v` avant d'en ajouter une.
La procédure est décrite par [Microsoft](https://learn.microsoft.com/windows/wsl/install/).

Dans **Ubuntu**, créer une copie Linux du projet et installer les outils :

```bash
sudo apt update
sudo apt install python3-venv python3-pip python3-tk git libgl1 libglib2.0-0
git clone https://github.com/Aissa-78/Concours-AIDGE.git ~/Concours-AIDGE
cd ~/Concours-AIDGE
bash Installer_Linux.sh app
bash Installer_Linux.sh aidge
```

Si le dossier existe déjà, le conserver et y faire la mise à jour ; ne pas
le remplacer sans sauvegarder les données. Le dossier Linux et celui de
Windows sont distincts. Copier les données privées dans la copie Linux en
conservant leur arborescence. Depuis Windows, elle est accessible dans
l'Explorateur via `\\wsl.localhost\Ubuntu-24.04\home\` puis le compte Linux.

Dans VS Code, ouvrir cette copie avec l'extension Microsoft WSL et sélectionner
le Python Linux approprié. Ne pas lancer les commandes Ubuntu dans PowerShell.

## 5. Aidge sous Linux natif ou dans Ubuntu/WSL

À la racine du projet dans Ubuntu :

```bash
bash Installer_Linux.sh aidge
```

L'installation utilise `.venv-aidge-linux` pour `aidge-core`, le backend CPU
et l'import ONNX. PyTorch reste dans `.venv-linux`.
L'installateur vérifie les imports, puis les sorties, gradients et une mise
à jour d'une petite convolution dont la réponse est connue. Il s'arrête en
cas d'erreur. Les versions réellement installées sont enregistrées dans
`.venv-aidge-linux/installed_versions.txt`.

Les paquets peuvent évoluer : ce test simple ne suffit pas à certifier notre
réseau complet. Vérifier celui-ci avant tout apprentissage long :

```bash
./.venv-aidge-linux/bin/python scripts/train_person_qqvga_aidge.py --run-dir dataset/detection_flowsense_equipe_2026-11-26/aidge --check-only
```

Pour utiliser les fichiers du **dernier entraînement**, récupérez le
[ZIP privé de l’équipe](../dataset/DATASET.md) et extrayez-le à la racine
de cette copie Linux du dépôt. Lancez le `--check-only` indiqué dans ce
guide avec `./.venv-aidge-linux/bin/python`. Ce contrôle doit réussir
avant l’apprentissage. Les modules pip Linux peuvent différer de la
version du Mac. Si vous voyez une erreur `Pad` ou `std::bad_cast`,
gardez le message et le fichier `installed_versions.txt`.

Pour Aidge natif Windows ou si les wheels Linux manquent, voir la
[procédure officielle Aidge](https://eclipse.dev/aidge/source/GetStarted/install.html).

## 6. Caméra : fichiers privés à récupérer avant lancement

Le paquet privé caméra de l'équipe doit fournir cette arborescence :

```text
.local/
├── colibry_sdk/
│   └── riscv_runtime/tools/colibry_interfaces/
│       ├── colibry_jtag_api.py
│       └── colibry_spis_api.py
└── colibry_camera/
    ├── demo.bin
    └── demo/demo
```

Le SDK ASYGN complet peut aussi être placé dans `colibry_sdk`.
`demo/demo` est le fichier ELF sans extension ; ce n'est pas un dossier vide.
Ces deux binaires sont le firmware de capture déjà compilé pour **la carte
Colibry de l'équipe**, pas pour le processeur du PC : ils peuvent donc être
transférés du Mac. Ils ne contiennent pas notre détecteur de personnes final.

Les fichiers ASYGN, leur licence et les binaires restent dans le partage privé
autorisé du groupe, jamais dans le dépôt public. Ne pas utiliser d'OpenOCD
Linux sur Windows : ce lanceur passe directement par le SDK Python.

## 7. Caméra sous Windows natif

PowerShell, dans le dossier Windows du projet :

```powershell
.\Installer_outils_Windows.bat hardware
.\.venv-colibry-hw\Scripts\python.exe scripts/check_colibry_connection.py --seconds 0
.\Ouvrir_camera_Windows.bat
```

La première commande installe les bibliothèques, pas les pilotes USB. Si la
sonde reste inaccessible, consulter les [instructions Windows PyFtdi](https://eblot.github.io/pyftdi/installation.html#windows).
Elles indiquent Zadig, le **parent composite** FTDI multicanal, et le pilote
**libusb-win32**. Ne pas sélectionner une interface isolée ni un autre
périphérique. Vérifier que le VID/PID de la sonde correspond à **0403:6043**
avant de remplacer son pilote, et noter l'ancien pilote pour pouvoir revenir
en arrière. Cette configuration peut affecter d'autres outils utilisant la
même sonde. Elle n'est jamais faite automatiquement par nos scripts.

PyFtdi décrit Windows comme non officiellement testé : si cela échoue,
conserver l'erreur et tester le matériel sous Linux natif. Le fait d'avoir
un port COM n'est pas suffisant : les canaux JTAG et SPI doivent être accessibles.

## 8. Caméra sous Linux natif

```bash
sudo apt install python3-venv python3-tk libusb-1.0-0
bash Installer_Linux.sh hardware
```

Pour une sonde d'identifiant **0403:6043**, créer une permission USB ciblée :

```bash
sudo groupadd -f plugdev
sudo usermod -aG plugdev "$USER"
sudo tee /etc/udev/rules.d/99-flowsense-colibry.rules <<'EOF'
SUBSYSTEM=="usb", ATTR{idVendor}=="0403", ATTR{idProduct}=="6043", GROUP="plugdev", MODE="0660"
EOF
sudo udevadm control --reload-rules
```

Fermer puis rouvrir la session Linux et rebrancher la sonde. Ne pas exécuter
toute l'application avec `sudo` pour contourner les permissions.

```bash
./.venv-colibry-linux/bin/python scripts/check_colibry_connection.py --seconds 0
bash Ouvrir_camera_Linux.sh
```

Sans affichage, on peut tester trois captures :

```bash
bash Ouvrir_camera_Linux.sh --headless --frames 3
```

## 9. Utilisation de la caméra et dépannage

Brancher la sonde, mettre la carte sur **ON**, sélecteur **JTAG**. Respecter
le manuel pour l'alimentation et le cavalier VBAT ; ne pas ajouter une batterie
si la sonde alimente déjà la carte. Fermer les autres applications utilisant
la même sonde avant le test.

Le contrôle JTAG attend `0x249511c3`. Le lanceur caméra charge le firmware
de diagnostic en **SRAM**, sans effacer ni programmer la flash.
La validation de l'ELF refuse les segments hors de cette SRAM.

- **S** : sauvegarder une photo brute dans `runs/colibry_hardware/captures/`.
- **R** : tourner l'aperçu uniquement, pas les pixels sauvegardés.
- **Échap** : fermer.

| Problème | Action |
|---|---|
| Python introuvable | Installer Python 3.11/3.12 puis rouvrir le terminal. |
| `venv` absent sous Ubuntu | Installer le paquet `python3-venv` correspondant au Python utilisé. |
| `tkinter` absent | Installer `python3-tk` sous Ubuntu ; vérifier Tcl/Tk dans Python Windows. |
| SDK ou firmware absent | Récupérer le paquet privé et vérifier l'arborescence de la section 6. |
| Accès USB refusé | Windows : pilote ciblé ; Linux : groupe/udev et nouvelle session. |
| Sonde occupée | Fermer les autres outils utilisant la carte. |
| JTAG invalide après un blocage | Débrancher/rebrancher l'USB, vérifier ON/JTAG puis réessayer. |
| Le test Aidge échoue | Garder le traceback et `installed_versions.txt` ; ne pas lancer l'apprentissage long. |

## 10. Ce qui reste à valider sur chaque PC

Les contrôles automatiques vérifient le code commun et les plans d'installation.
Le test numérique Aidge a été exécuté sur le Mac avec l'environnement existant.
Une installation propre Windows/Linux, le premier affichage caméra, la cadence
USB et les résultats du vrai réseau restent à confirmer sur les machines cibles.
Les lanceurs ne démontrent ni le déploiement du détecteur, ni les 5 mW.

Sources : [Aidge](https://eclipse.dev/aidge/source/GetStarted/install.html),
[PyFtdi](https://eblot.github.io/pyftdi/installation.html),
[Microsoft WSL](https://learn.microsoft.com/windows/wsl/install/).
