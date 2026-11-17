# Première connexion Colibry sur Mac — 17 novembre 2026

## Résultat vérifié

Le Mac Apple Silicon communique directement avec la ColibryProbe et la
ColibryNPU. La lecture JTAG retourne `0x249511c3`, l'identifiant attendu par
le SDK ASYGN. Le canal UART de Colibry transmet `ping pong` à 115200 bauds.
Cela confirme la communication et l'exécution du programme déjà présent
sur la carte. Notre détecteur, la caméra et la consommation ne sont pas
encore testés sur le matériel. Aucun firmware n'a été chargé lors de cet essai.

## Fichiers installés

- `.local/colibry_sdk/` : SDK extrait de l'archive fournie par ASYGN.
- `.venv-colibry-hw/` : environnement Python 3.12 dédié aux échanges USB.
- `requirements-colibry-hardware.txt` : dépendances de cet environnement.
- `scripts/check_colibry_connection.py` : lecture de l'identifiant et des messages.

Le SDK et le venv sont ignorés par Git. Les environnements d'entraînement
existants restent séparés. L'archive du Bureau est conservée.

## Refaire le test

Brancher la sonde avec la petite carte, sélecteur sur **JTAG** et interrupteur
de la petite carte sur **ON**. Le manuel décrit aussi le cavalier VBAT :
il alimente la carte depuis la sonde et doit être retiré si une batterie
alimente déjà la carte. Ne pas modifier ces branchements sans vérifier le manuel.

Dans le terminal de VS Code, à la racine du projet :

```sh
./.venv-colibry-hw/bin/python scripts/check_colibry_connection.py --report runs/colibry_hardware/connection.json
```

Résultats attendus : identifiant `0x249511c3`, puis messages `ping pong`
si le programme d'usine est toujours installé. L'absence de ces messages
ne prouve pas une panne si un autre programme a remplacé celui d'usine.
Si plusieurs sondes sont branchées, utiliser `--device` avec l'adresse affichée
par PyFtdi, sans le dernier `/1`, ou garder une seule sonde branchée.

## Compilation et exécution vérifiées sur ce Mac

Le compilateur CORE-V Embecosm GCC 14.1.0 pour Apple Silicon (distribution
20240530) est installé dans `.local/corev/`. L'exemple ASYGN
`10_uart_printf_blink` est copié dans `.local/colibry_examples/uart_mac/`.
Le seul changement de son code est le message de démarrage
`FlowSense Mac SRAM OK`. Le SDK original reste inchangé.

Commande à lancer depuis la racine du projet :

```sh
./.venv-colibry-hw/bin/python scripts/run_colibry_uart_demo.py --load-sram
```

Sans `--load-sram`, la commande compile uniquement. Avec cette option,
elle charge le programme en SRAM et démarre le CPU. Aucun effacement
ni écriture de la flash : une coupure puis remise sous tension permet de
recharger le programme qui reste en flash.

Résultat vérifié le 17 novembre 2026 : compilation réussie, fichier de
23 592 octets, message `FlowSense Mac SRAM OK`, puis `ping pong` sur UART.
Le rapport est `runs/colibry_hardware/uart_sram_test.json` et le journal de
compilation `runs/colibry_hardware/build_uart.log`.

Cette validation concerne l'exemple UART en `rv32imc_zicsr`, avec ce
compilateur alternatif. Elle ne démontre pas la compatibilité de tous les
exemples, des extensions optimisées ni de l'application NNPA complète.
Le binaire OpenOCD livré avec le SDK est Linux x86-64 et n'est pas utilisé
pour ce test. L'acquisition caméra, notre modèle et la puissance restent
à vérifier.

Référence du compilateur :
https://embecosm.com/downloads/tool-chain-downloads/ (CORE-V, Apple Silicon,
30 mai 2024).

## Caméra : capture et affichage fonctionnels

Caméra PAG7920LT, 160 × 120, niveaux de gris. Trois captures successives
différentes ont été récupérées et vérifiées le 17 novembre. Le script lit les
pixels sans DMA et les reconstruit en little-endian. Le transfert DMA de
l'exemple n'est pas validé sur ce Mac. La méthode par défaut regroupe demande
et réponse successive, une commande par échange SPI, à 1 MHz. Elle sert au
diagnostic et à la collecte, sans être la cadence du futur détecteur embarqué.

Pour ouvrir la fenêtre si le firmware caméra est déjà en SRAM :

```sh
./.venv-colibry-hw/bin/python scripts/view_colibry_camera.py
```

Après une coupure d'alimentation, recharger le firmware caméra compilé :

```sh
./.venv-colibry-hw/bin/python scripts/view_colibry_camera.py --load-sram
```

Fermer toute autre fenêtre qui utilise la sonde avant de lancer cette commande.
Si le JTAG ne répond pas, couper puis remettre l'alimentation USB.

Dans la fenêtre : **S** enregistre l'image brute, **R** tourne seulement
l'aperçu, **Échap** ou la croix ferme. Les captures sont dans
`runs/colibry_hardware/captures/`. Leur ajout au dataset demande ensuite
une sélection, une annotation et une séparation train/validation.

La copie du firmware est dans `.local/colibry_examples/camera_mac/`.
Elle part de l'exemple `40_video_numbers_ble`, désactive les commandes BLE
et les modes de veille pour ce diagnostic et embarque en SRAM la configuration
NNPA du modèle de gestes ASYGN pour conserver son chemin d'acquisition.
Elle ne contient pas notre modèle de personnes et n'affiche pas de rectangles.
Le chargement JTAG utilise des blocs de 4 kB. La flash reste inchangée.
Les sources et binaires ASYGN restent privés dans `.local/`.

Les bandes issues des premiers essais restent dans les fichiers de diagnostic,
pas dans les images d'entraînement. La consommation et la cadence finale
restent à optimiser et à mesurer. Le statut est consigné dans
`runs/colibry_hardware/camera_status.json`.

## Cadence mesurée après optimisation USB

Mesure du 17 novembre : trois comparaisons sur la même image ont confirmé
l'égalité des 4 800 mots (19 200 pixels), avec contrôle de chaque en-tête de
réponse. Temps de transfert : environ 1,44 s en lecture simple, 0,65 s avec
les lectures successives en pipeline. Le contrôle de pixels ne repose pas
sur une simple comparaison visuelle. Les regroupements de plusieurs commandes
sous un même chip-select et le burst DMA ont échoué et ne sont pas activés.

Huit captures complètes successives, avec décodage et sauvegarde, donnent
**1,24 image/s** sur le Mac. Sans transfert d'images, 24 messages de fin de
traitement sur 6,06 s donnent **3,96 images/s** pour le programme caméra avec
le modèle de gestes ASYGN. Ce n'est ni la cadence maximale du capteur, ni
celle de notre futur détecteur de personnes, ni une mesure énergétique.
Le firmware de diagnostic reste actif sans optimisation de veille.

La fenêtre affiche sa cadence réelle. Pour retrouver la lecture de référence :

```sh
./.venv-colibry-hw/bin/python scripts/view_colibry_camera.py --transport single
```

Rapports locaux : `pipeline_verification.json`, `viewer_timing_pipeline.json`
et `camera_timing.json` dans `runs/colibry_hardware/`. Le script refuse une
réponse dont l'en-tête est incorrect au lieu d'afficher des pixels corrompus.
