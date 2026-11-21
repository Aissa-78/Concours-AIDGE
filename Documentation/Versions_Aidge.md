# Versions Aidge de référence

Relevé sur le Mac de l'équipe le 21 novembre 2026, dans
`.venv-aidge-colibry`, à partir des métadonnées des paquets installés.

| Module | Version déclarée |
|---|---|
| aidge_core | 0.10.0.post3 |
| aidge_backend_cpu | 0.10.0.post3 |
| aidge_learning | 0.10.0.post3 |
| aidge_quantization | 0.10.0.post3 |
| aidge_onnx | 0.10.0-3 |
| aidge_export_cpp | 0.10.0-3 |
| aidge_export_colibry | 0.1.0 |

Il s'agit de la famille **0.10.0, révision 3**, compilée depuis les sources
avec des correctifs locaux. L'export Colibry est installé en mode éditable.
Le [correctif Mac conservé dans le dépôt](../patches/aidge_0.10.0_colibry_macos.patch)
ne doit pas être appliqué automatiquement sur Windows/Linux.

## Installateur Linux / Ubuntu WSL

Le profil `aidge` du nouvel installateur installe les modules CPU et ONNX
disponibles via pip, sans imposer les versions locales du Mac. **Ce n'est
donc pas un verrou de reproduction de cet environnement.** Les wheel Linux
et la disponibilité exacte de la révision locale ne sont pas certifiés ici.
Ne pas remplacer les paquets au hasard pour masquer un échec de vérification.

L'installation doit réussir le test numérique `check_aidge_training.py`,
puis le contrôle `--check-only` sur le vrai réseau préparé. Les versions
installées sont sauvegardées dans `.venv-aidge-linux/installed_versions.txt`.
En cas d'échec, conserver ce fichier et le traceback pour identifier une
différence de version, d'ABI ou d'API avant l'apprentissage.

Ces contrôles ne valident pas l'export INT8 Colibry. L'exporteur et les outils
ASYGN correspondants nécessitent leur propre installation et les ressources
privées autorisées.

Le [guide Windows/Linux](Windows_Linux_complet.md) distingue l'interface,
l'entraînement et le matériel.
