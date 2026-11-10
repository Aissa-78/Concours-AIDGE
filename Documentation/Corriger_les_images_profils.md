# Ouvrir la correction des images de profil

Dans le terminal de VS Code, à la racine de FlowSense-AIDGE :

macOS :

```sh
./.venv/bin/python scripts/review_profile_dataset.py
```

Windows, avec l'environnement Python installé sur ce PC :

```powershell
.\.venv\Scripts\python.exe scripts/review_profile_dataset.py
```

La fenêtre **FlowSense - annotation** affiche les images, pas une page web. La première peut être vide : sans personne visible, aucun rectangle n'est nécessaire ; S confirme cette annotation vide.

- Glisser le clic gauche : ajouter un rectangle autour d'une personne visible.
- Clic droit dans un rectangle : le supprimer. Sur Mac, clic avec deux doigts pour le clic droit.
- Pour refaire un rectangle : supprimer l'ancien puis redessiner.
- S : enregistrer et passer à la suivante. Si tout est juste, appuyer sur S sans modifier.
- P : revenir à la précédente.
- N : passer sans enregistrer / valider.
- Q ou croix de fermeture : quitter ; les changements de l'image courante non sauvés avec S sont perdus.

Les rectangles verts ici sont **les annotations à vérifier**, pas les prédictions de notre détecteur. Il faut un rectangle par personne visible, pas autour d'une chaise ou du décor. Deux personnes visibles nécessitent deux rectangles. Un rectangle correct doit rester correct même si notre modèle ne le retrouve pas ; ne pas le déplacer pour faire monter le score.

Le premier lancement crée `dataset/profile_mix_human_v1/`, une copie réelle de `dataset/profile_mix_v1/`. Seule la copie est modifiée : l'ancien entraînement et son bilan restent reproductibles. Le fichier `REVIEW.json` conserve les empreintes originales. Les deux sources COCO et portes restent séparées, comme train et validation.

Par défaut, on corrige les **34 images de validation locale**. La même commande reprend les images pas encore validées. Pour revoir aussi les images déjà sauvées :

```sh
./.venv/bin/python scripts/review_profile_dataset.py --include-reviewed
```

Pour corriger ensuite les **72 images d'entraînement local** :

```sh
./.venv/bin/python scripts/review_profile_dataset.py --split train
```

Sous Windows, remplacer `./.venv/bin/python` par `.\.venv\Scripts\python.exe`.

S ne réentraîne pas le modèle et ne recalcule pas le rapport automatiquement. Après la relecture, préparer une nouvelle évaluation / un nouvel essai à partir de `dataset/profile_mix_human_v1/`, sans intégrer la validation ou les vidéos de test à l'apprentissage. Le statut `reviewed=oui` dans `portes/manifest.csv` indique les images réellement validées par l'utilisateur ; créer la copie ne valide rien à lui seul.
