# Base d'images FlowSense

## Sous-ensemble initial : Wake Vision

La première base locale est construite avec **Wake Vision**, un dataset public de classification binaire spécialement conçu pour la présence de personnes en TinyML.

- Source : [Harvard-Edge/Wake-Vision](https://huggingface.co/datasets/Harvard-Edge/Wake-Vision)
- Labels du projet : `0 = vide` et `1 = personne`
- Licence annoncée par le dataset : CC BY 4.0
- Téléchargement : `python scripts/download_wake_vision_subset.py`

Le téléchargeur récupère par défaut 100 images `vide` et 100 images `personne` pour l'entraînement, puis 25 images de chaque classe pour la validation. Elles restent exclusivement sur le poste local : les fichiers image sont ignorés par Git.

## Limite importante

Wake Vision fournit une bonne base pour apprendre le fonctionnement de l'IA, mais ses images ne ressemblent pas forcément à la future caméra de porte FlowSense. Quand la caméra sera disponible, il faudra compléter la base avec des images prises dans le scénario réel : porte ou couloir, différents éclairages, personnes proches ou éloignées, et scènes réellement vides.

Ne placez jamais une même image dans `train` et dans `validation`.
