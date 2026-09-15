# Concours AIDGE 2026

## FlowSense

FlowSense est notre projet pour le Challenge AIDGE 2026. Nous développons un capteur vidéo embarqué capable de détecter et compter les passages de personnes à une porte ou dans un couloir, avec une consommation cible inférieure à 5 mW sur ColibryNPU.

Le traitement doit rester local. Le système ne transmet ni ne stocke les images ; il ne produit que des données agrégées telles que le nombre d'entrées, de sorties et l'occupation estimée.

## Documents du dépôt

- Vidéo explicative du projet : https://youtu.be/_fUpL8oQr6g
- [Fiche projet FlowSense](Fiche_projet_FlowSense.docx)
- [Plaquette du Challenge AIDGE](Plaquette_ChallengeAidge.pdf)
- [Règlement du Challenge AIDGE](Reglement-ChallengeAidge.pdf)

## État actuel

**Phase : cadrage et préparation**

L'équipe est à sa première séance de travail. Le matériel n'a pas encore été reçu : carte Colibry EvalboardTiny, caméra embarquée, documentation technique détaillée et outil de mesure énergétique.

Nous préparons donc le projet avant le déploiement matériel : environnement Aidge, données, modèle de référence, métriques et démonstrateur.

## Cas d'usage choisi

Une caméra fixe est installée à une porte ou dans un couloir. Le système :

1. détecte les personnes dans une zone définie ;
2. suit leur déplacement sur quelques images ;
3. détecte le franchissement d'une ligne virtuelle ;
4. compte les entrées et les sorties ;
5. estime l'occupation d'un espace ;
6. conserve le traitement vidéo local.

## Architecture envisagée

```text
Caméra embarquée
        ↓
Réduction de la résolution de l'image
        ↓
Détection de mouvement légère
        ↓
Petit réseau neuronal quantifié INT8 sur ColibryNPU
        ↓
Détection ou estimation du nombre de personnes
        ↓
Suivi court et ligne virtuelle sur CPU
        ↓
Entrées, sorties et occupation estimée
```

## Décisions déjà prises

- Le nom de travail du projet est **FlowSense**.
- Le scénario principal est le comptage de passages à une porte.
- Le traitement doit être effectué localement, sans cloud.
- Le modèle doit être compact et quantifié en INT8.
- Aidge servira à importer le modèle ONNX, analyser son graphe, l'optimiser, le quantifier et l'exporter.
- Nous commencerons par une version simple et fonctionnelle avant les optimisations avancées.

## Plan de progression

| Version | Objectif |
|---|---|
| MVP | Détecter la présence d'une personne dans une zone définie. |
| Version 2 | Distinguer le sens de passage grâce à une ligne virtuelle. |
| Version 3 | Compter plusieurs passages, limiter les doubles comptes et mesurer la consommation réelle. |

## Mesures de réussite

| Aspect | Mesures prévues |
|---|---|
| Détection | Précision, rappel, F1-score et faux positifs |
| Comptage | Erreur absolue, doubles comptes et erreurs entrée/sortie |
| Temps réel | Latence par inférence et images par seconde |
| Mémoire | Taille du modèle et mémoire maximale utilisée |
| Énergie | Énergie par inférence et puissance moyenne |
| Confidentialité | Aucune image transmise ou stockée |

## Prochaines actions

- [ ] Tester l'environnement et les premiers tutoriels Aidge.
- [ ] Choisir une baseline légère de détection de personnes.
- [ ] Préparer un dataset et les scripts d'évaluation.
- [ ] Répartir les rôles dans l'équipe.
- [ ] Préparer les questions pour les organisateurs.
- [ ] Déployer et mesurer le modèle après réception du matériel.

## Points à confirmer avec les organisateurs

- Les 5 mW incluent-ils la caméra, l'acquisition et le mode veille ?
- Le seuil concerne-t-il une puissance moyenne, un pic ou une énergie par inférence ?
- Quel scénario de comptage et quelle fréquence d'images seront évalués ?
- Quels opérateurs sont accélérés par ColibryNPU ?
- Existe-t-il un exporteur Aidge vers ColibryNPU ?
- Quel outil permettra de mesurer la consommation ?


