# Réglages de porte

Un réglage de porte ne modifie pas l'IA. Il indique seulement où se trouve la
porte pour une caméra fixe et quel côté correspond à l'intérieur.

Pour créer un réglage, ouvrez une image ou une vidéo représentative :

```bash
python src/calibrate_gate.py \
  --video videos_test/ma_video.mp4 \
  --frame 50 \
  --output configs/porte_01.json
```

Dans la fenêtre, cliquez dans cet ordre :

1. début de la première limite de porte ;
2. fin de la première limite ;
3. début de la deuxième limite ;
4. fin de la deuxième limite ;
5. un point situé du côté intérieur.

Appuyez sur `S` pour enregistrer. `U` annule le dernier point, `R` recommence
et `Q` quitte.

Les coordonnées sont enregistrées entre 0 et 1, ce qui permet d'utiliser le
même réglage lorsque la caméra enregistre avec une autre résolution.

`porte_demo_test05.json` est un exemple calibré pour la vidéo de démonstration
du dépôt : ce n'est pas un réglage universel.
