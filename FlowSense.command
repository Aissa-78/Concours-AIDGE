#!/bin/zsh

cd "${0:A:h}" || exit 1

if [[ ! -x ./.venv/bin/python ]]; then
    print "L'environnement Python .venv est introuvable."
    print "Ouvre d'abord le projet dans VS Code et installe les dépendances."
    read -k 1 "?Appuie sur une touche pour fermer..."
    exit 1
fi

./.venv/bin/python src/app.py
if [[ $? -ne 0 ]]; then
    print "FlowSense s'est arrêté avec une erreur."
    read -k 1 "?Appuie sur une touche pour fermer..."
fi
