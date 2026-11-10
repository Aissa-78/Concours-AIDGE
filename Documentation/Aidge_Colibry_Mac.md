# Aidge et Colibry sur le Mac du projet — état au 12 octobre 2026

La chaîne logicielle fonctionne **sur ce Mac** pour le petit modèle `TinyHeatmap`
à activation ReLU : PyTorch → ONNX → Aidge CPU → calibration INT8 → export
Colibry. Le bitstream est dans
`runs/colibry_relu/export_graph_int8_colibry/dnn/aidge_model_bitstream.nnpa`.
La carte, le firmware, la caméra et le comptage réel ne sont pas testés.

## Pour refaire l'essai

Ouvrir un terminal dans la racine de `FlowSense-AIDGE`, puis lancer les quatre
commandes de la section « Export logiciel Colibry » du `README.md`. Le premier
script exporte le checkpoint ReLU en ONNX ; le deuxième compare Aidge CPU à
PyTorch ; le troisième prépare 32 images réelles pour la calibration ; le
quatrième produit le bitstream.

`.venv-aidge-colibry/` contient l'installation locale (Python 3.12, Aidge
0.10.0.post3, `aidge_export_colibry` 0.1.0 et `colibry_npu` 0.1.2).
`.aidge-colibry-source/` contient le code de l'exporteur. Ces deux dossiers
sont volontairement ignorés par Git : **ne pas les copier sur Windows**. Le
code du projet et le patch de compatibilité, eux, peuvent être partagés.

## Pourquoi les correctifs étaient nécessaires

1. Sur macOS, Aidge Core construit depuis les sources manquait deux
   spécialisations de `PaddedConvDepthWise` utilisées par les bindings Python.
2. L'importeur ONNX construisait `Relu` avec `Node(ReLUOp())`. L'inférence CPU
   échouait ensuite avec `std::bad_cast`. La fabrique `aidge_core.ReLU()`
   produit le bon type de nœud ; la comparaison PyTorch/Aidge passe maintenant
   avec un écart maximal de `1,19 × 10⁻⁶` sur l'entrée de référence.
3. Pendant l'export, la recherche récursive d'opérations internes appelait
   `get_operator()` sur `Round`, ce qui interrompait Python. Pour ce graphe,
   la recherche n'entre désormais que dans les méta-opérations connues
   (`Quantizer`, `QConv`, `QFC`, `@NNPA`, etc.).

Ces trois changements sont conservés dans
`patches/aidge_0.10.0_colibry_macos.patch`, prévu pour le code source Aidge
version `v0.10.0`. Le patch a été vérifié avec `git apply --check` sur une
copie propre de cette version. Il s'agit d'un correctif local de compatibilité,
pas d'une modification officielle d'Aidge ; il faudra le réévaluer lors d'une
mise à jour d'Aidge.

## Ce que le résultat prouve, et ne prouve pas

L'exporteur a généré le bitstream sans erreur et estime 4 443 mots de données,
4 128 octets de configuration et 20,1 µJ par inférence. Cette dernière valeur
est une **estimation d'énergie**, pas une mesure de puissance moyenne. Elle ne
permet donc pas de conclure sur un objectif de 5 mW. La calibration utilise
des images réelles mais la variante ReLU reste moins bonne que le modèle
Hardtanh sur la porte B. La comparaison CPU après INT8 a depuis été réalisée
le 20 octobre : voir [le bilan](Bilan_INT8_porte_B.md). Le matériel reste à
tester. Le suivi des personnes et le franchissement de ligne sont
également des étapes distinctes de ce bitstream.
