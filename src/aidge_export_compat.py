"""Compatibilité locale à un processus d'export, sans modifier l'installation.

Le contournement macOS existant ne parcourt que des méta-opérateurs connus,
pour ne pas appeler get_operator() sur certains opérateurs feuilles fragiles.
Les convolutions paddées doivent aussi être parcourues pour leur export NNPA.
"""

KNOWN_METAOPS = {"@NNPA", "Quantizer", "QFC", "QConv", "QConvDw", "QConv2D",
                 "QConvDepthWise2D", "PaddedConv1D", "PaddedConv2D",
                 "PaddedConvDepthWise1D", "PaddedConvDepthWise2D",
                 "PadConv", "PadConvAct", "PadConvDw", "PadConvDwAct",
                 "ConvDwAct", "ConvAct", "QAdd", "AddAct"}


def find_node_in_known_metaops(node, node_type):
    if node.type() == node_type:
        return [node]
    if node.type() not in KNOWN_METAOPS:
        return []
    result = []
    for child in node.get_operator().get_micro_graph().get_nodes():
        result.extend(find_node_in_known_metaops(child, node_type))
    return result


def enable_padded_meta_lookup():
    """Appliquer avant l'import de l'exporteur ; aucun fichier de bibliothèque modifié."""
    import importlib

    for name in ("aidge_core.export_utils", "aidge_core.export_utils.export_utils"):
        module = importlib.import_module(name)
        module.get_node_from_metaop = find_node_in_known_metaops
