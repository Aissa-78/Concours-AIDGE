import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from aidge_export_compat import find_node_in_known_metaops


class FakeNode:
    def __init__(self, kind, children=()):
        self.kind, self.children = kind, children

    def type(self):
        return self.kind

    def get_operator(self):
        if self.kind == "Round":
            raise RuntimeError("Ne jamais introspecter cette feuille.")
        return SimpleNamespace(get_micro_graph=lambda: SimpleNamespace(get_nodes=lambda: self.children))


class ExportCompatTests(unittest.TestCase):
    def test_residual_add_is_found_inside_qadd(self):
        addition = FakeNode("Add")
        graph = FakeNode("QAdd", [addition, FakeNode("Round")])
        self.assertEqual(find_node_in_known_metaops(graph, "Add"), [addition])

    def test_padded_convolutions_are_found_recursively(self):
        convolution = FakeNode("ConvDepthWise2D")
        graph = FakeNode("@NNPA", [FakeNode("PaddedConvDepthWise2D", [FakeNode("Pad"), convolution])])
        self.assertEqual(find_node_in_known_metaops(graph, "ConvDepthWise2D"), [convolution])

    def test_round_leaf_is_never_inspected(self):
        self.assertEqual(find_node_in_known_metaops(FakeNode("Round"), "Conv2D"), [])


if __name__ == "__main__":
    unittest.main()
