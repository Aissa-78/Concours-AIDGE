"""Vérifie que le sens des clics ne change pas le comptage."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from count_video import Gate


class GateTests(unittest.TestCase):
    def test_second_line_can_be_drawn_backwards(self) -> None:
        config = {
            "gate": {
                "line_1": [[0.2, 0.4], [0.8, 0.4]],
                "line_2": [[0.8, 0.6], [0.2, 0.6]],
            },
            "inside_reference": [0.5, 0.8],
        }
        gate = Gate(config, 100, 100)

        self.assertAlmostEqual(gate.length, 60)
        self.assertEqual(gate.zone((50, 80)), "INTERIEUR")
        self.assertEqual(gate.zone((50, 20)), "EXTERIEUR")


if __name__ == "__main__":
    unittest.main()
