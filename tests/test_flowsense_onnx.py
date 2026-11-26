"""Vérifie le modèle publié sur les systèmes testés, avec une image synthétique."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from predict_flowsense_onnx import detect


class FlowSenseOnnxTest(unittest.TestCase):
    def test_decodes_coordinates_on_the_eight_pixel_grid(self):
        output = np.zeros((1, 5, 15, 20), dtype=np.float32)
        output[:, 4] = -50
        output[0, :, 5, 12] = (1, 1, 1, 1, 10)
        net = Mock()
        net.forward.return_value = output
        _, boxes = detect(net, np.zeros((120, 160), dtype=np.uint8))
        self.assertEqual(len(boxes), 1)
        np.testing.assert_array_equal(boxes[0][1], [92, 36, 108, 52])

    def test_creates_preview_with_the_published_model(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "image élève.png"
            output = Path(temp) / "résultat.png"
            Image.new("L", (160, 120), 96).save(source)
            result = subprocess.run(
                [sys.executable, "-X", "utf8", str(ROOT / "scripts/predict_flowsense_onnx.py"),
                 "--image", str(source), "--output", str(output)],
                capture_output=True, text=True, encoding="utf-8", timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("rectangle(s)", result.stdout)
            with Image.open(output) as preview:
                self.assertEqual(preview.size, (640, 480))
            with Image.open(source) as original:
                self.assertEqual(original.size, (160, 120))


if __name__ == "__main__":
    unittest.main()
