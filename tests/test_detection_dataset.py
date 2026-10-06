"""Tests des formats et coordonnées des annotations de détection."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate_tiny_heatmap import find_peaks
from prepare_detection_dataset import intersection_over_union
from review_detection_dataset import read_boxes, write_boxes


class DetectionDatasetTests(unittest.TestCase):
    def test_overlap(self) -> None:
        self.assertAlmostEqual(
            intersection_over_union((0, 0, 10, 10), (5, 0, 15, 10)),
            1 / 3,
        )

    def test_label_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            label_path = Path(directory) / "frame.txt"
            write_boxes(label_path, [[10, 20, 30, 60]], width=100, height=100)
            boxes = read_boxes(label_path, width=100, height=100)
            self.assertEqual(len(boxes), 1)
            for actual, expected in zip(boxes[0], [10, 20, 30, 60]):
                self.assertAlmostEqual(actual, expected, places=3)

    def test_peak_detection(self) -> None:
        scores = np.zeros((20, 20), dtype=np.float32)
        scores[3, 5] = 0.8
        scores[15, 16] = 0.9
        self.assertEqual(set(find_peaks(scores, 0.5, separation=2)), {(5.0, 3.0), (16.0, 15.0)})


if __name__ == "__main__":
    unittest.main()
