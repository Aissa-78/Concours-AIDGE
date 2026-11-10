import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from evaluate_yolo_camera_reference import match_boxes, summarize, common_nms


class CameraReferenceTests(unittest.TestCase):
    def test_matching_is_one_to_one_and_iou_based(self):
        truth = [(.5, .5, .25, .5)]
        box = np.array([60., 30., 100., 90.])
        result = match_boxes([(.9, box), (.8, box)], truth)
        self.assertEqual(result, {"tp": 1, "fp": 1, "fn": 0})
        miss = match_boxes([(.8, np.array([0., 0., 10., 10.]))], truth)
        self.assertEqual(miss, {"tp": 0, "fp": 1, "fn": 1})
        self.assertEqual(summarize([result, miss]),
                         {"tp": 1, "fp": 2, "fn": 1, "precision": 1/3, "recall": .5, "f1": .4})

    def test_empty_images_and_predictions_are_valid(self):
        self.assertEqual(match_boxes([], []), {"tp": 0, "fp": 0, "fn": 0})
        self.assertEqual(summarize([])["f1"], 0.)

    def test_nms_is_explicit_and_keeps_best_overlapping_box(self):
        box = np.array([60., 30., 100., 90.])
        other = np.array([0., 0., 10., 10.])
        kept = common_nms([(.8, box), (.9, box), (.7, other)])
        self.assertEqual([score for score, _ in kept], [.9, .7])


if __name__ == "__main__":
    unittest.main()
