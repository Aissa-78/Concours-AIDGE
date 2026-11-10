import sys
import unittest
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from benchmark_person_camera import prepare_variant, threshold_diagnostics, relocate_path
from evaluate_yolo_camera_reference import match_boxes


class CameraBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((64, 100, 3), np.uint8)
        self.frame[:, :, 0] = 40
        self.frame[10:50, 20:60, 1] = 220
        self.boxes = [(.4, 30/64, .4, 40/64)]

    def test_originals_remain_unchanged_and_grayscale_channels_match(self):
        before = self.frame.copy()
        color, boxes = prepare_variant(self.frame, self.boxes, None, False)
        np.testing.assert_array_equal(color, self.frame)
        self.assertEqual(boxes, self.boxes)
        gray, _ = prepare_variant(self.frame, self.boxes, None, True)
        for channel in range(3):
            np.testing.assert_array_equal(gray[:, :, channel], cv2.cvtColor(before, cv2.COLOR_BGR2GRAY))
        np.testing.assert_array_equal(self.frame, before)

    def test_resized_color_and_gray_have_identical_box_geometry(self):
        for size in ((320, 240), (160, 120)):
            color, first = prepare_variant(self.frame, self.boxes, size, False)
            gray, second = prepare_variant(self.frame, self.boxes, size, True)
            self.assertEqual(color.shape, (size[1], size[0], 3))
            self.assertEqual(gray.shape, color.shape)
            self.assertEqual(first, second)
            np.testing.assert_array_equal(gray[:, :, 0], gray[:, :, 2])

    def test_native_camera_adds_borders_without_resampling(self):
        frame, boxes = prepare_variant(self.frame, self.boxes, "native", False)
        self.assertEqual(frame.shape, (75, 100, 3))
        np.testing.assert_array_equal(frame[5:69], self.frame)
        np.testing.assert_allclose(boxes, [[.4, 35/75, .4, 40/75]])

    def test_matching_is_invariant_to_uniform_scaling(self):
        first = match_boxes([(.9, np.array([60., 30., 100., 90.]))], [(.5, .5, .25, .5)])
        second = match_boxes([(.9, np.array([120., 60., 200., 180.]))], [(.5, .5, .25, .5)], width=320, height=240)
        self.assertEqual(first, second)
        details = match_boxes([], self.boxes, width=100, height=64, return_details=True)
        self.assertEqual(details["missed"], [0])

    def test_threshold_diagnostics_do_not_change_ground_truth_or_predictions(self):
        predictions = [{"width": 160, "height": 120, "truth": [(.5, .5, .25, .5)],
                        "detections_above_005": [{"score": .6, "box": [60., 30., 100., 90.]},
                                                 {"score": .12, "box": [0., 0., 10., 10.]}]}]
        result = threshold_diagnostics(predictions, [{"source": "portes"}])
        self.assertEqual(result["0.25"]["overall"]["f1"], 1.)
        self.assertEqual(result["0.1"]["overall"]["fp"], 1)
        self.assertEqual(len(predictions[0]["detections_above_005"]), 2)
        with self.assertRaises(ValueError):
            threshold_diagnostics(predictions, [])

    def test_shared_paths_can_be_relocated_from_mac_or_windows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root/"dataset"/"example"
            dataset.mkdir(parents=True)
            for old in ("/nonexistent_computer/FlowSense/dataset/example", "Z:\\nonexistent_computer\\FlowSense\\dataset\\example"):
                self.assertEqual(relocate_path(old, root), dataset.resolve())
            with self.assertRaises(FileNotFoundError):
                relocate_path("/nonexistent_computer/dataset/../outside", root)


if __name__ == "__main__":
    unittest.main()
