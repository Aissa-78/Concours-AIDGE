import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from camera_preprocess import prepare_camera_frame


class CameraPreprocessTests(unittest.TestCase):
    def test_qqvga_is_one_channel_and_does_not_modify_the_source(self):
        frame = np.random.default_rng(7).integers(0, 256, (480, 640, 3), dtype=np.uint8)
        original = frame.copy()
        result, geometry = prepare_camera_frame(frame)
        self.assertEqual(result.shape, (120, 160))
        self.assertEqual(result.dtype, np.uint8)
        np.testing.assert_array_equal(frame, original)
        np.testing.assert_array_equal(result, cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 120), interpolation=cv2.INTER_AREA))
        np.testing.assert_allclose(geometry.transform_box((.5, .5, .2, .4)), (.5, .5, .2, .4))

    def test_portrait_is_not_stretched_or_cropped_and_boxes_follow(self):
        result, geometry = prepare_camera_frame(np.full((400, 200, 3), 200, dtype=np.uint8))
        self.assertEqual((geometry.resized_width, geometry.resized_height, geometry.left, geometry.top), (60, 120, 50, 0))
        self.assertTrue(np.all(result[:, :50] == 114))
        self.assertTrue(np.all(result[:, 50:110] == 200))
        np.testing.assert_allclose(geometry.transform_box((.25, .5, .2, .4)), (65/160, .5, 12/160, .4))

    def test_gray_input_is_accepted(self):
        result, _ = prepare_camera_frame(np.full((120, 160), 77, np.uint8))
        self.assertTrue(np.all(result == 77))

    def test_v2_80_pixel_preprocessing_remains_bit_identical(self):
        rng = np.random.default_rng(7)
        for height, width in ((200, 100), (100, 200), (83, 179), (32, 25)):
            frame = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
            factor = min(80 / width, 80 / height)
            rw, rh = max(1, round(width * factor)), max(1, round(height * factor))
            left, top = (80-rw)//2, (80-rh)//2
            expected = np.full((80, 80), 114, dtype=np.uint8)
            expected[top:top+rh, left:left+rw] = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (rw, rh), interpolation=cv2.INTER_AREA)
            result, _ = prepare_camera_frame(frame, 80, 80)
            np.testing.assert_array_equal(result, expected)

    def test_bad_resolution_input_or_box_is_rejected(self):
        frame = np.zeros((10, 10, 3), np.uint8)
        for width, height in ((321, 240), (320, 241), (0, 120), (160.5, 120)):
            with self.assertRaises(ValueError):
                prepare_camera_frame(frame, width, height)
        with self.assertRaises(ValueError):
            prepare_camera_frame(frame.astype(np.float32))
        _, geometry = prepare_camera_frame(frame)
        for box in ((.5, .5, 0, .4), (.1, .5, .8, .2), (np.nan, .5, .2, .2)):
            with self.assertRaises(ValueError):
                geometry.transform_box(box)


if __name__ == "__main__":
    unittest.main()
