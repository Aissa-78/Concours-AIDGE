from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from test_video_detectors import panel, sampled_frames


class VideoDiagnosticTests(unittest.TestCase):
    def test_panel_preserves_original_pixels(self):
        frame = np.full((120, 160), 100, np.uint8)
        before = frame.copy()
        view = panel(frame, [{"score": .8, "box": [20, 10, 100, 110]}], "Modele", "Gris")
        self.assertEqual(view.shape, (300, 400, 3))
        np.testing.assert_array_equal(before, frame)

    def test_panel_supports_original_color_and_no_detection(self):
        frame = np.zeros((464, 832, 3), np.uint8)
        self.assertEqual(panel(frame, [], "YOLO", "Original").shape, (300, 400, 3))

    def test_sampled_frames_keep_original_frame_numbers(self):
        class Capture:
            def __init__(self):
                self.index = 0
                self.released = False
            def isOpened(self):
                return True
            def read(self):
                value = self.index
                self.index += 1
                return (True, np.full((2, 3, 3), value, np.uint8)) if value < 13 else (False, None)
            def release(self):
                self.released = True
        capture = Capture()
        with patch("cv2.VideoCapture", return_value=capture):
            frames = list(sampled_frames(Path("example.mp4"), 6))
        self.assertEqual([index for index, _ in frames], [0, 6, 12])
        self.assertTrue(capture.released)


if __name__ == "__main__":
    unittest.main()
