import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from correct_confirmed_person_labels import EXPECTED, apply_confirmed_boxes


class ConfirmedLabelTests(unittest.TestCase):
    def test_duplicate_removal_preserves_two_other_boxes(self):
        stem = "testvideo_jour33_f00120"
        original = np.array(EXPECTED[stem])
        changed, _ = apply_confirmed_boxes(stem, original, 832, 464)
        np.testing.assert_array_equal(changed, original[:2])
        np.testing.assert_array_equal(original, EXPECTED[stem])

    def test_second_person_box_is_preserved(self):
        stem = "testvideo_jour33_f00210"
        original = np.array(EXPECTED[stem])
        changed, _ = apply_confirmed_boxes(stem, original, 832, 464)
        np.testing.assert_array_equal(changed[1], original[1])
        _, cx, cy, bw, bh = changed[0]
        np.testing.assert_allclose([(cx-bw/2)*832, (cy-bh/2)*464,
                                   (cx+bw/2)*832, (cy+bh/2)*464], [0, 176, 320, 464])
        self.assertEqual(len(changed), 2)
        np.testing.assert_array_equal(original, EXPECTED[stem])

    def test_unexpected_image_or_annotation_is_refused(self):
        stem = "testvideo_jour33_f00120"
        original = np.array(EXPECTED[stem])
        with self.assertRaises(ValueError):
            apply_confirmed_boxes(stem, original, 464, 832)
        original[0, 1] += .01
        with self.assertRaises(AssertionError):
            apply_confirmed_boxes(stem, original, 832, 464)


if __name__ == "__main__":
    unittest.main()
