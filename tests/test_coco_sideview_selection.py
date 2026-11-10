import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from coco_sideview_selection import inspect_image, inspect_person


def person():
    points = np.tile([250., 200., 2.], (17, 1))
    points[:5, 1] = [70, 75, 75, 80, 80]
    points[[5, 6]] = [[230, 130, 2], [270, 130, 2]]
    points[[11, 12]] = [[240, 250, 2], [260, 250, 2]]
    points[[13, 14], 1] = 350
    points[[15, 16], 1] = 430
    return {"id": 1, "iscrowd": 0, "bbox": [200, 50, 100, 400], "keypoints": points.ravel().tolist()}


class CocoSelectionTests(unittest.TestCase):
    def test_full_body_accepted(self):
        self.assertIsNotNone(inspect_person(person(), 640, 480))

    def test_cut_body_rejected(self):
        annotation = person()
        annotation["bbox"][0] = 0
        self.assertIsNone(inspect_person(annotation, 640, 480))

    def test_missing_ankles_rejected(self):
        annotation = person()
        annotation["keypoints"][15*3+2] = 0
        self.assertIsNone(inspect_person(annotation, 640, 480))

    def test_tiny_person_rejected(self):
        annotation = person()
        annotation["bbox"] = [200, 50, 10, 30]
        self.assertIsNone(inspect_person(annotation, 640, 480))

    def test_bad_second_person_rejects_whole_image(self):
        second = person()
        second["bbox"][0] = 0
        self.assertIsNone(inspect_image({"width": 640, "height": 480, "license": 4}, [person(), second], ["people walking"]))

    def test_orientation_is_only_a_hint(self):
        annotation = person()
        for index in (2, 4):
            annotation["keypoints"][index*3+2] = 0
        result = inspect_person(annotation, 640, 480)
        self.assertEqual(result["orientation"], "profil_probable")
        self.assertGreaterEqual(result["profile_proxy"], .6)

    def test_crowd_overlap_and_excluded_license(self):
        image = {"width": 640, "height": 480, "license": 4}
        self.assertIsNone(inspect_image(image, [person(), copy.deepcopy(person())], ["people walking"]))
        image["license"] = 1
        self.assertIsNone(inspect_image(image, [person()], ["person walking"]))

    def test_invalid_keypoints_and_dimensions(self):
        annotation = person()
        annotation["keypoints"][0] = float("nan")
        self.assertIsNone(inspect_person(annotation, 640, 480))
        self.assertIsNone(inspect_person(person(), 0, 480))


if __name__ == "__main__":
    unittest.main()
