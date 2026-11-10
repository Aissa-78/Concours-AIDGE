import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from person_training_data import assert_disjoint, balanced_order, augment_sample, selection_key
from person_detection_math import evaluate_boxes
from person_detection_math import targets_from_boxes


class PersonTrainingDataTests(unittest.TestCase):
    def test_no_augmentation_preserves_boundary_targets_exactly(self):
        image = np.full((1, 120, 160), .5, np.float32)
        boxes = [(.474999999, .3, .15, .2)]
        actual, targets, _ = augment_sample(image, boxes, np.random.default_rng(7), geometric=False, photometric=False)
        np.testing.assert_array_equal(actual, image)
        for first, second in zip(targets, targets_from_boxes(boxes)):
            np.testing.assert_array_equal(first, second)
    def test_every_image_is_used_and_sources_are_balanced(self):
        sources = np.array(["portes"]*179+["wepdtof"]*400)
        order = balanced_order(sources, np.random.default_rng(7))
        self.assertEqual(len(order), 800)
        self.assertEqual(set(order.tolist()), set(range(579)))
        for offset in range(0, len(order), 8):
            self.assertEqual(sum(sources[order[offset:offset+8]] == "portes"), 4)

    def test_mirrors_preserve_pixels_and_transform_targets(self):
        image = np.zeros((1, 120, 160), np.float32)
        image[:, 25:55, 30:70] = 1
        boxes = [(.3125, 1/3, .25, .25)]
        for seed in range(8):
            actual, (heat, distances, mask), transformed = augment_sample(image, boxes, np.random.default_rng(seed), photometric=False)
            self.assertEqual(actual.shape, image.shape)
            ys, xs = np.where(actual[0] == 1)
            cx, cy, bw, bh = transformed[0]
            np.testing.assert_allclose([(xs.min()+xs.max()+1)/320, (ys.min()+ys.max()+1)/240], [cx, cy], atol=1e-6)
            output = np.zeros((1, 5, 30, 40), np.float32)
            output[0, :4] = distances
            output[0, 4] = np.where(mask[0], 20, -20)
            metrics, _ = evaluate_boxes(output, [transformed])
            self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (1, 0, 0))
        np.testing.assert_array_equal(image[:, 25:55, 30:70], 1)

    def test_photometric_augmentation_is_finite_and_bounded(self):
        image = np.full((1, 120, 160), .5, np.float32)
        for seed in range(20):
            actual, _, boxes = augment_sample(image, [], np.random.default_rng(seed))
            self.assertEqual(actual.dtype, np.float32)
            self.assertTrue(actual.flags.c_contiguous)
            self.assertTrue(np.isfinite(actual).all())
            self.assertGreaterEqual(actual.min(), 0)
            self.assertLessEqual(actual.max(), 1)
            self.assertEqual(len(boxes), 0)

    def test_train_validation_leakage_is_refused(self):
        records = [{"source": "portes", "image_sha256": "a", "row": {"split": "train", "video": "a.mp4"}},
                   {"source": "portes", "image_sha256": "b", "row": {"split": "validation", "video": "b.mp4"}}]
        assert_disjoint(records)
        records[1]["image_sha256"] = "a"
        with self.assertRaises(ValueError):
            assert_disjoint(records)
        records[1]["image_sha256"] = "b"
        records[1]["row"]["video"] = "a.mp4"
        with self.assertRaises(ValueError):
            assert_disjoint(records)

    def test_selection_cannot_hide_the_weaker_source(self):
        first = {"portes": {"f1": .9}, "wepdtof": {"f1": .1}}
        second = {"portes": {"f1": .4}, "wepdtof": {"f1": .4}}
        self.assertGreater(selection_key(second, 2), selection_key(first, 1))


if __name__ == "__main__":
    unittest.main()
