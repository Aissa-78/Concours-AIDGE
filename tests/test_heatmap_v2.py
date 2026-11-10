"""Contrôles des coordonnées et du gradient pour l'entraînement V2."""

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from heatmap_v2 import HeatmapV2
from heatmap_v2_data import augment, read_sample, target_heatmap
from heatmap_metrics import weighted_bce, find_centers, evaluate_logits


class HeatmapV2Tests(unittest.TestCase):
    def test_metrics_reject_unpaired_images_and_annotations(self):
        with self.assertRaises(ValueError):
            evaluate_logits(np.zeros((2, 1, 20, 20), np.float32), [[]])

    def test_separation_threshold_has_no_rounding_duplicate(self):
        scores = np.zeros((20, 20), dtype=np.float32)
        scores[8, 3], scores[8, 5] = 0.9, 0.8
        self.assertEqual(len(find_centers(scores)), 1)

    def test_preprocessing_keeps_bbox_center(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cv2.imwrite(str(root / "image.png"), np.full((100, 200, 3), 200, np.uint8))
            (root / "label.txt").write_text("0 0.25 0.5 0.2 0.4\n")
            image, centers, _ = read_sample(root, {"image": "image.png", "label": "label.txt"})
            self.assertEqual(image.shape, (1, 128, 128))
            np.testing.assert_allclose(centers, [(0.25, 0.5)])
            self.assertEqual(float(image[:, :24].sum()), 0.0)
            target = target_heatmap(centers)
            self.assertEqual(target.shape, (1, 20, 20))
            self.assertGreater(float(target[0, 9, 4]), 0.7)

    def test_rotations_and_mirror_keep_point_on_object(self):
        rng = np.random.default_rng(15)
        image, target = np.zeros((1, 128, 128), np.float32), np.zeros((1, 20, 20), np.float32)
        gx, gy = 5, 4
        cx, cy = 26 + 4 * gx, 26 + 4 * gy
        image[0, cy-2:cy+2, cx-2:cx+2] = 1
        target[0, gy, gx] = 1
        for _ in range(20):
            augmented, transformed = augment(image, target, rng)
            yy, xx = np.where(augmented[0] > 0.5)
            ty, tx = np.unravel_index(np.argmax(transformed[0]), (20, 20))
            self.assertAlmostEqual(float(xx.mean() + 0.5), 26 + 4 * tx)
            self.assertAlmostEqual(float(yy.mean() + 0.5), 26 + 4 * ty)

    def test_receptive_field_and_grid_geometry(self):
        model = HeatmapV2()
        receptive, stride, size = 1, 1, 128
        for layer in model.layers:
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.MaxPool2d)):
                kernel = layer.kernel_size if isinstance(layer.kernel_size, int) else layer.kernel_size[0]
                step = layer.stride if isinstance(layer.stride, int) else layer.stride[0]
                receptive += (kernel - 1) * stride
                stride *= step
                size = (size - kernel) // step + 1
        self.assertEqual((receptive, stride, size), (52, 4, 20))
        self.assertEqual(receptive / 2, 24 + 80 / 20 / 2)

    def test_bce_gradient_matches_autograd(self):
        rng = np.random.default_rng(7)
        logits = rng.normal(size=(2, 1, 3, 3)).astype(np.float32)
        targets = rng.uniform(size=logits.shape).astype(np.float32)
        reference = torch.tensor(logits, requires_grad=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            reference, torch.tensor(targets), pos_weight=torch.tensor([10.0]))
        loss.backward()
        actual_loss, gradient = weighted_bce(logits, targets, 10.0)
        self.assertAlmostEqual(actual_loss, loss.item(), places=5)
        np.testing.assert_allclose(gradient, reference.grad.numpy(), atol=1e-7, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
