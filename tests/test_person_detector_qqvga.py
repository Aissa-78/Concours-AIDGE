import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from person_detector_qqvga import PersonDetectorQQVGA, make_person_detector
from person_detection_math import targets_from_boxes, detection_loss, evaluate_boxes, hard_negative_weights


class PersonDetectorTests(unittest.TestCase):
    def test_context_model_is_still_small(self):
        model = make_person_detector(channels=16, blocks=3, architecture="regular_context")
        self.assertEqual(sum(p.numel() for p in model.parameters()), 41189)
        self.assertEqual(tuple(model(torch.zeros(1, 1, 120, 160)).shape), (1, 5, 30, 40))

    def test_hard_negatives_and_gradient_match_autograd(self):
        rng = np.random.default_rng(21)
        raw = rng.normal(size=(2, 5, 30, 40)).astype(np.float32)
        targets = [targets_from_boxes([(.5, .5, .25, .4)]), targets_from_boxes([])]
        heat, boxes, mask = (np.stack([sample[i] for sample in targets]) for i in range(3))
        actual, gradient, _ = detection_loss(raw, heat, boxes, mask, classification="hard_bce")
        output = torch.tensor(raw, requires_grad=True)
        h, b, m = torch.tensor(heat), torch.tensor(boxes), torch.tensor(mask)
        pos = (h == 1).float()
        neg = (1-h)**4*(1-pos)
        log_n = torch.nn.functional.logsigmoid(-output[:, 4:5])
        weights = torch.zeros_like(neg)
        for i in range(2):
            valid = torch.nonzero(neg[i].flatten() > 0).flatten()
            count = min(len(valid), max(16, int(pos[i].sum())*16))
            chosen = valid[torch.topk((-neg[i]*log_n[i]).flatten()[valid], count).indices]
            weights[i].flatten()[chosen] = neg[i].flatten()[chosen]
        loss = -(pos*torch.nn.functional.logsigmoid(output[:, 4:5])).sum()/max(float(pos.sum()), 1)
        loss -= (weights*log_n).sum()/max(float(weights.sum()), 1)
        loss += (torch.nn.functional.smooth_l1_loss(output[:, :4], b, reduction="none")*m).sum()/max(float(m.sum())*4, 1)
        loss.backward()
        self.assertAlmostEqual(actual, float(loss.detach()), places=5)
        np.testing.assert_allclose(gradient, output.grad.numpy(), atol=2e-6, rtol=2e-4)
        self.assertEqual(int((weights[0] > 0).sum()), 16)
        self.assertEqual(int((weights[1] > 0).sum()), 16)
        self.assertFalse(bool((weights[pos > 0] != 0).any()))
    def test_regular_convolutions_keep_the_same_output_contract(self):
        torch.set_num_threads(2)
        compact = make_person_detector(channels=16, blocks=3, architecture="regular")
        dense = make_person_detector(channels=16, blocks=3, architecture="regular", training_dense=True)
        dense.copy_from_compact(compact)
        self.assertEqual(sum(p.numel() for p in compact.parameters()), 22757)
        image = torch.rand(2, 1, 120, 160)
        torch.testing.assert_close(compact(image), dense(image), atol=1e-6, rtol=1e-5)
        self.assertEqual(tuple(compact(image).shape), (2, 5, 30, 40))
        self.assertEqual(dense.parameter_masks(), {})

    def test_residual_dense_equivalence_and_gradients(self):
        torch.set_num_threads(2)
        torch.manual_seed(9)
        compact = PersonDetectorQQVGA(channels=4, blocks=2, residual=True)
        dense = PersonDetectorQQVGA(channels=4, blocks=2, training_dense=True, residual=True)
        dense.copy_from_compact(compact)
        image = torch.rand(2, 1, 120, 160)
        first, second = compact(image), dense(image)
        torch.testing.assert_close(first, second, atol=1e-5, rtol=1e-4)
        gradient = torch.rand_like(first)
        first.backward(gradient)
        second.backward(gradient)
        for name, parameter in compact.named_parameters():
            other = dict(dense.named_parameters())[name].grad
            if name.endswith("weight") and int(name.split(".")[1]) in compact.depthwise_indices:
                indices = torch.arange(other.shape[0])
                other = other[indices, indices][:, None]
            torch.testing.assert_close(parameter.grad, other, atol=3e-3, rtol=2e-3)

    def test_dense_and_depthwise_forward_and_useful_gradients_match(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        compact = PersonDetectorQQVGA(channels=4, blocks=2)
        dense = PersonDetectorQQVGA(channels=4, blocks=2, training_dense=True)
        dense.copy_from_compact(compact)
        image = torch.rand(2, 1, 120, 160)
        first, second = compact(image), dense(image)
        self.assertEqual(tuple(first.shape), (2, 5, 30, 40))
        torch.testing.assert_close(first, second, atol=1e-5, rtol=1e-4)
        gradient = torch.rand_like(first)
        first.backward(gradient)
        second.backward(gradient)
        for name, parameter in compact.named_parameters():
            other = dict(dense.named_parameters())[name].grad
            if name.endswith("weight") and int(name.split(".")[1]) in compact.depthwise_indices:
                indices = torch.arange(other.shape[0])
                other = other[indices, indices][:, None]
            torch.testing.assert_close(parameter.grad, other, atol=3e-3, rtol=2e-3)

    def test_analytic_loss_matches_autograd(self):
        rng = np.random.default_rng(7)
        raw = rng.normal(size=(2, 5, 30, 40)).astype(np.float32)
        targets = [targets_from_boxes([(.5, .5, .25, .4)]), targets_from_boxes([])]
        heat, boxes, mask = (np.stack([sample[i] for sample in targets]) for i in range(3))
        actual, gradient, _ = detection_loss(raw, heat, boxes, mask)
        output = torch.tensor(raw, requires_grad=True)
        z = output[:, 4:5]
        p, h, m, b = torch.sigmoid(z), torch.tensor(heat), torch.tensor(mask), torch.tensor(boxes)
        pos = (h == 1).float()
        focal = -(pos*(1-p)**2*torch.nn.functional.logsigmoid(z)
                  +(1-h)**4*(1-pos)*p**2*torch.nn.functional.logsigmoid(-z)).sum()/max(float(pos.sum()), 1)
        regression = (torch.nn.functional.smooth_l1_loss(output[:, :4], b, reduction="none")*m).sum()/max(float(m.sum())*4, 1)
        loss = focal+regression
        loss.backward()
        self.assertAlmostEqual(actual, float(loss.detach()), places=3)
        np.testing.assert_allclose(gradient, output.grad.numpy(), atol=2e-5, rtol=2e-4)

    def test_balanced_bce_matches_autograd_with_an_empty_image(self):
        rng = np.random.default_rng(19)
        raw = rng.normal(size=(2, 5, 30, 40)).astype(np.float32)
        targets = [targets_from_boxes([(.5, .5, .25, .4)]), targets_from_boxes([])]
        heat, boxes, mask = (np.stack([sample[i] for sample in targets]) for i in range(3))
        actual, gradient, _ = detection_loss(raw, heat, boxes, mask, classification="balanced_bce")
        output = torch.tensor(raw, requires_grad=True)
        z = output[:, 4:5]
        h, b, m = torch.tensor(heat), torch.tensor(boxes), torch.tensor(mask)
        positive = (h == 1).float()
        negative = (1-h)**4*(1-positive)
        presence = -(positive*torch.nn.functional.logsigmoid(z)).sum()/max(float(positive.sum()), 1)
        presence -= (negative*torch.nn.functional.logsigmoid(-z)).sum()/max(float(negative.sum()), 1)
        box_loss = (torch.nn.functional.smooth_l1_loss(output[:, :4], b, reduction="none")*m).sum()/max(float(m.sum())*4, 1)
        loss = presence+box_loss
        loss.backward()
        self.assertAlmostEqual(actual, float(loss.detach()), places=5)
        np.testing.assert_allclose(gradient, output.grad.numpy(), atol=2e-6, rtol=2e-4)

    def test_local_maximum_removes_untrained_neighbour_box(self):
        truth = [[(.5, .5, .3, .3)]]
        _, distances, mask = targets_from_boxes(truth[0])
        output = np.zeros((1, 5, 30, 40), np.float32)
        output[0, :4] = distances
        output[0, 4] = -20
        y, x = np.argwhere(mask[0] > 0)[0]
        output[0, 4, y, x] = 4
        output[0, 4, y, x+1] = 3
        output[0, :4, y, x+1] = 1
        plain, _ = evaluate_boxes(output, truth)
        peaks, _ = evaluate_boxes(output, truth, local_maximum=True)
        self.assertEqual((plain["tp"], plain["fp"]), (1, 1))
        self.assertEqual((peaks["tp"], peaks["fp"]), (1, 0))

    def test_targets_decode_back_to_the_annotated_box(self):
        truth = [[(.5, .5, .25, .4)]]
        heat, distances, mask = targets_from_boxes(truth[0])
        output = np.zeros((1, 5, 30, 40), np.float32)
        output[0, :4] = distances
        output[0, 4] = -20
        output[0, 4][mask[0] > 0] = 20
        metrics, _ = evaluate_boxes(output, truth)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (1, 0, 0))

    def test_collision_is_not_silently_dropped(self):
        with self.assertRaises(ValueError):
            targets_from_boxes([(.5, .5, .2, .2), (.5, .5, .3, .3)])

    def test_metrics_do_not_silently_drop_an_image(self):
        with self.assertRaises(ValueError):
            evaluate_boxes(np.zeros((2, 5, 30, 40), np.float32), [[]])

    def test_invalid_boxes_are_rejected(self):
        for box in ((-.2, .5, .2, .2), (.5, .5, 0, .2), (.1, .5, .8, .2), (np.nan, .5, .2, .2)):
            with self.assertRaises(ValueError):
                targets_from_boxes([box])


if __name__ == "__main__":
    unittest.main()
