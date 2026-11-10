import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from heatmap_balanced_protocol import balanced_batches, selection_key


class BalancedProtocolTests(unittest.TestCase):
    def test_every_batch_is_balanced_including_the_last(self):
        batches = balanced_batches(179, 400, 16, np.random.default_rng(7))
        all_indices = np.concatenate(batches)
        self.assertEqual(len(all_indices), 358)
        self.assertEqual(len(batches[-1]), 6)
        for batch in batches:
            self.assertEqual(sum(batch < 179), sum(batch >= 179))
            self.assertLessEqual(len(batch), 16)
        self.assertEqual(sorted(all_indices[all_indices < 179].tolist()), list(range(179)))
        self.assertEqual(len(set(all_indices[all_indices >= 179])), 179)
        self.assertLess(int(all_indices.max()), 579)

    def test_small_public_source_can_be_repeated_without_losing_balance(self):
        batches = balanced_batches(9, 2, 4, np.random.default_rng(7))
        for batch in batches:
            self.assertEqual(sum(batch < 9), sum(batch >= 9))
            self.assertLess(int(batch.max()), 11)

    def test_sampling_is_reproducible(self):
        first = balanced_batches(13, 20, 8, np.random.default_rng(7))
        second = balanced_batches(13, 20, 8, np.random.default_rng(7))
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)

    def test_invalid_sizes_are_rejected(self):
        for local, public, batch in ((0, 10, 16), (3, 0, 16), (3, 4, 1), (3, 4, 15)):
            with self.assertRaises(ValueError):
                balanced_batches(local, public, batch, np.random.default_rng(7))

    def test_a_collapsed_domain_is_not_hidden_by_a_high_average(self):
        balanced = selection_key(.40, .55, .4, .4)
        collapsed = selection_key(.95, .07, .2, .4)
        self.assertGreater(balanced, collapsed)

    def test_selection_is_symmetric_and_uses_documented_ties(self):
        self.assertEqual(selection_key(.4, .6, .2, .5), selection_key(.6, .4, .5, .2))
        self.assertGreater(selection_key(.4, .7, .9, .9), selection_key(.4, .6, .1, .1))
        self.assertGreater(selection_key(.4, .7, .2, .2), selection_key(.4, .7, .3, .3))

    def test_invalid_metrics_are_rejected(self):
        for values in ((np.nan, .4, .2, .2), (1.1, .4, .2, .2), (.3, -.1, .2, .2),
                       (.3, .4, -.1, .2), (.3, .4, .2, np.inf)):
            with self.assertRaises(ValueError):
                selection_key(*values)


if __name__ == "__main__":
    unittest.main()
