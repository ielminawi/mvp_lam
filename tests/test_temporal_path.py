import unittest
import numpy as np
import torch
from scripts.probe_temporal_path import segment_targets, score
from scripts.check_probe_bottleneck import train_mlp


class PathTests(unittest.TestCase):
    def test_segments_telescope_and_gripper_alignment(self):
        states = np.arange(12)[:, None]*np.arange(1, 8)[None, :]
        gripper = np.arange(12) % 2
        frames = np.array([[1, 4, 7, 10], [2, 5, 8, 11]])
        targets = segment_targets(states, gripper, frames).reshape(2, 3, 7)
        np.testing.assert_array_equal(targets[:, :, :6].sum(1), states[frames[:, -1], :6]-states[frames[:, 0], :6])
        np.testing.assert_array_equal(targets[:, :, 6], gripper[frames[:, 1:]-1])
        self.assertEqual(targets.shape, (2, 3, 7))
        with self.assertRaises(ValueError):
            segment_targets(states, gripper, np.array([[1, 4, 6, 10]]))

    def test_group_scores_and_constant_dimensions(self):
        rng = np.random.default_rng(9)
        truth = rng.normal(size=(100, 21)); truth[:, [6, 13, 20]] = 1
        mean = truth.mean(0)
        baseline = score([np.tile(mean, (len(truth), 1))], truth, mean)
        self.assertAlmostEqual(baseline['all_21']['nmse_mean'], 1)
        self.assertIsNone(baseline['gripper']['nmse_mean'])
        perfect = score([truth], truth, mean)
        self.assertEqual(perfect['translation']['nmse_mean'], 0)
        self.assertEqual(perfect['B_to_C']['nmse_mean'], 0)

    def test_mlp_supports_twenty_one_outputs(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(10)
        x, y = rng.normal(size=(16, 4)), rng.normal(size=(16, 21))
        _, predictions, _ = train_mlp(x, y, x[:4], y[:4], 42, epochs=2)
        self.assertEqual(predictions['validation'].shape, (4, 21))
        self.assertTrue(np.isfinite(predictions['validation']).all())


if __name__ == '__main__':
    unittest.main()
