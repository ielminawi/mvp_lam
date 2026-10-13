import unittest
import numpy as np
import torch
from scripts.diagnose_temporal_intermediates import CONDITIONS, intervene, paired_interval, probe_predict


class InterventionTests(unittest.TestCase):
    def test_endpoints_preserved_without_mutation(self):
        x = torch.arange(2*4*3).reshape(2, 4, 3)
        original = x.clone()
        for condition, order in CONDITIONS.items():
            result = intervene(x, condition)
            torch.testing.assert_close(result[:, 0], x[:, 0])
            torch.testing.assert_close(result[:, -1], x[:, -1])
            torch.testing.assert_close(result, x[:, order])
        torch.testing.assert_close(x, original)

    def test_episode_weighting_and_paired_null(self):
        records = np.array([0, 0, 0, 1])
        result = paired_interval(np.array([1., 1., 1., 3.]), records)
        self.assertEqual(result['episode_mean_difference'], 2)
        null = paired_interval(np.zeros(4), records)
        self.assertEqual(null['episode_bootstrap_95_percent_interval'], [0, 0])

    def test_saved_affine_probe_pipeline(self):
        features = np.array([[2., 4.], [3., 5.]])
        prep = {'feature_mean': np.array([1., 2.]), 'pca_components': np.eye(2)}
        weight = {'weight': torch.tensor([[2., 3.]]), 'bias': torch.tensor([4.])}
        actual = probe_predict(features, prep, [weight, weight])
        np.testing.assert_array_equal(actual[:, 0, 0], [12, 17])
        np.testing.assert_array_equal(actual[:, 0], actual[:, 1])


if __name__ == '__main__':
    unittest.main()
