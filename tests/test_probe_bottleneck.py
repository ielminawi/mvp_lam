import unittest
import numpy as np
import torch
from scripts.check_probe_bottleneck import ridge_fit, train_mlp
from scripts.analyze_temporal_prediction_changes import changes


class BottleneckTests(unittest.TestCase):
    def test_ridge_intercept_and_orthogonal_coordinates(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(80, 8))
        y = x@rng.normal(size=(8, 7))+3
        q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
        w, b = ridge_fit(x, y, 1)
        rotated_w, rotated_b = ridge_fit(x@q, y, 1)
        np.testing.assert_allclose(x@w+b, (x@q)@rotated_w+rotated_b, atol=1e-10)
        np.testing.assert_allclose((x@w+b).mean(0), y.mean(0), atol=1e-10)

    def test_mlp_seed_repeatable_and_finite(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(5)
        x, y = rng.normal(size=(32, 5)), rng.normal(size=(32, 7))
        first = train_mlp(x, y, x[:8], y[:8], 42, epochs=3, patience=2)
        second = train_mlp(x, y, x[:8], y[:8], 42, epochs=3, patience=2)
        np.testing.assert_array_equal(first[1]['validation'], second[1]['validation'])
        self.assertTrue(np.isfinite(first[1]['validation']).all())
        self.assertTrue(1 <= first[2]['selected_epoch'] <= 3)

    def test_prediction_changes_can_cancel_in_error(self):
        truth = np.zeros((2, 1))
        base = np.ones((2, 4, 1))
        prediction = -base
        result = changes(base, prediction, truth, np.array([0, 1]))
        self.assertEqual(result['normalized_prediction_rmse_change'], 2)
        self.assertEqual(result['normalized_action_squared_error_difference'], 0)
        self.assertEqual(result['episodes_unchanged'], 2)


if __name__ == '__main__':
    unittest.main()
