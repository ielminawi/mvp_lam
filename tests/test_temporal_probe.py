"""Probe fairness, normalization, overlays and held-out-data boundaries."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from scripts.probe_temporal_pilot import fit, fit_pca, group_scores, overlay_adaptation, VARIANTS
from scripts.temporal_pilot import RELEASE_HASH, sha256


class TemporalProbeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_baseline_metric_and_constant_component(self):
        rng = np.random.default_rng(1)
        truth = rng.normal(size=(20, 7))
        truth[:, 6] = 0
        mean = truth.mean(axis=0)
        scores = group_scores(np.broadcast_to(mean, truth.shape), truth, mean)
        self.assertAlmostEqual(scores["all_7"]["nmse"], 1)
        self.assertAlmostEqual(scores["translation"]["error_reduction_percent"], 0)
        self.assertIsNone(scores["gripper"]["nmse"])

    def test_pca_does_not_fit_validation(self):
        rng = np.random.default_rng(5)
        training = rng.normal(size=(130,512))
        validation = np.full((3,512), 10000.)
        _, mean, _, _, _ = fit_pca(training, validation)
        np.testing.assert_allclose(mean, training.mean(axis=0))

    def test_overlay_rejects_wrong_or_incomplete_parameters(self):
        model = torch.nn.Linear(2, 1)
        saved = {"variant": "two_frame", "base_checkpoint_sha256": RELEASE_HASH,
                 "update": 500, "config": {"updates": 500},
                 "adapted_parameters": {n: torch.zeros_like(p) for n,p in model.named_parameters()}}
        overlay_adaptation(model, saved, "two_frame")
        self.assertEqual(model.weight.abs().sum(), 0)
        with self.assertRaises(ValueError):
            overlay_adaptation(model, saved, "four_frame")
        saved["adapted_parameters"].pop("bias")
        with self.assertRaises(ValueError):
            overlay_adaptation(model, saved, "two_frame")

    def test_identical_features_give_identical_matched_results(self):
        rng = np.random.default_rng(8)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            features = root / "features"
            features.mkdir()
            norm = features / "action_normalization.npz"
            np.savez(norm, net_mean=np.zeros(7), net_scale=np.ones(7))
            (features / "manifest.json").write_text(json.dumps({"normalization_sha256": sha256(norm)}))
            arrays = {}
            for split, n, record in (("train",160,0),("validation",40,1)):
                z = rng.normal(size=(n, 7))
                x = np.zeros((n,512),dtype=np.float32)
                x[:,:7] = z
                arrays[split] = dict(features=x, targets_raw=z.astype(np.float32),
                    record_indices=np.full(n,record),frame_indices=np.tile([1,4,7,10],(n,1)))
            for variant in VARIANTS:
                folder = features / variant
                folder.mkdir()
                for split,data in arrays.items():
                    np.savez(folder / f"{split}.npz", **data)
                # Deliberately invalid: fitting must never try to read it.
                (folder / "test.npz").write_text("test is unavailable")
            output = root / "results"
            with contextlib.redirect_stdout(io.StringIO()):
                fit(SimpleNamespace(features_dir=features, output_dir=output))
            report = json.loads((output / "results.json").read_text())
            reference = report["variants"]["released"]["splits"]["validation"]
            self.assertLess(reference["least_squares"]["all_7"]["nmse"], 1e-10)
            for variant in VARIANTS:
                result = report["variants"][variant]["splits"]["validation"]
                self.assertEqual(result["summary"], reference["summary"])
                self.assertTrue((output / variant / "preprocessing.npz").exists())


if __name__ == "__main__":
    unittest.main()
