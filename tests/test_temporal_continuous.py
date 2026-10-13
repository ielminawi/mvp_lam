import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from scripts.compare_temporal_continuous import fit, identities_match
from scripts.temporal_pilot import sha256


class ContinuousTests(unittest.TestCase):
    def test_reject_misaligned_frames(self):
        reference = dict(record_indices=np.array([1]), frame_indices=np.array([[1,4,7,10]]), targets_raw=np.zeros((1,7)))
        changed = {**reference, 'frame_indices': np.array([[2,5,8,11]])}
        with self.assertRaises(AssertionError):
            identities_match(changed, reference)

    def test_fit_pipeline_matches_identical_representations(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); continuous = root/'features'; probes = root/'probes'
            (probes/'four_frame').mkdir(parents=True)
            np.savez(probes/'four_frame/preprocessing.npz', target_mean=np.zeros(7), target_scale=np.ones(7))
            (probes/'results.json').write_text(json.dumps({'feature_manifest': {'same': True}}))
            files = {}; rng = np.random.default_rng(12)
            shared = {}
            for split, n, record in [('train', 30, 1), ('validation', 12, 2)]:
                y = rng.normal(size=(n,7))
                x = np.zeros((n,512)); x[:, :7] = y
                shared[split] = dict(continuous=x, quantized=x, targets_raw=y,
                                     record_indices=np.full(n, record), frame_indices=np.tile([1,4,7,10],(n,1)))
            for variant in ('released', 'two_frame', 'four_frame'):
                (continuous/variant).mkdir(parents=True)
                for split in shared:
                    file = continuous/variant/f'{split}.npz'
                    np.savez(file, **shared[split]); files[f'{variant}/{split}.npz'] = sha256(file)
            (continuous/'manifest.json').write_text(json.dumps({'feature_manifest': {'same': True}, 'files':files}))
            def small_solver(x, y, xv, yv):
                # Avoid decomposing512x512 in a plumbing test; exact7D relation.
                w = np.zeros((512,7)); w[:7] = np.linalg.lstsq(x[:, :7], y, rcond=None)[0]
                return (0., 1., w, np.zeros(7)), [{'alpha':1., 'validation_mse':0.}]
            with patch('scripts.compare_temporal_continuous.ridge_grid', side_effect=small_solver):
                fit(SimpleNamespace(continuous_dir=continuous, probe_dir=probes, output_dir=root/'output'))
            report = json.loads((root/'output/results.json').read_text())
            self.assertEqual(report['paired_four_minus_two']['continuous']['episode_mean_difference'], 0.)
            self.assertLess(report['variants']['four_frame']['continuous']['validation']['all_7']['nmse_mean'], 1e-20)
            self.assertTrue((root/'output/released/continuous_ridge.npz').exists())


if __name__ == '__main__':
    unittest.main()
