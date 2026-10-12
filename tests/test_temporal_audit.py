import json
import sys
from unittest.mock import patch
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scripts.audit_temporal_pilot import inspect, usage, feature_usage, plot, main


class AuditTests(unittest.TestCase):
    def report(self):
        v = dict(pairs=4, code_counts=[1]*16, reconstruction_mse=.2, copy_initial_mse=.4)
        return dict(frozen_weights_unchanged=True, train_pairs=16, validation_pairs=4,
                    config=dict(updates=2, validation_every=2, micro_batch_size=1, accumulation_steps=2, gradient_clip=1),
                    history=[dict(update=0, validation=v),
                             dict(update=1, training_total=.3, training_reconstruction=.2, training_commitment=.4, gradient_norm_before_clip=2),
                             dict(update=2, training_total=.3, training_reconstruction=.2, training_commitment=.4, gradient_norm_before_clip=.5, validation=v)])

    def test_usage(self):
        self.assertAlmostEqual(usage([1]*16)['perplexity'], 16)
        self.assertAlmostEqual(usage([16]+[0]*15)['perplexity'], 1)
        with self.assertRaises(ValueError):
            usage([0]*16)

    def test_history_validation(self):
        r = self.report()
        s = inspect(r)
        self.assertEqual(s['clipping_fraction'], .5)
        self.assertEqual(s['equivalent_training_passes'], .25)
        r['history'][-1]['validation']['code_counts'] = [2]*16
        with self.assertRaises(ValueError):
            inspect(r)

    def test_full_audit_includes_untrained_released_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ('two_frame', 'four_frame'):
                report = self.report()
                report.update(variant=name, cache_manifest_sha256='cache', frozen_weight_sha256='frozen')
                (root/name).mkdir()
                (root/name/'results.json').write_text(json.dumps(report))
            for name in ('released', 'two_frame', 'four_frame'):
                dest = root/'features'/name
                dest.mkdir(parents=True)
                np.savez(dest/'train.npz', indices=np.tile(np.arange(16).reshape(4, 4), (4, 1)))
                np.savez(dest/'validation.npz', indices=np.arange(16).reshape(4, 4))
            argv = ['audit', '--two-frame-dir', str(root/'two_frame'),
                    '--four-frame-dir', str(root/'four_frame'),
                    '--features-dir', str(root/'features'), '--output-dir', str(root/'output')]
            with patch.object(sys, 'argv', argv), patch('scripts.audit_temporal_pilot.plot'):
                main()
            result = json.loads((root/'output/results.json').read_text())
            self.assertIn('validation', result['runs']['released']['extracted_code_usage'])
            self.assertNotIn('final_code_usage', result['runs']['released'])

    def test_slot_collapse_hidden_by_marginals_and_render(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            np.savez(p/'features.npz', indices=np.tile(np.arange(4), (10, 1)))
            s = feature_usage(p/'features.npz')
            self.assertEqual(s['distinct_four_code_tuples'], 1)
            self.assertTrue(all(x['perplexity'] == 1 for x in s['per_slot']))
            self.assertAlmostEqual(s['marginal']['perplexity'], 4)
            plot(dict(two_frame=self.report(), four_frame=self.report()), p, 1)
            self.assertGreater((p/'training_curves.png').stat().st_size, 1000)
            self.assertGreater((p/'code_usage.pdf').stat().st_size, 1000)


if __name__ == '__main__':
    unittest.main()
