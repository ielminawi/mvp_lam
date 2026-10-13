import unittest
from unittest.mock import patch
import numpy as np
import torch

from latent_action_model.genie.modules.temporal_pilot import WholeWindowLAM
from scripts.diagnose_temporal_readout import encode_all, decode_codes, ridge_grid
from scripts.check_probe_bottleneck import ridge_fit


class ReadoutTests(unittest.TestCase):
    def test_all_slots_final_equivalence_and_causality(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        with patch('torch.hub.load', return_value=torch.nn.Identity()):
            model = WholeWindowLAM(in_dim=3, model_dim=768, latent_dim=128,
                                  num_latents=16, patch_size=14, enc_blocks=1, dec_blocks=1,
                                  num_heads=12, dropout=0.).configure_pilot()
        model.requires_grad_(False)
        patches = torch.randn(1, 4, 256, 768)
        with torch.no_grad():
            continuous, quantized, indices = encode_all(model, patches)
            reference = model.reconstruct_window(patches)
            self.assertEqual(continuous.shape, (1, 4, 4, 128))
            torch.testing.assert_close(continuous[:, -1], reference['emb'])
            torch.testing.assert_close(quantized[:, -1:], reference['z_q'])
            torch.testing.assert_close(indices[:, -1], reference['indices'])
            torch.testing.assert_close(decode_codes(model, patches[:, :1], quantized[:, -1:]), reference['recon'])
            changed = patches.clone(); changed[:, -1] += torch.randn_like(changed[:, -1])
            altered, _, _ = encode_all(model, changed)
            torch.testing.assert_close(continuous[:, :3], altered[:, :3], rtol=0, atol=0)

    def test_spectral_ridge_matches_direct_solver(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(40, 8)); x[:, 7] = x[:, 0]  # Rank-deficient features.
        y, xv, yv = rng.normal(size=(40, 7)), rng.normal(size=(10, 8)), rng.normal(size=(10, 7))
        best, _ = ridge_grid(x, y, xv, yv)
        _, alpha, weight, bias = best
        expected_weight, expected_bias = ridge_fit(x, y, alpha)
        np.testing.assert_allclose(xv@weight+bias, xv@expected_weight+expected_bias, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
