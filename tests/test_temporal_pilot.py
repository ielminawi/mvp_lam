"""CPU checks with deterministic fake DINO features; no downloads/checkpoint needed."""
import unittest
from unittest.mock import patch

import torch
from torch import nn

from latent_action_model.genie.modules.temporal_pilot import WholeWindowLAM


class FakeDINO(nn.Module):
    def forward_features(self, videos):
        # Deterministic frame-dependent features; no actual visual backbone.
        values = videos.mean(dim=(1, 2, 3))[:, None, None]
        return {"x_norm_patchtokens": values.expand(-1, 256, 768)}


class TemporalPilotTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        with patch("torch.hub.load", return_value=FakeDINO()):
            self.model = WholeWindowLAM(in_dim=3, model_dim=768, latent_dim=128,
                num_latents=16, patch_size=14, enc_blocks=1, dec_blocks=1,
                num_heads=12, dropout=0.).configure_pilot()

    def test_two_frame_matches_original_encoder(self):
        videos = torch.randn(2, 2, 3, 28, 28)
        with torch.no_grad():
            original = self.model.vq_encode(videos)
            candidate = self.model.encode_window(original["patches"])
        for name in ("z_q", "z", "emb", "indices"):
            torch.testing.assert_close(candidate[name], original[name], rtol=0, atol=0)
        self.assertNotIn("encoder.pos_enc.pos_enc", self.model.state_dict())

    def test_decoder_gradient_and_frozen_weights(self):
        model = self.model
        frozen = {n: p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
        before = model.to_codebook.weight.detach().clone()
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-5)
        for times in (2, 4):
            optimizer.zero_grad(set_to_none=True)
            output = model.reconstruct_window(torch.randn(1, times, 256, 768))
            self.assertEqual(output["z_q"].shape, (1, 1, 4, 128))
            self.assertEqual(output["recon"].shape, (1, 1, 256, 768))
            # Reconstruction alone must train the encoder through the fixed decoder/quantizer.
            reconstruction = torch.nn.functional.mse_loss(output["recon"], output["target"])
            reconstruction.backward()
            for name in ("action_latent", "encoder.ffn.1.weight", "to_codebook.weight"):
                gradient = dict(model.named_parameters())[name].grad
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(gradient.abs().sum().item(), 0)
            self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))
            optimizer.step()
        self.assertFalse(torch.equal(before, model.to_codebook.weight))
        for name,p in model.named_parameters():
            if name in frozen:
                torch.testing.assert_close(p, frozen[name], rtol=0, atol=0)
        self.assertFalse(model.vq.code_restart)

    def test_invalid_window_rejected(self):
        with self.assertRaises(ValueError):
            self.model.encode_window(torch.randn(1, 3, 256, 768))


if __name__ == "__main__":
    unittest.main()
