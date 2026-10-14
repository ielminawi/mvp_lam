import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from latent_action_model.genie.modules.temporal_pilot import WholeWindowLAM
from latent_action_model.genie.modules.temporal_aggregator import AggregatedLAM, GatedSlotReadout
from scripts.temporal_pilot import frozen_hash
from scripts.train_temporal_aggregator import atomic_checkpoint, sample_stream


class AggregatorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(42)

    def test_initial_identity_gradients_and_frozen_backbone(self):
        with patch('torch.hub.load', return_value=torch.nn.Identity()):
            base = WholeWindowLAM(in_dim=3, model_dim=768, latent_dim=128, num_latents=16,
                                  patch_size=14, enc_blocks=1, dec_blocks=1, num_heads=12, dropout=0.).configure_pilot()
        model = AggregatedLAM(base)
        before = frozen_hash(model)
        patches = torch.randn(1, 4, 256, 768)
        with torch.no_grad():
            for frames in (patches[:, [0,3]], patches):
                actual, original = model.reconstruct_window(frames), base.reconstruct_window(frames)
                for key in ('recon','z_q','emb','indices'):
                    torch.testing.assert_close(actual[key], original[key], rtol=0, atol=0)
        optimizer = torch.optim.Adam(model.aggregator.parameters(), lr=.001)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = model.pilot_loss(model.reconstruct_window(patches))[0]
            loss.backward()
            self.assertGreater(model.aggregator.gate.grad.abs().item(), 0)
            if step == 1:
                self.assertGreater(model.aggregator.attention.in_proj_weight.grad.abs().sum().item(), 0)
                self.assertGreater(model.aggregator.queries.grad.abs().sum().item(), 0)
            self.assertTrue(all(p.grad is None for p in base.parameters()))
            optimizer.step()
        self.assertEqual(frozen_hash(model), before)
        model.train()
        self.assertFalse(base.training)

    def test_zero_gate_and_shared_input_lengths(self):
        readout = GatedSlotReadout(dimension=24, heads=3)
        for times in (2,4):
            slots = torch.randn(2,times,4,24)
            torch.testing.assert_close(readout(slots), slots[:, -1], rtol=0, atol=0)

    def test_checkpoint_restores_optimizer_and_sample_stream(self):
        readout = GatedSlotReadout(dimension=24, heads=3)
        optimizer = torch.optim.Adam(readout.parameters(), lr=.001)
        slots = torch.randn(2,4,4,24); target = torch.randn(2,4,24)
        def step(model, opt):
            opt.zero_grad(set_to_none=True)
            (model(slots)-target).square().mean().backward(); opt.step()
        step(readout,optimizer)
        gen = np.random.default_rng(42); order = gen.permutation(10)
        _, order, cursor = sample_stream(gen, order, 0, 8)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'latest.pt'
            atomic_checkpoint(path, {'aggregator':readout.state_dict(), 'optimizer':optimizer.state_dict(),
                                     'generator':gen.bit_generator.state,'order':torch.from_numpy(order),'cursor':cursor})
            saved = torch.load(path, weights_only=True)
            restored = GatedSlotReadout(dimension=24,heads=3)
            restored.load_state_dict(saved['aggregator'])
            restored_opt = torch.optim.Adam(restored.parameters(), lr=.001)
            restored_opt.load_state_dict(saved['optimizer'])
            step(readout,optimizer); step(restored,restored_opt)
            for key,value in readout.state_dict().items():
                torch.testing.assert_close(value,restored.state_dict()[key],rtol=0,atol=0)
            restored_gen=np.random.default_rng(); restored_gen.bit_generator.state=saved['generator']
            expected, _, _=sample_stream(gen,order,cursor,8)
            actual, _, _=sample_stream(restored_gen,saved['order'].numpy(),saved['cursor'],8)
            self.assertEqual(actual,expected)
            self.assertFalse(path.with_suffix('.tmp').exists())


if __name__ == '__main__':
    unittest.main()
