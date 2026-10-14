import unittest
from unittest.mock import patch

import torch
from torch import nn

from latent_action_model.genie.modules.temporal_pilot import WholeWindowLAM
from latent_action_model.genie.modules.temporal_aggregator import AggregatedLAM
from scripts.probe_temporal_aggregator import readouts, load_aggregator_state
from scripts.temporal_pilot import RELEASE_HASH, frozen_hash


class AggregatorProbeTests(unittest.TestCase):
    def test_zero_controls_match_and_gate_changes_only_aggregated(self):
        torch.set_num_threads(1); torch.manual_seed(42)
        with patch('torch.hub.load', return_value=nn.Identity()):
            base=WholeWindowLAM(in_dim=3,model_dim=768,latent_dim=128,num_latents=16,patch_size=14,
                                enc_blocks=1,dec_blocks=1,num_heads=12,dropout=0.)
        model=AggregatedLAM(base).eval()
        with torch.no_grad():
            for times in (2,4):
                patches=torch.randn(1,times,256,768)
                outputs, relative=readouts(model,patches)
                reference=base.encode_window(patches)
                torch.testing.assert_close(outputs['zero']['continuous'], reference['emb'].flatten(1),rtol=0,atol=0)
                torch.testing.assert_close(outputs['zero']['quantized'], reference['z_q'].flatten(1),rtol=0,atol=0)
                torch.testing.assert_close(outputs['zero']['continuous'],outputs['aggregated']['continuous'],rtol=0,atol=0)
                self.assertEqual(relative.item(),0.)
            model.aggregator.gate.fill_(.2)
            changed,relative=readouts(model,patches)
            torch.testing.assert_close(changed['zero']['continuous'],outputs['zero']['continuous'],rtol=0,atol=0)
            self.assertGreater(relative.item(),0)
            self.assertFalse(torch.equal(changed['aggregated']['continuous'],changed['zero']['continuous']))

    def test_checkpoint_report_final_update_required(self):
        model=nn.Module(); model.backbone=nn.Linear(2,2); model.backbone.requires_grad_(False)
        model.aggregator=nn.Linear(2,2)
        report={'config':{'updates':2},'history':[{'update':2}], 'variant':'two_frame',
                'cache_manifest_sha256':'cache','frozen_weight_sha256':frozen_hash(model),
                'base_checkpoint_sha256':RELEASE_HASH,'frozen_weights_unchanged':True}
        saved={'update':2,'run':report.copy(),'aggregator':model.aggregator.state_dict()}
        load_aggregator_state(model,saved,report,'two_frame','cache')
        saved['update']=1
        with self.assertRaises(ValueError):
            load_aggregator_state(model,saved,report,'two_frame','cache')


if __name__=='__main__':
    unittest.main()
