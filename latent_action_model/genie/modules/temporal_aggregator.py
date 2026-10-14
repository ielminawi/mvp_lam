"""Gated four-query readout; the entire released LAM stays frozen."""
import torch
from torch import nn
from latent_action_model.genie.modules.temporal_pilot import WholeWindowLAM


class GatedSlotReadout(nn.Module):
    def __init__(self, dimension=768, heads=12):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(4, dimension)*.02)
        self.norm = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(dimension, heads, dropout=0., batch_first=True)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, slots):
        # Shared readout for eight (two-frame) or sixteen (four-frame) slots.
        batch = slots.shape[0]
        memory = self.norm(slots.flatten(1, 2))
        query = self.queries[None].expand(batch, -1, -1)
        correction, _ = self.attention(query, memory, memory, need_weights=False)
        return slots[:, -1]+torch.tanh(self.gate)*correction


class AggregatedLAM(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.backbone.requires_grad_(False).eval()
        self.backbone.vq.code_restart = False
        self.aggregator = GatedSlotReadout()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def encode_window(self, patches):
        if patches.ndim != 4 or patches.shape[1] not in (2, 4) or patches.shape[2:] != (256, 768):
            raise ValueError('Expected [B,2 or4,256,768]')
        batch, times = patches.shape[:2]
        with torch.no_grad():
            slots = self.backbone.action_latent.expand(batch, times, -1, -1)
            encoded = self.backbone.encoder(torch.cat((slots, patches), dim=2))
            contextualized = encoded[:, :, :4]
        summary = self.aggregator(contextualized)
        # Frozen weights, but gradients through these operations must remain enabled.
        continuous = self.backbone.to_codebook(summary)
        quantized, selected, continuous, indices = self.backbone.vq(continuous)
        return {'z_q': quantized[:, None], 'z': selected, 'emb': continuous, 'indices': indices}

    def reconstruct_window(self, patches):
        output = self.encode_window(patches)
        tokens = torch.cat((self.backbone.action_up(output['z_q']), self.backbone.patch_up(patches[:, :1])), dim=2)
        output['recon'] = self.backbone.decoder(tokens)[:, :, -256:]
        output['target'] = patches[:, -1:]
        return output

    pilot_loss = staticmethod(WholeWindowLAM.pilot_loss)
