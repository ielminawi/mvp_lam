"""Whole-window adaptation; the released two-frame implementation stays separate."""
import torch
import torch.nn.functional as F

from latent_action_model.genie.modules.lam import DINOLatentActionModel


class WholeWindowLAM(DINOLatentActionModel):
    def configure_pilot(self):
        """Freeze visual features, vocabulary and decoder; adapt encoding only."""
        self.requires_grad_(False)
        self.action_latent.requires_grad_(True)
        self.encoder.requires_grad_(True)
        self.to_codebook.requires_grad_(True)
        self.vq.code_restart = False
        self.eval()
        return self

    def encode_window(self, patches):
        if patches.ndim != 4 or patches.shape[1] not in (2, 4) or patches.shape[2:] != (256, 768):
            raise ValueError("Expected [batch, 2 or 4, 256, 768] DINO features")
        batch, times = patches.shape[:2]
        slots = self.action_latent.expand(batch, times, -1, -1)
        tokens = torch.cat((slots, patches), dim=2)
        encoded = self.encoder(tokens)
        continuous = self.to_codebook(encoded[:, -1:, :self.num_codes])
        quantized, selected, continuous, indices = self.vq(continuous[:, 0])
        return {"z_q": quantized[:, None], "z": selected,
                "emb": continuous, "indices": indices}

    def reconstruct_window(self, patches):
        encoded = self.encode_window(patches)
        tokens = torch.cat((self.action_up(encoded["z_q"]), self.patch_up(patches[:, :1])), dim=2)
        encoded["recon"] = self.decoder(tokens)[:, :, -patches.shape[2]:]
        encoded["target"] = patches[:, -1:]
        return encoded

    @staticmethod
    def pilot_loss(output, beta=0.25):
        reconstruction = F.mse_loss(output["recon"], output["target"])
        commitment = F.mse_loss(output["emb"], output["z"].detach())
        return reconstruction + beta * commitment, reconstruction, commitment
