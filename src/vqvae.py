"""Small mel-spectrogram RVQ autoencoder for the custom representation path.

The model is a compact comparison point for the pretrained EnCodec interface.
It uses the ResidualVQ implementation from rvq.py and reconstructs a pooled-mel
target rather than waveform audio.

The training and export pipeline lives in scripts/6_train_mel_vqvae.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:  # works both as a package import and as `python src/vqvae.py`
    from .rvq import ResidualVQ
except ImportError:  # pragma: no cover
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.rvq import ResidualVQ


class MelEncoder(nn.Module):
    """log-mel [B, 1, M, T] -> latent [B, D, T'] (downsample along time)."""

    def __init__(self, n_mels=64, hidden=128, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden, 3, stride=(2, 2), padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, stride=(2, 2), padding=1), nn.ReLU(),
            nn.Conv2d(hidden, latent_dim, 3, stride=(2, 1), padding=1),
        )
        self.latent_dim = latent_dim

    def forward(self, mel):
        h = self.net(mel)                  # [B, D, M', T']
        return h.mean(dim=2)               # collapse freq -> [B, D, T']


class MelDecoder(nn.Module):
    """latent [B, D, T'] -> reconstructed pooled mel vector [B, n_mels]."""

    def __init__(self, latent_dim=64, hidden=128, n_mels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(latent_dim, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv1d(hidden, n_mels, 3, padding=1),
        )

    def forward(self, z):
        h = self.net(z)                    # [B, n_mels, T']
        return h.mean(dim=2)               # time-pooled mel proxy [B, n_mels]


class MelRVQVAE(nn.Module):
    def __init__(self, n_mels=64, latent_dim=64, num_quantizers=4,
                 codebook_size=512):
        super().__init__()
        self.encoder = MelEncoder(n_mels=n_mels, latent_dim=latent_dim)
        self.rvq = ResidualVQ(num_quantizers=num_quantizers,
                              num_embeddings=codebook_size,
                              embedding_dim=latent_dim)
        self.decoder = MelDecoder(latent_dim=latent_dim, n_mels=n_mels)

    def forward(self, mel):
        z = self.encoder(mel)
        q, vq_loss, indices, perplex = self.rvq(z)
        recon = self.decoder(q)
        return {"recon": recon, "vq_loss": vq_loss,
                "indices": indices, "perplexity": perplex, "latent": q}


if __name__ == "__main__":
    torch.manual_seed(0)
    model = MelRVQVAE(n_mels=64, latent_dim=64, num_quantizers=4, codebook_size=256)
    mel = torch.randn(4, 1, 64, 250)       # [B, 1, n_mels, T]
    out = model(mel)
    n_params = sum(p.numel() for p in model.parameters())
    print("params", f"{n_params/1e6:.2f}M")
    print("recon", tuple(out["recon"].shape), "vq_loss", round(float(out["vq_loss"]), 4))
    print("codebooks", len(out["indices"]), "idx0", tuple(out["indices"][0].shape))
