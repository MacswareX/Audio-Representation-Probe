"""Vector quantization modules: EMA-VQ and Residual VQ (RVQ).

ObsCodec used a single-codebook VQ-VAE; this module extends that implementation
to residual quantization. The benchmark probes the first-K RVQ codebook prefix
as an experimental discrete-code interface without assuming that early stages
are inherently more useful for the designed event groups.

These modules are intentionally small and self-contained so they run on a
single 8 GB GPU. They are unit-constructible and have a __main__ smoke test.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    """Standard VQ with EMA codebook updates and straight-through gradient.

    Input/Output convention: z of shape [B, D, T] (channels-first, 1-D time).
    """

    def __init__(self, num_embeddings=512, embedding_dim=64,
                 commitment_cost=0.25, decay=0.99, eps=1e-5,
                 reset_threshold=1.0):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps
        self.reset_threshold = reset_threshold
        embed = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer("embedding", embed)
        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_w", embed.clone())
        self.register_buffer("initted", torch.tensor(False))

    @torch.no_grad()
    def _data_init(self, flat):
        """Data-dependent codebook init from the first batch (anti-collapse)."""
        n = flat.shape[0]
        pick = torch.randint(0, n, (self.num_embeddings,), device=flat.device)
        chosen = flat[pick]
        self.embedding.copy_(chosen)
        self.ema_w.copy_(chosen)
        self.cluster_size.fill_(1.0)
        self.initted.fill_(True)

    @torch.no_grad()
    def _reset_dead(self, flat):
        """Replace codes that fell below usage threshold with live encoder vectors.
        This is the SoundStream/DAC-style inactive-code replacement trick."""
        dead = self.cluster_size < self.reset_threshold
        n_dead = int(dead.sum())
        if n_dead == 0:
            return
        pick = torch.randint(0, flat.shape[0], (n_dead,), device=flat.device)
        repl = flat[pick]
        self.embedding[dead] = repl
        self.ema_w[dead] = repl
        self.cluster_size[dead] = 1.0

    def forward(self, z):
        b, d, t = z.shape
        flat = z.permute(0, 2, 1).reshape(-1, d)  # [B*T, D]
        if self.training and not bool(self.initted):
            self._data_init(flat.detach())
        # squared L2 distance to each code
        dist = (flat.pow(2).sum(1, keepdim=True)
                - 2 * flat @ self.embedding.t()
                + self.embedding.pow(2).sum(1))
        idx = dist.argmin(1)                       # [B*T]
        onehot = F.one_hot(idx, self.num_embeddings).type(flat.dtype)
        quant = (onehot @ self.embedding).view(b, t, d).permute(0, 2, 1)

        if self.training:
            n = onehot.sum(0)
            self.cluster_size.mul_(self.decay).add_(n, alpha=1 - self.decay)
            dw = onehot.t() @ flat.detach()
            self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)
            total = self.cluster_size.sum()
            cluster = ((self.cluster_size + self.eps)
                       / (total + self.num_embeddings * self.eps) * total)
            self.embedding.copy_(self.ema_w / cluster.unsqueeze(1))
            self._reset_dead(flat.detach())

        commit = F.mse_loss(quant.detach(), z)
        loss = self.commitment_cost * commit
        quant_st = z + (quant - z).detach()        # straight-through

        avg = onehot.mean(0)
        perplexity = torch.exp(-(avg * (avg + 1e-10).log()).sum())
        return quant_st, loss, idx.view(b, t), perplexity


class ResidualVQ(nn.Module):
    """Stack of VQ layers, each quantizing the residual of the previous one.

    Returns the summed quantized latent, mean VQ loss, a list of index tensors
    (one [B, T] per codebook), and a list of per-codebook perplexities.
    The first k codebooks form a partial RVQ approximation.
    """

    def __init__(self, num_quantizers=4, num_embeddings=512, embedding_dim=64,
                 commitment_cost=0.25, decay=0.99):
        super().__init__()
        self.layers = nn.ModuleList([
            VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay)
            for _ in range(num_quantizers)
        ])

    def forward(self, z):
        residual = z
        quant_sum = torch.zeros_like(z)
        losses, indices, perplex = [], [], []
        for layer in self.layers:
            q, loss, idx, ppl = layer(residual)
            residual = residual - q
            quant_sum = quant_sum + q
            losses.append(loss)
            indices.append(idx)
            perplex.append(ppl)
        return quant_sum, torch.stack(losses).mean(), indices, perplex


if __name__ == "__main__":
    torch.manual_seed(0)
    rvq = ResidualVQ(num_quantizers=4, num_embeddings=256, embedding_dim=32)
    z = torch.randn(2, 32, 50)
    q, loss, idxs, ppl = rvq(z)
    print("input ", tuple(z.shape))
    print("quant ", tuple(q.shape), "vq_loss", round(float(loss), 4))
    print("codebooks", len(idxs), "indices[0]", tuple(idxs[0].shape))
    print("perplexities", [round(float(p), 1) for p in ppl])
