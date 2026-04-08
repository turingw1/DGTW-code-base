"""Time and pair-condition embeddings for DG-TWFD."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class TimeEmbedding(nn.Module):
    """Sinusoidal embedding for scalar times in `[0, 1]`.

    Later phases feed these embeddings into the student map `M_theta(t, s, x_t)`
    so the network is explicitly conditioned on the source and target times.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        if embed_dim < 4:
            raise ValueError("embed_dim must be at least 4")
        self.embed_dim = embed_dim
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, time_values: Tensor) -> Tensor:
        time_values = time_values.float().view(-1, 1)
        half_dim = self.embed_dim // 2
        freq = torch.exp(
            torch.linspace(
                0.0,
                math.log(10_000.0),
                steps=half_dim,
                device=time_values.device,
                dtype=time_values.dtype,
            )
            * (-1.0)
        )
        angles = time_values * freq.unsqueeze(0)
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if embedding.shape[-1] < self.embed_dim:
            embedding = torch.cat([embedding, embedding[:, :1]], dim=-1)
        return self.proj(embedding)


class PairTimeConditioner(nn.Module):
    """Encode `(t, s, delta=t-s)` into a conditioning vector.

    The student consumes this representation so `M_theta(t, s, x_t)` can depend
    on both endpoints and on the step width `delta`.
    """

    def __init__(self, embed_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.time_embedding = TimeEmbedding(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 3 + 3, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, t: Tensor, s: Tensor) -> Tensor:
        delta = t - s
        features = torch.cat(
            [
                self.time_embedding(t),
                self.time_embedding(s),
                self.time_embedding(delta),
                t.float().view(-1, 1),
                s.float().view(-1, 1),
                delta.float().view(-1, 1),
            ],
            dim=-1,
        )
        return self.mlp(features)
