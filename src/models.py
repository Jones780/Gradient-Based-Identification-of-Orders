"""Neural derivative surrogate used in the paper."""
from __future__ import annotations
import torch
import torch.nn as nn


class DeepONetAlpha(nn.Module):
    """DeepONet-style branch/trunk surrogate for the Caputo derivative."""
    def __init__(self, state_dim: int = 2, latent_dim: int = 64, hidden_dim: int = 64, num_layers: int = 4):
        super().__init__()
        layers_b = []
        in_dim = state_dim
        for _ in range(num_layers):
            layers_b.extend([nn.Linear(in_dim, hidden_dim), nn.GELU()])
            in_dim = hidden_dim
        layers_b.append(nn.Linear(hidden_dim, latent_dim))
        self.branch = nn.Sequential(*layers_b)

        layers_t = []
        in_dim = 2
        for _ in range(num_layers):
            layers_t.extend([nn.Linear(in_dim, hidden_dim), nn.GELU()])
            in_dim = hidden_dim
        layers_t.append(nn.Linear(hidden_dim, latent_dim))
        self.trunk = nn.Sequential(*layers_t)
        self.output_layer = nn.Linear(latent_dim, state_dim)

    def forward(self, state: torch.Tensor, t: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        b = self.branch(state)
        t_vec = self.trunk(torch.cat([t, alpha], dim=1))
        return self.output_layer(b * t_vec)
