"""Generate empirical reconstruction-loss landscapes for several true orders."""
from __future__ import annotations
import argparse
import os
import numpy as np
import torch

from src.data import set_seed
from src.fractional import fractional_integral_torch
from src.models import DeepONetAlpha
from src.order_estimation import noisy_trajectory
from src.systems import linear_system


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="figures/loss_landscapes.png")
    args = p.parse_args()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepONetAlpha(state_dim=2).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    import matplotlib.pyplot as plt
    true_orders = [0.65, 0.75, 0.85, 0.95]
    alpha_grid = np.linspace(0.05, 0.99, 100)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, true_alpha in zip(axes.ravel(), true_orders):
        ic = np.array([0.5, -0.5])
        t, clean, noisy = noisy_trajectory(linear_system, true_alpha, ic, noise_level=0.05)
        X = torch.tensor(noisy.T, dtype=torch.float32, device=device)
        tt = torch.tensor(t.reshape(-1, 1), dtype=torch.float32, device=device)
        target = torch.tensor((noisy - noisy[:, :1]).T, dtype=torch.float32, device=device)
        losses = []
        with torch.no_grad():
            for a in alpha_grid:
                av = torch.full((len(t), 1), float(a), device=device)
                d = model(X, tt, av)
                I = fractional_integral_torch(d, torch.tensor(float(a), device=device), 0.02, len(t))
                losses.append(float(torch.mean((I - target) ** 2).cpu()))
        ax.plot(alpha_grid, losses)
        ax.axvline(true_alpha, linestyle="--")
        ax.set_title(rf"True $\alpha={true_alpha}$")
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("Reconstruction loss")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
