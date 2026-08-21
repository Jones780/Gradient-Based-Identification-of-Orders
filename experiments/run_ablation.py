"""Run the derivative-source ablation for the linear benchmark."""
from __future__ import annotations
import argparse
import numpy as np
import torch

from src.data import set_seed
from src.models import DeepONetAlpha
from src.order_estimation import estimate_alpha_direct_rhs_grid, estimate_alpha_ffd_grid, estimate_alpha_exact, estimate_alpha_known_F
from src.systems import linear_system


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repeats", type=int, default=10)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepONetAlpha(state_dim=2).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    exact_err, direct_err, ffd_err = [], [], []
    for _ in range(args.repeats):
        ic = np.random.uniform(-2.0, 2.0, size=2)
        exact = estimate_alpha_exact(linear_system, 0.85, ic, noise_level=0.05)
        direct = estimate_alpha_direct_rhs_grid(linear_system, 0.85, 0.05, ic, np.arange(0.10, 0.99, 0.01))
        ffd = estimate_alpha_ffd_grid(linear_system, 0.85, 0.05, ic, np.arange(0.10, 0.99, 0.01))
        exact_err.append(abs(exact - 0.85))
        direct_err.append(abs(direct - 0.85))
        ffd_err.append(abs(ffd - 0.85))

    don_err = []
    for _ in range(args.repeats):
        ic = np.random.uniform(-2.0, 2.0, size=2)
        est, _ = estimate_alpha_known_F(linear_system, 0.85, 0.05, ic, model, device)
        don_err.append(abs(est - 0.85))

    for name, vals in [("Exact", exact_err), ("FFD", ffd_err), ("Direct RHS", direct_err), ("DeepONet", don_err)]:
        print(f"{name:12s}: {np.mean(vals):.4f} +- {np.std(vals):.4f}")


if __name__ == "__main__":
    main()
