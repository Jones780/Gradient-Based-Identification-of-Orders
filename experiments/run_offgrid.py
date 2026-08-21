"""Compare projected-gradient refinement with discrete grid search in an off-grid case."""
from __future__ import annotations
import argparse
import numpy as np
import torch
from scipy.stats import wilcoxon

from src.data import set_seed
from src.models import DeepONetAlpha
from src.order_estimation import estimate_alpha_known_F, estimate_alpha_direct_rhs_grid
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
    true_alpha = 0.8537
    grid = np.arange(0.10, 0.99, 0.001)
    grad_err, grid_err = [], []
    for _ in range(args.repeats):
        ic = np.random.uniform(-2.0, 2.0, size=2)
        est, _ = estimate_alpha_known_F(linear_system, true_alpha, 0.05, ic, model, device)
        g = estimate_alpha_direct_rhs_grid(linear_system, true_alpha, 0.05, ic, grid)
        grad_err.append(abs(est - true_alpha))
        grid_err.append(abs(g - true_alpha))
    stat = wilcoxon(grad_err, grid_err, alternative="two-sided", method="auto")
    print(f"gradient MAE: {np.mean(grad_err):.4f} +- {np.std(grad_err):.4f}")
    print(f"grid MAE:     {np.mean(grid_err):.4f} +- {np.std(grid_err):.4f}")
    print(f"Wilcoxon p-value: {stat.pvalue:.6g}")


if __name__ == "__main__":
    main()
