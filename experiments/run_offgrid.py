"""Paired off-grid comparison: projected gradient vs fine grid.

Both estimators use exactly the same noisy trajectory, the same
DeepONet surrogate, and the same integral reconstruction objective.
This isolates the effect of continuous projected-gradient refinement
from discrete grid search.
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
from scipy import stats

from src.data import set_seed
from src.models import DeepONetAlpha
from src.solver import solve_adams
from src.systems import linear_system
from src.fractional import fractional_integral_torch


def estimate_gradient_same_data(
    model: torch.nn.Module,
    X_noisy: np.ndarray,
    t: np.ndarray,
    h: float,
    initial_alpha: float,
    iters: int,
    lr: float,
    device: torch.device,
) -> float:
    """Estimate alpha by projected gradient on one fixed noisy trajectory."""
    X = torch.tensor(X_noisy, dtype=torch.float32, device=device)
    tt = torch.tensor(t.reshape(-1, 1), dtype=torch.float32, device=device)
    target = X - X[0:1, :]

    alpha_param = torch.nn.Parameter(
        torch.tensor([initial_alpha], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam([alpha_param], lr=lr)

    model.eval()
    for _ in range(iters):
        optimizer.zero_grad(set_to_none=True)

        alpha = torch.clamp(alpha_param, 0.01, 0.99)
        alpha_vec = alpha * torch.ones_like(tt)

        d_pred = model(X, tt, alpha_vec)
        I_alpha = fractional_integral_torch(
            d_pred, alpha, h, d_pred.shape[0]
        )
        loss = torch.mean((I_alpha - target) ** 2)

        loss.backward()
        optimizer.step()

    alpha_final = torch.clamp(alpha_param, 0.01, 0.99)
    return float(alpha_final.detach().cpu().item())


def estimate_grid_same_data(
    model: torch.nn.Module,
    X_noisy: np.ndarray,
    t: np.ndarray,
    h: float,
    alpha_grid: np.ndarray,
    device: torch.device,
) -> float:
    """Fine-grid estimation on the same noisy trajectory and same loss."""
    X = torch.tensor(X_noisy, dtype=torch.float32, device=device)
    tt = torch.tensor(t.reshape(-1, 1), dtype=torch.float32, device=device)
    target = X - X[0:1, :]

    best_alpha = None
    best_loss = np.inf

    model.eval()
    with torch.no_grad():
        for a in alpha_grid:
            alpha = torch.tensor(float(a), dtype=torch.float32, device=device)
            alpha_vec = torch.full_like(tt, float(a))

            d_pred = model(X, tt, alpha_vec)
            I_alpha = fractional_integral_torch(
                d_pred, alpha, h, d_pred.shape[0]
            )
            loss = torch.mean((I_alpha - target) ** 2)
            loss_value = float(loss.cpu().item())

            if loss_value < best_loss:
                best_loss = loss_value
                best_alpha = float(a)

    if best_alpha is None:
        raise RuntimeError("Grid search produced no candidate.")

    return best_alpha


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired off-grid comparison: gradient vs fine grid."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to models/deeponet_linear.pt",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--true-alpha", type=float, default=0.8537)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--h", type=float, default=0.02)
    parser.add_argument("--T", type=float, default=5.0)
    parser.add_argument("--initial-alpha", type=float, default=0.5)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.005)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"True alpha: {args.true_alpha}")
    print(f"Noise: {100 * args.noise:.1f}%")
    print("Fine-grid step: 0.001")
    print()

    model = DeepONetAlpha(state_dim=2).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    alpha_grid = np.arange(0.10, 0.9900001, 0.001)

    gradient_errors: list[float] = []
    grid_errors: list[float] = []

    print("=== Paired off-grid experiment ===")

    for rep in range(args.repeats):
        # Generate ONE initial condition and ONE noise realization.
        # The same noisy trajectory is passed to both estimators.
        ic = np.random.uniform(-2.0, 2.0, size=2)

        t, u_clean = solve_adams(
            linear_system,
            args.true_alpha,
            args.h,
            [0.0, args.T],
            ic,
        )

        rms = np.sqrt(np.mean(u_clean ** 2, axis=1, keepdims=True))
        u_noisy = (
            u_clean
            + args.noise * rms * np.random.randn(*u_clean.shape)
        )
        X_noisy = u_noisy.T

        est_gradient = estimate_gradient_same_data(
            model,
            X_noisy,
            t,
            args.h,
            args.initial_alpha,
            args.iters,
            args.lr,
            device,
        )
        est_grid = estimate_grid_same_data(
            model,
            X_noisy,
            t,
            args.h,
            alpha_grid,
            device,
        )

        err_gradient = abs(est_gradient - args.true_alpha)
        err_grid = abs(est_grid - args.true_alpha)

        gradient_errors.append(err_gradient)
        grid_errors.append(err_grid)

        print(
            f"Trial {rep + 1:2d}: "
            f"gradient alpha={est_gradient:.6f}, "
            f"grid alpha={est_grid:.6f}, "
            f"gradient error={err_gradient:.6f}, "
            f"grid error={err_grid:.6f}"
        )

    gradient_errors = np.asarray(gradient_errors, dtype=float)
    grid_errors = np.asarray(grid_errors, dtype=float)

    stat, p_value = stats.wilcoxon(
        gradient_errors,
        grid_errors,
        alternative="two-sided",
        zero_method="wilcox",
        method="auto",
    )

    print()
    print("=== Summary ===")
    print(
        f"Proposed gradient: MAE = {np.mean(gradient_errors):.6f} ± "
        f"{np.std(gradient_errors):.6f}"
    )
    print(
        f"Fine grid (Δα=0.001): MAE = {np.mean(grid_errors):.6f} ± "
        f"{np.std(grid_errors):.6f}"
    )
    print(
        f"Mean paired difference (grid - gradient) = "
        f"{np.mean(grid_errors - gradient_errors):.6f}"
    )
    print(f"Wilcoxon statistic = {stat:.6f}")
    print(f"Wilcoxon p-value = {p_value:.6f}")

    results = np.column_stack(
        [
            np.arange(1, args.repeats + 1),
            gradient_errors,
            grid_errors,
            grid_errors - gradient_errors,
        ]
    )
    np.savetxt(
        "offgrid_paired_results.csv",
        results,
        delimiter=",",
        header="trial,gradient_error,grid_error,grid_minus_gradient",
        comments="",
    )
    print("Saved: offgrid_paired_results.csv")


if __name__ == "__main__":
    main()
