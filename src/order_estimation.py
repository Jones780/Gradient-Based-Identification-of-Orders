"""Order-estimation routines and ablation baselines."""
from __future__ import annotations
import numpy as np
import torch

from .fractional import fractional_integral_torch, fractional_finite_difference
from .solver import solve_adams


def noisy_trajectory(system, true_alpha, ic, *, noise_level=0.05, h=0.02, T=5.0):
    t, clean = solve_adams(system, true_alpha, h, [0, T], ic)
    rms = np.sqrt(np.mean(clean**2, axis=1, keepdims=True))
    noisy = clean + noise_level * rms * np.random.randn(*clean.shape)
    return t, clean, noisy


def estimate_alpha_known_F(system, true_alpha, noise_level, ic, surrogate, device, initial_alpha=0.5, iters=300, lr=0.005, h=0.02, T=5.0):
    t_clean, u_clean, u_noisy = noisy_trajectory(system, true_alpha, ic, noise_level=noise_level, h=h, T=T)
    X_noisy = torch.tensor(u_noisy.T, dtype=torch.float32, device=device)
    t = torch.tensor(t_clean.reshape(-1, 1), dtype=torch.float32, device=device)
    target = torch.tensor((u_noisy - u_noisy[:, :1]).T, dtype=torch.float32, device=device)
    alpha_param = torch.nn.Parameter(torch.tensor([initial_alpha], dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam([alpha_param], lr=lr)
    surrogate.eval()
    history = []
    for _ in range(iters):
        optimizer.zero_grad(set_to_none=True)
        alpha = torch.clamp(alpha_param, 0.01, 0.99)
        a_vec = alpha * torch.ones_like(t)
        d_pred = surrogate(X_noisy, t, a_vec)
        I_alpha = fractional_integral_torch(d_pred, alpha, h, d_pred.shape[0])
        loss = torch.mean((I_alpha - target) ** 2)
        loss.backward()
        optimizer.step()
        history.append(float(alpha.detach().cpu().item()))
    return float(torch.clamp(alpha_param, 0.01, 0.99).detach().cpu().item()), history


def estimate_alpha_direct_rhs_grid(system, true_alpha, noise_level, ic, alpha_grid, h=0.02, T=5.0):
    t, clean, noisy = noisy_trajectory(system, true_alpha, ic, noise_level=noise_level, h=h, T=T)
    F_noisy = np.array([system(t[j], noisy[:, j]) for j in range(len(t))]).T
    F_tensor = torch.tensor(F_noisy.T, dtype=torch.float32)
    target = torch.tensor((noisy - noisy[:, :1]).T, dtype=torch.float32)
    best_alpha, best_loss = None, np.inf
    for a in alpha_grid:
        I = fractional_integral_torch(F_tensor, torch.tensor(float(a)), h, F_tensor.shape[0]).numpy()
        loss = float(np.mean((I - target.numpy()) ** 2))
        if loss < best_loss:
            best_loss, best_alpha = loss, float(a)
    return best_alpha


def estimate_alpha_ffd_grid(system, true_alpha, noise_level, ic, alpha_grid, h=0.02, T=5.0):
    t, clean, noisy = noisy_trajectory(system, true_alpha, ic, noise_level=noise_level, h=h, T=T)
    target = (noisy - noisy[:, :1]).T
    best_alpha, best_loss = None, np.inf
    for a in alpha_grid:
        d_ffd = fractional_finite_difference(noisy, float(a), h)
        d_tensor = torch.tensor(d_ffd.T, dtype=torch.float32)
        I = fractional_integral_torch(d_tensor, torch.tensor(float(a)), h, d_tensor.shape[0]).numpy()
        loss = float(np.mean((I - target) ** 2))
        if loss < best_loss:
            best_loss, best_alpha = loss, float(a)
    return best_alpha


def estimate_alpha_exact(system, true_alpha, ic, *, noise_level=0.05, h=0.02, T=5.0, iters=300, lr=0.005):
    """Oracle derivative baseline evaluated against the noisy observed trajectory."""
    t, clean = solve_adams(system, true_alpha, h, [0, T], ic)
    rms = np.sqrt(np.mean(clean**2, axis=1, keepdims=True))
    observed = clean + noise_level * rms * np.random.randn(*clean.shape)
    f_exact = np.array([system(t[j], clean[:, j]) for j in range(len(t))]).T
    f_tensor = torch.tensor(f_exact.T, dtype=torch.float32)
    target = torch.tensor((observed - observed[:, :1]).T, dtype=torch.float32)
    alpha_param = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.float32))
    optimizer = torch.optim.Adam([alpha_param], lr=lr)
    for _ in range(iters):
        optimizer.zero_grad(set_to_none=True)
        alpha = torch.clamp(alpha_param, 0.01, 0.99)
        I = fractional_integral_torch(f_tensor, alpha, h, f_tensor.shape[0])
        loss = torch.mean((I - target) ** 2)
        loss.backward()
        optimizer.step()
    return float(torch.clamp(alpha_param, 0.01, 0.99).item())
