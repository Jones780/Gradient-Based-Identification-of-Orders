"""Fractional-integral and derivative utilities."""
from __future__ import annotations
import numpy as np
import torch
from scipy.special import gamma


def fractional_integral_torch(F: torch.Tensor, alpha: torch.Tensor | float, h: float, Nt: int) -> torch.Tensor:
    """Differentiable left-sided fractional integral used by the experiments."""
    if not torch.is_tensor(alpha):
        alpha = torch.tensor(float(alpha), dtype=F.dtype, device=F.device)
    n = torch.arange(0, Nt, dtype=F.dtype, device=F.device).unsqueeze(1)
    j = torch.arange(0, Nt, dtype=F.dtype, device=F.device).unsqueeze(0)
    diff = (n - j) * h
    mask = (j < n).to(F.dtype)
    safe_diff = torch.clamp(diff, min=1e-12)
    weight = safe_diff ** (alpha - 1.0) * (h / torch.exp(torch.lgamma(alpha)))
    weight = weight * mask
    return weight @ F


def fractional_finite_difference(X: np.ndarray, alpha: float, h: float) -> np.ndarray:
    """Finite-difference approximation used as the paper baseline."""
    dim, m = X.shape
    DX = np.zeros_like(X)
    a = np.zeros(m)
    a[0] = 1.0
    for k in range(1, m - 1):
        a[k] = (k + 1) ** (1 - alpha) - 2 * k ** (1 - alpha) + (k - 1) ** (1 - alpha)
    if m > 1:
        a[m - 1] = (m - 1) ** (1 - alpha) - m ** (1 - alpha)
    factor = 1.0 / (h**alpha * gamma(2 - alpha))
    for r in range(dim):
        for n in range(m):
            DX[r, n] = factor * np.dot(a[: n + 1][::-1], X[r, : n + 1])
    return DX
