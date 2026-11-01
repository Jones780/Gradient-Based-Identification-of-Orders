"""Fractional-integral and derivative utilities."""
from __future__ import annotations

import numpy as np
import torch
from scipy.special import gamma


def fractional_integral_torch(
    F: torch.Tensor,
    alpha: torch.Tensor | float,
    h: float,
    Nt: int,
    rule: str = "rectangular",
) -> torch.Tensor:
    """Differentiable left-sided fractional integral.

    F has shape (Nt, state_dim). A scalar tensor alpha requiring gradients
    is supported for continuous order optimization. The rule is either
    'rectangular' (the original left-rectangle rule) or 'product'
    (piecewise-constant product integration).

    Product integration uses
      h**alpha / Gamma(alpha+1) *
      sum_j [(n-j)**alpha - (n-j-1)**alpha] F_j, j < n.
    It is exact at grid points for constant input data.
    """
    if not torch.is_tensor(F):
        raise TypeError("F must be a torch.Tensor.")
    if F.ndim != 2:
        raise ValueError(f"F must have shape (Nt, state_dim); got {tuple(F.shape)}.")
    if not isinstance(Nt, int) or Nt < 1 or Nt > F.shape[0]:
        raise ValueError(f"Nt must be an integer in [1, {F.shape[0]}].")
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be a finite positive number.")

    if not torch.is_tensor(alpha):
        alpha = torch.tensor(float(alpha), dtype=F.dtype, device=F.device)
    else:
        alpha = alpha.to(dtype=F.dtype, device=F.device)
    if alpha.numel() != 1:
        raise ValueError("alpha must be a scalar.")

    n = torch.arange(Nt, dtype=F.dtype, device=F.device).unsqueeze(1)
    j = torch.arange(Nt, dtype=F.dtype, device=F.device).unsqueeze(0)
    lag = n - j
    mask = (lag > 0).to(F.dtype)
    safe_lag = torch.clamp(lag, min=1.0)

    h_tensor = torch.tensor(float(h), dtype=F.dtype, device=F.device)
    if rule == "rectangular":
        weights = (
            h_tensor.pow(alpha)
            * safe_lag.pow(alpha - 1.0)
            / torch.exp(torch.lgamma(alpha))
        )
    elif rule == "product":
        previous_lag = torch.clamp(safe_lag - 1.0, min=1.0)
        increments = torch.where(
            safe_lag == 1.0,
            torch.ones_like(safe_lag),
            safe_lag.pow(alpha) - previous_lag.pow(alpha),
        )
        weights = (
            h_tensor.pow(alpha)
            * increments
            / torch.exp(torch.lgamma(alpha + 1.0))
        )
    else:
        raise ValueError("rule must be either 'rectangular' or 'product'.")

    weights = weights * mask
    return weights @ F[:Nt]


def fractional_finite_difference(X: np.ndarray, alpha: float, h: float) -> np.ndarray:
    """Finite-difference approximation used by the paper baseline.

    Center a trajectory with a nonzero initial condition before calling this
    function, e.g. pass X - X[:, :1].
    """
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
