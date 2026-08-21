"""Numerical solver used to generate synthetic fractional trajectories."""
from __future__ import annotations
import numpy as np
from scipy.optimize import fsolve
from scipy.special import gamma


def _u_step(n, h, alpha, t_vals, u_vals, f_vals, f, coef):
    s = ((n - 1) ** (alpha + 1) - (n - alpha - 1) * n**alpha) * f_vals[:, 0]
    for j in range(1, n):
        s += ((n - j + 1) ** (alpha + 1) - 2.0 * (n - j) ** (alpha + 1) + (n - j - 1) ** (alpha + 1)) * f_vals[:, j]

    def equation(x):
        return u_vals[:, 0] + coef * s + coef * f(t_vals[n], x) - x

    return fsolve(equation, x0=u_vals[:, n - 1])


def solve_adams(f, alpha, h, interval, init_vals):
    """Generate a trajectory using the fractional Adams predictor/corrector form used in the study."""
    t_vals = np.arange(interval[0], interval[1] + h, h)
    dim = len(init_vals)
    nodes = len(t_vals)
    u_vals = np.zeros((dim, nodes), dtype=float)
    f_vals = np.zeros((dim, nodes), dtype=float)
    u_vals[:, 0] = np.asarray(init_vals, dtype=float)
    f_vals[:, 0] = f(t_vals[0], u_vals[:, 0])
    coef = h**alpha / gamma(alpha + 2.0)
    for i in range(1, nodes):
        u_vals[:, i] = _u_step(i, h, alpha, t_vals, u_vals, f_vals, f, coef)
        f_vals[:, i] = f(t_vals[i], u_vals[:, i])
    return t_vals, u_vals
