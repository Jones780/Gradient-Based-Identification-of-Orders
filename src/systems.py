"""Fractional dynamical systems used in the paper experiments."""
from __future__ import annotations
import numpy as np


def linear_system(t: float, u: np.ndarray) -> np.ndarray:
    x, y = u[0], u[1]
    return np.array([-x - 2.0 * y, 2.0 * x - y], dtype=float)


def nonlinear_system(t: float, u: np.ndarray) -> np.ndarray:
    x, y = u[0], u[1]
    return np.array([1.0 - x * y**2, x**3 - y], dtype=float)


def brusselator_system(t: float, u: np.ndarray, A: float = 1.0, B: float = 3.0) -> np.ndarray:
    x, y = u[0], u[1]
    return np.array([A - (B + 1.0) * x + x**2 * y, B * x - x**2 * y], dtype=float)
