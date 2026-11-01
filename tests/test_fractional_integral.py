"""Unit tests for differentiable fractional-integral quadrature."""
import unittest

import torch
from scipy.special import gamma

from src.fractional import fractional_integral_torch


class FractionalIntegralTests(unittest.TestCase):
    def test_product_integration_is_exact_for_constant_input(self):
        alpha = 0.63
        h = 0.02
        nt = 31
        values = torch.ones((nt, 2), dtype=torch.float64)
        result = fractional_integral_torch(values, alpha, h, nt, rule="product")
        times = torch.arange(nt, dtype=torch.float64) * h
        expected = times.pow(alpha) / gamma(alpha + 1.0)
        self.assertTrue(torch.allclose(result[:, 0], expected, atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.allclose(result[:, 1], expected, atol=1e-12, rtol=1e-12))

    def test_product_integration_has_finite_order_gradient(self):
        values = torch.linspace(0.2, 1.0, 40, dtype=torch.float64).reshape(-1, 1)
        alpha = torch.tensor(0.72, dtype=torch.float64, requires_grad=True)
        result = fractional_integral_torch(values, alpha, 0.02, 40, rule="product")
        result.square().sum().backward()
        self.assertIsNotNone(alpha.grad)
        self.assertTrue(torch.isfinite(alpha.grad).all().item())

    def test_rectangular_default_is_backward_compatible(self):
        values = torch.randn((12, 2), dtype=torch.float64)
        default = fractional_integral_torch(values, 0.8, 0.02, 12)
        explicit = fractional_integral_torch(values, 0.8, 0.02, 12, rule="rectangular")
        self.assertTrue(torch.equal(default, explicit))

    def test_invalid_rule_raises(self):
        values = torch.ones((5, 1), dtype=torch.float64)
        with self.assertRaises(ValueError):
            fractional_integral_torch(values, 0.5, 0.02, 5, rule="unknown")


if __name__ == "__main__":
    unittest.main()
