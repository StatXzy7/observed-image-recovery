"""Finite-difference checks for the analytical gradients used by the solver."""

import unittest

import numpy as np

from geninv.operators import (
    adjoint_observation,
    forward_observation,
    gaussian_kernel,
    random_mask,
)
from geninv.generator import generator_forward
from geninv.synthetic import make_solvable_problem
from geninv.projection import project_onto_generator


class PixelGradientTests(unittest.TestCase):
    def test_residual_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(9)
        shape = (8, 10)
        kernel = gaussian_kernel(3, 0.9)
        mask = random_mask(shape, 0.35, seed=3)
        x = 0.6 * rng.standard_normal(shape)
        y = forward_observation(rng.standard_normal(shape), kernel, mask)

        def data_value(img):
            r = forward_observation(img, kernel, mask) - y
            return 0.5 * float(np.sum(r * r))

        analytic = adjoint_observation(
            forward_observation(x, kernel, mask) - y, kernel, mask
        )
        eps = 1e-6
        for _ in range(12):
            i = int(rng.integers(0, shape[0]))
            j = int(rng.integers(0, shape[1]))
            xp = x.copy(); xp[i, j] += eps
            xm = x.copy(); xm[i, j] -= eps
            numeric = (data_value(xp) - data_value(xm)) / (2 * eps)
            self.assertAlmostEqual(numeric, analytic[i, j], places=5)


class GeneratorGradientTests(unittest.TestCase):
    def setUp(self):
        self.problem, _ = make_solvable_problem(
            height=8, width=8, missing_fraction=0.2, seed=5,
            latent_dim=6, hidden_dim=18,
        )
        self.rng = np.random.default_rng(11)

    def test_generator_backprop_against_finite_differences(self):
        cfg = self.problem.generator
        kernel, mask = self.problem.kernel, self.problem.mask
        y = self.problem.observation
        z = 0.4 * self.rng.standard_normal(cfg.latent_dim)
        eps = 1e-6

        def value_and_grad(code):
            image, hidden, pre_output = generator_forward(code, cfg)
            residual = forward_observation(image, kernel, mask) - y
            value = 0.5 * float(np.sum(residual * residual))
            d_out = (
                adjoint_observation(residual, kernel, mask).ravel()
                * (1.0 - np.tanh(pre_output) ** 2)
            )
            d_hidden = d_out @ cfg.w2.T * (1.0 - hidden**2)
            grad = d_hidden @ cfg.w1.T
            return value, grad

        value, grad = value_and_grad(z)
        for k in range(cfg.latent_dim):
            zp = z.copy(); zp[k] += eps
            zm = z.copy(); zm[k] -= eps
            numeric = (value_and_grad(zp)[0] - value_and_grad(zm)[0]) / (
                2 * eps
            )
            self.assertAlmostEqual(numeric, grad[k], places=4)


if __name__ == "__main__":
    unittest.main()
