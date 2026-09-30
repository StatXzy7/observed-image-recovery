"""Adjoint-consistency tests for blur and the composed observation map."""

import unittest

import numpy as np

from geninv.operators import (
    blur,
    blur_adjoint,
    forward_observation,
    adjoint_observation,
    gaussian_kernel,
    random_mask,
)


class OperatorAdjointTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(123)
        self.shape = (9, 11)

    def _check_inner_product(self, forward, adjoint, shape_a, shape_b):
        x = self.rng.standard_normal(shape_a)
        y = self.rng.standard_normal(shape_b)
        lhs = float(np.vdot(forward(x), y))
        rhs = float(np.vdot(x, adjoint(y)))
        self.assertAlmostEqual(lhs, rhs, places=11)

    def test_blur_adjoint_gaussian(self):
        kernel = gaussian_kernel(3, sigma=0.9)
        for _ in range(5):
            self._check_inner_product(
                lambda x: blur(x, kernel),
                lambda y: blur_adjoint(y, kernel),
                self.shape,
                self.shape,
            )

    def test_blur_adjoint_asymmetric_and_rectangular(self):
        kernel = np.array(
            [[0.0, 0.2, 0.0], [0.1, 0.35, 0.05], [0.0, 0.25, 0.05]]
        )
        shape = (7, 5)
        self._check_inner_product(
            lambda x: blur(x, kernel),
            lambda y: blur_adjoint(y, kernel),
            shape,
            shape,
        )

    def test_blur_adjoint_five_tap_kernel(self):
        kernel = gaussian_kernel(5, sigma=1.2)
        self._check_inner_product(
            lambda x: blur(x, kernel),
            lambda y: blur_adjoint(y, kernel),
            self.shape,
            self.shape,
        )

    def test_zero_padding_actually_zeroes_boundaries(self):
        image = np.ones((5, 5))
        # Tap at the kernel bottom row (a=2, b=1) reads x[i-1, j]; at the
        # top image row it reaches beyond the boundary and must contribute
        # zero under zero-filled boundary conditions.
        kernel = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        out = blur(image, kernel)
        self.assertEqual(out[0, 2], 0.0)   # reads x[-1, 2] -> zero fill
        self.assertEqual(out[2, 2], 1.0)   # interior reads x[1, 2] = 1

    def test_observation_adjoint_with_mask(self):
        kernel = gaussian_kernel(3, 0.8)
        mask = random_mask(self.shape, missing_fraction=0.4, seed=7)
        for _ in range(5):
            x = self.rng.standard_normal(self.shape)
            y = self.rng.standard_normal(self.shape)
            lhs = np.vdot(forward_observation(x, kernel, mask), y)
            rhs = np.vdot(x, adjoint_observation(y, kernel, mask))
            self.assertAlmostEqual(float(lhs), float(rhs), places=11)

    def test_observation_adjoint_matches_direct_masking(self):
        kernel = gaussian_kernel(3, 0.8)
        mask = random_mask(self.shape, missing_fraction=0.4, seed=7)
        y = self.rng.standard_normal(self.shape)
        direct = blur_adjoint(mask.astype(float) * y, kernel)
        self.assertTrue(np.allclose(direct, adjoint_observation(y, kernel, mask)))

    def test_kernel_normalised(self):
        kernel = gaussian_kernel(5, sigma=1.2)
        self.assertAlmostEqual(float(kernel.sum()), 1.0, places=12)
        self.assertTrue(np.all(kernel >= 0.0))


if __name__ == "__main__":
    unittest.main()
