import unittest

import numpy as np

from grayrecon.operators import (
    MaskedBlur, OperatorError, blur_adjoint, blur_forward, normalize_kernel)


class TestBlurAdjoint(unittest.TestCase):
    """The observation operator and its adjoint must exactly agree."""

    def _check_adjoint(self, shape, kernel, mask, seed=0):
        op = MaskedBlur(shape, kernel, mask)
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(shape)
        y = rng.standard_normal(shape)
        ax = op.forward(x)
        aty = op.adjoint(y)
        lhs = float(np.vdot(ax, y))
        rhs = float(np.vdot(x, aty))
        self.assertAlmostEqual(lhs, rhs, places=10)

    def test_adjoint_identity_kernel_full_mask(self):
        self._check_adjoint(
            (7, 9), np.array([[1.0]]), np.ones((7, 9)), seed=1)

    def test_adjoint_box_blur_full_mask(self):
        self._check_adjoint(
            (10, 8), np.ones((3, 3)) / 9.0, np.ones((10, 8)), seed=2)

    def test_adjoint_with_missing_pixels(self):
        rng = np.random.default_rng(3)
        mask = (rng.random((8, 8)) > 0.35).astype(float)
        kernel = np.array([[0.0, 1.0, 0.0],
                           [1.0, 2.0, 1.0],
                           [0.0, 1.0, 0.0]])
        self._check_adjoint((8, 8), kernel, mask, seed=4)

    def test_adjoint_five_by_five_kernel(self):
        rng = np.random.default_rng(5)
        kernel = rng.random((5, 5))
        mask = (rng.random((6, 11)) > 0.5).astype(float)
        self._check_adjoint((6, 11), kernel, mask, seed=6)

    def test_zero_padding_zeroes_boundary(self):
        # A single bright pixel far from the kernel support spreads as the
        # normalized kernel while the image outside stays zero on input.
        kernel = np.ones((3, 3)) / 9.0
        image = np.zeros((5, 5))
        image[2, 2] = 9.0
        out = blur_forward(image, kernel)
        self.assertAlmostEqual(float(out.sum()), 9.0, places=10)
        self.assertEqual(float(out[0, 0]), 0.0)
        self.assertAlmostEqual(float(out[2, 2]), 1.0, places=10)

    def test_blur_forward_adjoint_shapes(self):
        k = normalize_kernel(np.ones((3, 3)))
        x = np.zeros((6, 6))
        self.assertEqual(blur_forward(x, k).shape, (6, 6))
        self.assertEqual(blur_adjoint(x, k).shape, (6, 6))

    def test_matrix_representation_matches_forward(self):
        rng = np.random.default_rng(7)
        mask = (rng.random((6, 6)) > 0.3).astype(float)
        op = MaskedBlur((6, 6), np.ones((3, 3)), mask)
        a = op.to_matrix()
        x = rng.standard_normal((6, 6))
        np.testing.assert_allclose(a @ x.ravel(),
                                   op.forward(x).ravel(), atol=1e-12)
        # Matrix-level adjoint identity A^T.
        y = rng.standard_normal((6, 6))
        np.testing.assert_allclose(a.T @ y.ravel(),
                                   op.adjoint(y).ravel(), atol=1e-12)

    def test_spectral_norm_is_nonexpansive(self):
        rng = np.random.default_rng(8)
        op = MaskedBlur((8, 8), np.ones((3, 3)),
                        (rng.random((8, 8)) > 0.4).astype(float))
        self.assertLessEqual(op.spectral_norm(), 1.0 + 1e-9)


class TestOperatorValidation(unittest.TestCase):
    def test_even_kernel_rejected(self):
        with self.assertRaises(OperatorError):
            MaskedBlur((4, 4), np.ones((2, 2)), np.ones((4, 4)))

    def test_non_positive_kernel_sum_rejected(self):
        with self.assertRaises(OperatorError):
            normalize_kernel(np.zeros((3, 3)))
        with self.assertRaises(OperatorError):
            normalize_kernel(np.full((3, 3), -1.0))

    def test_nonfinite_kernel_rejected(self):
        k = np.ones((3, 3))
        k[0, 0] = np.nan
        with self.assertRaises(OperatorError):
            normalize_kernel(k)

    def test_mask_shape_conflict_rejected(self):
        with self.assertRaises(OperatorError):
            MaskedBlur((4, 4), np.ones((3, 3)), np.ones((5, 5)))

    def test_forward_shape_conflict_rejected(self):
        op = MaskedBlur((4, 4), np.ones((3, 3)), np.ones((4, 4)))
        with self.assertRaises(OperatorError):
            op.forward(np.zeros((3, 4)))


if __name__ == "__main__":
    unittest.main()
