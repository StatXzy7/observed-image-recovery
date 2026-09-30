import unittest

import numpy as np

from grayrecon.data import box_kernel, random_mask
from grayrecon.generator import Generator
from grayrecon.operators import MaskedBlur
from grayrecon.projection import (
    ProjectionError, observation_objective, range_project, soft_threshold,
    split_objective)


def build_fixture(seed=0, shape=(6, 6), latent_dim=4, hidden_dim=7,
                  missing=0.3):
    rng = np.random.default_rng(seed)
    n = shape[0] * shape[1]
    w1 = rng.standard_normal((latent_dim, hidden_dim))
    b1 = rng.standard_normal(hidden_dim) * 0.2
    w2 = rng.standard_normal((hidden_dim, n))
    b2 = rng.standard_normal(n) * 0.1
    gen = Generator(w1, b1, w2, b2, shape)
    mask = random_mask(shape, missing, seed=seed + 100).astype(float)
    op = MaskedBlur(shape, box_kernel(3), mask)
    y = op(np.tanh(rng.standard_normal(shape)))
    return rng, gen, op, y


class TestSoftThreshold(unittest.TestCase):
    def test_thresholds_small_entries_to_zero(self):
        v = np.array([-0.5, -0.1, 0.0, 0.1, 0.5])
        np.testing.assert_allclose(
            soft_threshold(v, 0.2),
            np.array([-0.3, 0.0, 0.0, 0.0, 0.3]))

    def test_zero_threshold_is_identity(self):
        v = np.linspace(-1, 1, 9)
        np.testing.assert_array_equal(soft_threshold(v, 0.0), v)


class TestProjectionGradients(unittest.TestCase):
    """Finite-difference checks for all three layer-wise block gradients."""

    def setUp(self):
        self.rng, self.gen, self.op, self.y = build_fixture(seed=11)
        n = self.gen.output_dim
        self.z = self.rng.standard_normal(self.gen.latent_dim) * 0.3
        self.h = self.rng.standard_normal(self.gen.hidden_dim) * 0.3
        self.s = self.rng.standard_normal(n) * 0.1
        self.rho = 1.3

    def _fd(self, name):
        eps = 1e-7
        gen, op, y, rho = self.gen, self.op, self.y, self.rho

        def obj(zz, hh, ss):
            return split_objective(op, y, gen, zz, hh, ss, 0.0, rho)

        if name == "z":
            t1 = np.tanh(self.z @ gen.w1 + gen.b1)
            analytic = (-rho * (self.h - t1) * (1.0 - t1 ** 2)) @ gen.w1.T
            var = self.z
        elif name == "h":
            t1 = np.tanh(self.z @ gen.w1 + gen.b1)
            t2 = np.tanh(self.h @ gen.w2 + gen.b2)
            x = (t2 + self.s).reshape(gen.image_shape)
            grad_t2 = op.adjoint(op.forward(x) - y).ravel()
            analytic = (grad_t2 * (1.0 - t2 ** 2)) @ gen.w2.T \
                + rho * (self.h - t1)
            var = self.h
        else:
            t2 = np.tanh(self.h @ gen.w2 + gen.b2)
            x = (t2 + self.s).reshape(gen.image_shape)
            analytic = op.adjoint(op.forward(x) - y).ravel()
            var = self.s

        numeric = np.zeros_like(var)
        for i in range(var.size):
            vp, vm = var.copy(), var.copy()
            vp[i] += eps
            vm[i] -= eps
            if name == "z":
                numeric[i] = (obj(vp, self.h, self.s)
                              - obj(vm, self.h, self.s)) / (2 * eps)
            elif name == "h":
                numeric[i] = (obj(self.z, vp, self.s)
                              - obj(self.z, vm, self.s)) / (2 * eps)
            else:
                numeric[i] = (obj(self.z, self.h, vp)
                              - obj(self.z, self.h, vm)) / (2 * eps)
        err = np.max(np.abs(analytic - numeric))
        self.assertLess(err, 1e-6)

    def test_z_gradient_finite_difference(self):
        self._fd("z")

    def test_h_gradient_finite_difference(self):
        self._fd("h")

    def test_s_gradient_finite_difference(self):
        self._fd("s")


class TestProjectionBehavior(unittest.TestCase):
    def test_data_objective_is_observation_space(self):
        rng, gen, op, y = build_fixture(seed=3)
        image = rng.standard_normal(op.shape)
        expected = 0.5 * float(
            np.dot((op(image) - y).ravel(), (op(image) - y).ravel()))
        self.assertAlmostEqual(
            observation_objective(op, y, image), expected, places=12)

    def test_descent_decreases_observation_objective(self):
        rng, gen, op, y = build_fixture(seed=5)
        before = observation_objective(op, y, np.zeros(op.shape))
        result = range_project(op, y, gen, lam=0.0, inner_max_iters=60,
                               tol=0.0)
        self.assertLess(result.data_objective, before)

    def test_inner_iteration_cap_respected(self):
        rng, gen, op, y = build_fixture(seed=7)
        result = range_project(op, y, gen, inner_max_iters=5, tol=0.0)
        self.assertLessEqual(result.inner_iters, 5)
        self.assertEqual(len(result.history), 6)
        self.assertFalse(result.converged)

    def test_tolerance_stops_early(self):
        rng, gen, op, y = build_fixture(seed=9)
        result = range_project(op, y, gen, inner_max_iters=500,
                               tol=1e-4)
        self.assertTrue(result.converged)
        self.assertLess(result.inner_iters, 500)

    def test_l1_penalty_shrinks_offset(self):
        rng, gen, op, y = build_fixture(seed=13)
        unpenalized = range_project(
            op, y, gen, lam=0.0, inner_max_iters=120, tol=0.0)
        penalized = range_project(
            op, y, gen, lam=0.2, inner_max_iters=120, tol=0.0)
        self.assertGreater(unpenalized.offset_l1, 0.0)
        self.assertLess(penalized.offset_l1, unpenalized.offset_l1)

    def test_zero_lam_disables_sparsity_constraint(self):
        rng, gen, op, y = build_fixture(seed=17)
        result = range_project(op, y, gen, lam=0.0, inner_max_iters=40)
        # With no penalty the optimum offset is a dense vector in general.
        self.assertGreater(np.count_nonzero(result.s),
                           0.5 * result.s.size)

    def test_recovers_range_point_with_full_observation(self):
        rng = np.random.default_rng(21)
        shape = (6, 6)
        d, m, n = 3, 8, 36
        w1 = rng.standard_normal((d, m)) / np.sqrt(d)
        b1 = rng.standard_normal(m) * 0.1
        w2 = rng.standard_normal((m, n)) / np.sqrt(m)
        b2 = rng.standard_normal(n) * 0.05
        gen = Generator(w1, b1, w2, b2, shape)
        z_star = rng.standard_normal(d) * 0.4
        x_star, _, _, _ = gen.forward(z_star)
        op = MaskedBlur(shape, box_kernel(3), np.ones(shape))
        y = op(x_star)
        result = range_project(op, y, gen, lam=1e-2, rho=1.0,
                               inner_max_iters=2000, tol=0.0)
        self.assertLess(np.linalg.norm(op(result.x) - y), 1e-4)


class TestProjectionValidation(unittest.TestCase):
    def test_negative_lam_rejected(self):
        _, gen, op, y = build_fixture()
        with self.assertRaises(ProjectionError):
            range_project(op, y, gen, lam=-1.0)

    def test_non_positive_rho_rejected(self):
        _, gen, op, y = build_fixture()
        with self.assertRaises(ProjectionError):
            range_project(op, y, gen, rho=0.0)

    def test_zero_inner_iters_rejected(self):
        _, gen, op, y = build_fixture()
        with self.assertRaises(ProjectionError):
            range_project(op, y, gen, inner_max_iters=0)

    def test_shape_conflict_rejected(self):
        _, gen, op, y = build_fixture()
        with self.assertRaises(ProjectionError):
            range_project(op, y[:, :-1], gen)

    def test_nonfinite_y_rejected(self):
        _, gen, op, y = build_fixture()
        y_bad = y.copy()
        y_bad[0, 0] = np.inf
        with self.assertRaises(ProjectionError):
            range_project(op, y_bad, gen)


if __name__ == "__main__":
    unittest.main()
