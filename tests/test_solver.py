"""End-to-end behaviour tests for the alternating solver."""

import unittest

import numpy as np

from geninv.config import SolverConfig
from geninv.generator import generator_forward
from geninv.operators import forward_observation
from geninv.projection import project_onto_generator
from geninv.solver import (
    CONVERGED,
    UNOBSERVABLE,
    UNCONVERGED,
    solve,
)
from geninv.synthetic import make_solvable_problem


def _fast_config(**overrides):
    params = dict(
        pixel_step=0.8,
        sparse_penalty=1e-3,
        tie_weight=3.0,
        latent_step=0.3,
        hidden_step=1.5,
        inner_tol=1e-7,
        inner_max_iter=120,
        inner_min_iter=2,
        outer_tol=1e-9,
        outer_max_iter=60,
    )
    params.update(overrides)
    return SolverConfig(**params)


class SolvableSampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problem, cls.z_true = make_solvable_problem(
            height=9, width=9, missing_fraction=0.25, seed=42,
            latent_dim=10, hidden_dim=32,
        )

    def test_residual_and_image_improve(self):
        result = solve(
            self.problem.observation, self.problem.kernel, self.problem.mask,
            self.problem.generator, _fast_config(),
        )
        self.assertEqual(result.iterations, len(result.history))
        self.assertGreater(result.iterations, 1)
        self.assertLess(
            result.residual_norm, 0.25 * result.initial_residual_norm
        )
        rmse = np.sqrt(
            np.mean((result.reconstruction - self.problem.true_image) ** 2)
        )
        self.assertLess(rmse, 0.05)
        # Residual norms are recorded and (after warm-up) trend downward.
        residuals = [rec.residual_norm for rec in result.history]
        self.assertLess(residuals[-1], residuals[0])
        self.assertTrue(np.all(np.isfinite(result.reconstruction)))

    def test_records_contain_required_fields(self):
        result = solve(
            self.problem.observation, self.problem.kernel, self.problem.mask,
            self.problem.generator, _fast_config(outer_max_iter=8),
        )
        for rec in result.history:
            self.assertTrue(np.isfinite(rec.residual_norm))
            self.assertTrue(np.isfinite(rec.projection_objective))
            self.assertTrue(np.isfinite(rec.offset_l2))
            self.assertGreaterEqual(rec.inner_iterations, 1)
            self.assertLessEqual(rec.inner_iterations, 120)

    def test_projection_target_is_in_observed_domain(self):
        # The inner projection must reduce observed-domain distance rather
        # than merely pixel distance: perturb the image outside observed
        # support and confirm the projection follows observations.
        target = self.problem.true_image.copy()
        perturbation = np.zeros_like(target)
        perturbation[~self.problem.mask] = 0.5
        result = project_onto_generator(
            target_image=target + perturbation,
            observation=self.problem.observation,
            kernel=self.problem.kernel,
            mask=self.problem.mask,
            config=self.problem.generator,
            latent0=self.z_true,
            sparse_penalty=1e-3,
            latent_step=0.3,
            hidden_step=1.5,
            tol=1e-8,
            max_iter=400,
        )
        observed_after = forward_observation(
            result.image, self.problem.kernel, self.problem.mask
        )
        observed_target = forward_observation(
            target, self.problem.kernel, self.problem.mask
        )
        # Initialized from an offset that matches pixels exactly but the
        # target equals y, so observed fit must beat the perturbed start.
        start_fit = np.linalg.norm(
            forward_observation(
                target + perturbation, self.problem.kernel, self.problem.mask
            ) - self.problem.observation
        )
        self.assertLess(
            np.linalg.norm(observed_after - self.problem.observation),
            start_fit,
        )
        self.assertTrue(np.allclose(observed_target, self.problem.observation,
                                    atol=1e-10))

    def test_projection_fit_when_offset_is_free(self):
        # With a free zero-penalty offset the generator-plus-offset model can
        # fit the observations essentially exactly.
        result = project_onto_generator(
            target_image=self.problem.true_image,
            observation=self.problem.observation,
            kernel=self.problem.kernel,
            mask=self.problem.mask,
            config=self.problem.generator,
            latent0=self.z_true,
            sparse_penalty=0.0,
            latent_step=0.3,
            hidden_step=1.5,
            tol=1e-9,
            max_iter=500,
        )
        observed = forward_observation(
            result.image, self.problem.kernel, self.problem.mask
        )
        self.assertLess(
            np.linalg.norm(observed - self.problem.observation), 1e-4
        )


class SparsePenaltyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problem, cls.z_true = make_solvable_problem(
            height=8, width=8, missing_fraction=0.3, seed=7,
            latent_dim=8, hidden_dim=24,
        )

    def _run(self, penalty, outer_max_iter=20):
        return solve(
            self.problem.observation, self.problem.kernel, self.problem.mask,
            self.problem.generator,
            _fast_config(sparse_penalty=penalty, outer_max_iter=outer_max_iter),
        )

    def test_larger_penalty_shrinks_offset(self):
        weak = self._run(1e-4)
        strong = self._run(5e-2)
        self.assertLess(
            np.sum(np.abs(strong.offset)),
            np.sum(np.abs(weak.offset)),
        )
        self.assertLess(np.linalg.norm(strong.offset), np.linalg.norm(weak.offset))

    def test_zero_penalty_disables_constraint_and_keeps_offset_free(self):
        # Direct proximal check: with lambda=0 the prox is the identity.
        from geninv.projection import _soft_threshold

        values = np.array([-0.5, -0.01, 0.0, 0.02, 0.7])
        self.assertTrue(np.allclose(_soft_threshold(values, 0.0), values))
        result = self._run(0.0)
        for rec in result.history:
            self.assertTrue(np.isfinite(rec.offset_l2))
        # Free offset must not be forced to zero (problem is ill-conditioned
        # at unobserved pixels), while strong penalty is much sparser.
        strong = self._run(5e-2)
        self.assertGreaterEqual(
            np.sum(np.abs(result.offset)) / result.offset.size,
            np.sum(np.abs(strong.offset)) / strong.offset.size,
        )


class StoppingAndStatusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problem, cls.z_true = make_solvable_problem(
            height=8, width=8, missing_fraction=0.2, seed=11,
            latent_dim=8, hidden_dim=24,
        )

    def test_budget_exhaustion_reports_unconverged(self):
        config = _fast_config(outer_max_iter=3)
        result = solve(
            self.problem.observation, self.problem.kernel, self.problem.mask,
            self.problem.generator, config,
        )
        self.assertEqual(result.iterations, 3)
        self.assertEqual(result.status, UNCONVERGED)
        self.assertFalse(result.converged)

    def test_inner_budget_exhaustion_keeps_finite_result(self):
        result = project_onto_generator(
            target_image=np.zeros_like(self.problem.true_image),
            observation=self.problem.observation,
            kernel=self.problem.kernel,
            mask=self.problem.mask,
            config=self.problem.generator,
            latent0=np.zeros(self.problem.generator.latent_dim),
            sparse_penalty=1e-3,
            tol=1e-12,
            max_iter=3,
            min_iter=5,
        )
        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 3)
        self.assertIn("budget", result.message)
        self.assertTrue(np.all(np.isfinite(result.image)))
        self.assertTrue(np.all(np.isfinite(result.latent)))
        self.assertTrue(np.isfinite(result.objective))

    def test_all_missing_is_unobservable(self):
        mask = np.zeros(self.problem.true_image.shape, dtype=bool)
        result = solve(
            self.problem.observation, self.problem.kernel, mask,
            self.problem.generator, _fast_config(),
        )
        self.assertEqual(result.status, UNOBSERVABLE)
        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 0)
        self.assertEqual(result.observed_pixel_count, 0)
        self.assertIn("unconstrained", result.message)

    def test_shape_mismatch_rejected(self):
        kernel = self.problem.kernel
        bad_mask = np.ones((4, 4), dtype=bool)
        with self.assertRaises(ValueError):
            solve(
                self.problem.observation, kernel, bad_mask,
                self.problem.generator,
            )
        bad_obs = np.zeros((5, 5))
        good_mask = np.ones((5, 5), dtype=bool)
        with self.assertRaises(ValueError):
            solve(bad_obs, kernel, good_mask, self.problem.generator)

    def test_non_finite_inputs_rejected(self):
        y = self.problem.observation.copy()
        y[0, 0] = np.nan
        mask = np.ones_like(y, dtype=bool)
        with self.assertRaises(ValueError):
            solve(y, self.problem.kernel, mask, self.problem.generator)
        y2 = self.problem.observation.copy()
        y2[0, 0] = np.inf
        with self.assertRaises(ValueError):
            solve(y2, self.problem.kernel, mask, self.problem.generator)
        bad_kernel = self.problem.kernel.copy()
        bad_kernel[0, 0] = np.nan
        with self.assertRaises(ValueError):
            solve(self.problem.observation, bad_kernel, mask,
                  self.problem.generator)

    def test_illegal_step_sizes_rejected(self):
        for kwargs in (
            {"pixel_step": 0.0},
            {"pixel_step": -1.0},
            {"pixel_step": float("nan")},
            {"latent_step": 0.0},
            {"hidden_step": -0.5},
            {"sparse_penalty": -0.01},
            {"inner_tol": 0.0},
            {"outer_max_iter": 0},
            {"inner_max_iter": 0},
        ):
            with self.assertRaises(ValueError):
                SolverConfig(**kwargs).validate()

    def test_reconstruction_stays_in_valid_range(self):
        result = solve(
            self.problem.observation, self.problem.kernel, self.problem.mask,
            self.problem.generator, _fast_config(),
        )
        self.assertTrue(np.all(result.reconstruction >= -1.0 - 1e-12))
        self.assertTrue(np.all(result.reconstruction <= 1.0 + 1e-12))


if __name__ == "__main__":
    unittest.main()
