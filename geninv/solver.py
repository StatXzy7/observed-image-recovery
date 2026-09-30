"""Alternating solver: pixel residual step then generator-range projection."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import GeneratorConfig, SolverConfig
from .operators import adjoint_observation, forward_observation
from .projection import ProjectionResult, project_onto_generator

CONVERGED = "converged"
UNCONVERGED = "iterations_exhausted"
UNOBSERVABLE = "unobservable"
INVALID = "invalid_input"


@dataclass
class IterationRecord:
    iteration: int
    residual_norm: float
    relative_residual: float
    projection_objective: float
    offset_l2: float
    offset_l1: float
    inner_iterations: int
    inner_converged: bool


@dataclass
class SolverResult:
    reconstruction: np.ndarray
    generator_image: np.ndarray
    offset: np.ndarray
    latent: np.ndarray
    residual: np.ndarray
    residual_norm: float
    initial_residual_norm: float
    relative_residual: float
    status: str
    converged: bool
    inner_converged: bool
    iterations: int
    observed_pixel_count: int
    message: str
    history: list[IterationRecord] = field(default_factory=list)

    def records_as_dicts(self) -> list[dict]:
        return [
            {
                "iteration": rec.iteration,
                "residual_norm": rec.residual_norm,
                "relative_residual": rec.relative_residual,
                "projection_objective": rec.projection_objective,
                "offset_l2": rec.offset_l2,
                "offset_l1": rec.offset_l1,
                "inner_iterations": rec.inner_iterations,
                "inner_converged": rec.inner_converged,
            }
            for rec in self.history
        ]


def _validate_arrays(
    observation: np.ndarray, mask: np.ndarray, kernel: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observation = np.asarray(observation, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    kernel = np.asarray(kernel, dtype=np.float64)
    if observation.ndim != 2 or mask.ndim != 2:
        raise ValueError("observation and mask must be 2-D arrays")
    if observation.shape != mask.shape:
        raise ValueError("observation and mask shapes must match")
    if kernel.ndim != 2 or kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("kernel must be a 2-D array with odd dimensions")
    if not np.all(np.isfinite(observation)):
        raise ValueError("observation contains non-finite values")
    if not np.all(np.isfinite(kernel)):
        raise ValueError("kernel contains non-finite values")
    if not np.any(kernel):
        raise ValueError("kernel must not be all zeros")
    return observation, mask, kernel


def solve(
    observation: np.ndarray,
    kernel: np.ndarray,
    mask: np.ndarray,
    generator: GeneratorConfig,
    config: SolverConfig | None = None,
    initial_image: np.ndarray | None = None,
) -> SolverResult:
    """Run the alternating reconstruction.

    Every outer iteration performs (1) a pixel-space gradient step on the
    observation residual, and (2) an approximate generator-range projection
    whose target measures distance *after* the observation operator.

    Raises:
        ValueError: for non-finite inputs, mismatched shapes or illegal step
            sizes. When every pixel is missing a result with status
            ``unobservable`` is returned instead.
    """
    config = config or SolverConfig()
    config.validate()
    observation, mask, kernel = _validate_arrays(observation, mask, kernel)
    if observation.shape != generator.image_shape:
        raise ValueError("observation shape does not match generator config")

    observed_count = int(np.count_nonzero(mask))
    if observed_count == 0:
        zero = np.zeros_like(observation)
        return SolverResult(
            reconstruction=zero,
            generator_image=zero,
            offset=zero,
            latent=np.zeros(generator.latent_dim),
            residual=zero,
            residual_norm=0.0,
            initial_residual_norm=0.0,
            relative_residual=0.0,
            status=UNOBSERVABLE,
            converged=False,
            inner_converged=False,
            iterations=0,
            observed_pixel_count=0,
            message=(
                "no observed pixels: the inverse problem is unconstrained"
            ),
            history=[],
        )

    if initial_image is None:
        image = adjoint_observation(observation, kernel, mask)
        denom = float(np.count_nonzero(mask)) / observation.size
        image = image / max(denom, 1e-8)
        image = np.clip(image, -1.0, 1.0)
    else:
        image = np.asarray(initial_image, dtype=np.float64)
        if image.shape != observation.shape or not np.all(np.isfinite(image)):
            raise ValueError("initial_image must match shape and be finite")

    latent = np.zeros(generator.latent_dim, dtype=np.float64)
    residual = forward_observation(image, kernel, mask) - observation
    initial_residual_norm = float(np.linalg.norm(residual))
    observation_scale = max(float(np.linalg.norm(observation)), 1e-12)

    history: list[IterationRecord] = []
    prev_residual_norm = initial_residual_norm
    status = UNCONVERGED
    converged = False
    message = "outer iteration budget exhausted"
    iterations = 0
    last_inner: ProjectionResult | None = None

    for iterations in range(1, config.outer_max_iter + 1):
        # --- Phase 1: pixel-space observation-residual gradient step. ---
        grad = adjoint_observation(residual, kernel, mask)
        image = np.clip(image - config.pixel_step * grad, -1.0, 1.0)
        residual = forward_observation(image, kernel, mask) - observation

        # --- Phase 2: approximate projection in the observed domain. ---
        projection = project_onto_generator(
            target_image=image,
            observation=observation,
            kernel=kernel,
            mask=mask,
            config=generator,
            latent0=latent,
            sparse_penalty=config.sparse_penalty,
            tie_weight=config.tie_weight,
            latent_step=config.latent_step,
            hidden_step=config.hidden_step,
            tol=config.inner_tol,
            max_iter=config.inner_max_iter,
            min_iter=config.inner_min_iter,
        )
        latent = projection.latent
        image = projection.image
        residual = forward_observation(image, kernel, mask) - observation
        residual_norm = float(np.linalg.norm(residual))
        last_inner = projection

        record = IterationRecord(
            iteration=iterations,
            residual_norm=residual_norm,
            relative_residual=residual_norm / observation_scale,
            projection_objective=projection.objective,
            offset_l2=float(np.linalg.norm(projection.offset)),
            offset_l1=float(np.sum(np.abs(projection.offset))),
            inner_iterations=projection.iterations,
            inner_converged=projection.converged,
        )
        history.append(record)

        denom = max(prev_residual_norm, 1e-12)
        rel_change = abs(prev_residual_norm - residual_norm) / denom
        if iterations >= 2 and rel_change <= config.outer_tol:
            status = CONVERGED
            converged = True
            message = "outer residual reached tolerance"
            break
        prev_residual_norm = residual_norm
    else:
        status = UNCONVERGED
        converged = False
        message = "outer iteration budget exhausted"

    gen_image = (projection.image - projection.offset)
    inner_converged = bool(last_inner is not None and last_inner.converged)
    if converged and not inner_converged:
        message += "; inner projection did not reach tolerance"
    if not converged and last_inner is not None and not last_inner.converged:
        message += "; inner projection did not reach tolerance"

    return SolverResult(
        reconstruction=image,
        generator_image=np.clip(gen_image, -1.0, 1.0),
        offset=projection.offset,
        latent=latent,
        residual=residual,
        residual_norm=float(np.linalg.norm(residual)),
        initial_residual_norm=initial_residual_norm,
        relative_residual=float(np.linalg.norm(residual)) / observation_scale,
        status=status,
        converged=converged,
        inner_converged=inner_converged,
        iterations=iterations,
        observed_pixel_count=observed_count,
        message=message,
        history=history,
    )
