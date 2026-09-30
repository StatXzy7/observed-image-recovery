"""Alternating inverse-problem solver on CPU (small grayscale images).

Each outer iteration performs

1. a pixel-space residual gradient step
   ``x <- x - alpha * A^* (A x - y)``, then
2. an approximate projection onto the generative range, whose target
   compares distances **after** applying the observation operator.

The solver supports tolerance-based stopping, an iteration budget,
fixed-seed synthetic data, explicit unobservable (fully missing) problems
and strict rejection of non-finite inputs, shape conflicts and illegal
step sizes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import make_problem
from .generator import Generator, GeneratorError, default_weights
from .operators import MaskedBlur, OperatorError
from .projection import ProjectionError, range_project


class SolveError(ValueError):
    """Raised when the problem or its parameters are invalid."""


class Status:
    CONVERGED = "converged"
    MAX_ITERS = "max_iters_reached"
    UNOBSERVABLE = "unobservable"
    INNER_NOT_CONVERGED = "inner_not_converged"


@dataclass
class IterationRecord:
    iteration: int
    residual_norm: float       # ||A x - y||_2 at the projected iterate
    gradient_residual_norm: float  # ||A x - y||_2 right after the gradient step
    projection_objective: float  # 1/2 ||A x_proj - y||^2 (observation space)
    split_objective: float      # projection objective incl. penalties
    offset_l1: float
    inner_iters: int
    inner_converged: bool


@dataclass
class SolveResult:
    x: np.ndarray
    observation: np.ndarray
    mask: np.ndarray
    operator: MaskedBlur
    generator: Generator
    truth: np.ndarray | None
    status: str
    iterations: int
    records: list = field(default_factory=list)
    initial_residual_norm: float = 0.0
    final_residual_norm: float = 0.0
    config: dict = field(default_factory=dict)
    message: str = ""


def _as_finite_array(name, value, shape=None):
    arr = np.asarray(value, dtype=np.float64)
    if shape is not None and arr.shape != tuple(shape):
        raise SolveError(f"{name} shape {arr.shape} does not match {tuple(shape)}")
    if not np.all(np.isfinite(arr)):
        raise SolveError(f"{name} must contain finite values")
    return arr


def _validate_positive_scalar(name, value, allow_zero=False):
    value = float(value)
    if not np.isfinite(value) or value < 0.0 or (value == 0.0 and not allow_zero):
        comparator = "positive" if not allow_zero else "non-negative"
        raise SolveError(f"{name} must be a finite {comparator} number")
    return value


def solve_inverse(operator: MaskedBlur,
                  observation: np.ndarray,
                  generator: Generator,
                  *,
                  pixel_step: float = 0.5,
                  max_iters: int = 30,
                  tol: float = 1e-5,
                  lam: float = 0.0,
                  rho: float = 1.0,
                  inner_max_iters: int = 50,
                  inner_tol: float = 1e-7,
                  stall_tol: float = 1e-6,
                  stall_patience: int = 3,
                  x0: np.ndarray | None = None,
                  truth: np.ndarray | None = None):
    """Run the alternating gradient-step / generative-projection solver.

    Raises
    ------
    SolveError
        For non-finite inputs, shape conflicts or illegal step sizes.
    """
    if not isinstance(operator, MaskedBlur):
        raise SolveError("operator must be a MaskedBlur instance")
    if not isinstance(generator, Generator):
        raise SolveError("generator must be a Generator instance")
    if generator.image_shape != operator.shape:
        raise SolveError(
            f"generator image shape {generator.image_shape} does not match "
            f"operator shape {operator.shape}")

    y = _as_finite_array("observation", observation, operator.shape)
    y = operator.mask * y  # missing entries carry no information
    max_iters = int(max_iters)
    if max_iters < 0:
        raise SolveError("max_iters must be >= 0")
    tol = _validate_positive_scalar("tol", tol, allow_zero=True)
    lam = _validate_positive_scalar("lam", lam, allow_zero=True)
    rho = _validate_positive_scalar("rho", rho)
    inner_max_iters = int(inner_max_iters)
    if inner_max_iters < 1:
        raise SolveError("inner_max_iters must be >= 1")
    inner_tol = _validate_positive_scalar("inner_tol", inner_tol,
                                          allow_zero=True)
    stall_tol = _validate_positive_scalar("stall_tol", stall_tol,
                                          allow_zero=True)
    stall_patience = int(stall_patience)
    if stall_patience < 1:
        raise SolveError("stall_patience must be >= 1")

    # Fully missing data: the observation operator has no rows; the problem
    # is unconstrained and must be reported rather than "solved".
    if operator.n_observed == 0:
        zero = np.zeros(operator.shape, dtype=np.float64)
        return SolveResult(
            x=zero, observation=y, mask=operator.mask.astype(bool),
            operator=operator, generator=generator, truth=truth,
            status=Status.UNOBSERVABLE, iterations=0,
            initial_residual_norm=0.0, final_residual_norm=0.0,
            message=("all pixels are missing; the problem is unobservable "
                     "and x is unconstrained"),
        )

    operator_norm = operator.spectral_norm()
    alpha = float(pixel_step)
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise SolveError("pixel_step must be a positive finite number")
    upper = 2.0 / (operator_norm ** 2 + 1e-30)
    if alpha >= upper:
        # Stability for x <- x - alpha A*(Ax - y) requires 0 < a < 2/||A||^2.
        raise SolveError(
            f"pixel_step={alpha:g} is illegal; stability requires "
            f"0 < pixel_step < {upper:.6g} (2 / ||A||^2)")

    if x0 is None:
        # Minimum-norm data fit, zero on missing pixels.
        x = operator.adjoint(y)
    else:
        x = _as_finite_array("x0", x0, operator.shape).copy()

    z = np.zeros(generator.latent_dim, dtype=np.float64)
    h = s = None

    initial_residual = np.linalg.norm((operator(x) - y).ravel())
    prev_residual = initial_residual
    records = []
    status = Status.MAX_ITERS
    message = f"iteration budget {max_iters} exhausted"
    iterations = 0
    stalled = 0

    for iterations in range(1, max_iters + 1):
        # 1) pixel-space residual gradient step
        grad_residual = operator(x) - y
        x_grad = x - alpha * operator.adjoint(grad_residual)
        gradient_residual_norm = float(
            np.linalg.norm((operator(x_grad) - y).ravel()))

        # 2) approximate projection onto the generative range; the projection
        #    objective compares A x with y (observation space), not pixels.
        proj = range_project(
            operator, y, generator,
            lam=lam, rho=rho,
            inner_max_iters=inner_max_iters, tol=inner_tol,
            z=z, h=h, s=s)
        x = proj.x
        z, h, s = proj.z, proj.h, proj.s

        residual_norm = float(np.linalg.norm((operator(x) - y).ravel()))
        records.append(IterationRecord(
            iteration=iterations,
            residual_norm=residual_norm,
            gradient_residual_norm=gradient_residual_norm,
            projection_objective=proj.data_objective,
            split_objective=proj.objective,
            offset_l1=proj.offset_l1,
            inner_iters=proj.inner_iters,
            inner_converged=proj.converged,
        ))

        if not np.all(np.isfinite(x)):
            raise SolveError("iterate became non-finite; check step size")

        if residual_norm <= tol * max(1.0, np.linalg.norm(y.ravel())):
            status = Status.CONVERGED
            message = "observation residual reached absolute tolerance"
            break
        scale = max(prev_residual, 1e-300)
        improvement = (prev_residual - residual_norm) / scale
        prev_residual = residual_norm
        # A plateau is several consecutive outer iterations with negligible
        # relative residual improvement. If the inner projection itself never
        # reaches its tolerance we keep the finite result but flag the run as
        # not fully converged.
        if 0.0 <= improvement <= stall_tol:
            stalled += 1
        else:
            stalled = 0
        if stalled >= stall_patience:
            if proj.converged:
                status = Status.CONVERGED
                message = "residual improvement below tolerance for " \
                          f"{stall_patience} outer iterations"
            else:
                status = Status.INNER_NOT_CONVERGED
                message = ("outer progress stalled and the inner projection "
                           "did not meet its tolerance; finite result kept")
            break

    return SolveResult(
        x=x, observation=y, mask=operator.mask.astype(bool),
        operator=operator, generator=generator, truth=truth,
        status=status, iterations=iterations, records=records,
        initial_residual_norm=initial_residual,
        final_residual_norm=records[-1].residual_norm if records
        else initial_residual,
        config={"pixel_step": alpha, "max_iters": max_iters, "tol": tol,
                "lam": lam, "rho": rho,
                "inner_max_iters": inner_max_iters,
                "inner_tol": inner_tol,
                "stall_tol": stall_tol,
                "stall_patience": stall_patience,
                "operator_norm": operator_norm},
        message=message,
    )


def solve(shape=(16, 16), *, missing_fraction: float = 0.3,
          kernel_size: int = 3, noise_sigma: float = 0.0,
          latent_dim: int = 6, hidden_dim: int = 12,
          weight_seed: int = 2024, data_seed: int = 7, mask_seed: int = 11,
          kernel: np.ndarray | None = None,
          weights: dict | None = None,
          **kwargs) -> SolveResult:
    """Convenience driver: build fixed-seed synthetic data and solve it.

    ``kernel`` and ``weights`` may override the defaults; weights use the
    keys ``W1, b1, W2, b2``. No pretrained weights are ever loaded.
    """
    shape = (int(shape[0]), int(shape[1]))
    problem = make_problem(
        shape, missing_fraction=missing_fraction,
        kernel_size=kernel_size, noise_sigma=noise_sigma,
        seed=data_seed, mask_seed=mask_seed)
    if kernel is not None:
        operator = MaskedBlur(shape, kernel, problem["mask"].astype(float))
    else:
        operator = problem["operator"]

    if weights is None:
        weights = default_weights(shape, latent_dim=latent_dim,
                                  hidden_dim=hidden_dim, seed=weight_seed)
    try:
        generator = Generator(
            w1=weights["W1"], b1=weights["b1"],
            w2=weights["W2"], b2=weights["b2"],
            image_shape=shape)
    except (KeyError, GeneratorError) as exc:
        raise SolveError(f"invalid generator weights: {exc}") from exc

    return solve_inverse(operator, problem["observation"], generator,
                         truth=problem["truth"], **kwargs)
