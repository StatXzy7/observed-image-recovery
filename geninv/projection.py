"""Approximate projection onto the generator range.

The projection minimises the distance **after applying the observation
operator** (not a plain pixel distance)::

    F(z, h, s) = 0.5 * || H(g(z via h) + s) - y ||_2^2
                 + 0.5 * beta * || h - tanh(z W1 + b1) ||_2^2
                 + lambda * || s ||_1

where ``g = tanh(h W2 + b2)``. The optimisation is layer-wise block
coordinate descent over the latent code, the hidden activations and the
sparse offset; the L1 offset term is handled by a soft-threshold prox step
and is fully disabled when ``lambda`` is zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import GeneratorConfig
from .operators import adjoint_observation, forward_observation


@dataclass
class ProjectionResult:
    image: np.ndarray
    offset: np.ndarray
    latent: np.ndarray
    hidden: np.ndarray
    objective: float
    data_misfit: float
    offset_l1: float
    iterations: int
    converged: bool
    message: str


def _generator_from_hidden(hidden: np.ndarray, config: GeneratorConfig) -> np.ndarray:
    pre_output = hidden @ config.w2 + config.b2
    return np.tanh(pre_output)


def _objective(
    latent: np.ndarray,
    hidden: np.ndarray,
    offset: np.ndarray,
    observation: np.ndarray,
    kernel: np.ndarray,
    mask: np.ndarray,
    config: GeneratorConfig,
    sparse_penalty: float,
    tie_weight: float,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    gen_flat = _generator_from_hidden(hidden, config)
    image = (gen_flat + offset).reshape(config.image_shape)
    residual = forward_observation(image, kernel, mask) - observation
    data_misfit = 0.5 * float(np.sum(residual * residual))
    layer_one = np.tanh(latent @ config.w1 + config.b1)
    tie = 0.5 * tie_weight * float(np.sum((hidden - layer_one) ** 2))
    offset_l1 = sparse_penalty * float(np.sum(np.abs(offset)))
    objective = data_misfit + tie + offset_l1
    return objective, data_misfit, offset_l1, residual, layer_one


def _gradient_step(
    latent: np.ndarray,
    hidden: np.ndarray,
    offset: np.ndarray,
    residual: np.ndarray,
    layer_one: np.ndarray,
    kernel: np.ndarray,
    mask: np.ndarray,
    config: GeneratorConfig,
    tie_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    back = adjoint_observation(residual, kernel, mask).ravel()
    pre_output = hidden @ config.w2 + config.b2
    gen_flat = np.tanh(pre_output)
    d_gen = back * (1.0 - gen_flat**2)
    grad_h = d_gen @ config.w2.T + tie_weight * (hidden - layer_one)
    d_layer1 = (tie_weight * (layer_one - hidden)) * (1.0 - layer_one**2)
    grad_z = d_layer1 @ config.w1.T
    return grad_h, grad_z


def _soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def project_onto_generator(
    target_image: np.ndarray,
    observation: np.ndarray,
    kernel: np.ndarray,
    mask: np.ndarray,
    config: GeneratorConfig,
    latent0: np.ndarray | None = None,
    sparse_penalty: float = 1e-3,
    tie_weight: float = 1.0,
    latent_step: float = 0.05,
    hidden_step: float = 0.2,
    tol: float = 1e-5,
    max_iter: int = 200,
    min_iter: int = 2,
) -> ProjectionResult:
    """Run the layer-wise inner projection with a bounded iteration budget.

    Args:
        target_image: Current pixel iterate (used to initialise the offset).
        observation / kernel / mask: Observation model used inside the target,
            so the projection matches observed-domain, not pixel, distance.
        sparse_penalty: L1 weight; zero disables the sparse-offset constraint.
        max_iter: Hard cap on the number of inner iterations.

    A finite result is always returned; ``converged`` is False when the
    tolerance is not reached before the budget is exhausted.
    """
    if sparse_penalty < 0.0:
        raise ValueError("sparse_penalty must be non-negative")
    if latent_step <= 0.0 or hidden_step <= 0.0:
        raise ValueError("inner step sizes must be positive")
    if tol <= 0.0 or max_iter < 1:
        raise ValueError("tol must be positive and max_iter >= 1")
    target_image = np.asarray(target_image, dtype=np.float64)
    if target_image.shape != config.image_shape:
        raise ValueError("target_image shape does not match generator config")

    if latent0 is None:
        latent = np.zeros(config.latent_dim, dtype=np.float64)
    else:
        latent = np.array(latent0, dtype=np.float64, copy=True)
        if latent.shape != (config.latent_dim,):
            raise ValueError("latent0 has the wrong shape")

    layer_one = np.tanh(latent @ config.w1 + config.b1)
    gen0 = _generator_from_hidden(layer_one, config).reshape(config.image_shape)
    hidden = layer_one.copy()
    offset = (target_image - gen0).ravel()
    if sparse_penalty == 0.0:
        pass  # free offset: prox leaves it untouched
    else:
        offset = _soft_threshold(offset, sparse_penalty)

    objective, data_misfit, offset_l1, residual, layer_one = _objective(
        latent, hidden, offset, observation, kernel, mask, config,
        sparse_penalty, tie_weight,
    )
    initial_objective = objective
    history = [objective]
    converged = False
    iterations = 0

    for iterations in range(1, max_iter + 1):
        grad_h, grad_z = _gradient_step(
            latent, hidden, offset, residual, layer_one, kernel, mask,
            config, tie_weight,
        )

        # Block 1: latent code, backtracking line search.
        step = latent_step
        new_latent = latent - step * grad_z
        new_obj, _, _, new_res, new_layer1 = _objective(
            new_latent, hidden, offset, observation, kernel, mask, config,
            sparse_penalty, tie_weight,
        )
        for _ in range(30):
            if new_obj <= objective + 1e-12 * max(1.0, abs(objective)):
                break
            step *= 0.5
            new_latent = latent - step * grad_z
            new_obj, _, _, new_res, new_layer1 = _objective(
                new_latent, hidden, offset, observation, kernel, mask, config,
                sparse_penalty, tie_weight,
            )
        latent, objective, residual, layer_one = (
            new_latent, new_obj, new_res, new_layer1,
        )

        # Block 2: hidden activations, backtracking line search.
        grad_h, _ = _gradient_step(
            latent, hidden, offset, residual, layer_one, kernel, mask,
            config, tie_weight,
        )
        step = hidden_step
        new_hidden = hidden - step * grad_h
        new_obj, _, _, new_res, new_layer1 = _objective(
            latent, new_hidden, offset, observation, kernel, mask, config,
            sparse_penalty, tie_weight,
        )
        for _ in range(30):
            if new_obj <= objective + 1e-12 * max(1.0, abs(objective)):
                break
            step *= 0.5
            new_hidden = hidden - step * grad_h
            new_obj, _, _, new_res, new_layer1 = _objective(
                latent, new_hidden, offset, observation, kernel, mask, config,
                sparse_penalty, tie_weight,
            )
        hidden, objective, residual, layer_one = (
            new_hidden, new_obj, new_res, new_layer1,
        )

        # Block 3: sparse offset proximal step (closed form, exact minimiser
        # of the data term plus the L1 penalty).
        gen_image = _generator_from_hidden(hidden, config).reshape(
            config.image_shape
        )
        back = adjoint_observation(residual, kernel, mask).ravel()
        candidate = offset - hidden_step * back
        if sparse_penalty > 0.0:
            new_offset = _soft_threshold(candidate, hidden_step * sparse_penalty)
        else:
            new_offset = candidate
        new_image = (
            _generator_from_hidden(hidden, config) + new_offset
        ).reshape(config.image_shape)
        new_res = forward_observation(new_image, kernel, mask) - observation
        new_data = 0.5 * float(np.sum(new_res * new_res))
        new_tie = 0.5 * tie_weight * float(np.sum((hidden - layer_one) ** 2))
        new_l1 = sparse_penalty * float(np.sum(np.abs(new_offset)))
        new_obj = new_data + new_tie + new_l1
        if new_obj <= objective + 1e-12 * max(1.0, abs(objective)):
            offset, objective, residual = new_offset, new_obj, new_res
        else:
            new_obj = objective

        history.append(objective)
        denom = max(1.0, abs(initial_objective))
        improvement = history[-2] - history[-1]
        if iterations >= min_iter and improvement / denom <= tol:
            converged = True
            break

    final_obj, data_misfit, offset_l1, residual, _ = _objective(
        latent, hidden, offset, observation, kernel, mask, config,
        sparse_penalty, tie_weight,
    )
    gen_flat = _generator_from_hidden(hidden, config)
    image = (gen_flat + offset).reshape(config.image_shape)
    message = (
        "converged" if converged else "inner iteration budget exhausted"
    )
    return ProjectionResult(
        image=np.clip(image, -1.0, 1.0),
        offset=offset.reshape(config.image_shape),
        latent=latent,
        hidden=hidden,
        objective=float(final_obj),
        data_misfit=float(data_misfit),
        offset_l1=float(offset_l1),
        iterations=iterations,
        converged=converged,
        message=message,
    )
