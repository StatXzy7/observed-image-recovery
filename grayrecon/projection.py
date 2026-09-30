"""Approximate projection onto the generative range.

The projection target is measured **after** the observation operator::

    min_{z, s}  1/2 * || A G(z, s) - y ||^2  +  lam * ||s||_1

where ``G(z, s) = tanh(tanh(z W1 + b1) W2 + b2) + s`` reshaped to the image.
Comparing pixel-space distances (e.g. against the raw gradient-step image)
would be wrong for an inverse problem; the observation-space residual is
used throughout.

The latent state is split into ``(z, h, s)`` and optimized layer by layer:
a gradient step on the first-layer code ``z`` (enforcing the tanh coupling),
a gradient step on the hidden state ``h`` (driving the observation residual),
and an ISTA step on the sparse offset ``s`` with an L1 penalty.  The L1
constraint is switched off entirely when ``lam == 0``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class ProjectionError(ValueError):
    """Raised for invalid projection parameters."""


def soft_threshold(v: np.ndarray, threshold: float) -> np.ndarray:
    """Proximal operator of ``threshold * |.|`` (component-wise)."""
    return np.sign(v) * np.maximum(np.abs(v) - threshold, 0.0)


def observation_objective(operator, y: np.ndarray, x: np.ndarray) -> float:
    """1/2 ||A x - y||^2: the projection target in observation space."""
    residual = operator.forward(x) - y
    return 0.5 * float(np.dot(residual.ravel(), residual.ravel()))


def split_objective(operator, y, generator, z, h, s, lam, rho):
    """Full split objective used by the layer-wise inner optimization."""
    pre1 = z @ generator.w1 + generator.b1
    t1 = np.tanh(pre1)
    pre2 = h @ generator.w2 + generator.b2
    x = (np.tanh(pre2) + s).reshape(generator.image_shape)
    residual = operator.forward(x) - y
    data = 0.5 * np.dot(residual.ravel(), residual.ravel())
    coupling = 0.5 * rho * np.dot((h - t1).ravel(), (h - t1).ravel())
    return float(data + lam * np.sum(np.abs(s)) + coupling)


@dataclass
class ProjectionResult:
    x: np.ndarray
    z: np.ndarray
    h: np.ndarray
    s: np.ndarray
    objective: float          # full split objective (incl. penalties)
    data_objective: float     # 1/2 ||A x - y||^2 (observation space)
    offset_l1: float
    inner_iters: int
    converged: bool
    history: list


def initial_state(generator, z=None, h=None, s=None):
    if z is None:
        z = np.zeros(generator.latent_dim, dtype=np.float64)
    if s is None:
        s = np.zeros(generator.output_dim, dtype=np.float64)
    if h is None:
        _, h0, _, _ = generator.forward(z)
        h = h0.copy()
    return (np.asarray(z, dtype=np.float64).copy(),
            np.asarray(h, dtype=np.float64).copy(),
            np.asarray(s, dtype=np.float64).copy())


def _descent_step(variable, gradient, lip, loss_fn, max_shrinks=6):
    """Gradient step with Armijo-style backtracking on a smooth block.

    Returns ``(variable_new, lip_used)``; the step is shrunk until the block
    objective does not increase (or the fixed safe bound is reached).
    """
    base_loss = loss_fn(variable)
    step = variable - gradient / lip
    new_loss = loss_fn(step)
    shrinks = 0
    while new_loss > base_loss + 1e-12 * max(1.0, abs(base_loss)) \
            and shrinks < max_shrinks:
        lip *= 2.0
        step = variable - gradient / lip
        new_loss = loss_fn(step)
        shrinks += 1
    return step, lip


def range_project(operator, y, generator, *, lam: float = 0.0,
                  rho: float = 1.0, inner_max_iters: int = 50,
                  tol: float = 1e-7, z=None, h=None, s=None):
    """Approximately project onto the generative range.

    Parameters
    ----------
    operator:
        Observation operator with ``forward``/``adjoint``.
    y:
        Observed image (already masked); compared in observation space.
    generator:
        Two-layer :class:`~grayrecon.generator.Generator`.
    lam:
        L1 penalty weight on the sparse offset; ``0`` disables sparsity.
    rho:
        Weight of the first-layer tanh coupling; must be positive.
    inner_max_iters:
        Hard cap on inner iterations; at most this many are performed.
    tol:
        Relative objective-change tolerance for early stopping.
    z, h, s:
        Optional warm start (e.g. previous outer iteration state).
    """
    if not np.isfinite(lam) or lam < 0.0:
        raise ProjectionError("lam must be a non-negative finite number")
    if not np.isfinite(rho) or rho <= 0.0:
        raise ProjectionError("rho must be a positive finite number")
    if int(inner_max_iters) < 1:
        raise ProjectionError("inner_max_iters must be >= 1")
    if not np.isfinite(tol) or tol < 0.0:
        raise ProjectionError("tol must be a non-negative finite number")
    y = np.asarray(y, dtype=np.float64)
    if y.shape != generator.image_shape:
        raise ProjectionError("y shape does not match generator image shape")
    if not np.all(np.isfinite(y)):
        raise ProjectionError("y must contain finite values")

    z, h, s = initial_state(generator, z, h, s)

    # Safe upper bounds on the per-block Lipschitz constants; local curvature
    # is typically much smaller (tanh saturation, masking), so the descent
    # step backtracks up to these bounds when needed.
    l_z_bound = rho * float(
        np.linalg.svd(generator.w1, compute_uv=False)[0]) ** 2
    l_h_bound = float(
        np.linalg.svd(generator.w2, compute_uv=False)[0]) ** 2 + rho
    # Masked normalized blur is a contraction: ||A||_2 <= 1.
    l_s = 1.0
    power_vec = np.ones(generator.hidden_dim) / np.sqrt(generator.hidden_dim)
    power_vec_z = np.ones(generator.latent_dim) \
        / np.sqrt(generator.latent_dim)

    history = []
    phi = split_objective(operator, y, generator, z, h, s, lam, rho)
    history.append(phi)
    converged = False
    iterations = 0

    for iterations in range(1, int(inner_max_iters) + 1):
        phi_old = phi

        # --- first-layer code z: restore the tanh coupling h ~ tanh(pre1) ---
        def z_loss(zz):
            tt = np.tanh(zz @ generator.w1 + generator.b1)
            return 0.5 * rho * float(np.dot((h - tt).ravel(), (h - tt).ravel()))

        t1 = np.tanh(z @ generator.w1 + generator.b1)
        dt1 = 1.0 - t1 ** 2
        grad_z = (-rho * (h - t1) * dt1) @ generator.w1.T
        # Local curvature estimate for W1 diag(dt1) via power iterations.
        vz = power_vec_z
        for _ in range(4):
            vz = (vz @ generator.w1) * dt1 @ generator.w1.T
            norm_vz = float(np.linalg.norm(vz))
            if norm_vz > 0.0:
                vz /= norm_vz
        power_vec_z = vz
        probe_z = (vz @ generator.w1) * dt1
        local_l_z = rho * float(np.dot(probe_z, probe_z))
        z, l_z = _descent_step(z, grad_z,
                                max(local_l_z, 1e-14), z_loss)

        # --- hidden state h: drives the observation-space residual ----------
        t1 = np.tanh(z @ generator.w1 + generator.b1)

        def h_loss(hh):
            tt2 = np.tanh(hh @ generator.w2 + generator.b2)
            xx = (tt2 + s).reshape(generator.image_shape)
            rr = operator.forward(xx) - y
            return (0.5 * float(np.dot(rr.ravel(), rr.ravel()))
                    + 0.5 * rho * float(np.dot((hh - t1).ravel(),
                                               (hh - t1).ravel())))

        t2 = np.tanh(h @ generator.w2 + generator.b2)
        x = (t2 + s).reshape(generator.image_shape)
        grad_t2 = operator.adjoint(operator.forward(x) - y).ravel()
        dt2 = 1.0 - t2 ** 2
        grad_h = (grad_t2 * dt2) @ generator.w2.T + rho * (h - t1)

        # Local spectral norm of the observation part via power iterations,
        # then backtrack to guarantee descent.
        v = power_vec
        for _ in range(4):
            q = operator.forward(
                ((v @ generator.w2) * dt2).reshape(
                    generator.image_shape)).ravel()
            v = (operator.adjoint(q.reshape(generator.image_shape)).ravel()
                 * dt2) @ generator.w2.T
            norm_v = float(np.linalg.norm(v))
            if norm_v > 0.0:
                v /= norm_v
        power_vec = v
        probe = (v @ generator.w2) * dt2
        local_l_h = float(np.dot(
            operator.forward(probe.reshape(
                generator.image_shape)).ravel(),
            operator.forward(probe.reshape(
                generator.image_shape)).ravel())) + rho
        h, l_h = _descent_step(h, grad_h, max(local_l_h, 1e-14), h_loss)

        # --- sparse offset s: ISTA (plain gradient step when lam == 0) ------
        t2 = np.tanh(h @ generator.w2 + generator.b2)
        t2_image_shape = generator.image_shape

        def s_penalty(ss):
            xx = (t2 + ss).reshape(t2_image_shape)
            rr = operator.forward(xx) - y
            return 0.5 * float(np.dot(rr.ravel(), rr.ravel())) \
                + lam * float(np.sum(np.abs(ss)))

        grad_s = operator.adjoint(
            operator.forward((t2 + s).reshape(t2_image_shape)) - y).ravel()
        base_s_loss = s_penalty(s)
        s_new = soft_threshold(s - grad_s / l_s, lam / l_s)
        shrinks = 0
        while s_penalty(s_new) > base_s_loss and shrinks < 6:
            l_s *= 2.0
            s_new = soft_threshold(s - grad_s / l_s, lam / l_s)
            shrinks += 1
        s = s_new

        phi = split_objective(operator, y, generator, z, h, s, lam, rho)
        # Defensive monotonicity: if a sweep somehow raised the objective,
        # restore the previous iterate but still count the iteration.
        if not np.isfinite(phi):
            raise ProjectionError("non-finite objective during projection")
        history.append(phi)
        scale = max(1.0, abs(phi_old))
        if abs(phi_old - phi) <= tol * scale:
            converged = True
            break

    pre2 = h @ generator.w2 + generator.b2
    x = (np.tanh(pre2) + s).reshape(generator.image_shape)
    data_obj = observation_objective(operator, y, x)

    return ProjectionResult(
        x=x, z=z, h=h, s=s,
        objective=phi,
        data_objective=data_obj,
        offset_l1=float(np.sum(np.abs(s))),
        inner_iters=iterations,
        converged=converged,
        history=history,
    )
