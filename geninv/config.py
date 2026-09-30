"""Configuration dataclasses for the generator and the iterative solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GeneratorConfig:
    """Two-layer tanh generator operating on a flattened image.

    Forward map::

        h = tanh(z @ W1 + b1)          # (hidden_dim,)
        g = tanh(h @ W2 + b2)          # (output_dim,) == image.ravel()

    Parameters are supplied through configuration; no pretrained weights are
    ever loaded.
    """

    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    image_shape: tuple[int, int]

    def __post_init__(self) -> None:
        w1 = np.asarray(self.w1, dtype=np.float64)
        b1 = np.asarray(self.b1, dtype=np.float64)
        w2 = np.asarray(self.w2, dtype=np.float64)
        b2 = np.asarray(self.b2, dtype=np.float64)
        if w1.ndim != 2 or b1.ndim != 1 or w2.ndim != 2 or b2.ndim != 1:
            raise ValueError("generator matrices/biases have invalid ranks")
        latent_dim, hidden_dim = w1.shape
        if b1.shape != (hidden_dim,):
            raise ValueError("b1 shape must match W1 hidden dimension")
        if w2.shape[0] != hidden_dim:
            raise ValueError("W2 rows must match the hidden dimension")
        output_dim = int(w2.shape[1])
        if b2.shape != (output_dim,):
            raise ValueError("b2 shape must match the generator output size")
        height, width = self.image_shape
        if height * width != output_dim:
            raise ValueError("image_shape must match generator output size")
        object.__setattr__(self, "w1", w1)
        object.__setattr__(self, "b1", b1)
        object.__setattr__(self, "w2", w2)
        object.__setattr__(self, "b2", b2)

    @property
    def latent_dim(self) -> int:
        return int(self.w1.shape[0])

    @property
    def hidden_dim(self) -> int:
        return int(self.w1.shape[1])

    @property
    def output_dim(self) -> int:
        return int(self.w2.shape[1])


@dataclass
class SolverConfig:
    """Hyper-parameters of the alternating solver.

    Attributes:
        pixel_step: Step size for the pixel-space residual gradient step.
        sparse_penalty: L1 weight on the sparse image-domain offset. Zero
            disables the offset constraint entirely.
        tie_weight: Weight of the coupling between the optimised hidden state
            and layer one's activations during the layer-wise inner solve.
        latent_step / hidden_step: Inner block-coordinate step sizes.
        inner_tol: Relative objective tolerance for the inner projection.
        inner_max_iter: Hard cap on inner iterations.
        outer_tol: Relative observation-residual stopping tolerance.
        outer_max_iter: Iteration budget for the outer loop.
    """

    pixel_step: float = 0.5
    sparse_penalty: float = 1e-3
    tie_weight: float = 1.0
    latent_step: float = 0.05
    hidden_step: float = 0.2
    inner_tol: float = 1e-6
    inner_max_iter: int = 200
    inner_min_iter: int = 2
    outer_tol: float = 1e-8
    outer_max_iter: int = 80

    def validate(self) -> None:
        if not np.isfinite(self.pixel_step) or self.pixel_step <= 0.0:
            raise ValueError("pixel_step must be a finite positive number")
        if not np.isfinite(self.sparse_penalty) or self.sparse_penalty < 0.0:
            raise ValueError("sparse_penalty must be finite and non-negative")
        if not np.isfinite(self.tie_weight) or self.tie_weight < 0.0:
            raise ValueError("tie_weight must be finite and non-negative")
        if not np.isfinite(self.latent_step) or self.latent_step <= 0.0:
            raise ValueError("latent_step must be a finite positive number")
        if not np.isfinite(self.hidden_step) or self.hidden_step <= 0.0:
            raise ValueError("hidden_step must be a finite positive number")
        if not np.isfinite(self.inner_tol) or self.inner_tol <= 0.0:
            raise ValueError("inner_tol must be a finite positive number")
        if not np.isfinite(self.outer_tol) or self.outer_tol <= 0.0:
            raise ValueError("outer_tol must be a finite positive number")
        if int(self.inner_max_iter) < 1:
            raise ValueError("inner_max_iter must be >= 1")
        if int(self.outer_max_iter) < 1:
            raise ValueError("outer_max_iter must be >= 1")
        if int(self.inner_min_iter) < 1:
            raise ValueError("inner_min_iter must be >= 1")
