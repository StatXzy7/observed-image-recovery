"""Two-layer small generative map with tanh activations.

The map is fully configurable via explicit matrices/biases (no pretrained
weights are ever loaded)::

    x = tanh(tanh(z W1 + b1) W2 + b2),   z in R^d,  x in R^n (reshaped to HxW)

The output range is (-1, 1)^n; callers rescale to [0, 1] gray levels when
presenting images (metrics always use the solver's native scale).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class GeneratorError(ValueError):
    """Raised for malformed generator parameters or latent codes."""


@dataclass(frozen=True)
class Generator:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    image_shape: tuple

    @property
    def latent_dim(self) -> int:
        return self.w1.shape[0]

    @property
    def hidden_dim(self) -> int:
        return self.w1.shape[1]

    @property
    def output_dim(self) -> int:
        return self.w2.shape[1]

    def __post_init__(self):
        w1 = np.asarray(self.w1, dtype=np.float64)
        b1 = np.asarray(self.b1, dtype=np.float64)
        w2 = np.asarray(self.w2, dtype=np.float64)
        b2 = np.asarray(self.b2, dtype=np.float64)
        for name, arr, expected_ndim in (
                ("W1", w1, 2), ("b1", b1, 1),
                ("W2", w2, 2), ("b2", b2, 1)):
            if arr.ndim != expected_ndim:
                raise GeneratorError(f"{name} has wrong number of dimensions")
            if not np.all(np.isfinite(arr)):
                raise GeneratorError(f"{name} must contain finite values")
        if b1.shape != (w1.shape[1],):
            raise GeneratorError("b1 length must equal W1 number of columns")
        if w2.shape[0] != w1.shape[1]:
            raise GeneratorError("W2 rows must equal W1 columns (hidden dim)")
        if b2.shape != (w2.shape[1],):
            raise GeneratorError("b2 length must equal W2 number of columns")
        h, w = self.image_shape
        if w2.shape[1] != int(h) * int(w):
            raise GeneratorError("output dim W2 must equal H * W")
        object.__setattr__(self, "w1", w1)
        object.__setattr__(self, "b1", b1)
        object.__setattr__(self, "w2", w2)
        object.__setattr__(self, "b2", b2)
        object.__setattr__(self, "image_shape", (int(h), int(w)))

    def forward(self, z: np.ndarray):
        """Return ``(x, h, pre1, pre2)`` for a latent vector ``z``."""
        z = np.asarray(z, dtype=np.float64)
        if z.shape != (self.latent_dim,):
            raise GeneratorError(
                f"latent code shape {z.shape} != ({self.latent_dim},)")
        pre1 = z @ self.w1 + self.b1
        h = np.tanh(pre1)
        pre2 = h @ self.w2 + self.b2
        x = np.tanh(pre2).reshape(self.image_shape)
        return x, h, pre1, pre2


def default_weights(image_shape, latent_dim: int = 6,
                    hidden_dim: int = 12, seed: int = 2024) -> dict:
    """Fixed-seed weight matrices/biases for a map of the given dimensions."""
    rng = np.random.default_rng(seed)
    n = int(image_shape[0]) * int(image_shape[1])
    w1 = rng.standard_normal((latent_dim, hidden_dim)) / np.sqrt(latent_dim)
    b1 = rng.standard_normal(hidden_dim) * 0.1
    w2 = rng.standard_normal((hidden_dim, n)) / np.sqrt(hidden_dim)
    b2 = rng.standard_normal(n) * 0.05
    return {"W1": w1, "b1": b1, "W2": w2, "b2": b2}
