"""Observation operators: masked zero-padded convolution blur.

The observation model is  y = M (K x),  where ``K`` is a small-kernel
convolution with zero boundary padding and ``M`` applies a missing-pixel
mask (missing observations contribute zero to both the forward map and its
adjoint, so the adjoint stays exact: <Kx, My> = <K* M y, x>).
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class OperatorError(ValueError):
    """Raised for malformed operator inputs (shape, kernel, values)."""


def normalize_kernel(kernel: np.ndarray) -> np.ndarray:
    """Validate and normalize a 2-D blur kernel so its coefficients sum to 1."""
    k = np.asarray(kernel, dtype=np.float64)
    if k.ndim != 2:
        raise OperatorError("blur kernel must be 2-D")
    if k.shape[0] % 2 == 0 or k.shape[1] % 2 == 0:
        raise OperatorError("blur kernel dimensions must be odd")
    if not np.all(np.isfinite(k)):
        raise OperatorError("blur kernel must be finite")
    total = float(k.sum())
    if total <= 0.0:
        raise OperatorError("blur kernel must have a positive sum")
    return k / total


def blur_forward(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2-D valid correlation on a zero-padded image (same shape output)."""
    if image.ndim != 2:
        raise OperatorError("image must be 2-D")
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.zeros((image.shape[0] + 2 * ph, image.shape[1] + 2 * pw),
                      dtype=np.float64)
    padded[ph:ph + image.shape[0], pw:pw + image.shape[1]] = image
    windows = sliding_window_view(padded, (kh, kw))
    return np.einsum("ijkl,kl->ij", windows, kernel)


def blur_adjoint(field: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Exact adjoint of :func:`blur_forward` (correlation with flipped kernel)."""
    if field.ndim != 2:
        raise OperatorError("field must be 2-D")
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    out = np.zeros((field.shape[0] + 2 * ph, field.shape[1] + 2 * pw),
                   dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            out[i:i + field.shape[0], j:j + field.shape[1]] += \
                field * kernel[i, j]
    return out[ph:ph + field.shape[0], pw:pw + field.shape[1]]


class MaskedBlur:
    """Composition of zero-padded blur and a binary missing-pixel mask.

    Parameters
    ----------
    shape:
        ``(H, W)`` image shape.
    kernel:
        2-D kernel with odd dimensions and positive sum.
    mask:
        Boolean/float array where 1 = observed pixel, 0 = missing.
    """

    def __init__(self, shape, kernel: np.ndarray, mask: np.ndarray):
        self.shape = (int(shape[0]), int(shape[1]))
        self.kernel = normalize_kernel(kernel)
        m = np.asarray(mask, dtype=np.float64)
        if m.shape != self.shape:
            raise OperatorError(
                f"mask shape {m.shape} does not match image shape {self.shape}")
        if not np.all(np.isfinite(m)) or np.any((m != 0.0) & (m != 1.0)):
            raise OperatorError("mask must contain only 0/1 finite values")
        self.mask = m

    @property
    def n_pixels(self) -> int:
        return self.shape[0] * self.shape[1]

    @property
    def n_observed(self) -> int:
        return int(self.mask.sum())

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.shape != self.shape:
            raise OperatorError(
                f"image shape {x.shape} does not match {self.shape}")
        return self.mask * blur_forward(x, self.kernel)

    __call__ = forward

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        if y.shape != self.shape:
            raise OperatorError(
                f"field shape {y.shape} does not match {self.shape}")
        return blur_adjoint(self.mask * y, self.kernel)

    def to_matrix(self) -> np.ndarray:
        """Dense matrix representation (n_pixels, n_pixels); n is small."""
        n = self.n_pixels
        a = np.empty((n, n), dtype=np.float64)
        for j in range(n):
            basis = np.zeros(self.shape, dtype=np.float64)
            basis.flat[j] = 1.0
            a[:, j] = self.forward(basis).ravel()
        return a

    def spectral_norm(self) -> float:
        """Largest singular value of the observation operator."""
        a = self.to_matrix()
        return float(np.linalg.svd(a, compute_uv=False)[0])
