"""Observation operators: zero-padded small-kernel blur and pixel masking.

All images are float64 2-D arrays on the internal scale. The blur uses
zero-filled boundaries, and :func:`blur_adjoint` is the exact linear adjoint
of :func:`blur`, which the test-suite verifies numerically.
"""

from __future__ import annotations

import numpy as np


def _check_kernel(kernel: np.ndarray) -> np.ndarray:
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.ndim != 2:
        raise ValueError("kernel must be a 2-D array")
    if kernel.shape[0] < 1 or kernel.shape[1] < 1:
        raise ValueError("kernel dimensions must be positive")
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("kernel dimensions must be odd")
    if not np.all(np.isfinite(kernel)):
        raise ValueError("kernel contains non-finite values")
    if not np.any(kernel):
        raise ValueError("kernel must not be all zeros")
    return kernel


def blur(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply ``kernel`` to ``image`` with zero-filled boundaries.

    Uses the convolution convention centered at the kernel's middle tap.
    Output shape equals the input image shape.
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError("image must be 2-D")
    kernel = _check_kernel(kernel)
    kh, kw = kernel.shape
    off_h, off_w = kh // 2, kw // 2
    height, width = image.shape
    out = np.zeros_like(image)
    for a in range(kh):
        for b in range(kw):
            weight = kernel[a, b]
            if weight == 0.0:
                continue
            # y[i, j] += w * x[i - (a-off_h), j - (b-off_w)]
            shift_h, shift_w = a - off_h, b - off_w
            ys, ye = max(0, shift_h), min(height, height + shift_h)
            xs, xe = max(0, shift_w), min(width, width + shift_w)
            out[ys:ye, xs:xe] += weight * image[
                ys - shift_h : ye - shift_h, xs - shift_w : xe - shift_w
            ]
    return out


def blur_adjoint(field: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Exact adjoint of :func:`blur`.

    Implemented directly (transpose of the zero-padded convolution), which is
    equivalent to blurring with the kernel rotated by 180 degrees.
    """
    field = np.asarray(field, dtype=np.float64)
    if field.ndim != 2:
        raise ValueError("field must be 2-D")
    kernel = _check_kernel(kernel)
    kh, kw = kernel.shape
    off_h, off_w = kh // 2, kw // 2
    height, width = field.shape
    out = np.zeros_like(field)
    for a in range(kh):
        for b in range(kw):
            weight = kernel[a, b]
            if weight == 0.0:
                continue
            shift_h, shift_w = a - off_h, b - off_w
            # x[i, j] += w * y[i + shift_h, j + shift_w]
            ys, ye = max(0, shift_h), min(height, height + shift_h)
            xs, xe = max(0, shift_w), min(width, width + shift_w)
            out[ys - shift_h : ye - shift_h, xs - shift_w : xe - shift_w] += (
                weight * field[ys:ye, xs:xe]
            )
    return out


def forward_observation(
    image: np.ndarray, kernel: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Observation operator ``H(x) = mask * blur(x)`` (zero-filled at gaps)."""
    mask = np.asarray(mask)
    if mask.shape != np.asarray(image).shape:
        raise ValueError("mask and image shapes must match")
    return mask.astype(np.float64) * blur(image, kernel)


def adjoint_observation(
    field: np.ndarray, kernel: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Adjoint ``H^T(y) = blur_adjoint(mask * y)`` of the observation map."""
    mask = np.asarray(mask)
    if mask.shape != np.asarray(field).shape:
        raise ValueError("mask and field shapes must match")
    return blur_adjoint(mask.astype(np.float64) * np.asarray(field, dtype=np.float64), kernel)


def gaussian_kernel(size: int = 3, sigma: float = 1.0) -> np.ndarray:
    """Return a normalised, finite-supported Gaussian blur kernel."""
    if int(size) < 1 or int(size) % 2 == 0:
        raise ValueError("kernel size must be a positive odd integer")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be a finite positive number")
    radius = int(size) // 2
    axis = np.arange(-radius, radius + 1, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis)
    kernel = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    return kernel


def random_mask(
    shape: tuple[int, int], missing_fraction: float, seed: int | None = None
) -> np.ndarray:
    """Boolean mask of *observed* pixels with a fixed-seed random dropout."""
    if not 0.0 <= missing_fraction <= 1.0:
        raise ValueError("missing_fraction must lie in [0, 1]")
    rng = np.random.default_rng(seed)
    height, width = shape
    total = height * width
    n_missing = int(round(missing_fraction * total))
    flat = np.ones(total, dtype=bool)
    if n_missing > 0:
        missing_idx = rng.choice(total, size=n_missing, replace=False)
        flat[missing_idx] = False
    return flat.reshape(height, width)
