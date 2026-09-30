"""Small two-layer tanh generator defined entirely by supplied configuration.

Forward map (all parameters are NumPy arrays provided at call time)::

    h = tanh(z @ W1 + b1)
    g = tanh(h @ W2 + b2)          reshaped to the configured image size

No pretrained weights are loaded anywhere in the package.
"""

from __future__ import annotations

import numpy as np

from .config import GeneratorConfig


def generator_forward(
    z: np.ndarray, config: GeneratorConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(image, hidden, pre_output)`` for latent code ``z``.

    ``image`` has shape ``config.image_shape``; ``hidden`` are the layer-one
    activations and ``pre_output`` the pre-tanh layer-two activations; both are
    reused by the layer-wise projection routine.
    """
    z = np.asarray(z, dtype=np.float64)
    if z.shape != (config.latent_dim,):
        raise ValueError(
            f"z must have shape ({config.latent_dim},), got {z.shape}"
        )
    pre_hidden = z @ config.w1 + config.b1
    hidden = np.tanh(pre_hidden)
    pre_output = hidden @ config.w2 + config.b2
    image = np.tanh(pre_output).reshape(config.image_shape)
    return image, hidden, pre_output


def random_generator_config(
    image_shape: tuple[int, int],
    latent_dim: int = 8,
    hidden_dim: int = 32,
    seed: int | None = 0,
    init_scale: float = 1.0,
) -> GeneratorConfig:
    """Build a fresh, fixed-seed generator configuration.

    Weights are scaled down so that random latent codes produce images with
    meaningful (non-saturated) tanh activations.
    """
    height, width = image_shape
    output_dim = height * width
    if latent_dim < 1 or hidden_dim < 1:
        raise ValueError("latent_dim and hidden_dim must be positive")
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal((latent_dim, hidden_dim)) * (
        init_scale / np.sqrt(latent_dim)
    )
    b1 = rng.standard_normal(hidden_dim) * 0.01
    w2 = rng.standard_normal((hidden_dim, output_dim)) * (
        init_scale / np.sqrt(hidden_dim)
    )
    b2 = rng.standard_normal(output_dim) * 0.01
    return GeneratorConfig(w1=w1, b1=b1, w2=w2, b2=b2, image_shape=image_shape)
