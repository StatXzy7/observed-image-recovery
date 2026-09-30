"""Small-scale grayscale inverse-problem solver (NumPy/Pillow, CPU only).

The solver alternates a pixel-space observation-residual gradient step with
an approximate projection onto the range of a small two-layer tanh generator.
Images live on the internal scale [-1, 1]; exporters map that scale to PNG.
"""

from .config import GeneratorConfig, SolverConfig
from .operators import (
    blur,
    blur_adjoint,
    forward_observation,
    adjoint_observation,
    random_mask,
    gaussian_kernel,
)
from .generator import generator_forward, random_generator_config
from .synthetic import synthetic_image, make_problem, make_solvable_problem
from .projection import project_onto_generator
from .solver import (
    solve,
    SolverResult,
    CONVERGED,
    UNCONVERGED,
    UNOBSERVABLE,
    INVALID,
)
from .report import (
    save_observation_png,
    save_reconstruction_png,
    save_difference_png,
    save_report,
    save_outputs,
    internal_to_uint8,
)

__all__ = [
    "GeneratorConfig",
    "SolverConfig",
    "blur",
    "blur_adjoint",
    "forward_observation",
    "adjoint_observation",
    "random_mask",
    "gaussian_kernel",
    "generator_forward",
    "random_generator_config",
    "synthetic_image",
    "make_problem",
    "make_solvable_problem",
    "project_onto_generator",
    "solve",
    "SolverResult",
    "CONVERGED",
    "UNCONVERGED",
    "UNOBSERVABLE",
    "INVALID",
    "save_observation_png",
    "save_reconstruction_png",
    "save_difference_png",
    "save_report",
    "save_outputs",
    "internal_to_uint8",
]
