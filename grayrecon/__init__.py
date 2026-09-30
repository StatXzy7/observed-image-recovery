"""Small-scale grayscale inverse-problem solver (NumPy + Pillow only)."""

from .generator import Generator
from .operators import MaskedBlur, blur_adjoint, blur_forward, normalize_kernel
from .projection import range_project, soft_threshold
from .reporting import save_outputs, build_report
from .solver import SolveError, Status, solve, solve_inverse

__all__ = [
    "Generator", "MaskedBlur", "blur_adjoint", "blur_forward",
    "normalize_kernel", "range_project", "soft_threshold",
    "save_outputs", "build_report", "solve", "solve_inverse",
    "SolveError", "Status",
]
