"""Command-line entry point: build a synthetic problem and run the solver."""

from __future__ import annotations

import argparse
import sys

from .config import SolverConfig
from .report import save_outputs
from .solver import UNOBSERVABLE, solve
from .synthetic import make_problem, make_solvable_problem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Alternating residual/projection solver for small grayscale "
            "inverse problems."
        )
    )
    parser.add_argument("--height", type=int, default=12)
    parser.add_argument("--width", type=int, default=12)
    parser.add_argument("--missing", type=float, default=0.3,
                        help="fraction of missing pixels in [0, 1)")
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pixel-step", type=float, default=0.5)
    parser.add_argument("--sparse-penalty", type=float, default=1e-3)
    parser.add_argument("--inner-tol", type=float, default=1e-7)
    parser.add_argument("--inner-max-iter", type=int, default=300)
    parser.add_argument("--outer-tol", type=float, default=1e-8)
    parser.add_argument("--outer-max-iter", type=int, default=120)
    parser.add_argument("--latent-dim", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=40)
    parser.add_argument(
        "--solvable",
        action="store_true",
        help="use a model-consistent ground truth (generator-generated)",
    )
    parser.add_argument("--out-dir", default="outputs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.solvable:
        problem, _ = make_solvable_problem(
            height=args.height,
            width=args.width,
            missing_fraction=args.missing,
            kernel_size=args.kernel_size,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            seed=args.seed,
        )
    else:
        problem = make_problem(
            height=args.height,
            width=args.width,
            missing_fraction=args.missing,
            kernel_size=args.kernel_size,
            seed=args.seed,
        )
    config = SolverConfig(
        pixel_step=args.pixel_step,
        sparse_penalty=args.sparse_penalty,
        inner_tol=args.inner_tol,
        inner_max_iter=args.inner_max_iter,
        outer_tol=args.outer_tol,
        outer_max_iter=args.outer_max_iter,
    )
    result = solve(
        problem.observation, problem.kernel, problem.mask,
        problem.generator, config,
    )
    paths = save_outputs(
        args.out_dir, result, problem.observation, problem.mask,
        problem.true_image, config,
    )
    print(f"status: {result.status} ({result.message})")
    print(
        f"residual: {result.initial_residual_norm:.6f} -> "
        f"{result.residual_norm:.6f} over {result.iterations} iterations"
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 2 if result.status == UNOBSERVABLE else 0


if __name__ == "__main__":
    sys.exit(main())
