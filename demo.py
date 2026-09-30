#!/usr/bin/env python3
"""Run the fixed-seed demo problem and write figures + report to ./outputs."""

import json
import os

from grayrecon.reporting import save_outputs
from grayrecon.solver import solve


def main():
    outdir = os.path.join(os.path.dirname(__file__), "outputs")
    result = solve(
        shape=(12, 12),
        missing_fraction=0.3,
        kernel_size=3,
        latent_dim=6,
        hidden_dim=16,
        pixel_step=1.9,
        max_iters=100,
        tol=5e-4,
        lam=0.0,
        rho=1.0,
        inner_max_iters=100,
        inner_tol=1e-11,
        stall_tol=1e-8,
        stall_patience=5,
        data_seed=7,
        mask_seed=11,
        weight_seed=2024,
    )
    report = save_outputs(result, outdir, crop_size=20, scale=20)
    print(f"status            : {result.status}")
    print(f"message           : {result.message}")
    print(f"outer iterations  : {result.iterations}")
    print(f"residual norm     : "
          f"{result.initial_residual_norm:.6f} -> "
          f"{result.final_residual_norm:.6e}")
    print(f"observed pixels   : {result.operator.n_observed}/"
          f"{result.operator.n_pixels}")
    print(f"RMSE vs truth     : "
          f"{report['metrics']['rmse_vs_truth_uncropped']:.6f} "
          "(uncropped)")
    print(f"outputs written   : {outdir}")
    print("files: " + ", ".join(
        sorted(os.listdir(outdir))))


if __name__ == "__main__":
    main()
