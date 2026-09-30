"""PNG figures and a machine-readable numerical report.

Figures are explicitly annotated whenever they show a crop (header text plus
a black frame around the cropped region). All reported metrics are computed
on the **uncropped** arrays; cropping is presentation-only.
"""

from __future__ import annotations

import json

import numpy as np
from PIL import Image, ImageDraw

HEADER_PX = 13
BORDER_PX = 2
DEFAULT_CROP = 20
DEFAULT_SCALE = 16


def to_uint8(array: np.ndarray) -> np.ndarray:
    """Clip a float array to [0, 1] and convert to uint8."""
    return np.clip(np.rint(np.asarray(array, dtype=np.float64) * 255.0),
                   0, 255).astype(np.uint8)


def crop_center(array: np.ndarray, crop_size: int):
    """Return ``(cropped, r0, r1, c0, c1, was_cropped)`` (center crop)."""
    h, w = array.shape
    if crop_size is not None and max(h, w) > crop_size:
        ch = min(crop_size, h)
        cw = min(crop_size, w)
        r0 = (h - ch) // 2
        c0 = (w - cw) // 2
        r1, c1 = r0 + ch, c0 + cw
        return array[r0:r1, c0:c1], r0, r1, c0, c1, True
    return array, 0, h, 0, w, False


def render_panel(array_u8: np.ndarray, title: str, scale: int = DEFAULT_SCALE,
                 crop_size: int = DEFAULT_CROP) -> Image.Image:
    """Render a grayscale panel with a header and crop frame.

    The header always states whether the panel is a crop (with row/column
    range) or the full image, so cropping cannot be mistaken for content.
    """
    cropped, r0, r1, c0, c1, was_cropped = crop_center(array_u8, crop_size)
    tag = f"CROP r{r0}:{r1} c{c0}:{c1}" if was_cropped else "FULL"
    header = f"{title} | {tag}"

    ch, cw = cropped.shape
    panel_w = cw * scale + 2 * BORDER_PX
    panel_h = HEADER_PX + ch * scale + 2 * BORDER_PX
    panel = Image.new("L", (panel_w, panel_h), color=255)
    draw = ImageDraw.Draw(panel)
    draw.text((2, 1), header, fill=0)

    pixels = Image.fromarray(cropped, mode="L").resize(
        (cw * scale, ch * scale), Image.Resampling.NEAREST)
    panel.paste(pixels, (BORDER_PX, HEADER_PX + BORDER_PX))
    # Black frame marks the boundary of the shown (possibly cropped) region.
    draw.rectangle(
        [BORDER_PX - 1, HEADER_PX + BORDER_PX - 1,
         BORDER_PX + cw * scale, HEADER_PX + BORDER_PX + ch * scale],
        outline=0)
    return panel


def difference_array(reconstruction: np.ndarray,
                     reference: np.ndarray) -> np.ndarray:
    return np.asarray(reconstruction, dtype=np.float64) - \
        np.asarray(reference, dtype=np.float64)


def difference_uint8(diff: np.ndarray) -> tuple:
    """Map |diff| to grayscale with a shared scale; return ``(uint8, scale)``."""
    peak = float(np.max(np.abs(diff)))
    scale = 1.0 / peak if peak > 1e-12 else 1.0
    return to_uint8(np.abs(diff) * scale), scale


def _metric_dict(result, truth):
    y = result.observation
    residual = result.operator.forward(result.x) - y
    metrics = {
        "initial_residual_norm": result.initial_residual_norm,
        "final_residual_norm": result.final_residual_norm,
        "residual_relative_reduction": (
            (result.initial_residual_norm - result.final_residual_norm)
            / result.initial_residual_norm
            if result.initial_residual_norm > 0 else 0.0),
        "residual_rms": float(np.sqrt(np.mean(residual ** 2))),
        "n_observed": result.operator.n_observed,
        "n_pixels": result.operator.n_pixels,
    }
    if truth is not None:
        err = result.x - np.asarray(truth, dtype=np.float64)
        metrics.update({
            "rmse_vs_truth_uncropped": float(np.sqrt(np.mean(err ** 2))),
            "max_abs_error_uncropped": float(np.max(np.abs(err))),
        })
    return metrics


def build_report(result, *, truth=None, crop_size=DEFAULT_CROP,
                 diff_scale=None) -> dict:
    """Assemble the full, JSON-serializable numerical report (uncropped)."""
    truth = result.truth if truth is None and result.truth is not None else truth
    if truth is None:
        diff_reference = result.operator.adjoint(result.observation)
        reference_name = "zero-filled observed data"
    else:
        diff_reference = np.asarray(truth, dtype=np.float64)
        reference_name = "synthetic truth"
    diff = difference_array(result.x, diff_reference)
    _, auto_scale = difference_uint8(diff)

    return {
        "status": result.status,
        "message": result.message,
        "iterations": result.iterations,
        "image_shape": list(result.operator.shape),
        "config": {k: float(v) for k, v in result.config.items()},
        "metrics": _metric_dict(result, truth),
        "difference": {
            "reference": reference_name,
            "visual_scale": float(diff_scale if diff_scale is not None
                                  else auto_scale),
            "mean_uncropped": float(np.mean(diff)),
            "max_abs_uncropped": float(np.max(np.abs(diff))),
            "rms_uncropped": float(np.sqrt(np.mean(diff ** 2))),
        },
        "figures": {
            "crop_size": int(crop_size),
            "cropping_is_display_only": True,
            "metrics_use_uncropped_values": True,
        },
        "iterations_log": [
            {
                "iteration": rec.iteration,
                "residual_norm": rec.residual_norm,
                "gradient_residual_norm": rec.gradient_residual_norm,
                "projection_objective": rec.projection_objective,
                "split_objective": rec.split_objective,
                "offset_l1": rec.offset_l1,
                "inner_iters": rec.inner_iters,
                "inner_converged": rec.inner_converged,
            }
            for rec in result.records
        ],
    }


def save_outputs(result, outdir, *, truth=None,
                 crop_size=DEFAULT_CROP, scale=DEFAULT_SCALE):
    """Save observation/reconstruction/difference PNGs and report.json.

    Observed panels show mask gaps as black; every panel labels its crop.
    Returns the parsed report dictionary.
    """
    import os
    os.makedirs(outdir, exist_ok=True)
    truth = result.truth if truth is None and result.truth is not None else truth

    observed_display = result.operator.mask * result.observation
    reference = (np.asarray(truth, dtype=np.float64) if truth is not None
                 else result.operator.adjoint(result.observation))
    diff = difference_array(result.x, reference)
    diff_u8, diff_scale = difference_uint8(diff)

    panels = {
        "observation.png": (to_uint8(observed_display), "observed (mask)"),
        "reconstruction.png": (to_uint8(result.x), "reconstruction"),
        "difference.png": (
            diff_u8, f"|diff| x{diff_scale:g}" if diff_scale != 1.0
            else "|diff|"),
    }
    for filename, (pixels, title) in panels.items():
        render_panel(pixels, title, scale=scale,
                     crop_size=crop_size).save(os.path.join(outdir, filename))

    report = build_report(result, truth=truth, crop_size=crop_size,
                          diff_scale=diff_scale)
    with open(os.path.join(outdir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    return report
