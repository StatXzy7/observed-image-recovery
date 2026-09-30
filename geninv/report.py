"""PNG exporters and machine-readable numeric reports.

Images use the internal [-1, 1] scale, converted to uint8 grayscale. The
difference figure explicitly marks the cropped inset; every reported metric
is computed on the full, uncropped image.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def internal_to_uint8(image: np.ndarray) -> np.ndarray:
    """Map an internal [-1, 1] image to uint8 on the display [0, 255] scale."""
    image = np.asarray(image, dtype=np.float64)
    scaled = np.rint((np.clip(image, -1.0, 1.0) * 0.5 + 0.5) * 255.0)
    return scaled.astype(np.uint8)


def _to_gray_png(image: np.ndarray) -> Image.Image:
    return Image.fromarray(internal_to_uint8(image), mode="L")


def _scale_nearest(image: Image.Image, factor: int) -> Image.Image:
    if factor <= 1:
        return image.copy()
    return image.resize(
        (image.width * factor, image.height * factor), Image.NEAREST
    )


def _center_crop_bounds(
    height: int, width: int, crop_fraction: float
) -> tuple[int, int, int, int]:
    crop_fraction = float(np.clip(crop_fraction, 0.05, 1.0))
    ch = max(2, int(round(height * crop_fraction)))
    cw = max(2, int(round(width * crop_fraction)))
    top = (height - ch) // 2
    left = (width - cw) // 2
    return left, top, left + cw, top + ch


def _draw_label(draw: ImageDraw.ImageDraw, text: str, position: tuple[int, int]) -> None:
    draw.text(position, text, fill=255, stroke_width=2, stroke_fill=0)


LABEL_HEIGHT = 22


def _compose_with_label(
    panel: Image.Image, label: str, label_height: int = LABEL_HEIGHT
) -> Image.Image:
    """Place ``panel`` under a dark annotation band so no pixels are hidden.

    The canvas is widened when the label is wider than the panel; the panel
    is then centered so every image pixel and every label glyph is visible.
    """
    measurer = ImageDraw.Draw(Image.new("RGB", (4, 4)))
    bbox = measurer.textbbox((0, 0), label, stroke_width=1)
    text_width = bbox[2] - bbox[0] + 8
    width = max(panel.width, text_width)
    canvas = Image.new(
        "RGB", (width, panel.height + label_height), color=(30, 30, 30)
    )
    canvas.paste(panel.convert("RGB"), ((width - panel.width) // 2, label_height))
    ImageDraw.Draw(canvas).text(
        (4, 4), label, fill=(255, 255, 255),
        stroke_width=1, stroke_fill=(0, 0, 0),
    )
    return canvas


def save_observation_png(
    path: str | Path,
    observation: np.ndarray,
    mask: np.ndarray,
    scale: int = 4,
) -> Path:
    """Save the observation: observed grayscale pixels, missing pixels black.

    A single-pixel dark border outlines each missing pixel so black-on-black
    gaps stay visible. The annotation lives in a separate band above the
    image, leaving every image pixel unobscured.
    """
    observation = np.asarray(observation, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    canvas = np.where(mask, observation, -1.0)
    image = _scale_nearest(_to_gray_png(canvas), scale).convert("RGB")
    draw = ImageDraw.Draw(image)
    height, width = mask.shape
    for i in range(height):
        for j in range(width):
            if not mask[i, j]:
                draw.rectangle(
                    [j * scale, i * scale, (j + 1) * scale - 1, (i + 1) * scale - 1],
                    outline=(40, 40, 40),
                )
    image = _compose_with_label(image, "observation: black = missing")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def save_reconstruction_png(
    path: str | Path, reconstruction: np.ndarray, scale: int = 4
) -> Path:
    """Save the reconstruction as an upscaled grayscale PNG."""
    image = _scale_nearest(_to_gray_png(reconstruction), scale)
    image = _compose_with_label(image, "reconstruction: uncropped")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def save_difference_png(
    path: str | Path,
    reference: np.ndarray,
    estimate: np.ndarray,
    mask: np.ndarray | None = None,
    crop_fraction: float = 0.6,
    scale: int = 4,
) -> Path:
    """Save a two-panel absolute-error PNG.

    Left panel: full uncropped error map with a rectangle marking the crop
    region. Right panel: enlarged center crop of that same error map. Both
    panels share the same per-unit scaling, computed from uncropped values.
    """
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if reference.shape != estimate.shape:
        raise ValueError("reference and estimate shapes must match")
    error = np.abs(reference - estimate)
    height, width = error.shape
    left, top, right, bottom = _center_crop_bounds(
        height, width, crop_fraction
    )
    emax = float(error.max())
    shown = np.ones_like(error)
    if emax > 0.0:
        shown = error / emax
    error_uint8 = np.rint(np.clip(shown, 0.0, 1.0) * 255.0).astype(np.uint8)

    full_img = _scale_nearest(Image.fromarray(error_uint8, mode="L"), scale)
    crop_img = _scale_nearest(
        Image.fromarray(error_uint8[top:bottom, left:right], mode="L"), scale
    )
    pad = 6
    label_h = 18
    panel_w = full_img.width + crop_img.width + 3 * pad
    panel_h = max(full_img.height, crop_img.height) + 2 * pad + label_h
    canvas = Image.new("RGB", (panel_w, panel_h), color=(30, 30, 30))
    full_pos = (pad, pad + label_h)
    crop_pos = (2 * pad + full_img.width, pad + label_h)
    canvas.paste(full_img.convert("RGB"), full_pos)
    canvas.paste(crop_img.convert("RGB"), crop_pos)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        [
            full_pos[0] + left * scale,
            full_pos[1] + top * scale,
            full_pos[0] + right * scale - 1,
            full_pos[1] + bottom * scale - 1,
        ],
        outline=(255, 255, 0),
        width=2,
    )
    def _centered_label(text: str, panel_x: int, panel_width: int) -> None:
        bbox = draw.textbbox((0, 0), text)
        text_w = bbox[2] - bbox[0]
        x = panel_x + max(0, (panel_width - text_w) // 2)
        draw.text(
            (x, pad), text, fill=(255, 255, 255),
            stroke_width=1, stroke_fill=(0, 0, 0),
        )

    _centered_label("full", full_pos[0], full_img.width)
    _centered_label("zoom", crop_pos[0], crop_img.width)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


def build_report(
    result,
    observation: np.ndarray,
    mask: np.ndarray,
    true_image: np.ndarray | None = None,
    config=None,
) -> dict:
    """Build a JSON-serialisable report; all metrics use uncropped arrays."""
    mask_bool = np.asarray(mask, dtype=bool)
    obs_residual = np.asarray(result.residual, dtype=np.float64)
    report = {
        "status": result.status,
        "converged": bool(result.converged),
        "inner_converged": bool(result.inner_converged),
        "message": result.message,
        "image_shape": list(observation.shape),
        "observed_pixel_count": int(result.observed_pixel_count),
        "total_pixel_count": int(observation.size),
        "missing_fraction": float(1.0 - np.mean(mask_bool)),
        "iterations": int(result.iterations),
        "residual_norm": float(result.residual_norm),
        "initial_residual_norm": float(result.initial_residual_norm),
        "relative_residual": float(result.relative_residual),
        "offset_l2": float(np.linalg.norm(result.offset)),
        "offset_l1": float(np.sum(np.abs(result.offset))),
        "observed_rmse_uncropped": float(
            np.sqrt(
                np.sum(obs_residual[mask_bool] ** 2)
                / max(1, result.observed_pixel_count)
            )
        ) if result.observed_pixel_count else float("nan"),
        "history": result.records_as_dicts(),
    }
    if true_image is not None:
        pixel_error = np.asarray(result.reconstruction) - np.asarray(true_image)
        report["reconstruction_rmse_uncropped"] = float(
            np.sqrt(np.mean(pixel_error**2))
        )
        report["reconstruction_max_abs_error_uncropped"] = float(
            np.max(np.abs(pixel_error))
        )
    if config is not None:
        report["config"] = {
            "pixel_step": float(config.pixel_step),
            "sparse_penalty": float(config.sparse_penalty),
            "tie_weight": float(config.tie_weight),
            "latent_step": float(config.latent_step),
            "hidden_step": float(config.hidden_step),
            "inner_tol": float(config.inner_tol),
            "inner_max_iter": int(config.inner_max_iter),
            "outer_tol": float(config.outer_tol),
            "outer_max_iter": int(config.outer_max_iter),
        }
    return _json_safe(report)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_report(path: str | Path, report: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return path


def save_outputs(
    out_dir: str | Path,
    result,
    observation: np.ndarray,
    mask: np.ndarray,
    true_image: np.ndarray | None = None,
    config=None,
    scale: int = 4,
) -> dict[str, Path]:
    """Save observation/reconstruction/difference PNGs and a JSON report."""
    out_dir = Path(out_dir)
    paths = {
        "observation": save_observation_png(
            out_dir / "observation.png", observation, mask, scale=scale
        ),
        "reconstruction": save_reconstruction_png(
            out_dir / "reconstruction.png", result.reconstruction, scale=scale
        ),
        "difference": save_difference_png(
            out_dir / "difference.png",
            reference=(
                true_image if true_image is not None else np.zeros_like(
                    result.reconstruction
                )
            ),
            estimate=result.reconstruction,
            mask=mask,
            scale=scale,
        ),
        "report": save_report(
            out_dir / "report.json",
            build_report(result, observation, mask, true_image, config),
        ),
    }
    return paths
