"""Export tests: decoded PNG pixels must match the solver arrays."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from geninv.config import SolverConfig
from geninv.report import (
    build_report,
    internal_to_uint8,
    save_difference_png,
    save_observation_png,
    save_reconstruction_png,
    save_report,
)
from geninv.solver import UNOBSERVABLE, solve
from geninv.synthetic import make_problem, make_solvable_problem


SCALE = 4
from geninv.report import LABEL_HEIGHT as LABEL_H  # image starts here


def _nearest_tiles(uint8_image: np.ndarray, scale: int) -> np.ndarray:
    image = Image.fromarray(uint8_image, mode="L")
    up = image.resize(
        (image.width * scale, image.height * scale), Image.NEAREST
    )
    return np.array(up)


class PixelExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        rng = np.random.default_rng(0)
        self.image = np.clip(rng.standard_normal((7, 8)) * 0.7, -1.0, 1.0)
        self.mask = np.ones((7, 8), dtype=bool)
        self.mask[5, 1] = False
        self.mask[6, 6] = False
        self.observation = np.where(self.mask, self.image, 0.0)

    def tearDown(self):
        self.tmp.cleanup()

    def test_uint8_mapping_is_exact(self):
        values = np.array([[-1.0, 0.0, 1.0], [-0.5, 0.25, 0.75]])
        expected = np.rint((values * 0.5 + 0.5) * 255).astype(np.uint8)
        self.assertTrue(np.array_equal(internal_to_uint8(values), expected))

    def test_reconstruction_png_pixels_match_array(self):
        path = save_reconstruction_png(
            self.out / "reconstruction.png", self.image, scale=SCALE
        )
        decoded = np.array(Image.open(path).convert("L"))
        expected = _nearest_tiles(internal_to_uint8(self.image), SCALE)
        body = decoded[LABEL_H:]
        # The panel is centered when the label widens the canvas; crop it.
        x0 = (body.shape[1] - expected.shape[1]) // 2
        panel = body[:, x0 : x0 + expected.shape[1]]
        self.assertTrue(np.array_equal(panel, expected))

    def test_observation_png_marks_missing_pixels_black(self):
        path = save_observation_png(
            self.out / "observation.png", self.observation, self.mask,
            scale=SCALE,
        )
        decoded = np.array(Image.open(path).convert("L"))
        body = decoded[LABEL_H:]
        x0 = (body.shape[1] - self.image.shape[1] * SCALE) // 2
        for i, j in [(5, 1), (6, 6)]:
            interior = body[
                i * SCALE + 1 : (i + 1) * SCALE - 1,
                x0 + j * SCALE + 1 : x0 + (j + 1) * SCALE - 1,
            ]
            self.assertTrue(np.all(interior == 0))
        observed_canvas = np.where(self.mask, self.image, -1.0)
        expected = _nearest_tiles(internal_to_uint8(observed_canvas), SCALE)
        body = decoded[LABEL_H:]
        x0 = (body.shape[1] - expected.shape[1]) // 2
        panel = body[:, x0 : x0 + expected.shape[1]]
        observed_mask_up = np.repeat(
            np.repeat(self.mask, SCALE, axis=0), SCALE, axis=1
        )
        self.assertTrue(
            np.array_equal(panel[observed_mask_up], expected[observed_mask_up])
        )

    def test_difference_png_panels_and_crop_mark(self):
        estimate = np.clip(self.image + 0.2, -1.0, 1.0)
        pad = 6
        label_h = 18
        path = save_difference_png(
            self.out / "difference.png",
            reference=self.image,
            estimate=estimate,
            mask=self.mask,
            crop_fraction=0.5,
            scale=SCALE,
        )
        rgb = np.array(Image.open(path).convert("RGB"))
        height, width = self.image.shape
        ch = max(2, int(round(height * 0.5)))
        cw = max(2, int(round(width * 0.5)))
        top, left = (height - ch) // 2, (width - cw) // 2
        error = np.abs(estimate - self.image)
        gray = np.rint(error / error.max() * 255).astype(np.uint8)
        full_up = _nearest_tiles(gray, SCALE)
        crop_up = _nearest_tiles(gray[top : top + ch, left : left + cw], SCALE)

        # Interior blocks avoid panel borders; the yellow crop rectangle is
        # masked out of the comparison below.
        def interior(tile, border=3):
            return tile[border:-border, border:-border]

        full_decoded = rgb[
            pad + label_h : pad + label_h + full_up.shape[0],
            pad : pad + full_up.shape[1],
            :,
        ]
        full_interior = interior(full_decoded)
        gray_interior = interior(full_up)
        not_yellow = ~(
            (full_interior[..., 0] > 200)
            & (full_interior[..., 1] > 200)
            & (full_interior[..., 2] < 100)
        )
        self.assertTrue(
            np.array_equal(
                full_interior[..., 0][not_yellow], gray_interior[not_yellow]
            )
        )
        x0 = 2 * pad + full_up.shape[1]
        crop_decoded = rgb[
            pad + label_h : pad + label_h + crop_up.shape[0],
            x0 : x0 + crop_up.shape[1],
            :,
        ]
        self.assertTrue(
            np.array_equal(
                interior(crop_decoded), np.stack([interior(crop_up)] * 3, -1)
            )
        )
        yellow = (rgb[..., 0] > 200) & (rgb[..., 1] > 200) & (rgb[..., 2] < 100)
        self.assertGreater(int(yellow.sum()), 0)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_json_report_roundtrip_and_metrics(self):
        problem, _ = make_solvable_problem(
            height=8, width=8, missing_fraction=0.2, seed=3,
            latent_dim=8, hidden_dim=20,
        )
        config = SolverConfig(outer_max_iter=5, inner_max_iter=60)
        result = solve(
            problem.observation, problem.kernel, problem.mask,
            problem.generator, config,
        )
        report = build_report(
            result, problem.observation, problem.mask,
            problem.true_image, config,
        )
        path = save_report(self.out / "report.json", report)
        parsed = json.loads(path.read_text())
        self.assertIn("status", parsed)
        self.assertEqual(parsed["iterations"], 5)
        self.assertEqual(len(parsed["history"]), 5)
        self.assertAlmostEqual(
            parsed["residual_norm"], result.residual_norm, places=12
        )
        self.assertAlmostEqual(
            parsed["reconstruction_rmse_uncropped"],
            float(np.sqrt(np.mean(
                (result.reconstruction - problem.true_image) ** 2
            ))),
            places=12,
        )
        self.assertGreater(parsed["missing_fraction"], 0.0)
        self.assertLess(parsed["missing_fraction"], 1.0)
        last = parsed["history"][-1]
        for key in (
            "residual_norm",
            "projection_objective",
            "offset_l2",
            "inner_iterations",
        ):
            self.assertIn(key, last)

    def test_end_to_end_outputs(self):
        problem = make_problem(height=8, width=8, missing_fraction=0.35,
                               seed=2)
        result = solve(
            problem.observation, problem.kernel, problem.mask,
            problem.generator, SolverConfig(outer_max_iter=4),
        )
        from geninv.report import save_outputs

        paths = save_outputs(
            self.out, result, problem.observation, problem.mask,
            problem.true_image,
        )
        for path in paths.values():
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 100)
        parsed = json.loads(paths["report"].read_text())
        self.assertEqual(parsed["image_shape"], [8, 8])
        rec = np.array(Image.open(paths["reconstruction"]).convert("L"))
        self.assertTrue(rec.max() > rec.min())

    def test_unobservable_report_is_explicit(self):
        problem = make_problem(height=6, width=6, missing_fraction=0.0,
                               seed=4)
        mask = np.zeros_like(problem.mask)
        result = solve(
            problem.observation, problem.kernel, mask, problem.generator,
        )
        self.assertEqual(result.status, UNOBSERVABLE)
        report = build_report(result, problem.observation, mask)
        self.assertEqual(report["status"], UNOBSERVABLE)
        self.assertIsNone(report["observed_rmse_uncropped"])
        self.assertEqual(report["observed_pixel_count"], 0)


if __name__ == "__main__":
    unittest.main()
