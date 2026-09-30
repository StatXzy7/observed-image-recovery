import json
import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from grayrecon.reporting import (
    BORDER_PX, DEFAULT_CROP, HEADER_PX, build_report,
    crop_center, difference_uint8, render_panel, save_outputs, to_uint8)
from grayrecon.solver import Status, solve


def run_solver(**kwargs):
    defaults = dict(shape=(12, 12), missing_fraction=0.3,
                    latent_dim=6, hidden_dim=16,
                    max_iters=10, pixel_step=1.0,
                    lam=0.02, inner_max_iters=25, tol=1e-6)
    defaults.update(kwargs)
    return solve(**defaults)


class TestPngExports(unittest.TestCase):
    def setUp(self):
        self.result = run_solver()
        self.tmp = tempfile.mkdtemp()
        self.report = save_outputs(self.result, self.tmp,
                                   crop_size=8, scale=8)

    def test_all_files_are_valid_pngs(self):
        for name in ("observation.png", "reconstruction.png",
                     "difference.png", "report.json"):
            self.assertTrue(os.path.exists(os.path.join(self.tmp, name)))
        for name in ("observation.png", "reconstruction.png",
                     "difference.png"):
            with Image.open(os.path.join(self.tmp, name)) as im:
                self.assertEqual(im.format, "PNG")
                self.assertEqual(im.mode, "L")
                im.load()

    def test_exported_pixels_match_arrays(self):
        # Reconstruction panel pixels (excluding header/frame) must equal the
        # uint8-clipped solver output for the center crop.
        with Image.open(os.path.join(self.tmp, "reconstruction.png")) as im:
            panel = np.array(im)
        crop, r0, r1, c0, c1, _ = crop_center(self.result.x, 8)
        expected = to_uint8(crop)
        block = panel[HEADER_PX + BORDER_PX:
                      HEADER_PX + BORDER_PX + expected.shape[0] * 8:8,
                      BORDER_PX:BORDER_PX + expected.shape[1] * 8:8]
        np.testing.assert_array_equal(block, expected)

    def test_observation_has_black_missing_pixels(self):
        with Image.open(os.path.join(self.tmp, "observation.png")) as im:
            panel = np.array(im)
        crop_mask, _, _, _, _ = crop_center(
            self.result.mask.astype(np.uint8), 8)[:5]
        block = panel[HEADER_PX + BORDER_PX::8, BORDER_PX::8]
        block = block[:crop_mask.shape[0], :crop_mask.shape[1]]
        self.assertTrue(np.all(block[crop_mask == 0] == 0))

    def test_crop_is_labeled_in_header_and_frame(self):
        with Image.open(os.path.join(self.tmp, "observation.png")) as im:
            panel = np.array(im)
        # Dark text pixels exist in the header band (CROPPED label).
        self.assertGreater(int((panel[:HEADER_PX, :] < 100).sum()), 0)
        # Black frame immediately surrounds the pixel region.
        frame_row = HEADER_PX + BORDER_PX - 1
        self.assertTrue(np.any(panel[frame_row, :] == 0))

    def test_full_image_panel_is_labeled_full(self):
        tmp = tempfile.mkdtemp()
        save_outputs(self.result, tmp, crop_size=99, scale=4)
        with Image.open(os.path.join(tmp, "reconstruction.png")) as im:
            panel = np.array(im)
        self.assertGreater(int((panel[:HEADER_PX, :] < 100).sum()), 0)
        h, w = self.result.x.shape
        self.assertEqual(
            panel.shape[1], w * 4 + 2 * BORDER_PX)
        self.assertEqual(
            panel.shape[0], h * 4 + 2 * BORDER_PX + HEADER_PX)

    def test_difference_png_pixels_match_scaled_difference(self):
        with Image.open(os.path.join(self.tmp, "difference.png")) as im:
            panel = np.array(im)
        diff = self.result.x - self.result.truth
        diff_u8, visual_scale = difference_uint8(diff)
        crop, *_ = crop_center(diff_u8, 8)
        block = panel[HEADER_PX + BORDER_PX::8, BORDER_PX::8]
        block = block[:crop.shape[0], :crop.shape[1]]
        np.testing.assert_array_equal(block, crop)
        self.assertAlmostEqual(
            self.report["difference"]["visual_scale"], visual_scale)


class TestNumericalReport(unittest.TestCase):
    def test_report_is_parseable_and_uncropped(self):
        result = run_solver()
        tmp = tempfile.mkdtemp()
        save_outputs(result, tmp, crop_size=6, scale=8)
        with open(os.path.join(tmp, "report.json"), encoding="utf-8") as fh:
            report = json.load(fh)

        self.assertEqual(report["image_shape"], list(result.operator.shape))
        self.assertTrue(report["figures"]["metrics_use_uncropped_values"])
        self.assertTrue(report["figures"]["cropping_is_display_only"])

        truth = result.truth
        expected_rmse = float(np.sqrt(np.mean((result.x - truth) ** 2)))
        self.assertAlmostEqual(
            report["metrics"]["rmse_vs_truth_uncropped"], expected_rmse,
            places=12)

        expected_diff_rms = float(
            np.sqrt(np.mean((result.x - truth) ** 2)))
        self.assertAlmostEqual(
            report["difference"]["rms_uncropped"], expected_diff_rms,
            places=12)

    def test_iteration_log_records_metrics(self):
        result = run_solver(max_iters=4)
        report = build_report(result, crop_size=DEFAULT_CROP)
        self.assertEqual(len(report["iterations_log"]), 4)
        for entry, rec in zip(report["iterations_log"], result.records):
            self.assertEqual(entry["iteration"], rec.iteration)
            self.assertAlmostEqual(
                entry["residual_norm"], rec.residual_norm, places=15)
            self.assertAlmostEqual(
                entry["projection_objective"],
                rec.projection_objective, places=15)
            self.assertAlmostEqual(
                entry["offset_l1"], rec.offset_l1, places=15)
            self.assertEqual(entry["inner_iters"], rec.inner_iters)

    def test_unobservable_report_round_trip(self):
        result = solve(shape=(8, 8), missing_fraction=1.0, max_iters=0)
        self.assertEqual(result.status, Status.UNOBSERVABLE)
        tmp = tempfile.mkdtemp()
        report = save_outputs(result, tmp, crop_size=99, scale=4)
        self.assertEqual(report["status"], Status.UNOBSERVABLE)
        self.assertEqual(report["metrics"]["n_observed"], 0)


if __name__ == "__main__":
    unittest.main()
