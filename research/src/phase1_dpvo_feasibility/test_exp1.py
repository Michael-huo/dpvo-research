"""Focused unit checks for the single final Experiment 1 protocol."""
from __future__ import annotations

import json
import inspect
import math
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import torch

from .behavior import (
    BehaviorRecorder, GlobalBottomK, OBSERVATION_DTYPE, ScalarHistogram,
    build_comparison, compact_artifact_arrays, correlation_metrics, global_sim3_window_errors,
    ordinal_label,
)
from .run_exp1 import controlled_output_root
from .runtime import SCHEMA_VERSION, probe_perturbation, resolve_config, run_sanity_gate, run_sequence, splitmix64, splitmix64_array, trajectory_difference


class Experiment1Tests(unittest.TestCase):
    def test_fixed_config_and_controlled_output(self) -> None:
        config = resolve_config(sequence="MH_01_easy", max_frames=3)
        self.assertEqual(config["experiment"]["sequence"], "MH_01_easy")
        self.assertEqual(config["experiment"]["stride"], 2)
        root = Path(config["repo_root"])
        self.assertEqual(controlled_output_root(root), root / "research/results/phase1-dpvo-feasibility/exp1")

    def test_splitmix_sampling_and_verified_correlation_layout(self) -> None:
        values = np.asarray([0, 1, 2, 1234, 2**63], dtype=np.uint64)
        np.testing.assert_array_equal(splitmix64_array(values), np.asarray([splitmix64(int(item)) for item in values], dtype=np.uint64))
        rows = np.zeros(100, dtype=OBSERVATION_DTYPE); rows["score"] = np.arange(100, dtype=np.uint64)[::-1]; rows["factor_uid"] = np.arange(100)
        sampler = GlobalBottomK(10); sampler.add(rows); np.testing.assert_array_equal(sampler.values()["score"], np.arange(10))
        shaped = torch.zeros(1, 7, 7, 3, 3, 2); shaped[0, 5, 1, :, :, 0] = 4; shaped[0, 2, 3, :, :, 0] = 1.5
        metrics = correlation_metrics(shaped.reshape(1, 1, 882))
        self.assertAlmostEqual(float(metrics["margin"][0, 0]), 2.5)
        self.assertAlmostEqual(float(metrics["offset"][0, 0]), math.sqrt(8))

    def test_histogram_lifetime_and_compressed_dtype_contract(self) -> None:
        histogram = ScalarHistogram(0, 1, 10); histogram.add(np.asarray([-1., 0., .5, 1., 2., np.nan]))
        self.assertEqual(histogram.as_dict()["accounted_count"], 6)
        recorder = BehaviorRecorder({"experiment": {"seed": 1234}, "behavior": {"patches_per_frame": 2, "correlation_temperature": 1., "observation_bottom_k": 10}})
        try:
            recorder.record_patches(0, np.ones((2, 3), dtype=np.float32), 16, 16)
            recorder.append_factors([7], [np.uint64(0)], [1], append_frame=5)
            recorder.active_first_update[:] = 2; recorder.active_last_update[:] = 4; recorder.active_update_count[:] = 3
            recorder.remove_factors(np.asarray([True]), "window_inactive")
            self.assertEqual(recorder.factor_lifetime_histogram[3], 1)
        finally:
            recorder.abort()
        arrays = compact_artifact_arrays({"stream_index": np.asarray([0, 2000]), "weight_mean": np.asarray([.2, .8]), "accepted": np.asarray([True, False])})
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "a.npz"; np.savez_compressed(artifact, **arrays)
            with zipfile.ZipFile(artifact) as archive: self.assertTrue(all(member.compress_type == zipfile.ZIP_DEFLATED for member in archive.infolist()))

    def test_global_sim3_and_non_blocking_probe_perturbation(self) -> None:
        count = 60; time = np.linspace(0., 2., count); reference = np.column_stack((time, np.sin(time), np.cos(time)))
        estimate = (reference - np.asarray([2., -1., .5])) / 1.7; estimate[20:40, 1] += .2
        gt = {"timestamp_ns": np.arange(count, dtype=np.int64) * 100_000_000 + 1_000_000_000, "position": reference, "quaternion_wxyz": np.tile(np.asarray([1., 0., 0., 0.]), (count, 1))}
        result = global_sim3_window_errors(estimate, gt["timestamp_ns"], gt)
        self.assertGreater(result["windows"][1]["translation_rmse"], .05)
        poses = np.zeros((3, 7)); poses[:, 6] = 1
        arrays = {"poses": poses, "internal_tstamps": np.arange(3), "euroc_timestamps_ns": np.arange(3, dtype=np.uint64)}
        baseline = {"ate": {"translation_rmse": 1.}}
        for ate, label, representativeness in ((1.049, "small", "normal"), (1.05, "moderate", "caution"), (1.10, "large", "low")):
            diagnostic = probe_perturbation(baseline, {"ate": {"translation_rmse": ate}}, arrays, arrays)
            self.assertTrue(diagnostic["probe_valid"])
            self.assertEqual(diagnostic["perturbation_label"], label)
            self.assertEqual(diagnostic["probe_representativeness"], representativeness)
        invalid = dict(arrays); invalid["internal_tstamps"] = np.asarray([1, 2, 3])
        self.assertFalse(probe_perturbation(baseline, {"ate": {"translation_rmse": 1.2}}, arrays, invalid)["probe_valid"])

    def test_sanity_and_runtime_have_no_repeatability_envelope(self) -> None:
        sanity_source = inspect.getsource(run_sanity_gate)
        sequence_source = inspect.getsource(run_sequence)
        self.assertIn('"sets_full_sequence_tolerance": False', sanity_source)
        self.assertIn('"sanity_validation"] = True', sanity_source)
        self.assertNotIn("baseline" + "_repeat", sanity_source + sequence_source)
        self.assertNotIn("tolerance", sequence_source)

    def test_simple_descriptive_ordinal_labels(self) -> None:
        self.assertEqual(ordinal_label([.1, .11, .12], expected_direction=1, metric="weight_mean"), "monotonic_degradation")
        self.assertEqual(ordinal_label([.1, .101, .102], expected_direction=1, metric="weight_mean"), "no_meaningful_change")
        self.assertEqual(ordinal_label([-.814, -.816, -.810], expected_direction=0, metric="entropy_weight_spearman"), "no_meaningful_change")
        self.assertEqual(SCHEMA_VERSION, 1)

    def test_trajectory_difference_ignores_quaternion_sign(self) -> None:
        left = np.asarray([[0., 0., 0., 0., 0., 0., 1.]])
        right = left.copy(); right[:, 3:7] *= -1
        self.assertEqual(trajectory_difference(left, right)["rotation_rmse_rad"], 0.)

    def test_three_sequence_comparison_fixture_schema_and_figures(self) -> None:
        root = Path(__file__).resolve().parents[3]
        fixture = root / "research/results/phase1-dpvo-feasibility/exp1"
        if not all((fixture / name).is_dir() for name in ("MH_01_easy", "MH_03_medium", "MH_05_difficult")):
            self.skipTest("no local Experiment 1 fixture available")
        with tempfile.TemporaryDirectory(prefix="exp1-test-comparison-") as temporary:
            output = Path(temporary) / "comparison"
            build_comparison({name: fixture / name for name in ("MH_01_easy", "MH_03_medium", "MH_05_difficult")}, output)
            summary = json.loads((output / "summary.json").read_text())
            comparison = json.loads((output / "comparison.json").read_text())
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual({summary["schema_version"], comparison["schema_version"], manifest["schema_version"]}, {SCHEMA_VERSION})
            for section in ("sequence_headlines", "probe_perturbation", "mismatch", "patch_diagnostics", "frozen_threshold_ratios", "difficulty_response"):
                self.assertEqual(set(comparison[section]), {"MH_01_easy", "MH_03_medium", "MH_05_difficult"})
            self.assertEqual(len(list((output / "figures").glob("*.png"))), 5)
            self.assertFalse(list(output.glob("*.npz")))
            self.assertEqual(manifest["sequence_runs"]["MH_01_easy"], {"sequence_id": "MH_01_easy", "result_path": "MH_01_easy"})
            self.assertEqual(manifest["mh01_reference"]["metrics_path"], "MH_01_easy/report/metrics.json")
            report = (output / "REPORT.md").read_text().lower()
            self.assertNotIn("experiment 1" + "b", report)
            self.assertNotIn("easy vs " + "difficult", report)


if __name__ == "__main__":
    unittest.main()
