"""Lightweight, CPU-only contract checks for final Experiment 3."""
from __future__ import annotations

import unittest
from pathlib import Path

from .analyze import SPARSE_TRAJECTORY_NOTE, common_accepted, recovery_metrics
from .run_exp3 import FINAL_INTERVAL, FINAL_METHODS, FINAL_SEQUENCES, build_anchor_schedule, final_run_matrix, repo_root, resolve_config, sequence_config
from .runtime import source_patch_slots_are_valid


def _method(*, schedule: list[int], accepted: list[int], bootstrap: int) -> dict:
    return {"scheduled_anchor_timestamps_ns": schedule, "accepted_graph_timestamps_ns": accepted, "scheduled_anchor_candidate_indices": list(range(len(schedule))), "accepted_anchor_candidate_indices": list(range(len(accepted))), "bootstrap_end_candidate_index": bootstrap}


class FinalExperiment3Tests(unittest.TestCase):
    def test_frozen_final_config(self) -> None:
        config = resolve_config(repo_root())
        self.assertEqual(tuple(config["experiment"]["sequences"]), FINAL_SEQUENCES)
        self.assertEqual(config["experiment"]["post_bootstrap_anchor_interval"], 8)
        self.assertEqual(config["experiment"]["nominal_rgb_upload_ratio"], .125)

    def test_three_sequence_k8_matrix(self) -> None:
        rows = final_run_matrix()
        self.assertEqual(len(rows), 9)
        self.assertEqual({row["sequence"] for row in rows}, set(FINAL_SEQUENCES))
        self.assertEqual({row["method"] for row in rows}, set(FINAL_METHODS))
        self.assertTrue(all(row["anchor_interval"] in {None, FINAL_INTERVAL} for row in rows))

    def test_schedule_includes_bootstrap_then_every_eighth_candidate(self) -> None:
        self.assertEqual(build_anchor_schedule(30, 7), tuple(range(8)) + (8, 16, 24))
        with self.assertRaises(ValueError):
            build_anchor_schedule(5, 5)

    def test_sequence_config_resolves_all_final_assets(self) -> None:
        base = resolve_config(repo_root())
        config = sequence_config(base, "MH_03_medium")
        self.assertEqual(config["experiment"]["sequence"], "MH_03_medium")
        self.assertGreater(config["experiment"]["processed_frames"], 0)
        self.assertTrue(Path(config["paths"]["groundtruth"]).is_file())

    def test_common_anchor_intersection_and_coverage(self) -> None:
        sparse = _method(schedule=[1, 2, 3, 4], accepted=[1, 2, 4], bootstrap=1)
        oracle = _method(schedule=[1, 2, 3, 4], accepted=[1, 2, 3], bootstrap=1)
        contract = common_accepted(sparse, oracle)
        self.assertEqual(contract["common_accepted_anchor_timestamps_ns"], [1, 2])
        self.assertEqual(contract["common_scheduled_coverage"], .5)
        self.assertFalse(contract["accepted_timestamps_exact"])

    def test_gap_recovery_and_undefined_branch(self) -> None:
        values = recovery_metrics(1.0, 5.0, 3.0)
        self.assertEqual(values["sparse_degradation_abs"], 4.0)
        self.assertEqual(values["oracle_recovery_abs"], 2.0)
        self.assertEqual(values["gap_recovery"], .5)
        self.assertIsNone(recovery_metrics(5.0, 1.0, .5)["gap_recovery"])

    def test_fixed_m_placeholder_source_prohibition(self) -> None:
        kinds = ["anchor", "latent", "anchor"]
        self.assertTrue(source_patch_slots_are_valid(kinds, [0, 95, 192], 96))
        self.assertFalse(source_patch_slots_are_valid(kinds, [96, 191], 96))
        self.assertTrue(source_patch_slots_are_valid(["anchor", "latent", "anchor"], [0, 7, 16], 8))
        self.assertFalse(source_patch_slots_are_valid(["anchor", "latent", "anchor"], [8], 8))

    def test_sparse_plot_semantics_are_explicit(self) -> None:
        self.assertIn("not a dense estimated trajectory", SPARSE_TRAJECTORY_NOTE)


if __name__ == "__main__":
    unittest.main()
