from __future__ import annotations

import unittest
import inspect

import numpy as np
import torch

from .run_h1 import CONDITIONS, TRAINING_SEQUENCE, _metadata, load_config
from .training_runtime import ResidentRows


class H1ContractTest(unittest.TestCase):
    def test_frozen_bridge_recipe(self) -> None:
        config, _ = load_config()
        self.assertEqual(config["experiment"]["bridge_initialization_seed"], 1236)
        self.assertEqual(config["bridge"]["passes"], 2)
        self.assertEqual(config["bridge"]["epochs_per_pass"], 30)
        self.assertTrue(config["bridge"]["reset_optimizer_and_grad_scaler_between_passes"])
        self.assertEqual(config["experiment"]["training_sequence"], TRAINING_SEQUENCE)

    def test_h1_conditions_are_oracle_interface_not_deployment(self) -> None:
        self.assertEqual(CONDITIONS, (
            "full_rgb", "sparse_rgb", "true_fmap", "oracle_jepa_bridge",
        ))
        metadata = _metadata()
        self.assertTrue(metadata["oracle_jepa_bridge"]["offline_reference_only"])
        self.assertFalse(metadata["oracle_jepa_bridge"]["strict_deployment"])
        self.assertEqual(metadata["oracle_jepa_bridge"]["input_source"],
                         "offline_oracle_hidden_jepa_to_bridge")

    def test_formal_h1_uses_parallel_preparation_residency_and_sequential_dpvo(self) -> None:
        from . import run_h1
        source = inspect.getsource(run_h1.run)
        self.assertIn("extract_parallel(", source)
        self.assertIn("ResidentH1View(", source)
        self.assertNotIn("require_full", source)
        self.assertIn("run_sequential_trajectory_jobs(", source)
        self.assertNotIn("run_formal_jobs(", source)
        self.assertNotIn("training-data-backend", source)

    def test_resident_native_fp16_gather_matches_memmap_conversion(self) -> None:
        native = np.arange(7 * 3 * 2, dtype=np.float16).reshape(7, 3, 2)
        resident = ResidentRows(
            {"x": native}, range(7), device=torch.device("cpu"),
        )
        rows = np.asarray([6, 1, 4], dtype=np.int64)
        expected = torch.from_numpy(np.asarray(native[rows], dtype=np.float32))
        self.assertTrue(torch.equal(resident.batch("x", rows), expected))
        self.assertEqual(resident.diagnostics["batch_h2d_bytes"], 0)
        resident.close()

if __name__ == "__main__":
    unittest.main()
