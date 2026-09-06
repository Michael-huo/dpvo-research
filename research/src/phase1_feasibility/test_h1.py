from __future__ import annotations

import unittest

from .run_h1 import CONDITIONS, TRAINING_SEQUENCE, _metadata, load_config


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

if __name__ == "__main__":
    unittest.main()
