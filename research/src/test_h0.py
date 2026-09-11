from __future__ import annotations

import inspect
import unittest
from pathlib import Path

from .protocol import FrameIdentity, post_bootstrap_ratio_roles, ratio_schedule_payload
from .run_h0 import CONDITIONS, load_config
from .schema import VISUAL_STATE_CONTRACT, VISUAL_STATE_CONTRACT_SHA256


def identities(count: int = 40) -> list[FrameIdentity]:
    return [FrameIdentity("euroc", "machine_hall", "sequence", "MH_01_easy",
                          index * 2, index, 1_000_000_000 + index)
            for index in range(count)]


class DecompositionProtocolTest(unittest.TestCase):
    def test_ratio_point_two_has_frozen_phase(self) -> None:
        frames = identities()
        ratio = post_bootstrap_ratio_roles(
            frames, bootstrap_end_candidate_index=8, anchor_ratio=.2,
        )
        anchors = [item.candidate_index for item in frames if ratio[item.key] == "anchor"]
        self.assertEqual(anchors, [*range(9), 9, 14, 19, 24, 29, 34, 39])

    def test_future_sweep_ratios_are_supported_and_deterministic(self) -> None:
        frames = identities(200)
        for value in (1.0, .5, .25, .1, .05):
            left = post_bootstrap_ratio_roles(
                frames, bootstrap_end_candidate_index=8, anchor_ratio=value,
            )
            right = post_bootstrap_ratio_roles(
                frames, bootstrap_end_candidate_index=8, anchor_ratio=value,
            )
            self.assertEqual(left, right)
            payload = ratio_schedule_payload(
                frames, bootstrap_end_candidate_index=8, anchor_ratio=value,
            )
            self.assertEqual(payload["requested_post_bootstrap_anchor_ratio"], value)
            self.assertEqual(payload["anchor_count"], sum(role == "anchor" for role in left.values()))

    def test_visual_state_contract_is_fmap_only(self) -> None:
        self.assertEqual(VISUAL_STATE_CONTRACT["stored_hidden_visual_state"], ["fmap"])
        self.assertEqual(VISUAL_STATE_CONTRACT["imap"], "deterministic_zero")
        self.assertIn("removed", VISUAL_STATE_CONTRACT["colors"])
        self.assertEqual(len(VISUAL_STATE_CONTRACT_SHA256), 64)

    def test_final_conditions_and_config(self) -> None:
        config, _ = load_config()
        self.assertEqual(CONDITIONS, ("full_rgb", "sparse_rgb", "true_fmap"))
        self.assertEqual(config["experiment"]["anchor_ratio"], .2)
        source = inspect.getsource(__import__(
            "research.src.run_h0",
            fromlist=["run"],
        ))
        self.assertNotIn("jepa_fmap", source)
        self.assertNotIn("from .predictor import", source)
        self.assertNotIn("from .jepa_fmap import", source)

    def test_only_three_public_phase1_runners_and_configs_exist(self) -> None:
        source_root = Path(__file__).parent
        runners = {path.name for path in source_root.glob("run_*.py")}
        self.assertEqual(runners, {"run_h0.py", "run_h1.py", "run_h2.py"})
        config_root = source_root.parent / "configs"
        configs = {path.name for path in config_root.glob("phase1_feasibility_*.yaml")}
        self.assertEqual(configs, {
            "phase1_feasibility_h0.yaml", "phase1_feasibility_h1.yaml",
            "phase1_feasibility_h2.yaml",
        })


if __name__ == "__main__":
    unittest.main()
