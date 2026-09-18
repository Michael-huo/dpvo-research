"""B1 sample assembly and legacy MH01 regression contracts."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from latent_vslam import train
from latent_vslam.b1_training import prepare_predictor_samples
from latent_vslam.bridge_checkpoint import load_bridge_config
from latent_vslam.inference_runtime import load_config as load_inference_config
from latent_vslam.jepa_fmap import contiguous_split, hidden_split_keys
from latent_vslam.protocol import REPO_ROOT, load_sequence_records, post_bootstrap_ratio_roles
from prediction.predictor import build_anchor_intervals, split_anchor_intervals


class B1TrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, _ = load_inference_config()
        cls.records = {sequence: load_sequence_records(cls.config, sequence)
                       for sequence in ("MH_01_easy", "MH_02_easy")}

    def test_mh01_predictor_population_matches_existing_split(self):
        source = self.records["MH_01_easy"]
        roles = post_bootstrap_ratio_roles(
            [row.identity for row in source], bootstrap_end_candidate_index=7,
            anchor_ratio=.2,
        )
        old_split, old_payload = split_anchor_intervals(build_anchor_intervals(source, roles))
        _, budget, preparation = prepare_predictor_samples(
            self.records, {"MH_01_easy": 7}, ("MH_01_easy",),
        )
        self.assertEqual(old_payload["split_sha256"],
                         "68eff0d531cb6d3f2d75c14d20a369fc32ef3c51edcfedcb9cae79c5dc364bda")
        self.assertEqual(preparation["per_sequence"]["MH_01_easy"]["split"], old_payload)
        for name, original in old_split.items():
            assembled = budget["split"][name]
            self.assertEqual([row.anchor_keys for row in assembled],
                             [row.anchor_keys for row in original])
            self.assertEqual([row.local_interval_index for row in assembled],
                             [row.interval_index for row in original])
        self.assertEqual(budget["sample_counts"]["total"], {
            "train": {"intervals": 219, "queries": 876},
            "validation": {"intervals": 72, "queries": 288},
            "test": {"intervals": 73, "queries": 292},
            "all": {"intervals": 364, "queries": 1456},
        })

    def test_mh01_bridge_population_matches_existing_split(self):
        config, _ = load_bridge_config()
        rows = load_sequence_records(config, "MH_01_easy")
        identities = [row.identity for row in rows]
        roles = post_bootstrap_ratio_roles(
            identities, bootstrap_end_candidate_index=7, anchor_ratio=.2,
        )
        split = contiguous_split(identities)
        self.assertEqual(split["split_sha256"],
                         "b55a790cf8c26b52064f0f07a4c05bb5aa33ec7b95f7d2613b1303d2e9dde07b")
        self.assertEqual({name: len(keys) for name, keys in
                          hidden_split_keys(identities, roles, split).items()},
                         config["split"]["expected_hidden_counts"])

    def test_two_sequences_keep_local_identity_and_unique_global_index(self):
        sequences = ("MH_01_easy", "MH_02_easy")
        _, budget, _ = prepare_predictor_samples(
            self.records, {name: 7 for name in sequences}, sequences,
        )
        intervals = [row for rows in budget["split"].values() for row in rows]
        self.assertEqual(len({row.interval_index for row in intervals}), len(intervals))
        self.assertTrue(all(row.global_interval_index == row.interval_index for row in intervals))
        self.assertTrue(all({row.anchor0.sequence, row.anchor1.sequence,
                             *(query.identity.sequence for query in row.hidden)}
                            == {row.anchor0.sequence} for row in intervals))
        self.assertEqual(budget["sample_counts"]["total"], {
            "train": {"intervals": 400, "queries": 1600},
            "validation": {"intervals": 131, "queries": 524},
            "test": {"intervals": 133, "queries": 532},
            "all": {"intervals": 664, "queries": 2656},
        })

    def test_b1_config_dispatch(self):
        path = REPO_ROOT / "configs/train/b1_bridge.yaml"
        with patch.object(train, "run_bridge_training", return_value={"status": "complete"}) as bridge:
            train.run(path, sequences_override=["MH_01_easy"])
        self.assertEqual(bridge.call_args.args[0], ("MH_01_easy",))
        self.assertTrue(bridge.call_args.kwargs["b1"])
        with patch.object(train, "run_b1_predictor", return_value={"status": "complete"}) as predictor:
            train.run(REPO_ROOT / "configs/train/b1_predictor.yaml")
        self.assertEqual(predictor.call_args.args[0], ("MH_01_easy", "MH_02_easy"))


if __name__ == "__main__":
    unittest.main()
