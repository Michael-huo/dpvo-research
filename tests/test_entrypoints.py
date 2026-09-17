"""CPU contracts for the config-led Train and Infer interfaces."""
from __future__ import annotations

import copy
import unittest
from pathlib import Path
from unittest.mock import patch

from latent_vslam import infer, train
from latent_vslam.inference_runtime import CONDITIONS, load_config
from latent_vslam.manifests import config_protocol_fingerprint, empty_index
from latent_vslam.protocol import REPO_ROOT, load_sequence_records, sha256_file
from latent_vslam.stride_sweep import load_protocol


class EntrypointContractsTest(unittest.TestCase):
    def test_config_names_and_four_modes(self):
        infer_config = REPO_ROOT / "configs/infer/mh01_four_modes.yaml"
        bridge_config = REPO_ROOT / "configs/train/bridge_mh01.yaml"
        self.assertEqual(infer.parser().parse_args(["--config", str(infer_config)]).config, infer_config)
        self.assertEqual(train.parser().parse_args(["--config", str(bridge_config)]).config, bridge_config)
        self.assertEqual(set(infer.MODES.values()), set(CONDITIONS))
        self.assertEqual(len(load_sequence_records(load_config()[0], "MH_01_easy")), 1841)

    def test_train_dispatches_separate_models(self):
        bridge = REPO_ROOT / "configs/train/bridge_mh01.yaml"
        sweep = REPO_ROOT / "configs/train/predictor_stride_sweep_mh01.yaml"
        with patch.object(train, "run_bridge_training", return_value={"training": "bridge"}) as run_bridge, \
             patch.object(train, "train_stride_sweep", return_value={"training": "predictor"}) as run_predictor:
            self.assertEqual(train.run(bridge)["training"], "bridge")
            self.assertEqual(train.run(sweep)["training"], "predictor")
        self.assertEqual(run_bridge.call_args.kwargs["config_path"], bridge)
        run_predictor.assert_called_once_with(sweep)

    def test_infer_preflights_before_runtime(self):
        config = REPO_ROOT / "configs/infer/mh01_four_modes.yaml"
        with patch.object(infer, "preflight_inference") as preflight, \
             patch.object(infer, "run_four_modes", return_value={"status": "complete"}) as execute:
            self.assertEqual(infer.run(config)["infer_modes"], list(infer.MODES))
        preflight.assert_called_once()
        self.assertEqual(execute.call_args.kwargs["config_path"], config)

    def test_missing_predictor_fails_before_model_load(self):
        config, _ = load_config()
        missing = copy.deepcopy(config)
        missing["paths"]["predictor"] = "/tmp/a5-missing-predictor.pt"
        with patch.object(infer, "load_canonical_predictor") as load:
            with self.assertRaisesRegex(FileNotFoundError, "Predictor checkpoint is missing"):
                infer.preflight_inference(missing)
        load.assert_not_called()

    def test_stride_sweep_requires_training_manifest_before_gpu(self):
        sweep = REPO_ROOT / "configs/infer/mh01_stride_sweep.yaml"
        protocol, canonical, resolved = load_protocol(sweep)
        self.assertEqual(tuple(protocol["anchor_strides"]), (3, 5, 10))
        with patch.object(infer, "TRAINING_ROOT", Path("/tmp/a5-no-training-manifest")), \
             patch("latent_vslam.cuda_devices.CudaDevicePool.discover") as discover:
            with self.assertRaisesRegex(FileNotFoundError, "stride training manifest is missing"):
                infer.preflight_stride_sweep(protocol, canonical, resolved)
        discover.assert_not_called()
        self.assertEqual(sha256_file(sweep), sha256_file(
            REPO_ROOT / "configs/train/predictor_stride_sweep_mh01.yaml"))

    def test_paths_and_display_names_do_not_change_scientific_config_hash(self):
        config, _ = load_config()
        modified = copy.deepcopy(config)
        modified["experiment"]["name"] = "another_display_name"
        modified["paths"]["output_root"] = "/tmp/another-result-root"
        modified["jepa"]["checkpoint"] = "/tmp/same-weight.pt"
        self.assertEqual(config_protocol_fingerprint(config)["config_protocol_sha256"],
                         config_protocol_fingerprint(modified)["config_protocol_sha256"])
        self.assertEqual(set(empty_index("inference", ["MH_01_easy"])),
                         {"schema_version", "module", "run_policy", "requested_sequences",
                          "model_manifest", "sequences"})

    def test_old_entrypoints_and_result_trees_are_absent(self):
        self.assertFalse((REPO_ROOT / "research/src").exists())
        for name in ("run_h0", "run_h1", "run_h2", "run_anchor_budget"):
            self.assertFalse((REPO_ROOT / "latent_vslam" / f"{name}.py").exists())
        for name in ("h0-state", "h1-interface", "h2-prediction", "anchor-budget"):
            self.assertFalse((REPO_ROOT / "research/results" / name).exists())


if __name__ == "__main__":
    unittest.main()
