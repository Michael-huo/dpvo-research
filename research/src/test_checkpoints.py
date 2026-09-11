from __future__ import annotations

import copy
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from . import bridge_checkpoint, registry, run_h0, run_h1, run_h2, scientific_lineage
from .jepa_runtime import state_dict_sha256
from .predictor import predictor_state_sha256
from .protocol import REPO_ROOT, canonical_sha256, repo_path, sha256_file
from .schema import VISUAL_STATE_CONTRACT_SHA256


class _FakeModel:
    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        self._state = state

    def cuda(self): return self
    def eval(self): return self
    def requires_grad_(self, value): return self
    def load_state_dict(self, state, strict=True): self._state = state
    def state_dict(self): return self._state


class FreshCheckpointPolicyTest(unittest.TestCase):
    def _bridge_checkpoint(
        self, state: dict[str, torch.Tensor], training_input: dict,
    ) -> dict:
        lineage = {
            "training_input": training_input,
            "bootstrap_end_candidate_index": 7,
            "coordinate_protocol": {},
        }
        lineage["training_lineage_sha256"] = canonical_sha256(lineage)
        return {
            "schema_version": 2, "training_recipe": {},
            "architecture": bridge_checkpoint.BRIDGE_ARCHITECTURE,
            "layer_zero_based": 5,
            "state_dict": state,
            "state_dict_sha256": state_dict_sha256(state),
            "training_lineage": lineage,
            "coordinate_protocol": {},
        }

    def test_h2_accepts_only_bridge_matching_current_h1_training_input(self) -> None:
        state = {"weight": torch.arange(4, dtype=torch.float32)}
        expected = {"config_protocol_sha256": "current", "source_sha256": "current"}
        checkpoint = self._bridge_checkpoint(state, expected)
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root / "bridge.pt"
            torch.save(checkpoint, path)
            with patch.object(bridge_checkpoint, "REPO_ROOT", root), \
                 patch.object(bridge_checkpoint, "build_bridge", return_value=_FakeModel(state)):
                model, metadata, _ = bridge_checkpoint.load_compatible_bridge(
                    path, object(), hidden_channels=160,
                    expected_training_input=expected,
                )
                self.assertTrue(torch.equal(model.state_dict()["weight"], state["weight"]))
                self.assertEqual(metadata["file_sha256"], bridge_checkpoint.sha256_file(path))
                with self.assertRaisesRegex(RuntimeError, "run run_h1 first"):
                    bridge_checkpoint.load_compatible_bridge(
                        path, object(), hidden_channels=160,
                        expected_training_input={**expected, "source_sha256": "new-source"},
                    )

    def test_bridge_integrity_failure_is_fail_closed(self) -> None:
        state = {"weight": torch.ones(1)}
        checkpoint = self._bridge_checkpoint(state, {"source_sha256": "current"})
        checkpoint["state_dict_sha256"] = "corrupt"
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root / "bridge.pt"
            torch.save(checkpoint, path)
            with patch.object(bridge_checkpoint, "REPO_ROOT", root):
                with self.assertRaisesRegex(RuntimeError, "run run_h1 first"):
                    bridge_checkpoint.load_compatible_bridge(
                        path, object(), hidden_channels=160,
                        expected_training_input={"source_sha256": "current"},
                    )

    def test_fresh_predictor_validation_binds_current_bridge_hash(self) -> None:
        config = {"training": {
            "cosine_weight": 1.0,
            "smooth_l1_weight": .1,
            "checkpoint_selector": "lowest_validation_total",
        }}
        state = {"weight": torch.arange(3, dtype=torch.float32)}
        calibration = {"threshold": 1.0}
        calibration["calibration_sha256"] = canonical_sha256(calibration)
        lineage = {
            "training_input": {"h1_bridge_sha256": "bridge-a"},
            "bootstrap_end_candidate_index": 7,
            "train_only_calibration_sha256": calibration["calibration_sha256"],
        }
        lineage["training_lineage_sha256"] = canonical_sha256(lineage)
        checkpoint = {
            "schema_version": 1,
            "training_recipe": dict(config["training"]) | {
                "target": "offline_oracle_hidden_jepa_block5",
            },
            "state_dict": state,
            "state_dict_sha256": predictor_state_sha256(state),
            "architecture": {},
            "deployment_protocol": {"mode": "delayed_bracketed", "timestamp_causal": False},
            "train_only_calibration": calibration,
            "training_lineage": lineage,
        }
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "predictor.pt"
            torch.save(checkpoint, path)
            with patch.object(run_h2, "_new_predictor", return_value=_FakeModel(state)), \
                 patch.object(run_h2, "predictor_metadata", return_value={}):
                validated = run_h2._validate_fresh_predictor(path, config, lineage)
                self.assertEqual(validated["training_lineage"], lineage)
                incompatible = {
                    **lineage,
                    "training_input": {"h1_bridge_sha256": "bridge-b"},
                }
                with self.assertRaisesRegex(RuntimeError, "training_lineage_mismatch"):
                    run_h2._validate_fresh_predictor(path, config, incompatible)

    def test_runners_have_no_checkpoint_or_sequence_reuse_path(self) -> None:
        for runner in (run_h0, run_h1, run_h2):
            source = inspect.getsource(runner.run)
            self.assertNotIn("load_index", source)
            self.assertNotIn("validate_sequence_entry", source)
            self.assertNotIn("reused_sequences", source)
            self.assertIn("fresh_sequences", source)
            self.assertIn("publish_current_canonical", source)
        self.assertFalse(hasattr(run_h1, "_load_cached_bridge"))
        self.assertFalse(hasattr(run_h2, "_load_cached_predictor"))
        bridge_source = inspect.getsource(run_h2._load_bridge)
        self.assertIn("h1_training_context", bridge_source)
        self.assertIn("load_compatible_bridge", bridge_source)
        self.assertNotIn("train_bridge", bridge_source)


class ScientificLineageTest(unittest.TestCase):
    def test_legacy_manifest_preserves_fail_closed_scientific_contracts(self) -> None:
        manifest = json.loads(Path(scientific_lineage.__file__).with_name(
            "legacy_scientific_lineage.json",
        ).read_text())
        for module, migrations in manifest.items():
            fingerprint = scientific_lineage.scientific_fingerprint(module)
            expected = {"source_sha256": fingerprint["source_sha256"], "dataset_sha256": "data"}
            for source in migrations:
                actual = expected | {"source_sha256": source}
                self.assertTrue(scientific_lineage.compatible_training_input(actual, expected, module=module))
                self.assertFalse(scientific_lineage.compatible_training_input(
                    actual | {"dataset_sha256": "changed"}, expected, module=module,
                ))
            self.assertFalse(scientific_lineage.compatible_training_input(
                expected | {"source_sha256": "unknown"}, expected, module=module,
            ))
            for key in ("architecture", "objective", "training"):
                with patch.dict(scientific_lineage.MODULE_CONTRACTS[module], {key: "changed"}):
                    changed = scientific_lineage.scientific_fingerprint(module)["source_sha256"]
                    self.assertNotEqual(changed, expected["source_sha256"])
                    self.assertFalse(scientific_lineage.compatible_training_input(
                        actual, expected | {"source_sha256": changed}, module=module,
                    ))


class CanonicalCheckpointCompatibilityTest(unittest.TestCase):
    """Read-only CPU checks against installed canonical checkpoints and dataset."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = REPO_ROOT / "research/results/phase1-feasibility"
        cls.bridge_path = cls.root / "h1_interface/bridge.pt"
        cls.predictor_path = cls.root / "h2_prediction/predictor.pt"
        if not cls.bridge_path.is_file() or not cls.predictor_path.is_file():
            raise unittest.SkipTest("canonical H1/H2 artifacts are not installed")
        cls.before = cls._artifact_hashes()

    @classmethod
    def _artifact_hashes(cls) -> dict:
        return {str(path.relative_to(cls.root)): sha256_file(path)
                for path in cls.root.rglob("*") if path.is_file()}

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._artifact_hashes() != cls.before:
            raise AssertionError("canonical artifact file set or SHA256 changed during CPU validation")

    def test_canonical_bridge_loads_on_cpu_with_rebuilt_scientific_lineage(self) -> None:
        config, _ = bridge_checkpoint.load_h1_config()
        records, base, provenance = bridge_checkpoint.h1_training_context(config)
        checkpoint = torch.load(self.bridge_path, map_location="cpu", weights_only=False)
        calibration = np.loadtxt(repo_path(config["paths"]["calibration"]), delimiter=" ")
        details = run_h1._training_details(
            records, checkpoint["training_lineage"]["bootstrap_end_candidate_index"],
            calibration, config, base,
        )
        self.assertEqual(details["lineage"], checkpoint["training_lineage"])
        with patch.object(torch.nn.Module, "cuda", lambda model: model):
            model, metadata, loaded = bridge_checkpoint.load_compatible_bridge(
                self.bridge_path, details["transform"],
                hidden_channels=int(config["bridge"]["hidden_channels"]),
                expected_training_input=bridge_checkpoint.h1_training_input(base),
                expected_training_lineage=details["lineage"],
            )
        self.assertEqual(state_dict_sha256(model.state_dict()), loaded["state_dict_sha256"])
        self.assertEqual(metadata["file_sha256"], self.before["h1_interface/bridge.pt"])
        self.assertTrue(all(Path(path).parent == Path("research/src")
                            for path in provenance["sources"]["files"]))
        for key in ("dataset_sha256", "config_protocol_sha256", "source_sha256",
                    "h0_state_contract_sha256"):
            with self.subTest(field=key), self.assertRaisesRegex(RuntimeError, "incompatible"):
                bridge_checkpoint.load_compatible_bridge(
                    self.bridge_path, details["transform"], hidden_channels=160,
                    expected_training_input=bridge_checkpoint.h1_training_input(base) | {key: "changed"},
                )

    def test_canonical_predictor_validates_on_cpu_with_rebuilt_scientific_lineage(self) -> None:
        config, _ = run_h2.load_config()
        training_config = copy.deepcopy(config)
        training_config.pop("evaluation", None)
        training_config.pop("diagnostics", None)
        # H2 scientific identity is selected by the runner name; source paths are provenance.
        base, provenance = registry.base_lineage(
            training_config, run_h2.TRAINING_SEQUENCE, [Path(run_h2.__file__)],
            h0_contract_sha256=VISUAL_STATE_CONTRACT_SHA256,
            h1_bridge_sha256=sha256_file(self.bridge_path),
        )
        checkpoint = torch.load(self.predictor_path, map_location="cpu", weights_only=False)
        records = run_h2.load_sequence_records(config, run_h2.TRAINING_SEQUENCE)
        calibration = np.loadtxt(repo_path(config["paths"]["calibration"]), delimiter=" ")
        details = run_h2._predictor_training_details(
            records, checkpoint["training_lineage"]["bootstrap_end_candidate_index"],
            calibration, config, base,
        )
        lineage = details["lineage"] | {
            "train_only_calibration_sha256": checkpoint["train_only_calibration"]["calibration_sha256"],
        }
        lineage["training_lineage_sha256"] = canonical_sha256(lineage)
        validated = run_h2._validate_fresh_predictor(self.predictor_path, config, lineage)
        self.assertEqual(validated["training_lineage"], lineage)
        self.assertEqual(validated["training_recipe"], dict(config["training"]) | {
            "seed": 1234, "training_sequence": run_h2.TRAINING_SEQUENCE,
            "target": "offline_oracle_hidden_jepa_block5",
            "checkpoint_contains_optimizer_or_scaler": False,
        })
        self.assertEqual(set(provenance["sources"]["files"]), {"research/src/run_h2.py"})
        for key in ("split_sha256", "coordinate_transform_sha256",
                    "transport_calibration_protocol_sha256"):
            with self.subTest(field=key), self.assertRaisesRegex(RuntimeError, "lineage_mismatch"):
                run_h2._validate_fresh_predictor(
                    self.predictor_path, config, lineage | {key: "changed"},
                )


if __name__ == "__main__":
    unittest.main()
