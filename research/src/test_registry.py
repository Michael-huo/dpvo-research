from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .protocol import atomic_write_json, sha256_file
from .registry import (
    CONDITION_ORDER, LINEAGE_FIELDS, SEQUENCE_FILES, complete_lineage,
    dataset_fingerprint, empty_index, publish_current_canonical,
    render_aggregate_summary, render_sequence_summary, sequence_entry,
    validate_module_manifest,
)


def _summary_result(module: str) -> dict:
    conditions = {}
    for index, key in enumerate(CONDITION_ORDER[module]):
        condition = {
            "canonical_evaluation": {
                "ate_rmse_m": 0.12345678901234566 + index,
                "translation_rpe_rmse_m": 0.23456789012345678 + index,
                "rotation_rpe_rmse_deg": 10.345678901234567 + index,
            },
            "canonical_coverage": {
                "canonical_pose_coverage": 0.9123456789012345 + index / 100.0,
            },
            "runtime": {"final_node_count_before_terminate": 101 + index},
        }
        if key == "predicted_jepa_hidden":
            condition.update({
                "strict_deployment": True,
                "timestamp_causal": False,
                "closing_anchor_online_available": True,
            })
            condition["runtime"].update({
                "rgb_uploaded_frame_count": 23,
                "hidden_online_rgb_violation_count": 0,
            })
        conditions[key] = condition
    result = {
        "candidate_count": 101,
        "hidden_count": 78,
        "schedule": {"actual_full_sequence_anchor_ratio": 0.2039123456789},
        "conditions": conditions,
    }
    if module == "h2_prediction":
        result["efficiency"] = {
            "transmission": {"anchor_ratio": .2039123456789,
                             "encoded_byte_reduction": .75},
            "matched_online_wall_clock": {
                "full_rgb_total_s": 10.0, "h2_total_s": 12.0,
                "h2_over_full_rgb_ratio": 1.2, "extra_cloud_compute_s": 2.0,
            },
            "h2_stage_profile": {
                "stages": {
                    "jepa_encoder": {"total_ms": 1000.0},
                    "jepa_predictor": {"total_ms": 2000.0},
                    "bridge": {"total_ms": 300.0},
                    "dpvo_graph_runtime": {"total_ms": 4000.0},
                },
                "context_wait_ms": {"mean_ms": 250.0, "p95_ms": 400.0},
            },
            "break_even_uplink_bandwidth": {
                "status": "finite", "break_even_uplink_bandwidth_mbps": 8.0,
            },
        }
    return result


class CanonicalManifestTest(unittest.TestCase):
    def _lineage(self, module: str) -> dict:
        base = {
            "dataset_sha256": "dataset", "config_protocol_sha256": "config",
            "source_sha256": "source", "schedule_sha256": None,
            "h0_state_contract_sha256": "contract",
            "h1_bridge_sha256": "bridge" if module != "h0_state" else None,
            "h2_predictor_sha256": "predictor" if module == "h2_prediction" else None,
        }
        self.assertEqual(tuple(base), LINEAGE_FIELDS)
        return complete_lineage(base, "schedule")

    def _module(
        self, root: Path, module: str, sequences: tuple[str, ...],
    ) -> dict:
        (root / "sequences").mkdir(parents=True)
        index = empty_index(module, sequences)
        for sequence in sequences:
            directory = root / "sequences" / sequence
            directory.mkdir()
            lineage = self._lineage(module)
            result = _summary_result(module)
            for filename in SEQUENCE_FILES[module]:
                if filename == "results.json":
                    atomic_write_json(directory / filename, {
                        "schema_version": 1, "module": module, "sequence": sequence,
                        "lineage": lineage, "provenance": {"sources": "current"},
                        "result": result,
                    })
                else:
                    (directory / filename).write_bytes(filename.encode())
            index["sequences"][sequence] = sequence_entry(
                directory, module, lineage, bootstrap_end_candidate_index=7,
            )
        checkpoint_name = {
            "h0_state": None, "h1_interface": "bridge.pt", "h2_prediction": "predictor.pt",
        }[module]
        if checkpoint_name is not None:
            (root / checkpoint_name).write_bytes(b"fresh checkpoint")
            index["canonical_checkpoint"] = {
                "file": checkpoint_name,
                "file_sha256": sha256_file(root / checkpoint_name),
            }
        (root / {"h0_state": "SUMMARY_H0.md", "h1_interface": "SUMMARY_H1.md",
                 "h2_prediction": "SUMMARY_H2.md"}[module]).write_text("summary")
        atomic_write_json(root / "INDEX.json", index)
        return index

    def test_manifest_records_current_provenance_and_artifact_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name) / "h0_state"
            index = self._module(root, "h0_state", ("MH_01_easy",))
            validate_module_manifest(root, "h0_state", index)
            self.assertEqual(index["run_policy"], "fresh_current_canonical_replace")
            self.assertEqual(index["requested_sequences"], ["MH_01_easy"])
            entry = index["sequences"]["MH_01_easy"]
            self.assertEqual(entry["status"], "valid")
            self.assertEqual(entry["lineage"]["dataset_sha256"], "dataset")
            (root / "sequences/MH_01_easy/trajectory.png").write_bytes(b"corrupt")
            with self.assertRaisesRegex(RuntimeError, "artifact hash mismatch"):
                validate_module_manifest(root, "h0_state", index)

    def test_manifest_fails_closed_on_result_lineage_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name) / "h0_state"
            index = self._module(root, "h0_state", ("MH_01_easy",))
            path = root / "sequences/MH_01_easy/results.json"
            payload = json.loads(path.read_text())
            payload["lineage"]["source_sha256"] = "different"
            atomic_write_json(path, payload)
            entry = index["sequences"]["MH_01_easy"]
            entry["artifacts"]["results.json"] = sha256_file(path)
            with self.assertRaisesRegex(RuntimeError, "lineage mismatch"):
                validate_module_manifest(root, "h0_state", index)

    def test_manifest_sequence_set_is_exactly_the_request(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name) / "h0_state"
            index = self._module(root, "h0_state", ("MH_03_medium",))
            index["requested_sequences"].append("MH_01_easy")
            with self.assertRaisesRegex(RuntimeError, "sequence set mismatch"):
                validate_module_manifest(root, "h0_state", index)

    def test_manifest_detects_checkpoint_artifact_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name) / "h1_interface"
            index = self._module(root, "h1_interface", ("MH_01_easy",))
            (root / "bridge.pt").write_bytes(b"different checkpoint")
            with self.assertRaisesRegex(RuntimeError, "checkpoint artifact hash mismatch"):
                validate_module_manifest(root, "h1_interface", index)

    def test_current_canonical_publish_removes_unrequested_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            parent = Path(name)
            destination = parent / "h2_prediction"
            staged = parent / "staged"
            self._module(destination, "h2_prediction", ("MH_01_easy",))
            index = self._module(staged, "h2_prediction", ("MH_03_medium",))
            validate_module_manifest(staged, "h2_prediction", index)
            publish_current_canonical(staged, destination)
            self.assertEqual(
                {path.name for path in (destination / "sequences").iterdir()},
                {"MH_03_medium"},
            )
            self.assertFalse(staged.exists())
            self.assertFalse((parent / ".h2_prediction.previous").exists())

    def test_failed_publish_restores_previous_canonical_module(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            parent = Path(name)
            destination = parent / "h0_state"
            staged = parent / "staged"
            self._module(destination, "h0_state", ("MH_01_easy",))
            self._module(staged, "h0_state", ("MH_03_medium",))
            real_rename = Path.rename

            def fail_staged(path: Path, target: Path) -> Path:
                if path == staged:
                    raise OSError("injected publish failure")
                return real_rename(path, target)

            with patch.object(Path, "rename", autospec=True, side_effect=fail_staged):
                with self.assertRaisesRegex(OSError, "injected publish failure"):
                    publish_current_canonical(staged, destination)
            self.assertTrue((destination / "sequences/MH_01_easy").is_dir())
            self.assertFalse((destination / "sequences/MH_03_medium").exists())

    def test_sequence_and_module_summaries_share_human_readable_format(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for module in CONDITION_ORDER:
                result = _summary_result(module)
                sequence = "MH_01_easy"
                directory = root / module / "sequences" / sequence
                directory.mkdir(parents=True)
                (directory / "results.json").write_text(json.dumps({"result": result}))
                sequence_summary = render_sequence_summary(module, sequence, result)
                module_summary = render_aggregate_summary(
                    root / module, module,
                    {"requested_sequences": [sequence], "sequences": {sequence: {}}},
                )
                header = "| Condition | ATE RMSE (m) | translation RPE@1s (m) | rotation RPE@1s (deg) | coverage | nodes |"
                row_count = 2 + len(CONDITION_ORDER[module])
                sequence_lines = sequence_summary.splitlines()
                module_lines = module_summary.splitlines()
                sequence_start = sequence_lines.index(header)
                module_start = module_lines.index(header)
                self.assertEqual(
                    sequence_lines[sequence_start:sequence_start + row_count],
                    module_lines[module_start:module_start + row_count],
                )
                self.assertIn(
                    "| Full RGB | 0.1235 | 0.2346 | 10.35 | 91.2% | 101 |",
                    sequence_summary,
                )

    def test_h2_deployment_summary_uses_result_values_at_both_levels(self) -> None:
        result = _summary_result("h2_prediction")
        sequence = "MH_01_easy"
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            directory = root / "sequences" / sequence
            directory.mkdir(parents=True)
            (directory / "results.json").write_text(json.dumps({"result": result}))
            sequence_summary = render_sequence_summary("h2_prediction", sequence, result)
            module_summary = render_aggregate_summary(
                root, "h2_prediction",
                {"requested_sequences": [sequence], "sequences": {sequence: {}}},
            )
        for line in (
            "- Communication: anchor ratio 20.39%; encoded byte reduction 75.00%",
            "- Cloud compute: Full RGB 10.000 s; H2 12.000 s; H2 / Full RGB 1.200x; extra 2.000 s",
            "- H2 stage totals: JEPA encoder 1.000 s; predictor 2.000 s; bridge 0.300 s; DPVO graph CUDA-event span 4.000 s",
            "- Latency: context wait mean 250.00 ms; P95 400.00 ms",
            "- System: break-even uplink bandwidth 8.000 Mbps",
        ):
            self.assertIn(line, sequence_summary)
            self.assertIn(line, module_summary)

    def test_dataset_fingerprint_does_not_hash_rgb_payloads(self) -> None:
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            camera = root / "cam0"
            (camera / "data").mkdir(parents=True)
            rgb = camera / "data" / "1.png"; rgb.write_bytes(b"rgb payload")
            (camera / "data.csv").write_text("1,1.png\n")
            calibration = root / "calib.txt"; calibration.write_text("calibration")
            groundtruth = root / "MH_01_easy.txt"; groundtruth.write_text("groundtruth")
            identity = SimpleNamespace(
                key="euroc/test/MH_01_easy/0/1", candidate_index=0,
            )
            record = SimpleNamespace(identity=identity, rgb_path=str(rgb))
            config = {
                "dataset": {"groundtruth_pattern": str(groundtruth),
                            "calibration": str(calibration)},
                "paths": {},
            }
            hashed: list[Path] = []
            from . import registry
            real_hash = registry.sha256_file

            def recording_hash(path):
                hashed.append(Path(path)); return real_hash(path)

            with patch.object(registry, "load_sequence_records", return_value=[record]), \
                 patch.object(registry, "sha256_file", side_effect=recording_hash):
                result = dataset_fingerprint(config, "MH_01_easy")
            self.assertFalse(result["rgb_payload_hashed"])
            self.assertNotIn(rgb, hashed)
            self.assertEqual(set(hashed), {camera / "data.csv", calibration, groundtruth})


if __name__ == "__main__":
    unittest.main()
