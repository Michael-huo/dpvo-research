"""Fresh replacement failure injection; all writes stay in temporary directories."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .artifact_runtime import (
    publish_current_canonical, staged_directory, validate_lightweight_results,
    without_worker_log_paths,
)
from .anchor_budget_artifacts import inventory


class FreshCurrentReplaceTest(unittest.TestCase):
    def test_persisted_metadata_omits_logs_without_mutating_runtime_payload(self):
        job = {"worker_log": "/tmp/.research_h2_run/job/worker.log",
               "elapsed_seconds": 1.234567890123, "logical_device": 0,
               "cleanup": {"success": True}, "status": "complete"}
        runtime = {"jobs": [job], "nested": ({"worker_log":
                   "research/results/.research_h1_run/job/worker.log"},),
                   "checkpoint": {"relative_path": "research/checkpoints/h2-prediction/predictor.pt",
                                  "sha256": "checkpoint-hash"}, "missing": None}
        before = json.dumps(runtime, sort_keys=True)
        persisted = without_worker_log_paths(runtime)
        self.assertNotIn(".research_", json.dumps(persisted))
        self.assertNotIn('"worker_log"', json.dumps(persisted))
        self.assertEqual(persisted["jobs"], [{key: value for key, value in job.items()
                                              if key != "worker_log"}])
        self.assertEqual(persisted["nested"], ({},))
        self.assertEqual(persisted["checkpoint"], runtime["checkpoint"])
        self.assertIsNone(persisted["missing"])
        self.assertEqual(json.dumps(runtime, sort_keys=True), before)
        persisted["jobs"][0]["cleanup"]["success"] = False
        self.assertTrue(job["cleanup"]["success"])

    def _tree(self, root, values):
        root.mkdir(parents=True)
        for name, value in values.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value)
        return root

    def test_success_replaces_existing_and_drops_unrequested_sequences(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name) / "results/h0-state"
            self._tree(root, {"MH01/results.json": "old", "MH03/results.json": "old", "MH05/results.json": "old"})
            with staged_directory(root) as staging:
                self._tree(staging / "MH01", {"results.json": "current"})
                publish_current_canonical(staging, root)
            self.assertEqual(set(inventory(root)), {"MH01/results.json"})
            self.assertEqual((root / "MH01/results.json").read_text(), "current")
            self.assertEqual([p.name for p in root.parent.iterdir()], ["h0-state"])

    def test_failed_run_retains_previous_and_cleans_staging(self):
        for module in ("h0-state", "h1-interface", "h2-prediction", "anchor-budget"):
            with self.subTest(module=module), tempfile.TemporaryDirectory() as name:
                root = self._tree(Path(name) / module, {"results.json": "old"})
                before = inventory(root)
                with self.assertRaisesRegex(RuntimeError, "training failed"):
                    with staged_directory(root) as staging:
                        (staging / "worker.log").write_text("unfinished")
                        raise RuntimeError("training failed")
                self.assertEqual(before, inventory(root))
                self.assertEqual(list(root.parent.iterdir()), [root])

    def test_checkpoint_results_rollback_at_every_rename(self):
        for fail_at in (1, 2, 3, 4):
            with self.subTest(fail_at=fail_at), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                results = self._tree(root / "results/h2-prediction", {"results.json": "old results"})
                checkpoints = self._tree(root / "checkpoints/h2-prediction", {"predictor.pt": "old model"})
                before = inventory(root)
                real_rename, calls = Path.rename, []
                def rename(source, target):
                    calls.append((source, target))
                    if len(calls) == fail_at:
                        raise OSError("injected rename failure")
                    return real_rename(source, target)
                with self.assertRaisesRegex(OSError, "injected rename failure"):
                    with staged_directory(results) as staged, staged_directory(checkpoints) as weights:
                        (staged / "results.json").write_text("new results")
                        (weights / "predictor.pt").write_text("new model")
                        with patch.object(Path, "rename", autospec=True, side_effect=rename):
                            publish_current_canonical(staged, results, checkpoint_staged=weights, checkpoint_destination=checkpoints)
                self.assertEqual(before, inventory(root))
                self.assertFalse(any("staging-" in p.name or "backup-" in p.name for p in root.rglob("*")))

    def test_failure_with_no_previous_directories_restores_absence(self):
        with tempfile.TemporaryDirectory() as name:
            results, checkpoints = Path(name) / "results/h1-interface", Path(name) / "checkpoints/h1-interface"
            real_rename = Path.rename
            with self.assertRaisesRegex(OSError, "checkpoint publish failed"):
                with staged_directory(results) as staged, staged_directory(checkpoints) as weights:
                    (staged / "results.json").write_text("new")
                    (weights / "bridge.pt").write_text("new")
                    def rename(source, target):
                        if source == weights:
                            raise OSError("checkpoint publish failed")
                        return real_rename(source, target)
                    with patch.object(Path, "rename", autospec=True, side_effect=rename):
                        publish_current_canonical(staged, results, checkpoint_staged=weights, checkpoint_destination=checkpoints)
            self.assertFalse(results.exists())
            self.assertFalse(checkpoints.exists())

    def test_all_model_suffixes_are_excluded_from_results(self):
        for suffix in (".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".onnx"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                (root / ("model" + suffix)).write_bytes(b"model")
                with self.assertRaisesRegex(RuntimeError, "must not be published"):
                    validate_lightweight_results(root)

    def test_old_phase_trees_are_never_merged_or_changed(self):
        with tempfile.TemporaryDirectory() as name:
            parent = Path(name)
            old = self._tree(parent / "phase1-feasibility/h0_state", {"MH03/results.json": "old phase"})
            before = inventory(old)
            current = parent / "h0-state"
            with staged_directory(current) as staged:
                (staged / "results.json").write_text("fresh")
                publish_current_canonical(staged, current)
            self.assertEqual(inventory(old), before)
            self.assertEqual(set(inventory(current)), {"results.json"})

    def test_h2_and_budget_formal_execution_fail_closed_before_work(self):
        import torch
        from . import run_h2, run_anchor_budget
        for count in (1, 2):
            with self.subTest(count=count), patch.object(torch.cuda, "device_count", return_value=count), patch.object(run_h2, "load_sequence_records", side_effect=AssertionError("no preparation")), patch.object(run_anchor_budget, "load_sequence_records", side_effect=AssertionError("no preparation")):
                for run in (lambda: run_h2.run(["MH_01_easy"]), lambda: run_anchor_budget.run(execute=True)):
                    with self.assertRaisesRegex(RuntimeError, "requires 3 visible CUDA devices"):
                        run()

    def test_h0_h1_h2_mock_preparation_failure_keeps_both_old_trees(self):
        import copy
        from types import SimpleNamespace
        from . import run_h0, run_h1, run_h2
        for runner, module in ((run_h0, "h0_state"), (run_h1, "h1_interface"), (run_h2, "h2_prediction")):
            with self.subTest(module=module), tempfile.TemporaryDirectory() as name, contextlib.ExitStack() as stack:
                root = Path(name)
                result_root = self._tree(root / "results" / module.replace("_", "-"), {"INDEX.json": "old"})
                checkpoint_root = self._tree(root / "checkpoints" / module.replace("_", "-"), {"weight.pt": "old"})
                before = inventory(root)
                config, config_path = runner.load_config()
                config = copy.deepcopy(config)
                config["paths"]["output_root"] = str(result_root)
                stack.enter_context(patch.object(runner, "load_config", return_value=(config, config_path)))
                stack.enter_context(patch.object(runner, "initialize_formal_main_process", return_value={"hardware": {}, "runtime_settings": {}}))
                stack.enter_context(patch.object(runner, "cpu_numa_layout", return_value={}))
                stack.enter_context(patch.object(runner, "fixed_cpu_profile", return_value={"stage_c": {}}))
                stack.enter_context(patch.object(runner, "run_sequential_trajectory_jobs", side_effect=RuntimeError("mock preparation failed")))
                if module == "h0_state":
                    stack.enter_context(patch.object(runner, "load_sequence_records", return_value=()))
                else:
                    if module == "h1_interface":
                        stack.enter_context(patch.dict(runner.CHECKPOINT_ROOTS, {module: checkpoint_root}))
                        stack.enter_context(patch.object(runner, "h1_training_context", return_value=((), {}, {})))
                    else:
                        stack.enter_context(patch.object(runner, "load_sequence_records", return_value=[None]))
                        stack.enter_context(patch.object(runner, "sequence_geometry", return_value=(None, {})))
                        stack.enter_context(patch.object(runner, "_load_bridge", return_value=(SimpleNamespace(state_dict=lambda: {}, cpu=lambda: None), {"file_sha256": "bridge"})))
                        stack.enter_context(patch.object(runner, "release_cuda_training_state", return_value={"cleanup_complete": True}))
                        stack.enter_context(patch.object(runner, "load_canonical_predictor", return_value=({"state_dict": {}, "train_only_calibration": {}}, checkpoint_root / "weight.pt", "hash")))
                        stack.enter_context(patch.object(runner, "base_lineage", return_value=({}, {})))
                with self.assertRaisesRegex(RuntimeError, "mock preparation failed"):
                    runner.run(["MH_01_easy"])
                self.assertEqual(before, inventory(root))
                self.assertFalse(any("staging-" in p.name or p.name.startswith(".research_") for p in root.rglob("*")))

    def test_anchor_budget_cli_mock_dispatch_new_request_set(self):
        import sys
        from . import run_anchor_budget
        with patch.object(sys, "argv", ["run_anchor_budget", "--strides", "5", "7", "8", "--execute"]), patch.object(run_anchor_budget, "run", return_value={"status": "complete", "sequence": "MH_01_easy", "anchor_strides": [5,7,8]}) as run, contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(run_anchor_budget.main(), 0)
        self.assertEqual(run.call_args.args[1], [5,7,8])
        self.assertTrue(run.call_args.kwargs["execute"])
        self.assertIn("Anchor Budget — Communication-Budget Sensitivity", output.getvalue())


if __name__ == "__main__":
    unittest.main()
