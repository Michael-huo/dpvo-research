from __future__ import annotations

import inspect
import queue
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import torch

from . import execution_runtime, jepa_runtime, parallel_runtime, training_runtime
from .execution_runtime import (
    FormalExecution, require_lifecycle_cleanup, release_cuda_training_state,
)
from .h2_pipeline import CanonicalH2Pipeline
from .staged_transfer import GPUWorker, PinnedTransfer, SharedSlot
from .training_runtime import ResidentRows, resident_correspondence


class FormalCleanupLifecycleTest(unittest.TestCase):
    def test_worker_launchers_use_functional_modules_and_original_gpu_mapping(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            with patch.object(execution_runtime, "capture_runtime", return_value={}), \
                 patch.object(jepa_runtime.JepaSidecar, "_receive", return_value={
                     "status": "ready", "provenance": {},
                 }), patch("subprocess.Popen") as popen:
                jepa_runtime.JepaSidecar({"runtime": {"jepa_python": sys.executable}}, root)
                command = popen.call_args.args[0]
                self.assertEqual(command[:3], [sys.executable, "-m", "research.src.jepa_worker"])
                self.assertEqual(popen.call_args.kwargs["cwd"], jepa_runtime.REPO_ROOT)

            for component, device in (("encoder", 2), ("predictor", 1)):
                ready = {"status": "ready", "provenance": {
                    "cuda_visible_devices": str(device), "logical_cuda_device_count": 1,
                    "current_logical_cuda_device": 0,
                }}
                with patch("subprocess.Popen") as popen, \
                     patch.object(threading.Thread, "start"), \
                     patch.object(GPUWorker, "receive", return_value=ready):
                    GPUWorker(python=sys.executable, device=device, component=component,
                              config_path=root / "worker.json")
                    command = popen.call_args.args[0]
                    self.assertEqual(command[:3], [sys.executable, "-m", "research.src.pipeline_worker"])
                    self.assertEqual(command[-2:], ["--component", component])
                    self.assertEqual(popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], str(device))
                    self.assertEqual(popen.call_args.kwargs["cwd"], jepa_runtime.REPO_ROOT)

            with patch("subprocess.run") as launch, patch.object(torch, "save"):
                parallel_runtime._launch({}, root, 0)
                self.assertEqual(launch.call_args.args[0][:3], [
                    sys.executable, "-m", "research.src.parallel_runtime",
                ])
                self.assertEqual(launch.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "0")
                self.assertEqual(launch.call_args.kwargs["cwd"], jepa_runtime.REPO_ROOT)

    def _health_check_pipeline(self):
        pipeline = object.__new__(CanonicalH2Pipeline)
        pipeline.failures = queue.Queue()
        pipeline.stop = threading.Event()
        pipeline.identities = [SimpleNamespace(key=key) for key in ("a", "b", "c")]
        pipeline.yielded_candidate_keys = ["a"]
        return pipeline

    def test_healthy_poll_does_not_scan_trajectory(self):
        class NoIteration:
            def __iter__(self):
                raise AssertionError("healthy polling must not scan trajectory history")
        pipeline = self._health_check_pipeline()
        pipeline.identities = pipeline.yielded_candidate_keys = NoIteration()
        for _ in range(100):
            pipeline._check()

    def test_failed_poll_preserves_error_and_unconsumed_order(self):
        pipeline = self._health_check_pipeline()
        pipeline.yielded_candidate_keys = ["b"]
        pipeline.failures.put("predictor failed")
        pipeline.stop.set()
        with self.assertRaises(RuntimeError) as caught:
            pipeline._check()
        self.assertEqual(str(caught.exception),
                         "predictor failed\nunconsumed_identity=['a', 'c']")
        self.assertEqual(pipeline.yielded_candidate_keys, ["b"])

    def test_cancelled_poll_builds_consumed_set_once(self):
        class CountedHistory(list):
            iterations = 0
            def __iter__(self):
                self.iterations += 1
                return super().__iter__()
        pipeline = self._health_check_pipeline()
        pipeline.yielded_candidate_keys = CountedHistory(["a", "b", "a"])
        pipeline.stop.set()
        with self.assertRaisesRegex(RuntimeError,
                                    "H2 pipeline cancelled; unconsumed_identity=\\['c'\\]"):
            pipeline._check()
        self.assertEqual(pipeline.yielded_candidate_keys.iterations, 1)

    def test_wait_and_queue_poll_still_fail_on_worker_error(self):
        pipeline = self._health_check_pipeline()
        pipeline.execution = FormalExecution()
        pipeline.waits = []
        pipeline.failures.put("worker exited")
        with self.assertRaisesRegex(RuntimeError, "worker exited"):
            pipeline._get(queue.Queue())
        ready = threading.Event()
        ready.set()
        pipeline.stop.set()
        with self.assertRaisesRegex(RuntimeError, "pipeline cancelled"):
            pipeline._wait(ready, "prediction_ready")

    def test_nonzero_cuda_memory_is_diagnostic_only(self) -> None:
        with patch.object(torch.cuda, "is_initialized", return_value=True), patch.object(
            torch.cuda, "current_device", return_value=0,
        ), patch.object(torch.cuda, "synchronize") as synchronize, patch.object(
            torch.cuda, "empty_cache",
        ) as empty_cache, patch.object(
            torch.cuda, "memory_allocated", return_value=123,
        ), patch.object(torch.cuda, "memory_reserved", return_value=456):
            result = release_cuda_training_state()
        require_lifecycle_cleanup(result)
        self.assertEqual(result["devices"][0]["allocated_bytes"], 123)
        self.assertEqual(result["devices"][0]["reserved_bytes"], 456)
        self.assertTrue(result["memory_telemetry_only"])
        synchronize.assert_called_once_with(0)
        empty_cache.assert_called_once_with()
        self.assertNotIn("ipc_collect", inspect.getsource(release_cuda_training_state))

    def test_owned_resource_close_failure_remains_lifecycle_error(self) -> None:
        owner = MagicMock()
        owner.close.side_effect = RuntimeError("owner did not close")
        with patch.object(torch.cuda, "is_initialized", return_value=False):
            result = release_cuda_training_state(owner)
        self.assertFalse(result["cleanup_complete"])
        with self.assertRaisesRegex(RuntimeError, "lifecycle cleanup failed"):
            require_lifecycle_cleanup(result)

    def test_nvml_failure_is_telemetry_not_admission(self) -> None:
        with patch.object(
            execution_runtime.subprocess, "check_output",
            side_effect=OSError("NVML unavailable"),
        ):
            result = execution_runtime.validate_formal_hardware()
        self.assertTrue(result["telemetry_only"])
        self.assertIn("NVML unavailable", result["nvidia_smi_inventory_error"])
        self.assertIn("NVML unavailable", result["nvidia_smi_p2p_error"])

    def test_h2_requires_three_logical_cuda_devices(self) -> None:
        with patch.object(
            execution_runtime, "validate_formal_hardware", return_value={"devices": []},
        ), patch.object(
            execution_runtime, "capture_runtime", return_value={"seed": 1234},
        ), patch.object(torch.cuda, "device_count", return_value=2):
            with self.assertRaisesRegex(RuntimeError, "logical cuda:0/1/2"):
                execution_runtime.initialize_formal_main_process(
                    training_device=None, required_logical_devices=(0, 1, 2),
                )

    def test_numa_query_failure_uses_deterministic_cpuset_fallback(self) -> None:
        hardware = {"devices": [
            {"physical_index": 0, "numa_node": 0},
            {"physical_index": 1, "numa_node": 1},
            {"physical_index": 2, "numa_node": 1},
        ]}
        with patch.object(
            execution_runtime, "_physical_cpus_for_node",
            side_effect=OSError("sysfs unavailable"),
        ), patch.object(
            execution_runtime, "_allowed_physical_cpus",
            return_value=[0, 1, 2, 3, 4, 5],
        ):
            layout = execution_runtime.cpu_numa_layout(hardware)
        self.assertEqual(layout["affinity_basis"], "allowed_cpuset_fallback")
        self.assertEqual(layout["stage_c"]["cpus"], [0, 1, 2])
        self.assertTrue(layout["sets_disjoint"])

    def test_worker_binding_validation_uses_logical_runtime(self) -> None:
        valid = {
            "cuda_visible_devices": "2", "logical_cuda_device_count": 1,
            "current_logical_cuda_device": 0,
        }
        parallel_runtime._validate_isolated_worker_binding(valid, 2)
        with self.assertRaisesRegex(RuntimeError, "binding mismatch"):
            parallel_runtime._validate_isolated_worker_binding(
                valid | {"current_logical_cuda_device": 1}, 2,
            )

    def test_external_gpu_process_snapshot_does_not_gate_job(self) -> None:
        process_snapshot = {
            "rows": [{"pid": 999, "gpu_uuid": "shared", "used_gpu_memory_mib": 1200}],
            "error": "telemetry warning",
        }

        def fake_launch(_task, path, _device):
            torch.save({
                "sequence": "sequence", "condition": "bootstrap_schedule",
                "worker_runtime": {
                    "cuda_visible_devices": "0", "logical_cuda_device_count": 1,
                    "current_logical_cuda_device": 0,
                },
                "cleanup": {"cleanup_complete": True},
                "started_ns": time.monotonic_ns(),
                "completed_ns": time.monotonic_ns(),
            }, path / "result.pt")
            return path

        with tempfile.TemporaryDirectory() as name, patch.object(
            parallel_runtime, "_gpu_process_telemetry", return_value=process_snapshot,
        ), patch.object(parallel_runtime, "_launch", side_effect=fake_launch):
            rows, execution = parallel_runtime.run_sequential_trajectory_jobs(
                ({"kind": "materialize_schedule", "sequence": "sequence"},),
                Path(name) / "jobs", hardware={"devices": []},
            )
        self.assertEqual(len(rows), 1)
        self.assertEqual(execution["job_count"], 1)
        self.assertEqual(
            rows[0]["sequential_execution"]["processes_before"], process_snapshot,
        )

    def test_telemetry_exception_is_recorded_without_gating(self) -> None:
        with patch(
            "research.src.efficiency_profiling.gpu_process_snapshot",
            side_effect=RuntimeError("diagnostic command failed"),
        ):
            result = parallel_runtime._gpu_process_telemetry()
        self.assertEqual(result["rows"], [])
        self.assertTrue(result["telemetry_only"])
        self.assertIn("diagnostic command failed", result["error"])

    def test_resident_upload_does_not_query_free_memory(self) -> None:
        source = inspect.getsource(ResidentRows.__init__)
        self.assertNotIn("mem_get_info", source)
        self.assertNotIn("free_bytes", inspect.signature(ResidentRows).parameters)
        resident = ResidentRows(
            {"x": np.zeros((3, 2), dtype=np.float16)}, range(3),
            device=torch.device("cpu"),
        )
        self.assertEqual(resident.diagnostics["resident_rows"], 3)
        resident.close()
        resident.close()
        self.assertTrue(resident.closed)

    def test_partial_resident_oom_preserves_original_exception(self) -> None:
        original = torch.cuda.OutOfMemoryError("synthetic resident OOM")
        calls = 0
        real_to = torch.Tensor.to

        def fail_second(tensor, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise original
            return real_to(tensor, *args, **kwargs)

        with patch.object(torch.Tensor, "to", new=fail_second), patch.object(
            training_runtime.gc, "collect", side_effect=RuntimeError("cleanup diagnostic"),
        ):
            with self.assertRaises(torch.cuda.OutOfMemoryError) as raised:
                ResidentRows(
                    {"first": np.zeros((2, 2), np.float16),
                     "second": np.zeros((2, 2), np.float16)},
                    range(2), device=torch.device("cpu"),
                )
        self.assertIs(raised.exception, original)
        self.assertTrue(
            raised.exception.phase1_resident_cleanup["resident_tensors_cleared"],
        )
        self.assertIn(
            "cleanup diagnostic",
            " ".join(raised.exception.phase1_resident_cleanup["errors"]),
        )

    def test_partial_correspondence_oom_preserves_original_exception(self) -> None:
        from .transport import RobustCorrespondence
        row = RobustCorrespondence(*(
            torch.zeros(1, dtype=torch.float16)
            for _ in RobustCorrespondence.__dataclass_fields__
        ))
        store = SimpleNamespace(calibration={}, rows={0: row})
        original = torch.cuda.OutOfMemoryError("synthetic correspondence OOM")
        calls = 0
        real_to = torch.Tensor.to

        def fail_second(tensor, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise original
            return real_to(tensor, *args, **kwargs)

        with patch.object(torch.Tensor, "to", new=fail_second):
            with self.assertRaises(torch.cuda.OutOfMemoryError) as raised:
                resident_correspondence(store, torch.device("cpu"))
        self.assertIs(raised.exception, original)
        self.assertTrue(
            raised.exception.phase1_correspondence_resident_cleanup[
                "resident_tensors_cleared"
            ],
        )

    def test_resident_cleanup_failure_does_not_replace_oom(self) -> None:
        original = torch.cuda.OutOfMemoryError("primary OOM")
        with patch.object(
            torch.Tensor, "to", side_effect=original,
        ), patch.object(
            ResidentRows, "_release_owned", side_effect=RuntimeError("cleanup failed"),
        ):
            with self.assertRaises(torch.cuda.OutOfMemoryError) as raised:
                ResidentRows(
                    {"x": np.zeros((1, 1), np.float16)}, (0,),
                    device=torch.device("cpu"),
                )
        self.assertIs(raised.exception, original)
        self.assertIn(
            "cleanup failed",
            " ".join(raised.exception.phase1_resident_cleanup["errors"]),
        )

    def test_shared_slot_close_is_idempotent_and_reports_busy_state(self) -> None:
        slot = SharedSlot((1,), dtype=np.float32)
        slot.acquire()
        first = slot.close()
        second = slot.close()
        self.assertTrue(first["was_busy"])
        self.assertTrue(second["already_closed"])

    def test_pinned_transfer_releases_owned_buffers(self) -> None:
        transfer = object.__new__(PinnedTransfer)
        transfer.closed = False
        transfer.stream = MagicMock()
        transfer.buffers = {"owned": object()}
        first = transfer.close()
        second = transfer.close()
        transfer.stream.synchronize.assert_called_once_with()
        self.assertEqual(transfer.buffers, {})
        self.assertFalse(first["already_closed"])
        self.assertTrue(second["already_closed"])

    def test_gpu_worker_that_does_not_exit_fails_closed(self) -> None:
        process = MagicMock()
        process.poll.return_value = None
        process.returncode = None
        process.stdin = MagicMock()
        process.stdout = MagicMock()
        process.stderr = MagicMock()
        process.wait.return_value = None
        worker = object.__new__(GPUWorker)
        worker.closed = False
        worker.process = process
        worker.reader = MagicMock()
        worker.error_reader = MagicMock()
        worker.reader.is_alive.return_value = False
        worker.error_reader.is_alive.return_value = False
        worker.request = MagicMock(return_value={"status": "closed"})
        with self.assertRaisesRegex(RuntimeError, "lifecycle cleanup failed"):
            worker.close(require_graceful=True)

    def test_unfinished_pipeline_queue_fails_lifecycle_cleanup(self) -> None:
        class FakeWorker:
            def __init__(self, **_kwargs):
                self.process = SimpleNamespace(pid=123)

            def close(self, *, require_graceful=True):
                return {
                    "closed": True, "graceful": require_graceful,
                    "returncode": 0, "reader_threads_stopped": True,
                }

        class FakeSlot:
            def __init__(self, *_args, **_kwargs):
                self.busy = False

            def close(self):
                return {"closed": True, "already_closed": False, "was_busy": False}

        class FakeTransfer:
            def __init__(self, **_kwargs):
                pass

            def close(self):
                return {"closed": True, "already_closed": False}

        pipeline = object.__new__(CanonicalH2Pipeline)
        pipeline.config = {"runtime": {"jepa_python": "python"}}
        pipeline.settings = {}
        pipeline.predictor_checkpoint = "predictor.pt"
        pipeline.predictor_state_sha256 = "state"
        pipeline.transport_calibration = {}
        pipeline.transform = SimpleNamespace(
            token_grid_height=2, token_grid_width=2,
            padded_height=4, padded_width=4,
        )
        pipeline.execution = FormalExecution()
        pipeline.cpu_profile = None
        pipeline.intervals = (SimpleNamespace(hidden=(object(),)),)
        pipeline.workers = {}
        pipeline.slots = []
        pipeline.threads = []
        pipeline.stop = threading.Event()
        pipeline.encode_queue = queue.Queue(2)
        pipeline.consume_queue = queue.Queue(2)
        pipeline.transfer = None
        pipeline.lifecycle_cleanup = None
        with tempfile.TemporaryDirectory() as name:
            pipeline.temporary = Path(name) / "online"
            with patch.object(
                CanonicalH2Pipeline, "predictor_config", new_callable=PropertyMock,
            ) as predictor_config, patch(
                "research.src.h2_pipeline.dataclasses.asdict",
                return_value={},
            ), patch(
                "research.src.h2_pipeline.GPUWorker", FakeWorker,
            ), patch(
                "research.src.h2_pipeline.SharedSlot", FakeSlot,
            ), patch(
                "research.src.h2_pipeline.PinnedTransfer", FakeTransfer,
            ):
                predictor_config.return_value = {}
                with self.assertRaisesRegex(RuntimeError, "unfinished tasks"):
                    with pipeline.online_session():
                        thread = MagicMock()
                        thread.name = "stuck-thread"
                        thread.is_alive.return_value = True
                        pipeline.threads.append(thread)
                        pipeline.encode_queue.put(object())
        self.assertFalse(pipeline.lifecycle_cleanup["complete"])
        self.assertEqual(pipeline.lifecycle_cleanup["threads_alive"], ["stuck-thread"])

    def test_matched_timing_code_does_not_poll_gpu_telemetry(self) -> None:
        from .runtime import run_deployment_observations
        source = inspect.getsource(run_deployment_observations)
        self.assertNotIn("gpu_process_snapshot", source)
        self.assertNotIn("nvidia-smi", source)


if __name__ == "__main__":
    unittest.main()
