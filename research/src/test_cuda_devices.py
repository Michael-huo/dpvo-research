"""CPU/mock and fresh-spawn checks for inherited logical CUDA resources."""
from __future__ import annotations

import importlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import cuda_devices, execution_runtime, parallel_runtime
from .cuda_devices import CudaDevicePool, ONLINE_DEVICE_ERROR
from .protocol import FrameIdentity, canonical_sha256, sha256_file


def _spawn_mapping(connection, count, logical):
    with connection, patch.object(torch.cuda, "device_count", return_value=count), patch.object(torch.cuda, "set_device") as select:
        for name in ("parallel_runtime", "pipeline_worker", "jepa_worker"):
            importlib.import_module("research.src." + name)
        cuda_devices.select_worker_device(logical)
        connection.send({"mask": os.environ.get("CUDA_VISIBLE_DEVICES"),
                         "pool": CudaDevicePool.discover().provenance(),
                         "selected": [call.args[0] for call in select.call_args_list],
                         "cuda_initialized": torch.cuda.is_initialized()})


class CudaDevicePoolTest(unittest.TestCase):
    def test_visible_counts_one_two_three_four(self):
        for count in (1, 2, 3, 4):
            with self.subTest(count=count), patch.object(torch.cuda, "device_count", return_value=count):
                pool = CudaDevicePool.discover()
                self.assertEqual(pool.primary_device, "cuda:0")
                self.assertEqual(pool.worker_devices, tuple(f"cuda:{i}" for i in range(1, count)))
                self.assertEqual(pool.preparation_devices, tuple(range(count)))
                self.assertEqual(pool.provenance()["preparation_mode"], "serial" if count == 1 else "sharded")
                if count < 3:
                    with self.assertRaisesRegex(RuntimeError, ONLINE_DEVICE_ERROR):
                        execution_runtime.FormalExecution.from_pool(pool)
                else:
                    options = execution_runtime.FormalExecution.from_pool(pool)
                    self.assertEqual((options.consumer_device, options.predictor_device, options.encoder_device), ("0", "1", "2"))
                    self.assertEqual(set(pool.online_mapping().values()), {"cuda:0", "cuda:1", "cuda:2"})

    def test_zero_visible_devices_fail_primary(self):
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            CudaDevicePool(0).primary_device

    def test_training_selects_only_primary_for_one_and_three(self):
        for count in (1, 3):
            with self.subTest(count=count), patch.object(torch.cuda, "device_count", return_value=count), patch.object(torch.cuda, "set_device") as selected, patch.object(execution_runtime, "validate_formal_hardware", return_value={}), patch.object(execution_runtime, "apply_runtime"), patch.object(execution_runtime, "capture_runtime", return_value={}):
                result = execution_runtime.initialize_formal_main_process()
                selected.assert_called_once_with(0)
                self.assertEqual(result["main_process_device"], 0)

    def test_fresh_spawn_inherits_reordered_mask_and_selects_only_assigned_device(self):
        context = multiprocessing.get_context("spawn")
        for count, logical in ((1, 0), (2, 1), (3, 2), (4, 3)):
            mask = {1: "2", 2: "2,0", 3: "2,0,1", 4: "2,0,1,3"}[count]
            with self.subTest(count=count), patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": mask}):
                parent, child = context.Pipe(duplex=False)
                process = context.Process(target=_spawn_mapping, args=(child, count, logical))
                try:
                    process.start(); child.close()
                    self.assertTrue(parent.poll(30))
                    result = parent.recv()
                    process.join(10)
                    self.assertEqual(process.exitcode, 0)
                    self.assertEqual(result["mask"], mask)
                    self.assertEqual(result["selected"], [logical])
                    self.assertFalse(result["cuda_initialized"])
                    self.assertEqual(result["pool"]["visible_devices"], [f"cuda:{i}" for i in range(count)])
                finally:
                    parent.close(); child.close()
                    if process.is_alive():
                        process.terminate(); process.join(5)

    def test_uuid_pci_mapping_uses_driver_ordinals_not_mask_numbers(self):
        import ctypes
        import uuid
        class Driver:
            def cuInit(self, flags): return 0
            def cuDeviceGetCount(self, output): output._obj.value = 3; return 0
            def cuDeviceGet(self, output, logical): output._obj.value = logical; return 0
            def cuDeviceGetUuid(self, output, device):
                output._obj[:] = bytes([device.value + 1]) * 16
                return 0
            def cuDeviceGetPCIBusId(self, output, size, device):
                output.value = f"0000:0{device.value + 1}:00.0".encode(); return 0
            def cuDeviceGetName(self, output, size, device): output.value = b"mock GPU"; return 0
        physical = [{"uuid": "GPU-" + str(uuid.UUID(bytes=bytes([i+1])*16)), "physical_index": physical_id}
                    for i, physical_id in enumerate((7, 9, 4))]
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,0,1"}), patch.object(ctypes, "CDLL", return_value=Driver()), patch.object(torch.cuda, "set_device", side_effect=AssertionError("telemetry must not create contexts")):
            rows = cuda_devices.logical_device_telemetry(3, physical)
        self.assertEqual([r["logical_index"] for r in rows], [0, 1, 2])
        self.assertEqual([r["physical_index"] for r in rows], [7, 9, 4])
        self.assertEqual([r["pci_bus_id"] for r in rows], ["0000:01:00.0", "0000:02:00.0", "0000:03:00.0"])

    def test_failed_telemetry_does_not_invent_physical_mapping(self):
        with patch("ctypes.CDLL", side_effect=OSError("no driver")), patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,0,1"}):
            rows = cuda_devices.logical_device_telemetry(3)
        self.assertEqual([r["physical_index"] for r in rows], [None] * 3)
        self.assertTrue(all("no driver" in r["mapping_error"] for r in rows))

    def test_preparation_batch_identity_order_and_hash_are_exact_one_vs_three(self):
        values = list(range(19))
        results = []
        for count in (1, 3):
            devices = CudaDevicePool(count).preparation_devices
            shards = parallel_runtime.shard_batches(values, 4, devices)
            # Deterministic synthetic native feature payloads; original batches
            # must stay intact including the final partial batch.
            batches = [shard[i:i+4] for shard in shards for i in range(0, len(shard), 4)]
            self.assertEqual(sorted(batches), [values[i:i+4] for i in range(0, len(values), 4)])
            arrays = [[(str(i), np.full((2, 3), i, dtype=np.float16)) for i in shard] for shard in shards]
            ordered, manifest = parallel_runtime.merge_identity_rows([str(i) for i in values], list(reversed(arrays)))
            results.append((ordered, manifest))
        self.assertEqual(results[0][1], results[1][1])
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(results[0][0], results[1][0])))

    def test_launch_task_preserves_visibility_and_passes_logical_rank(self):
        with tempfile.TemporaryDirectory() as name, patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,0,1"}), patch.object(subprocess, "run") as launch:
            directory = Path(name)
            parallel_runtime._launch({"kind": "h2"}, directory, 2)
            task = torch.load(directory / "task.pt", weights_only=False)
            self.assertEqual(task["logical_device"], 2)
            self.assertNotIn("physical_device", task)
            self.assertEqual(launch.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "2,0,1")

    def test_preparation_automatically_uses_pool_and_merges_native_rows(self):
        from types import SimpleNamespace
        identities = [FrameIdentity("euroc", "machine_hall", "s", "MH_01_easy", i, i, i+1) for i in range(9)]
        records = [SimpleNamespace(identity=i) for i in identities]
        transform = SimpleNamespace(token_grid_height=1, token_grid_width=1)
        config = {"jepa": {"batch_size": 4}}
        manifests = []
        def launch(task, directory, device):
            keys = [i.key for i in task["identities"]]
            tokens = np.stack([np.full((1, 768), i.candidate_index, dtype=np.float16) for i in task["identities"]])
            import hashlib
            hashes = {key: {"block5": hashlib.sha256(tokens[j].tobytes()).hexdigest()} for j, key in enumerate(keys)}
            np.save(directory / "block5.npy", tokens)
            (directory / "manifest.json").write_text(json.dumps({"keys": keys, "token_keys": keys, "hashes": hashes, "runtime": {"logical_device": device}}))
            return directory
        for count in (1, 3):
            with tempfile.TemporaryDirectory() as name, patch.object(torch.cuda, "device_count", return_value=count), patch.object(parallel_runtime, "_launch", side_effect=launch):
                store, meta = parallel_runtime.extract_parallel(records, identities, None, config, Path(name), transform)
                self.assertEqual(len(meta["workers"]), count)
                self.assertEqual([r["logical_device"] for r in meta["shard_assignment"]], list(range(count)))
                manifests.append((meta["identity_sha256"], meta["ordered_content_sha256"], store.values.copy()))
                store.close()
        self.assertEqual(manifests[0][:2], manifests[1][:2])
        self.assertTrue(np.array_equal(manifests[0][2], manifests[1][2]))


if __name__ == "__main__":
    unittest.main()
