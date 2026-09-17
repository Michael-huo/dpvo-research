"""Small CPU checks for the package boundary and frozen runtime contracts."""

from __future__ import annotations

import copy
import importlib
import os
import queue
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import torch

from latent_vslam.bridge_checkpoint import (
    bridge_training_context, bridge_training_input, load_compatible_bridge, load_bridge_config,
)
from latent_vslam.dataset_paths import LOCAL_PATHS_CONFIG
from latent_vslam.cuda_devices import CudaDevicePool
from latent_vslam.execution_runtime import FormalExecution
from latent_vslam import execution_runtime
from latent_vslam.prediction_pipeline import PredictionPipeline
from latent_vslam.jepa_fmap import build_bridge, derive_full_fov_transform
from latent_vslam.protocol import REPO_ROOT, load_sequence_records
from latent_vslam import parallel_runtime
from latent_vslam.manifests import config_protocol_fingerprint, dataset_fingerprint
from latent_vslam.inference_runtime import load_config
from latent_vslam.staged_transfer import GPUWorker
from latent_vslam.uniform_admission import uniform_ordinals
from prediction.jepa_runtime import JepaSidecar
from prediction.predictor import RobustTransportBlock5Predictor
from prediction.predictor_checkpoint import load_canonical_predictor
from prediction.transport import RobustCorrespondence, robust_transport_interpolation


class CoreContractsTest(unittest.TestCase):
    def test_core_has_no_legacy_import(self):
        for package in ("prediction", "latent_vslam"):
            for path in (REPO_ROOT / package).glob("*.py"):
                self.assertNotIn("research.src", path.read_text(), path.name)

    def test_dataset_and_scientific_identity_survive_runtime_path_changes(self):
        self.assertEqual(LOCAL_PATHS_CONFIG, REPO_ROOT / "configs/paths.local.yaml")
        config, _ = load_config()
        self.assertEqual(len(load_sequence_records(config, "MH_01_easy")), 1841)
        self.assertEqual(
            dataset_fingerprint(config, "MH_01_easy")["dataset_sha256"],
            "1297657019af0c1fdeaa2ebeaa3a8ecb2b97dc1a8e685069e539cc130c742510",
        )
        changed = copy.deepcopy(config)
        changed["runtime"] = {"jepa_python": "/different/env/bin/python"}
        self.assertEqual(
            config_protocol_fingerprint(config)["config_protocol_sha256"],
            config_protocol_fingerprint(changed)["config_protocol_sha256"],
        )

    def test_canonical_checkpoints_load_without_gpu(self):
        config, _ = load_config()
        checkpoint, _, _ = load_canonical_predictor(config)
        self.assertEqual(checkpoint["state_dict_sha256"],
                         config["canonical_predictor"]["state_dict_sha256"])
        h1_config, _ = load_bridge_config()
        _, base, _ = bridge_training_context(h1_config)
        transform = derive_full_fov_transform(64, 64, target_height=64)
        with patch.object(torch.nn.Module, "cuda", lambda model: model):
            bridge, metadata, _ = load_compatible_bridge(
                REPO_ROOT / config["paths"]["bridge"], transform,
                hidden_channels=160, expected_training_input=bridge_training_input(base),
            )
        self.assertFalse(bridge.training)
        self.assertEqual(metadata["architecture"], h1_config["bridge"]["architecture"])

    def test_prediction_transport_bridge_and_admission_cpu(self):
        self.assertEqual(uniform_ordinals(5), (2, 4))
        self.assertEqual(uniform_ordinals(10), (2, 4, 6, 8))
        j0 = torch.ones(1, 768, 2, 2)
        j1 = 3 * j0
        displacement = torch.zeros(1, 2, 2, 2)
        confidence = torch.ones(1, 1, 2, 2)
        mask = torch.ones(2, 2, dtype=torch.bool)
        corr = RobustCorrespondence(
            displacement, displacement, confidence, confidence,
            mask, mask, confidence, confidence, confidence, confidence,
            confidence, confidence,
        )
        with torch.inference_mode():
            result = robust_transport_interpolation(j0, j1, torch.tensor([0.5]), corr, mask)
            self.assertTrue(torch.equal(result.field, 2 * j0))
            predictor = RobustTransportBlock5Predictor().eval()
            output = predictor(
                result.field, result.warped_difference,
                result.warp0.coverage, result.warp1.coverage,
                result.fused_confidence, torch.tensor([0.5]), torch.tensor([1.0]),
            )
            self.assertTrue(torch.equal(output, result.field))
            transform = derive_full_fov_transform(64, 64, target_height=64)
            bridge = build_bridge(transform).eval()
            self.assertEqual(tuple(bridge(torch.zeros(1, 4 * 4, 768)).shape),
                             (1, 128, 16, 16))

    def test_vjepa_provider_imports_from_local_package(self):
        code = """
import sys
from pathlib import Path
import prediction.jepa_worker
from prediction.vjepa2.runtime import load_vitb_encoder
assert callable(load_vitb_encoder)
assert str(Path(load_vitb_encoder.__code__.co_filename).resolve()).startswith(str(Path.cwd()))
assert not any(name.startswith(('app.vjepa_2_1', 'src.models')) for name in sys.modules)
"""
        subprocess.run(
            [sys.executable, "-c", code], cwd=REPO_ROOT,
            env=os.environ | {"CUDA_VISIBLE_DEVICES": ""},
            capture_output=True, text=True, check=True, timeout=30,
        )

    def test_runtime_provenance_keeps_python_environment(self):
        settings = {"seed": 1234, "autocast_dtype": "float16"}
        with patch.object(execution_runtime, "capture_runtime", return_value=settings), \
             patch.object(torch.cuda, "is_available", return_value=False), \
             patch.object(torch.backends.cudnn, "version", return_value=None):
            provenance = execution_runtime.runtime_provenance(
                settings, component="v_jepa_encoder",
            )
        self.assertEqual(provenance["python_executable"], sys.executable)
        self.assertEqual(provenance["python_prefix"], sys.prefix)
        self.assertEqual(provenance["logical_cuda_device_count"], 0)

    def test_vjepa_sidecar_and_pipeline_worker_commands(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            with patch("latent_vslam.execution_runtime.capture_runtime", return_value={}), \
                 patch.object(torch.cuda, "current_device", return_value=0), \
                 patch.object(JepaSidecar, "_receive", return_value={
                     "status": "ready", "provenance": {},
                 }), patch("prediction.jepa_runtime.subprocess.Popen") as popen:
                JepaSidecar({}, root)
                self.assertEqual(popen.call_args.args[0][:3],
                                 [sys.executable, "-m", "prediction.jepa_worker"])
            for component, device in (("encoder", 2), ("predictor", 1)):
                ready = {"status": "ready", "provenance": {
                    "cuda_visible_devices": "0,1,2", "logical_cuda_device_count": 3,
                    "current_logical_cuda_device": device, "settings": {"seed": 1234},
                }}
                with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2"}), \
                     patch("latent_vslam.staged_transfer.subprocess.Popen") as popen, \
                     patch.object(CudaDevicePool, "discover",
                                  return_value=SimpleNamespace(device_count=3)), \
                     patch.object(threading.Thread, "start"), \
                     patch.object(GPUWorker, "receive", return_value=ready):
                    GPUWorker(python=sys.executable, device=device, component=component,
                              config_path=root / "worker.json")
                    self.assertEqual(popen.call_args.args[0][:3],
                                     [sys.executable, "-m", "latent_vslam.pipeline_worker"])
                    self.assertEqual(popen.call_args.args[0][-2:],
                                     ["--logical-device", str(device)])

    def test_trajectory_worker_stays_a_separate_process(self):
        with tempfile.TemporaryDirectory() as name, \
             patch.object(torch, "save"), \
             patch("latent_vslam.parallel_runtime.subprocess.run") as launch:
            parallel_runtime._launch({}, Path(name), 0)
        self.assertEqual(launch.call_args.args[0][:3],
                         [sys.executable, "-m", "latent_vslam.parallel_runtime"])
        self.assertEqual(launch.call_args.kwargs["cwd"], REPO_ROOT)

    def test_inference_retains_two_workers_and_three_gpu_mapping(self):
        launches = []

        class Worker:
            def __init__(self, **kwargs):
                launches.append(kwargs)
                self.process = SimpleNamespace(pid=123)

            def close(self, **_kwargs):
                return {"closed": True}

        class Slot:
            def __init__(self, *_args):
                pass

            def close(self):
                return {"was_busy": False}

        class Transfer:
            def __init__(self, **_kwargs):
                pass

            def close(self):
                return {"closed": True}

        pipeline = object.__new__(PredictionPipeline)
        pipeline.config = {"jepa": {}}
        pipeline.settings = {}
        pipeline.predictor_checkpoint = "predictor.pt"
        pipeline.predictor_state_sha256 = "state"
        pipeline.transport_calibration = {}
        pipeline.transform = SimpleNamespace(
            token_grid_height=2, token_grid_width=2, padded_height=4, padded_width=4,
        )
        pipeline.execution = FormalExecution()
        pipeline.cpu_profile = None
        pipeline.intervals = (SimpleNamespace(hidden=(object(),)),)
        pipeline.workers, pipeline.slots, pipeline.threads = {}, [], []
        pipeline.stop = threading.Event()
        pipeline.encode_queue, pipeline.consume_queue = queue.Queue(2), queue.Queue(2)
        pipeline.transfer = None
        with tempfile.TemporaryDirectory() as name, \
             patch.object(PredictionPipeline, "predictor_config", new_callable=PropertyMock,
                          return_value={}), \
             patch("latent_vslam.prediction_pipeline.dataclasses.asdict", return_value={}), \
             patch("latent_vslam.prediction_pipeline.GPUWorker", Worker), \
             patch("latent_vslam.prediction_pipeline.SharedSlot", Slot), \
             patch("latent_vslam.prediction_pipeline.PinnedTransfer", Transfer):
            pipeline.temporary = Path(name) / "online"
            with pipeline.online_session():
                self.assertEqual(len(pipeline.workers), 2)
        self.assertEqual(
            [(row["component"], row["device"], row["python"]) for row in launches],
            [("encoder", "2", sys.executable), ("predictor", "1", sys.executable)],
        )
        self.assertEqual(pipeline.execution.consumer_device, "0")
        self.assertTrue(pipeline.lifecycle_cleanup["complete"])


if __name__ == "__main__":
    unittest.main()
