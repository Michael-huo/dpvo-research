"""CPU dry execution of fixed-checkpoint H2 and separated scientific contracts."""
import contextlib
import copy
import json
import tempfile
import subprocess
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import numpy as np

from . import run_h2, h2_training, execution_runtime
from .predictor_checkpoint import training_contract
from .protocol import atomic_write_json, sha256_file
from .scientific_lineage import deployment_lineage
from .test_h2 import frame


class CanonicalH2Test(unittest.TestCase):
    def test_strict_replay_records_stage_c_provenance_after_online_completion(self):
        config, _ = run_h2.load_config()
        settings = {"seed": 1234, "autocast_dtype": "float16"}
        usage = {
            "worker_provenance": {
                name: {"status": "ready", "provenance": {"settings": dict(settings)},
                       "logical_device": device, "requested_global_logical_device": device,
                       "cpu_profile": None}
                for name, device in (("encoder", 2), ("predictor", 1))},
            "anchor_encoded_exactly_once": True,
            "candidate_yielded_exactly_once": True,
            "hidden_consumed_exactly_once": True,
            "contains_hidden_rgb_or_path_capability": False,
            "contains_hidden_reference_or_groundtruth_capability": False,
        }
        provider = SimpleNamespace(
            settings=settings, consumed_hidden_keys=["h1", "h2", "h3", "h4"],
            online_session=contextlib.nullcontext, observations=lambda: iter(()),
            on_tracked=lambda *args: None, usage_payload=lambda: usage,
            jepa_peak_online_vram_bytes=0, worker_diagnostics={"predictor": {"peak_vram_bytes": 0}},
        )
        runtime = {"admission": {"generated_hidden_count": 4},
                   "observation_sampling_provenance": {}, "peak_gpu_vram_bytes": 0}
        record = SimpleNamespace(identity=frame(0), rgb_path="anchor.png")
        bridge = torch.nn.Linear(1, 1).eval().requires_grad_(False)
        with tempfile.TemporaryDirectory() as name, contextlib.ExitStack() as stack:
            replacements = {
                "load_dpvo_domain": lambda *args: (np.zeros((16, 16, 3)), None),
                "OnlineProfiler": lambda: SimpleNamespace(payload=lambda **kwargs: {}),
                "CanonicalH2Pipeline": lambda **kwargs: provider,
                "capture_runtime": lambda *args: settings,
                "warmup_dpvo_frontend": lambda *args, **kwargs: {},
                "run_deployment_observations": lambda *args, **kwargs: (runtime, {}),
            }
            for key, value in replacements.items():
                stack.enter_context(patch.object(run_h2, key, value))
            stack.enter_context(patch.object(execution_runtime, "capture_runtime", return_value=settings))
            stack.enter_context(patch.object(torch.cuda, "is_available", return_value=False))
            _, _, profile = run_h2._run_strict_replay(
                [record], {record.identity.key: "anchor"}, [], np.ones(4), object(),
                bridge, bridge, {}, config, Path(name) / "online",
                predictor_checkpoint=Path(name) / "predictor.pt", predictor_state_hash="weights")
            usage["worker_provenance"]["predictor"]["provenance"]["settings"]["seed"] = 99
            with self.assertRaisesRegex(RuntimeError, "pipeline worker scientific seed mismatch"):
                run_h2._run_strict_replay(
                    [record], {record.identity.key: "anchor"}, [], np.ones(4), object(),
                    bridge, bridge, {}, config, Path(name) / "wrong_seed",
                    predictor_checkpoint=Path(name) / "predictor.pt", predictor_state_hash="weights")
        self.assertEqual(profile["stage_c_runtime"]["component"], "stage_c_native_frontend_bridge_dpvo")
        self.assertEqual(profile["stage_c_runtime"]["settings"]["seed"], 1234)
        self.assertEqual(runtime["observation_sampling_provenance"]["worker_seeds"],
                         {"encoder": 1234, "predictor": 1234})

    def test_transport_and_predictor_numerics_match_merge_base(self):
        from . import transport, predictor, pipeline_worker
        from .protocol import REPO_ROOT
        baseline = "16d5d5fc114778f891fc39b2c21b9dfd62d96377"
        for module in (predictor, pipeline_worker, h2_training):
            path = Path(module.__file__)
            original = subprocess.check_output(["git", "show", baseline + ":" + str(path.relative_to(REPO_ROOT))], cwd=REPO_ROOT)
            self.assertEqual(path.read_bytes(), original)
        reference = types.ModuleType("research.src._transport_reference")
        source = subprocess.check_output(["git", "show", baseline + ":research/src/transport.py"], cwd=REPO_ROOT)
        with patch.dict(sys.modules, {reference.__name__: reference}), torch.random.fork_rng(devices=[]):
            exec(compile(source, "transport_reference", "exec"), reference.__dict__)
            torch.manual_seed(9)
            left = torch.randn(1, 768, 4, 4)
            right = left + .01 * torch.randn_like(left)
            mask = torch.ones(4, 4, dtype=torch.bool)
            calibration = dict(max_displacement_chebyshev_tokens=4, similarity_min=-1.,
                               margin_min=-1., margin_scale=1., cycle_max_tokens=100.)
            outputs = []
            model = predictor.RobustTransportBlock5Predictor().eval()
            # Exercise the learned residual path, not only the zero initialization.
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.uniform_(-.1, .1)
            for module in (reference, transport):
                corr = module.estimate_robust_correspondence(left, right, mask, calibration)
                out = module.robust_transport_interpolation(left, right, torch.tensor([.4]), corr, mask)
                inputs = (out.field, out.warped_difference, out.warp0.coverage,
                          out.warp1.coverage, out.fused_confidence)
                outputs.append((*inputs, model(*inputs, torch.tensor([.4]), torch.tensor([.5]))))
            for before, after in zip(*outputs):
                self.assertTrue(torch.equal(before, after))

    def test_admission_changes_only_deployment_fingerprint(self):
        config, _ = run_h2.load_config()
        kwargs = dict(predictor_sha256="weights", training_lineage_sha256="training",
                      bridge_sha256="bridge", dpvo_sha256="dpvo", dpvo_config_sha256="config", schedule_sha256="schedule")
        before = deployment_lineage(config, **kwargs)
        changed = copy.deepcopy(config)
        changed["admission"]["ordinal_rule"] = "different scientific deployment"
        self.assertEqual(training_contract(changed), training_contract(config))
        self.assertNotEqual(before["deployment_lineage_sha256"], deployment_lineage(changed, **kwargs)["deployment_lineage_sha256"])
        changed["training"]["learning_rate"] *= 2
        self.assertNotEqual(training_contract(changed), training_contract(config))

    def test_missing_checkpoint_fails_before_cuda_and_never_trains(self):
        with patch.object(run_h2, "load_canonical_predictor", side_effect=FileNotFoundError("canonical predictor")), \
             patch.object(run_h2, "initialize_formal_main_process", side_effect=AssertionError("CUDA")), \
             patch.object(h2_training, "train_predictor", side_effect=AssertionError("training")):
            with self.assertRaises(FileNotFoundError):
                run_h2.run(["MH_01_easy"])

    def test_complete_mock_evaluation_uses_exact_weights_and_never_republishes_checkpoint(self):
        config, _ = run_h2.load_config()
        records = [SimpleNamespace(identity=frame(i)) for i in range(21)]
        calls = []
        model = torch.nn.Linear(1, 1)
        state = {"weight": torch.ones(1)}
        checkpoint = {"state_dict": state, "state_dict_sha256": "verified-weights",
                      "train_only_calibration": {"frozen": True}, "best_epoch": 12,
                      "training_recipe": {"seed": 1234},
                      "training_lineage": {"training_lineage_sha256": "training"}}
        class Audit:
            def __init__(self, *args, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def phase(self, *args): return contextlib.nullcontext()
            def payload(self): return {}
        def jobs(tasks, *args, **kwargs):
            calls.append(tasks)
            if tasks[0]["kind"] == "materialize_schedule":
                return [{"sequence": "MH_01_easy", "schedule": {"bootstrap_end_candidate_index": 0}}], {}
            self.assertEqual([t["kind"] for t in tasks], ["formal_h2_full", "formal_h2_sparse",
                "formal_h2_representation_control", "formal_h2_representation_control", "formal_h2_predicted"])
            self.assertIs(tasks[-1]["predictor_state"], state)
            self.assertEqual(tasks[-1]["thresholds"], {"frozen": True})
            self.assertEqual(tasks[-1]["execution"]["predictor_device"], "1")
            self.assertEqual(tasks[-1]["execution"]["encoder_device"], "2")
            return [{"sequence": "MH_01_easy"}], {"maximum_concurrent_dpvo_instances": 1}
        def assemble(*args, **kwargs):
            args[9].mkdir(parents=True)
            return {"conditions": {"predicted_jepa_hidden": {"runtime": {
                "elapsed_seconds": 1., "provider_usage": {}}}},
                "efficiency": {"h2_stage_profile": {}}, "schedule": {"bootstrap_end_candidate_index": 0}}
        with tempfile.TemporaryDirectory() as name, contextlib.ExitStack() as stack:
            root = Path(name)
            cp = root / "checkpoint/predictor.pt"
            cp.parent.mkdir(); cp.write_bytes(b"verified immutable checkpoint")
            before = sha256_file(cp)
            config = copy.deepcopy(config)
            config["paths"]["output_root"] = str(root / "results")
            bridge = root / "bridge.pt"; bridge.write_bytes(b"bridge")
            config["paths"]["h1_bridge"] = str(bridge)
            bindings = {
                "REPO_ROOT": root, "load_config": lambda: (config, root / "config.yaml"),
                "load_canonical_predictor": lambda _: (checkpoint, cp, before),
                "initialize_formal_main_process": lambda **_: {"hardware": {}},
                "load_sequence_records": lambda *_: records,
                "sequence_geometry": lambda *_: (None, {}),
                "_load_bridge": lambda *_: (model, {"file_sha256": sha256_file(bridge)}),
                "release_cuda_training_state": lambda: {"cleanup_complete": True},
                "cpu_numa_layout": lambda _: {}, "fixed_cpu_profile": lambda _: {"stage_c": {}},
                "run_sequential_trajectory_jobs": jobs, "PersistentPerformanceAudit": Audit,
                "performance_diagnosis": lambda _: {}, "condition_runtime_diagnostics": lambda _: {},
                "add_cuda_worker_mapping": lambda *_: None,
                "_run_sequence": assemble, "_repository_provenance": lambda: {},
                "execution_provenance": lambda: {},
                "base_lineage": lambda *_args, **_kwargs: ({}, {"config_protocol": {
                    "dpvo_checkpoint_sha256": "dpvo", "dpvo_config_sha256": "cfg"}}),
                "complete_lineage": lambda *args: {},
                "write_sequence_metadata": lambda directory, module, sequence, result, *args: atomic_write_json(directory / "results.json", {"result": result}),
                "sequence_entry": lambda *args, **kwargs: {"status": "valid"},
                "write_registry_and_summary": lambda directory, module, index: atomic_write_json(directory / "INDEX.json", index),
                "validate_module_manifest": lambda *args: None,
            }
            (root / "config.yaml").write_text("test configuration")
            for key, value in bindings.items():
                stack.enter_context(patch.object(run_h2, key, value))
            stack.enter_context(patch.object(torch.cuda, "device_count", return_value=3))
            stack.enter_context(patch.object(h2_training, "train_predictor", side_effect=AssertionError("training")))
            stack.enter_context(patch.object(h2_training, "calibrate_train_only_thresholds", side_effect=AssertionError("recalibration")))
            result = run_h2.run(["MH_01_easy"])
            self.assertEqual(result["checkpoint_training"], "not_performed")
            self.assertEqual(sha256_file(cp), before)
            self.assertEqual(len(calls), 2)
            index = json.loads((root / "results/INDEX.json").read_text())
            self.assertFalse(index["canonical_checkpoint"]["training_performed"])
            self.assertEqual(index["canonical_checkpoint"]["best_epoch"], 12)
            self.assertEqual(index["deployment_lineages"]["MH_01_easy"]["admission"], config["admission"])
