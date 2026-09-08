from __future__ import annotations

import inspect
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from . import benchmark_h2_efficiency, efficiency_profiling, run_h0, run_h1, run_h2
from .benchmark_h2_efficiency import _copy_mean, _new_output, run_smoke
from .efficiency_profiling import (
    PerformanceRecorder, PersistentPerformanceAudit, TransferLedger,
    canonical_artifact_snapshot,
    bootstrap_pipeline, logical_physical_mapping, parse_nvidia_smi_row,
    parse_nvidia_topology, performance_diagnosis, simulate_pipeline,
    validate_output_path,
)
from .predictor import RobustTransportBlock5Predictor, prediction_loss
from .transport import estimate_robust_correspondence, robust_transport_interpolation


def _calibration() -> dict[str, float | int]:
    return {
        "similarity_min": -1.0, "margin_min": -1.0, "margin_scale": 1.0,
        "cycle_max_tokens": 100.0, "max_displacement_chebyshev_tokens": 12,
    }


class H2EfficiencyTest(unittest.TestCase):
    def test_cpu_and_cuda_domains_reconcile_independently(self) -> None:
        recorder = PerformanceRecorder(enable_cuda=False)
        recorder.begin_outer()
        with recorder.stage("one", cuda=False):
            sum(range(100))
        recorder.finish_outer()
        payload = recorder.payload()
        self.assertTrue(payload["domains_must_not_be_combined"])
        self.assertGreaterEqual(payload["cpu"]["uninstrumented_ms"], 0.0)
        self.assertFalse(payload["cuda"]["available"])
        self.assertEqual(payload["cuda"]["outer"]["count"], 0)

    def test_exclusive_profiler_rejects_nesting(self) -> None:
        recorder = PerformanceRecorder(enable_cuda=False)
        with recorder.stage("outer", cuda=False):
            with self.assertRaisesRegex(RuntimeError, "nesting is forbidden"):
                with recorder.stage("inner", cuda=False):
                    pass

    def test_profiled_transport_is_scientifically_identical_on_cpu(self) -> None:
        torch.manual_seed(7)
        left = torch.randn(1, 32, 3, 5)
        right = left + .01 * torch.randn_like(left)
        mask = torch.ones(3, 5, dtype=torch.bool)
        plain = estimate_robust_correspondence(left, right, mask, _calibration())
        recorder = PerformanceRecorder(enable_cuda=False)
        recorder.begin_outer()
        profiled = estimate_robust_correspondence(
            left, right, mask, _calibration(), profiler=recorder,
        )
        alpha = torch.tensor([.4])
        plain_transport = robust_transport_interpolation(left, right, alpha, plain, mask)
        profiled_transport = robust_transport_interpolation(
            left, right, alpha, profiled, mask, profiler=recorder,
        )
        recorder.finish_outer()
        for name in plain.__dataclass_fields__:
            self.assertTrue(torch.equal(getattr(plain, name), getattr(profiled, name)), name)
        for name in plain_transport.__dataclass_fields__:
            plain_value, profiled_value = getattr(plain_transport, name), getattr(profiled_transport, name)
            if isinstance(plain_value, torch.Tensor):
                self.assertTrue(torch.equal(plain_value, profiled_value), name)
            elif hasattr(plain_value, "__dataclass_fields__"):
                for field in plain_value.__dataclass_fields__:
                    self.assertTrue(torch.equal(
                        getattr(plain_value, field), getattr(profiled_value, field),
                    ), f"{name}.{field}")
        payload = recorder.payload()
        self.assertIn("coarse_residual_estimation", payload["cpu"]["exclusive_stages"])
        self.assertIn(
            "coarse_local_count_gpu_to_cpu_sync",
            payload["cpu"]["nested_wait_diagnostics_excluded_from_reconciliation"],
        )
        self.assertGreaterEqual(payload["cpu"]["uninstrumented_ms"], 0.0)

    def test_profile_scope_preserves_gradient_and_optimizer_step(self) -> None:
        torch.manual_seed(11)
        left = RobustTransportBlock5Predictor()
        right = RobustTransportBlock5Predictor()
        right.load_state_dict(left.state_dict())
        optim_left = torch.optim.AdamW(left.parameters(), lr=1e-4, weight_decay=1e-4)
        optim_right = torch.optim.AdamW(right.parameters(), lr=1e-4, weight_decay=1e-4)
        transport = torch.randn(2, 768, 2, 3)
        difference = torch.randn_like(transport)
        reliability = torch.ones(2, 1, 2, 3)
        alpha, delta = torch.tensor([.25, .75]), torch.tensor([.5, .5])
        target, mask = torch.randn_like(transport), torch.ones(2, 3, dtype=torch.bool)

        optim_left.zero_grad(set_to_none=True)
        output_left = left(transport, difference, reliability, reliability,
                           reliability, alpha, delta)
        loss_left = prediction_loss(output_left, target, mask)["total"]
        loss_left.backward(); optim_left.step()

        recorder = PerformanceRecorder(enable_cuda=False); recorder.begin_outer()
        optim_right.zero_grad(set_to_none=True)
        with recorder.stage("neural_residual_predictor_forward", cuda=False):
            output_right = right(transport, difference, reliability, reliability,
                                 reliability, alpha, delta)
        with recorder.stage("forward_loss", cuda=False):
            loss_right = prediction_loss(output_right, target, mask)["total"]
        with recorder.stage("backward", cuda=False):
            loss_right.backward()
        with recorder.stage("optimizer_scaler", cuda=False):
            optim_right.step()
        recorder.finish_outer()
        self.assertTrue(torch.equal(output_left, output_right))
        self.assertTrue(torch.equal(loss_left, loss_right))
        for (name_left, value_left), (name_right, value_right) in zip(
                left.state_dict().items(), right.state_dict().items()):
            self.assertEqual(name_left, name_right)
            self.assertTrue(torch.equal(value_left, value_right), name_left)

    def test_transfer_ledger_keeps_cpu_and_cuda_separate(self) -> None:
        ledger = TransferLedger()
        ledger.add("copy", byte_count=16, cpu_ms=2.0, cuda_ms=.5)
        payload = ledger.payload()["operations"]["copy"]
        self.assertEqual(payload["total_bytes"], 16)
        self.assertEqual(payload["cpu_wall"]["total_ms"], 2.0)
        self.assertEqual(payload["cuda_copy"]["total_ms"], .5)

    def test_persistent_performance_audit_is_exclusive_and_diagnostic_only(self) -> None:
        topology = {
            "logical_devices": [{
                "logical_index": 0, "physical_index": 2, "uuid": "GPU-two",
                "pci_bus_id": "C", "name": "RTX", "mapping_basis": "test",
                "mapping_error": None,
            }],
        }
        sampler_payload = {"sample_period_ms": 200, "error": None, "devices": {}}
        with patch.object(efficiency_profiling, "gpu_topology_audit", return_value=topology), \
                patch.object(efficiency_profiling, "current_cuda_device", return_value={
                    "available": True, "logical_index": 0, "pid": 7,
                }), patch.object(
                    efficiency_profiling, "gpu_process_snapshot",
                    return_value={"rows": [], "error": None},
                ), patch.object(
                    efficiency_profiling.NvidiaSmiSampler, "start",
                ), patch.object(
                    efficiency_profiling.NvidiaSmiSampler, "stop",
                ), patch.object(
                    efficiency_profiling.NvidiaSmiSampler, "payload",
                    return_value=sampler_payload,
                ):
            with PersistentPerformanceAudit(
                "h0_state", "sequence:test", components=("dpvo",),
            ) as audit:
                with audit.phase("one"):
                    sum(range(100))
        payload = audit.payload()
        self.assertTrue(payload["diagnostic_only"])
        self.assertTrue(payload["excluded_from_scientific_metrics_and_decisions"])
        self.assertGreaterEqual(payload["cpu_wall"]["uninstrumented_ms"], 0.0)
        self.assertEqual(payload["devices"]["components"]["dpvo"]["uuid"], "GPU-two")

    def test_persistent_diagnosis_keeps_cpu_and_cuda_rankings_separate(self) -> None:
        payload = {
            "cpu_wall": {"exclusive_phases": {}}, "condition_runtime": {},
            "strict_h2_predictor": {
                "cpu": {"exclusive_stages": {
                    "coarse": {"total_ms": 10.0}, "forward": {"total_ms": 2.0},
                }},
                "cuda": {"exclusive_stages": {
                    "coarse": {"total_ms": 8.0}, "forward": {"total_ms": 3.0},
                }},
            },
        }
        diagnosis = performance_diagnosis(payload)
        self.assertTrue(diagnosis["cpu_cuda_values_are_not_combined"])
        self.assertEqual(diagnosis["strict_h2_predictor_cpu_ranking"][0][0], "coarse")
        self.assertEqual(diagnosis["strict_h2_predictor_cuda_ranking"][0][1], 8.0)

    def test_regular_runners_persist_performance_diagnostics(self) -> None:
        for module in (run_h0, run_h1, run_h2):
            source = inspect.getsource(module.run)
            self.assertIn("PersistentPerformanceAudit", source)
            self.assertIn('performance_diagnostics', source)

    def test_nvidia_smi_and_topology_parsers(self) -> None:
        row = parse_nvidia_smi_row(
            "2026/09/07 12:00:00.000, 1, GPU-abc, 00000000:02:00.0, "
            "NVIDIA GeForce RTX 4090, 87, 32, 1024, 24564, 300.5, 450.0"
        )
        self.assertEqual(row["index"], 1)
        self.assertEqual(row["utilization.gpu"], 87.0)
        topology = parse_nvidia_topology(
            "        GPU0    GPU1    GPU2    CPU Affinity\n"
            "GPU0     X      PHB     SYS     0-15\n"
            "GPU1    PHB      X      PIX     0-15\n"
            "GPU2    SYS     PIX      X      16-31\n"
        )
        self.assertIn({"source": "GPU1", "destination": "GPU2", "link": "PIX"}, topology)
        self.assertEqual(len(topology), 6)

        ansi_topology = parse_nvidia_topology(
            "\x1b[4mGPU0 GPU1 GPU2 CPU Affinity\x1b[0m\n"
            "GPU0 X SYS SYS 0-15\n"
            "GPU1 SYS X NODE 16-31\n"
            "GPU2 SYS NODE X 16-31\n"
        )
        self.assertIn(
            {"source": "GPU1", "destination": "GPU2", "link": "NODE"},
            ansi_topology,
        )

    def test_logical_physical_mapping_uses_nvml_index(self) -> None:
        inventory = [
            {"physical_index": 0, "uuid": "GPU-zero", "pci_bus_id": "A", "name": "a"},
            {"physical_index": 2, "uuid": "GPU-two", "pci_bus_id": "C", "name": "c"},
        ]
        with patch("torch.cuda._get_nvml_device_index", side_effect=[2, 0]):
            rows = logical_physical_mapping(2, inventory)
        self.assertEqual(rows[0]["uuid"], "GPU-two")
        self.assertEqual(rows[1]["pci_bus_id"], "A")
        self.assertEqual(rows[0]["mapping_basis"], "torch.cuda._get_nvml_device_index")

    def test_transfer_model_never_assumes_zero_cost(self) -> None:
        direct = {"pairs": [{"source": 0, "destination": 1, "measurements": {
            "endpoint_block5": {"path": "direct_device_copy",
                                "cuda_copy_ms": {"mean_ms": .25}},
        }}]}
        staged = {"pairs": [{"source": 1, "destination": 2, "measurements": {
            "bridge_fmap": {"path": "pinned_host_staging",
                            "cpu_wall_ms": {"mean_ms": 1.5}},
        }}]}
        self.assertEqual(_copy_mean(direct, 0, 1, "endpoint_block5"), .25)
        self.assertEqual(_copy_mean(staged, 1, 2, "bridge_fmap"), 1.5)
        with self.assertRaisesRegex(RuntimeError, "no measured transfer model"):
            _copy_mean({"pairs": []}, 0, 1, "endpoint_block5")

    def test_pipeline_obeys_dependencies_and_sensor_cadence(self) -> None:
        rows = [
            {"anchor_available_ms": 100.0, "encode_ms": 3.0, "predict_ms": 7.0,
             "bridge_ms": 1.0, "transfer_ms": .5, "dpvo_ms": 5.0},
            {"anchor_available_ms": 200.0, "encode_ms": 3.0, "predict_ms": 7.0,
             "bridge_ms": 1.0, "transfer_ms": .5, "dpvo_ms": 5.0},
        ]
        compute = simulate_pipeline(rows, sensor_paced=False)
        sensor = simulate_pipeline(rows, sensor_paced=True)
        self.assertEqual(compute["makespan_ms"], 25.0)
        self.assertEqual(sensor["makespan_ms"], 216.5)
        self.assertLess(compute["events"][1]["encode_ready_ms"],
                        compute["events"][0]["dpvo_ready_ms"])
        self.assertIn("uplink_transmission", sensor["excludes"])
        first = bootstrap_pipeline(rows, sensor_paced=False, repetitions=20, seed=3)
        second = bootstrap_pipeline(rows, sensor_paced=False, repetitions=20, seed=3)
        self.assertEqual(first, second)
        self.assertEqual(first["makespan_ms"]["count"], 20)

    def test_canonical_output_path_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "canonical results"):
            validate_output_path(
                Path("research/results/phase1-feasibility/h2_prediction/audit")
            )
        with tempfile.TemporaryDirectory() as name:
            self.assertEqual(validate_output_path(name), Path(name).resolve())
            (Path(name) / "existing").write_text("occupied", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "absent or empty"):
                _new_output(name)

    def test_artifact_snapshot_is_stable_and_smoke_is_non_mutating(self) -> None:
        before = canonical_artifact_snapshot()
        payload = run_smoke()
        after = canonical_artifact_snapshot()
        self.assertEqual(before, after)
        self.assertEqual(payload["mode"], "smoke")

    def test_benchmark_does_not_import_or_call_formal_h2_run(self) -> None:
        source = inspect.getsource(benchmark_h2_efficiency)
        self.assertNotIn("publish_current_canonical", source)
        self.assertNotIn("evaluate_paired_trajectory", source)
        self.assertNotIn("run_formal_mode", source)
        self.assertNotIn("from .run_h2 import run", source)

    def test_current_component_mapping_is_static_cuda_zero(self) -> None:
        from . import h2_deployment, jepa_worker
        deployment_source = inspect.getsource(h2_deployment.DelayedDeploymentProvider)
        worker_source = inspect.getsource(jepa_worker._load)
        self.assertIn('device="cuda"', deployment_source)
        self.assertIn('torch.device("cuda:0")', worker_source)
        self.assertNotIn("set_device", deployment_source + worker_source)


if __name__ == "__main__":
    unittest.main()
