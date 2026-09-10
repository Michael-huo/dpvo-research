from __future__ import annotations

import inspect
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from . import h2_deployment
from . import run_h2
from .h2_deployment import DelayedDeploymentProvider
from .jepa_runtime import CompactFeatureStore, JepaSidecar, RestrictedFeatureView
from .oracle_packet import FMapZeroContextPacket
from .predictor import (AnchorInterval, HiddenQuery, RobustTransportBlock5Predictor,
                        build_anchor_intervals, prediction_loss,
                        split_anchor_intervals)
from .profiling import (OnlineProfiler, break_even_payload, context_wait_values,
                        distribution_ms, graph_workload_payload,
                        matched_wall_clock_payload)
from .protocol import FrameIdentity, post_bootstrap_ratio_roles
from .registry import (CONDITION_LABELS, CONDITION_ORDER, render_sequence_summary,
                       write_sequence_metadata)
from .run_h2 import CONDITIONS, TRAINING_ANCHOR_RATIO, _metadata, load_config
from .runtime import (AnchorRGBObservation, PacketObservation,
                      run_deployment_observations, warmup_dpvo_frontend)
from .transport import (effective_sample_count, forward_soft_splat,
                        normalized_descriptors)


def frame(index: int) -> FrameIdentity:
    return FrameIdentity("euroc", "machine_hall", "sequence", "MH_01_easy",
                         index * 2, index, 1_000_000_000 + index * 100_000_000)


def interval() -> AnchorInterval:
    left, hidden, right = frame(0), frame(1), frame(2)
    return AnchorInterval(0, left, right, (
        HiddenQuery(hidden, 1, .5, .2),
    ))


class H2ContractTest(unittest.TestCase):
    def _summary_results(self) -> dict:
        values = {
            key: (
                0.12345678901234566 + index,
                0.23456789012345678 + index,
                10.345678901234567 + index,
                0.9123456789012345 + index / 100.0,
                101 + index,
            )
            for index, key in enumerate(CONDITION_ORDER["h2_prediction"])
        }
        conditions = {}
        for key, (ate, translation, rotation, coverage, nodes) in values.items():
            conditions[key] = {
                "canonical_evaluation": {
                    "ate_rmse_m": ate,
                    "translation_rpe_rmse_m": translation,
                    "rotation_rpe_rmse_deg": rotation,
                    "sim3": {"must_not_appear": True},
                },
                "canonical_coverage": {"canonical_pose_coverage": coverage},
                "runtime": {
                    "final_node_count_before_terminate": nodes,
                    "hidden_online_rgb_violation_count": 0,
                    "rgb_uploaded_frame_count": 23,
                },
                "strict_deployment": key == "predicted_jepa_hidden",
                "timestamp_causal": key != "predicted_jepa_hidden",
                "closing_anchor_online_available": True,
            }
        return {
            "candidate_count": 101,
            "hidden_count": 78,
            "schedule": {
                "actual_full_sequence_anchor_ratio": 0.22772277227722773,
                },
            "conditions": conditions,
            "efficiency": {
                "transmission": {"anchor_ratio": .22772277227722773,
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
                    "stage_c_timing": {
                        "cpu_wall_exclusive": {
                            name: {"total_ms": float(index + 1) * 10.0}
                            for index, name in enumerate((
                                "stage_c_queue_wait", "stage_c_transfer_wait",
                                "bridge_compute", "native_frontend_compute",
                                "dpvo_graph_compute", "dpvo_sync_wait",
                                "stage_c_python_other",
                            ))
                        },
                        "cuda_event": {
                            "bridge_compute": {"total_ms": 30.0},
                            "native_frontend_compute": {"total_ms": 40.0},
                            "dpvo_graph_compute": {"total_ms": 50.0},
                        },
                        "queue_wait_breakdown": {
                            "prediction_ready_wait": {"total_ms": 1.0},
                            "consume_queue_wait": {"total_ms": 2.0},
                            "worker_flush_wait": {"total_ms": 3.0},
                        },
                    },
                    "producer_queue_backpressure": {
                        "total_ms": 4.0, "p95_ms": 2.0,
                    },
                },
                "break_even_uplink_bandwidth": {
                    "status": "finite", "break_even_uplink_bandwidth_mbps": 8.0,
                },
            },
        }

    def _provider(self, *, anchor_paths: dict[str, str] | None = None) -> DelayedDeploymentProvider:
        row = interval()
        config, _ = load_config()
        with tempfile.TemporaryDirectory() as name:
            return DelayedDeploymentProvider(
                anchor_paths=anchor_paths or {row.anchor0.key: "/anchor0.png",
                                              row.anchor1.key: "/anchor1.png"},
                identities=(row.anchor0, row.hidden[0].identity, row.anchor1),
                intervals=(row,), transform=SimpleNamespace(),
                calibration=np.ones(4), config=config, temporary=Path(name),
                bridge=SimpleNamespace(), predictor=SimpleNamespace(),
                transport_calibration={}, profiler=OnlineProfiler(),
            )

    def test_hidden_rgb_capability_is_rejected(self) -> None:
        row = interval()
        with self.assertRaisesRegex(PermissionError, "hidden RGB capability"):
            self._provider(anchor_paths={row.anchor0.key: "/a.png",
                                         row.hidden[0].identity.key: "/hidden.png",
                                         row.anchor1.key: "/b.png"})

    def test_hidden_identity_cannot_be_preprocessed(self) -> None:
        row = interval(); provider = self._provider()
        with self.assertRaisesRegex(PermissionError, "no uploaded anchor RGB"):
            provider._preprocess(row.hidden[0].identity)

    def test_closing_anchor_must_be_available(self) -> None:
        row = interval(); provider = self._provider()
        provider._last_anchor_key = row.anchor0.key
        provider._last_anchor_field = torch.zeros(1)
        with self.assertRaisesRegex(RuntimeError, "closing anchor A5"):
            provider._predict_interval(row, torch.zeros(1))

    def test_delayed_replay_encodes_only_anchors_and_consumes_hidden_once(self) -> None:
        row = interval(); provider = self._provider()
        encoded: list[str] = []

        class FakeSidecar:
            provenance = {"peak_gpu_memory_allocated_bytes": 0}

            def __init__(self, *_args, **_kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *_args): return None

        def preprocess(_identity):
            return (np.zeros((768, 2, 2), np.float32),
                    torch.zeros(3, 16, 16, dtype=torch.uint8), torch.ones(4))

        def encode(identity, _request_id, _jepa_input):
            encoded.append(identity.key)
            provider.encoded_anchor_keys.append(identity.key)
            return torch.zeros(1, 768, 2, 2)

        def packet(field):
            return FMapZeroContextPacket(torch.zeros(field.shape[0], 1, 128, 2, 2))

        def predict(current, right):
            self.assertIn(current.anchor1.key, provider.available_anchor_keys)
            self.assertEqual(right.shape, (1, 768, 2, 2))
            return torch.zeros(len(current.hidden), 768, 2, 2)

        provider._warmup = lambda _identity: None
        provider._preprocess = preprocess
        provider._encode_preprocessed = encode
        provider._bridge_packet = packet
        provider._predict_interval = predict
        with patch.object(
            h2_deployment, "JepaSidecar", FakeSidecar,
        ), patch.object(torch.cuda, "empty_cache"), patch.object(
            torch.cuda, "reset_peak_memory_stats",
        ):
            with provider.online_session():
                observations = list(provider.observations())
        self.assertEqual(encoded, [row.anchor0.key, row.anchor1.key])
        self.assertEqual([item.identity.key for item in observations],
                         [row.anchor0.key, row.hidden[0].identity.key, row.anchor1.key])
        self.assertIsInstance(observations[0], AnchorRGBObservation)
        self.assertIsInstance(observations[1], PacketObservation)
        self.assertIsInstance(observations[2], AnchorRGBObservation)
        self.assertEqual(observations[1].availability_timestamp_ns,
                         row.anchor1.timestamp_ns)
        self.assertTrue(provider.usage_payload()["hidden_consumed_exactly_once"])
        self.assertTrue(provider.usage_payload()["anchor_encoded_exactly_once"])
        self.assertTrue(provider.usage_payload()["candidate_yielded_exactly_once"])

    def test_provider_capabilities_exclude_hidden_references(self) -> None:
        provider = self._provider()
        attributes = set(vars(provider))
        self.assertFalse(any(name in attributes for name in (
            "hidden_rgb", "oracle_store", "true_fmap_store", "groundtruth",
            "pose", "depth", "graph_state",
        )))
        usage = provider.usage_payload()
        self.assertFalse(usage["contains_hidden_rgb_or_path_capability"])
        self.assertFalse(usage["contains_hidden_reference_or_groundtruth_capability"])
        self.assertEqual(set(provider.config), {"runtime", "jepa"})
        self.assertNotIn("dataset", provider.config)
        self.assertNotIn("paths", provider.config)
        self.assertEqual(set(provider.anchor_paths), {interval().anchor0.key,
                                                     interval().anchor1.key})

    def test_h2_metadata_is_delayed_and_only_prediction_is_strict(self) -> None:
        self.assertEqual(CONDITIONS, (
            "full_rgb_reference", "sparse_rgb_reference",
            "oracle_jepa_hidden_reference", "anchor_jepa_only",
            "predicted_jepa_hidden",
        ))
        metadata = _metadata()
        strict = [name for name, value in metadata.items() if value["strict_deployment"]]
        self.assertEqual(strict, ["predicted_jepa_hidden"])
        self.assertFalse(metadata["predicted_jepa_hidden"]["timestamp_causal"])
        self.assertTrue(metadata["predicted_jepa_hidden"]["closing_anchor_online_available"])
        self.assertEqual(
            metadata["predicted_jepa_hidden"]["input_source"],
            "native_anchor_rgb_plus_predicted_hidden_jepa_bridge",
        )
        self.assertEqual(TRAINING_ANCHOR_RATIO, 0.2)
        config, _ = load_config()
        self.assertEqual(config["training"]["cosine_weight"], 1.0)
        self.assertEqual(config["training"]["smooth_l1_weight"], .1)
        self.assertEqual(config["training"]["checkpoint_selector"], "lowest_validation_total")
        self.assertFalse(any(
            key.startswith("fmap_") and key.endswith("_weight")
            for key in config["training"]
        ))

    def test_sequence_summary_formats_display_without_rounding_results_json(self) -> None:
        result = self._summary_results()
        with tempfile.TemporaryDirectory() as name:
            output = Path(name) / "MH_01_easy"
            output.mkdir()
            write_sequence_metadata(
                output, "h2_prediction", "MH_01_easy", result, {}, {},
            )
            results_path = output / "results.json"
            summary_path = output / "SUMMARY_H2.md"
            persisted_results = json.loads(results_path.read_text(encoding="utf-8"))
            summary = summary_path.read_text(encoding="utf-8")
        self.assertEqual(persisted_results["result"], result)
        lines = summary.splitlines()
        self.assertEqual(lines[2], (
            "| Condition | ATE RMSE (m) | translation RPE@1s (m) | "
            "rotation RPE@1s (deg) | coverage | nodes |"
        ))

        table_rows = lines[4:9]
        self.assertEqual(
            [row.split("|")[1].strip() for row in table_rows],
            [CONDITION_LABELS[key] for key in CONDITION_ORDER["h2_prediction"]],
        )
        self.assertEqual(
            table_rows[0],
            "| Full RGB | 0.1235 | 0.2346 | 10.35 | 91.2% | 101 |",
        )

        self.assertIn("## Efficiency", summary)
        self.assertIn("anchor ratio 22.77%; encoded byte reduction 75.00%", summary)
        self.assertIn("Full RGB 10.000 s; H2 12.000 s", summary)
        self.assertIn("JEPA encoder 1.000 s; predictor 2.000 s", summary)
        self.assertIn("context wait mean 250.00 ms; P95 400.00 ms", summary)
        self.assertIn("DPVO graph CUDA-event span 4.000 s", summary)
        self.assertIn("Producer queue backpressure: 4.00 ms total", summary)
        self.assertIn("8.000 Mbps", summary)

    def test_sequence_summary_is_atomic_and_contains_no_low_frequency_details(self) -> None:
        result = self._summary_results()
        with tempfile.TemporaryDirectory() as name:
            output = Path(name)
            with patch.object(Path, "read_text", side_effect=AssertionError("unexpected read")):
                write_sequence_metadata(
                    output, "h2_prediction", "MH_01_easy", result, {}, {},
                )
            summary = (output / "SUMMARY_H2.md").read_text(encoding="utf-8")
        lowered = summary.lower()
        for forbidden in ("sim3", "sha256", "training", "profiling"):
            self.assertNotIn(forbidden, lowered)
        self.assertIn("`results.json` is the authoritative sequence result", summary)

    def test_h2_does_not_import_h1_runner_or_results(self) -> None:
        module = __import__("research.src.phase1_feasibility.run_h2", fromlist=["run"])
        source = inspect.getsource(module)
        self.assertNotIn("from .run_h1 import", source)
        self.assertNotIn("h1_interface/results.json", source)
        self.assertIn('"h1_results_json_reads": 0', source)

    def test_latency_and_break_even_aggregation(self) -> None:
        stats = distribution_ms([1.0, 2.0, 3.0])
        self.assertEqual(stats["total_ms"], 6.0)
        profiler = OnlineProfiler()
        for stage in profiler.samples:
            profiler.add(stage, 1.0)
        payload = profiler.payload(peak_online_vram_bytes=123)
        self.assertEqual(payload["profiled_stage_subtotal_ms"], 6.0)
        self.assertEqual(set(payload["stages"]), {
            "anchor_decode_preprocess", "jepa_encoder", "jepa_predictor",
            "bridge", "native_dpvo_frontend", "dpvo_graph_runtime",
        })
        self.assertNotIn("artifact_io", payload["stages"])
        self.assertEqual(payload["peak_online_vram_bytes"], 123)
        finite = break_even_payload(
            encoded_full_bytes=1000, encoded_anchor_bytes=500,
            extra_cloud_compute_s=.01,
        )
        self.assertEqual(finite["status"], "finite")
        self.assertEqual(finite["break_even_uplink_bandwidth_bps"], 400_000)
        unneeded = break_even_payload(
            encoded_full_bytes=1000, encoded_anchor_bytes=500,
            extra_cloud_compute_s=-.005,
        )
        self.assertEqual(unneeded["status"], "no_extra_cloud_compute")

    def test_matched_wall_and_graph_workload_schema(self) -> None:
        wall = matched_wall_clock_payload(
            full_rgb_seconds=10.0, h2_seconds=12.5, sparse_rgb_seconds=3.0,
        )
        self.assertEqual(wall["h2_over_full_rgb_ratio"], 1.25)
        self.assertEqual(wall["extra_cloud_compute_s"], 2.5)
        runtime = {
            "processed_observation_count": 5,
            "final_node_count_before_terminate": 4,
            "final_patch_count_before_terminate": 384,
            "factor_count_allocated": 100,
            "final_active_factor_count": 20,
            "dpvo_graph_runtime_total_ms": 25.0,
        }
        workload = graph_workload_payload(runtime)
        self.assertEqual(workload["dpvo_graph_mean_ms_per_processed_observation"], 5.0)
        self.assertEqual(workload["cumulative_factor_count"], 100)

    def test_context_wait_is_not_part_of_profiled_compute(self) -> None:
        profiler = OnlineProfiler()
        profiler.context_wait_ms.extend([100.0, 200.0])
        profiler.add("bridge", 3.0)
        payload = profiler.payload(peak_online_vram_bytes=0)
        self.assertEqual(payload["profiled_stage_subtotal_ms"], 3.0)
        self.assertEqual(payload["context_wait_ms"]["total_ms"], 300.0)

    def test_stage_c_cpu_categories_reconcile_and_cuda_is_separate(self) -> None:
        profiler = OnlineProfiler()
        samples = {
            "stage_c_queue_wait": 2.0,
            "stage_c_transfer_wait": 3.0,
            "bridge_compute": 5.0,
            "native_frontend_compute": 7.0,
            "dpvo_graph_compute": 11.0,
            "dpvo_sync_wait": 13.0,
            "stage_c_python_other": 17.0,
        }
        for category, value in samples.items():
            profiler.add_stage_c_cpu(category, value)
        profiler.add_stage_c_cuda("dpvo_graph_compute", 101.0)
        profiler.finalize_stage_c(100.0)
        timing = profiler.payload(peak_online_vram_bytes=0)["stage_c_timing"]
        self.assertAlmostEqual(timing["exclusive_cpu_total_ms"], 100.0)
        self.assertAlmostEqual(timing["reconciliation_error_ms"], 0.0)
        self.assertAlmostEqual(
            timing["cpu_wall_exclusive"]["stage_c_python_other"]["total_ms"],
            59.0,
        )
        self.assertEqual(
            timing["cuda_event"]["dpvo_graph_compute"]["total_ms"], 101.0,
        )
        self.assertTrue(timing["cpu_and_cuda_domains_must_not_be_added"])

    def test_formal_h2_uses_fixed_cpu_profiles(self) -> None:
        source = inspect.getsource(run_h2.run)
        self.assertIn('"selection": "fixed_same_as_h0_h1"', source)
        self.assertIn('"dynamic_calibration": False', source)
        self.assertIn('"components": fixed_cpu_profile(layout)', source)
        self.assertIn(
            'cpu_profile=selected_profile["components"]["stage_c"]', source,
        )

    def test_dpvo_timing_excludes_packet_conversion_and_explicit_sync(self) -> None:
        source = inspect.getsource(run_deployment_observations)
        conversion = source.index("native_packet = packet_to_native")
        graph = source.index("slam.track_packet")
        self.assertLess(conversion, graph)
        self.assertIn('"stage_c_python_other"', source[conversion - 500:graph])
        self.assertIn('"dpvo_graph_compute"', source[graph:graph + 800])
        self.assertIn('"dpvo_sync_wait"', source[graph:graph + 1200])
        self.assertIn("CUDA event span bounded immediately", source)
        self.assertIn("host dispatch gaps inside those calls", source)

    def test_sidecar_online_ready_and_flush_handshakes(self) -> None:
        sidecar = JepaSidecar.__new__(JepaSidecar)
        sidecar.process = SimpleNamespace(stdin=io.StringIO())
        sidecar.provenance = {"peak_gpu_memory_allocated_bytes": 99}
        sidecar._receive = Mock(side_effect=[
            {"status": "online_ready", "request_id": "online",
             "worker_cuda_synchronized": True, "worker_peak_memory_reset": True},
            {"status": "online_flushed", "request_id": "online",
             "worker_cuda_synchronized": True, "peak_gpu_memory_allocated_bytes": 123},
        ])
        sidecar.prepare_online(); sidecar.flush_online()
        actions = [json.loads(line)["action"]
                   for line in sidecar.process.stdin.getvalue().splitlines()]
        self.assertEqual(actions, ["prepare_online", "flush_online"])
        self.assertEqual(sidecar.provenance["peak_gpu_memory_allocated_bytes"], 123)

    def test_cross_process_timer_barrier_order_is_explicit(self) -> None:
        source = inspect.getsource(run_deployment_observations)
        self.assertLess(source.index("worker_barrier.prepare_online"),
                        source.index("matched_started = time.perf_counter()"))
        self.assertLess(source.index("worker_barrier.flush_online"),
                        source.index("elapsed_seconds = float"))
        self.assertIn("main_cuda_synchronized_after_worker_flush_before_stop", source)

    def test_dpvo_warmup_declares_throwaway_fresh_state(self) -> None:
        source = inspect.getsource(warmup_dpvo_frontend)
        self.assertIn("del slam", source)
        self.assertIn('"timed_dpvo_starts_from_fresh_state": True', source)
        self.assertIn('"stateful_dpvo_graph_warmed": False', source)

    def test_context_wait_uses_integer_timestamps(self) -> None:
        self.assertEqual(context_wait_values((interval(),)), [100.0])

    def test_runtime_is_fail_closed(self) -> None:
        runtime_source = inspect.getsource(run_deployment_observations)
        self.assertIn("hidden RGB capability", runtime_source)
        self.assertIn('"hidden_online_rgb_violation_count": 0', runtime_source)
        self.assertIn("processed_keys != expected_keys", runtime_source)

    def test_h2_uses_only_canonical_training_module(self) -> None:
        module = __import__("research.src.phase1_feasibility.run_h2", fromlist=["run"])
        source = inspect.getsource(module)
        self.assertNotIn("run_checkpoint_validation", source)
        self.assertIn("from .h2_training import", source)

    def test_canonical_interval_split(self) -> None:
        identities = [
            FrameIdentity("euroc", "machine_hall", "sequence", "MH_01_easy",
                          index * 2, index, 1_000_000_000 + index * 50_000_000)
            for index in range(1841)
        ]
        roles = post_bootstrap_ratio_roles(
            identities, bootstrap_end_candidate_index=7, anchor_ratio=.2,
        )
        intervals = build_anchor_intervals(identities, roles)
        split, payload = split_anchor_intervals(intervals)
        self.assertEqual(len(intervals), 366)
        self.assertEqual(
            {name: (len(rows), sum(len(row.hidden) for row in rows))
             for name, rows in split.items()},
            {"train": (219, 876), "validation": (72, 288), "test": (73, 292)},
        )
        self.assertEqual(payload["dropped_boundary_interval_indices"], [219, 292])
        anchors = {
            name: {key for row in rows for key in row.anchor_keys}
            for name, rows in split.items()
        }
        self.assertFalse(anchors["train"] & anchors["validation"])
        self.assertFalse(anchors["validation"] & anchors["test"])

    def test_matching_normalizes_but_transport_preserves_raw_values(self) -> None:
        raw = torch.randn(1, 8, 3, 5)
        self.assertTrue(torch.allclose(
            normalized_descriptors(raw), normalized_descriptors(raw * 7), atol=1e-6,
        ))
        displacement = torch.zeros(1, 3, 5, 2)
        confidence = torch.ones(1, 3, 5)
        mask = torch.ones(3, 5, dtype=torch.bool)
        warped = forward_soft_splat(raw, displacement, confidence, torch.tensor([.5]), mask)
        self.assertTrue(torch.allclose(warped.field, raw, atol=1e-6))
        self.assertFalse(torch.allclose(warped.field, normalized_descriptors(raw).transpose(1, 2).reshape_as(raw)))

    def test_predictor_zero_initialization_and_gradient(self) -> None:
        model = RobustTransportBlock5Predictor()
        transport = torch.randn(2, 768, 3, 5)
        difference = torch.randn_like(transport)
        reliability = torch.ones(2, 1, 3, 5)
        predicted = model(
            transport, difference, reliability, reliability, reliability,
            torch.tensor([.25, .75]), torch.tensor([.5, .5]),
        )
        self.assertTrue(torch.equal(predicted, transport))
        target = torch.randn_like(predicted)
        mask = torch.ones(3, 5, dtype=torch.bool)
        metrics = prediction_loss(predicted, target, mask)
        selected_prediction = predicted.float().permute(0, 2, 3, 1)[:, mask]
        selected_target = target.float().permute(0, 2, 3, 1)[:, mask]
        expected = (
            1 - torch.nn.functional.cosine_similarity(
                selected_prediction, selected_target, dim=-1, eps=1e-8,
            ).mean()
            + .1 * torch.nn.functional.smooth_l1_loss(
                selected_prediction, selected_target,
            )
        )
        self.assertTrue(torch.equal(metrics["total"], expected))
        loss = metrics["total"]
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(value).all() for value in gradients))
        self.assertTrue(any(bool((value != 0).any()) for value in gradients))

    def test_held_out_teacher_capability_rejects_out_of_scope_and_closed_access(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            held_out_identity, out_of_scope_identity = frame(1), frame(2)
            store = CompactFeatureStore(
                Path(name), "teacher", (held_out_identity,), (2, 2, 2),
            )
            store.put(held_out_identity, np.ones((2, 2, 2), np.float32)); store.finalize()
            view = RestrictedFeatureView(
                store, {held_out_identity.key}, "held_out_test_true_fmap_diagnostic",
            )
            self.assertEqual(view.get(held_out_identity).shape, (2, 2, 2))
            with self.assertRaises(PermissionError):
                view.get(out_of_scope_identity)
            self.assertEqual(view.usage_payload()["read_count"], 1)
            store.close()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                view.get(held_out_identity)

    def test_true_fmap_is_diagnostic_only_and_deployment_isolated(self) -> None:
        from . import run_h2
        from . import h2_training
        source = inspect.getsource(run_h2.run)
        selected = source.index("predictor, training_summary = train_predictor")
        saved = source.index("checkpoint = _save_predictor")
        validated = source.index("_validate_fresh_predictor")
        held_out_teacher = source.index("test_teacher_store, test_teacher_extraction")
        self.assertLess(selected, saved)
        self.assertLess(saved, validated)
        self.assertLess(validated, held_out_teacher)
        self.assertIn('"test_was_read_during_training_or_selection": False', source)
        self.assertIn('"stores_closed_before_deployment": stores_closed', source)
        self.assertNotIn("extract_true_fmap_store", source[:held_out_teacher])
        provider_attributes = set(vars(self._provider()))
        self.assertFalse(any("teacher" in name for name in provider_attributes))
        self.assertFalse(any(name in provider_attributes for name in (
            "true_fmap", "oracle_store", "reference_store",
        )))
        training_source = inspect.getsource(h2_training.train_predictor)
        parameters = inspect.signature(h2_training.train_predictor).parameters
        self.assertNotIn("bridge", parameters)
        self.assertFalse(any("teacher" in name for name in parameters))
        self.assertIn("prediction_loss", training_source)
        self.assertIn('validation["total"]', training_source)
        self.assertIn('"train_total"', training_source)
        self.assertIn('"validation_total"', training_source)
        for training_function in (h2_training.tiny_overfit, h2_training._validation):
            function_parameters = inspect.signature(training_function).parameters
            self.assertNotIn("bridge", function_parameters)
            self.assertFalse(any("teacher" in name for name in function_parameters))
            self.assertIn("prediction_loss", inspect.getsource(training_function))

        held_out_source = inspect.getsource(h2_training.held_out_representation)
        self.assertIn("_diagnostic_teacher_batch", held_out_source)
        self.assertIn('"fmap_target": "offline_true_fmap_teacher"', held_out_source)

        sequence_source = inspect.getsource(run_h2._run_sequence)
        run_source = inspect.getsource(run_h2.run)
        self.assertIn('"closed_before_strict_deployment": True', sequence_source)
        self.assertNotIn("run_formal_jobs(", sequence_source)
        self.assertIn("run_sequential_trajectory_jobs(", run_source)
        self.assertIn('"maximum_concurrent_dpvo_instances": 1', sequence_source)

    def test_checkpoint_recipe_and_h1_bridge_lineage_remain_canonical(self) -> None:
        from . import run_h2
        save_source = inspect.getsource(run_h2._save_predictor)
        run_source = inspect.getsource(run_h2.run)
        self.assertIn('"target": "offline_oracle_hidden_jepa_block5"', save_source)
        self.assertIn('"h1_bridge_sha256": bridge_meta["file_sha256"]', run_source)
        self.assertNotIn("objective", save_source)

    def test_effective_sample_count_formula(self) -> None:
        weights = torch.tensor([1.0, 2.0, 3.0])
        expected = weights.sum().square() / (weights.square().sum() + 1e-6)
        self.assertTrue(torch.allclose(effective_sample_count(weights), expected))

    def test_formal_h2_uses_residency_parallel_precompute_and_canonical_pipeline(self) -> None:
        from . import run_h2
        source = inspect.getsource(run_h2.run)
        self.assertIn("extract_parallel(", source)
        self.assertIn("correspondence_parallel(", source)
        self.assertIn("ResidentH2View(", source)
        replay = inspect.getsource(run_h2._run_strict_replay)
        self.assertIn("CanonicalH2Pipeline(", replay)
        self.assertNotIn("DelayedDeploymentProvider(", replay)

    def test_formal_h2_mapping_and_trace_defaults_are_canonical(self) -> None:
        from dataclasses import asdict
        from .execution_runtime import FormalExecution, fixed_cpu_profile
        execution = FormalExecution()
        self.assertEqual(
            (execution.encoder_device, execution.predictor_device,
             execution.consumer_device),
            ("2", "1", "0"),
        )
        self.assertEqual(FormalExecution(**asdict(execution)), execution)
        self.assertFalse(execution.verify_transfers)
        self.assertFalse(execution.decision_trace)
        layout = {
            "stage_c": {"device": 0, "numa_node": 0, "cpus": [0, 1, 2, 3]},
            "predictor": {"device": 1, "numa_node": 1, "cpus": [4, 5]},
            "encoder": {"device": 2, "numa_node": 1, "cpus": [6, 7, 8]},
        }
        components = fixed_cpu_profile(layout)
        for name, threads in (("stage_c", 4), ("predictor", 1), ("encoder", 1)):
            with self.subTest(component=name):
                self.assertEqual(components[name], {
                    **layout[name], "intraop_threads": threads, "interop_threads": 1,
                    "omp_num_threads": threads, "mkl_num_threads": threads,
                })
        sets = [set(components[name]["cpus"])
                for name in ("stage_c", "predictor", "encoder")]
        self.assertFalse(sets[0] & sets[1])
        self.assertFalse(sets[0] & sets[2])
        self.assertFalse(sets[1] & sets[2])

    def test_h2_formal_trajectory_order_is_sequence_major_and_serial(self) -> None:
        from . import run_h2
        source = inspect.getsource(run_h2.run)
        positions = [source.index(f'"condition": "{name}"')
                     for name in ("oracle_jepa_hidden_reference", "anchor_jepa_only")]
        self.assertLess(*positions)
        self.assertIn('"kind": "formal_h2_full"', source)
        self.assertIn('"kind": "formal_h2_sparse"', source)
        self.assertIn('"kind": "formal_h2_predicted"', source)
        self.assertNotIn("ThreadPoolExecutor", source)

    def test_correspondence_decision_trace_has_exact_discrete_contract(self) -> None:
        from .transport import (
            decision_trace_payload, estimate_robust_correspondence,
            robust_transport_interpolation,
        )
        torch.manual_seed(9)
        left = torch.randn(1, 8, 4, 4)
        right = left + 0.01 * torch.randn_like(left)
        mask = torch.ones(4, 4, dtype=torch.bool)
        calibration = {
            "max_displacement_chebyshev_tokens": 4,
            "similarity_min": -1.0,
            "margin_min": -1.0,
            "margin_scale": 1.0,
            "cycle_max_tokens": 100.0,
        }
        trace = {}
        correspondence = estimate_robust_correspondence(
            left, right, mask, calibration, decision_trace=trace,
        )
        robust_transport_interpolation(
            left, right, torch.tensor([0.5]), correspondence, mask,
            decision_trace=trace,
        )
        payload = decision_trace_payload(trace)
        fields = payload["fields"]
        for name in (
            "selected_top2_indices_0_to_1", "selected_top2_indices_1_to_0",
            "candidate_mask_0", "candidate_mask_1", "accepted_mask_0",
            "accepted_mask_1", "global_fallback_0", "global_fallback_1",
            "local_fallback_0", "local_fallback_1", "transport_fallback_mask",
        ):
            self.assertIn(name, fields)
            self.assertEqual(len(fields[name]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
