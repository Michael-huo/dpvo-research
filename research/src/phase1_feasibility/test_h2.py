from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from .h2_deployment import DelayedDeploymentProvider
from .oracle_packet import FMapZeroContextPacket
from .predictor import (AnchorInterval, HiddenQuery, RobustTransportBlock5Predictor,
                        build_anchor_intervals, prediction_loss,
                        split_anchor_intervals)
from .profiling import (OnlineProfiler, break_even_payload, context_wait_values,
                        distribution_ms)
from .protocol import FrameIdentity, post_bootstrap_ratio_roles
from .registry import (CONDITION_LABELS, CONDITION_ORDER, render_sequence_summary,
                       write_sequence_metadata)
from .run_h2 import CONDITIONS, TRAINING_ANCHOR_RATIO, _metadata, load_config
from .runtime import AnchorRGBObservation, PacketObservation, run_deployment_observations
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
            "efficiency": {"profiling": "must_not_appear"},
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
        with patch(
            "research.src.phase1_feasibility.h2_deployment.JepaSidecar",
            FakeSidecar,
        ), patch.object(torch.cuda, "empty_cache"), patch.object(
            torch.cuda, "reset_peak_memory_stats",
        ):
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
            "full_rgb_reference", "sparse_rgb_reference", "anchor_jepa_only",
            "oracle_jepa_hidden_reference", "predicted_jepa_hidden",
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

        metadata = {
            line[2:].split(": ", 1)[0]: line[2:].split(": ", 1)[1]
            for line in lines if line.startswith("- ")
        }
        predicted = result["conditions"]["predicted_jepa_hidden"]["runtime"]
        self.assertEqual(metadata["Anchor ratio"], "22.77%")
        uploaded, candidates = metadata["Uploaded anchors / total candidates"].split(" / ")
        self.assertEqual(int(uploaded), predicted["rgb_uploaded_frame_count"])
        self.assertEqual(int(candidates), result["candidate_count"])
        self.assertEqual(int(metadata["Hidden count"]), result["hidden_count"])
        self.assertEqual(int(metadata["Hidden RGB violation count"]),
                         predicted["hidden_online_rgb_violation_count"])
        self.assertEqual(metadata["Deployment mode"], "delayed/bracketed, non-causal")

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
        self.assertEqual(payload["cloud_compute_ms"], 6.0)
        self.assertEqual(set(payload["stages"]), {
            "anchor_decode_preprocess", "jepa_encoder", "jepa_predictor",
            "bridge", "native_dpvo_frontend", "dpvo_graph_runtime",
        })
        self.assertNotIn("artifact_io", payload["stages"])
        self.assertEqual(payload["peak_online_vram_bytes"], 123)
        finite = break_even_payload(
            encoded_full_bytes=1000, encoded_anchor_bytes=500,
            h2_cloud_compute_ms=20, full_rgb_cloud_compute_ms=10,
        )
        self.assertEqual(finite["status"], "finite")
        self.assertEqual(finite["break_even_uplink_bandwidth_bps"], 400_000)
        unneeded = break_even_payload(
            encoded_full_bytes=1000, encoded_anchor_bytes=500,
            h2_cloud_compute_ms=5, full_rgb_cloud_compute_ms=10,
        )
        self.assertEqual(unneeded["status"], "infinite_or_not_required")

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
        loss = prediction_loss(predicted, target, torch.ones(3, 5, dtype=torch.bool))["total"]
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(value).all() for value in gradients))
        self.assertTrue(any(bool((value != 0).any()) for value in gradients))

    def test_effective_sample_count_formula(self) -> None:
        weights = torch.tensor([1.0, 2.0, 3.0])
        expected = weights.sum().square() / (weights.square().sum() + 1e-6)
        self.assertTrue(torch.allclose(effective_sample_count(weights), expected))


if __name__ == "__main__":
    unittest.main()
