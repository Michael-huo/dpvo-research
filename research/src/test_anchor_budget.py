"""CPU contracts for the budget pilot; no training or SLAM execution."""

from __future__ import annotations

import copy
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from . import anchor_budget as budget_protocol
from . import anchor_budget_training as training
from . import h2_training, run_anchor_budget as runner
from .anchor_budget_results import evaluation_population
from .evaluation import build_rpe_pairs, evaluate_paired_trajectory, freeze_evaluation_population
from .h2_deployment import DelayedDeploymentProvider
from .h2_pipeline import CanonicalH2Pipeline
from .oracle_packet import FMapZeroContextPacket
from .predictor import (RobustTransportBlock5Predictor, build_anchor_intervals,
                        effective_records, split_anchor_intervals)
from .profiling import OnlineProfiler
from .protocol import (FrameIdentity, PrivateFrameRecord, canonical_sha256,
                       load_sequence_records, repo_path)
from .runtime import PacketObservation
from .transport import RobustCorrespondence


def records(count=1841):
    # Integer timestamps and frame IDs from the frozen MH01 candidate manifest.
    return tuple(PrivateFrameRecord(
        FrameIdentity("euroc", "machine_hall", "sequence", "MH_01_easy", i * 2, i,
                      1403636579763555584 + i * 100_000_000), f"/unused/{i}.png")
                 for i in range(count))


class AnchorBudgetProtocolTest(unittest.TestCase):
    def setUp(self):
        self.protocol, self.config, _ = budget_protocol.load_protocol()
        self.records = records()

    def test_stride5_exact_against_frozen_accumulator_and_split(self):
        # Independent snapshot of the pre-generalization ratio accumulator.
        expected = {}
        accumulator = 1.0 - 0.2
        for row in self.records:
            if row.identity.candidate_index <= 7:
                expected[row.identity.key] = "anchor"
                continue
            accumulator += 0.2
            if accumulator >= 1.0 - 1e-12:
                expected[row.identity.key] = "anchor"
                accumulator -= 1.0
            else:
                expected[row.identity.key] = "hidden"
        actual = budget_protocol.stride_roles([r.identity for r in self.records], 7, 5)
        self.assertEqual(actual, expected)
        legacy = build_anchor_intervals(self.records, expected)
        current = budget_protocol.build_budget(self.records, 5, self.protocol["fixed_split"])
        self.assertEqual(current["intervals"], legacy)
        split, payload = split_anchor_intervals(legacy)
        self.assertEqual(current["split"], split)
        self.assertEqual(payload["split_sha256"], self.protocol["fixed_split"]["source_split_sha256"])
        self.assertEqual(current["split_payload"]["dropped_boundary_interval_indices"], [219, 292])
        order = budget_protocol.online_identity_order(current["records"], current["roles"], legacy)
        self.assertEqual(order, [r.identity.key for r in current["records"]])
        self.assertEqual(len(order), len(set(order)))

    def test_fixed_time_populations_and_no_shared_anchor(self):
        expected = {3: (619, 1220, 610, (365, 119, 121), 5),
                    5: (375, 1464, 366, (219, 72, 73), 2),
                    10: (192, 1647, 183, (109, 36, 36), 2)}
        for stride, (anchors, hidden, count, split_counts, dropped) in expected.items():
            with self.subTest(stride=stride):
                budget = budget_protocol.build_budget(self.records, stride, self.protocol["fixed_split"])
                self.assertEqual(len(budget["records"]), 1839)
                self.assertEqual(sum(v == "anchor" for v in budget["roles"].values()), anchors)
                self.assertEqual(sum(v == "hidden" for v in budget["roles"].values()), hidden)
                self.assertEqual(len(budget["intervals"]), count)
                self.assertTrue(all(len(i.hidden) == stride-1 for i in budget["intervals"]))
                self.assertEqual(tuple(len(budget["split"][n]) for n in budget_protocol.SPLITS), split_counts)
                self.assertEqual(len(budget["split_payload"]["dropped_boundary_interval_indices"]), dropped)
                seen = set()
                for name, rows in budget["split"].items():
                    region = self.protocol["fixed_split"]["regions"][name]
                    self.assertTrue(all(region["timestamp_start_ns"] <= i.anchor0.timestamp_ns
                                        < i.anchor1.timestamp_ns <= region["timestamp_end_ns"] for i in rows))
                    current = {key for i in rows for key in i.anchor_keys}
                    self.assertFalse(current & seen)
                    seen.update(current)
                for interval in budget["intervals"]:
                    for query in interval.hidden:
                        self.assertEqual(query.alpha, (query.identity.timestamp_ns-interval.anchor0.timestamp_ns)
                                         / (interval.anchor1.timestamp_ns-interval.anchor0.timestamp_ns))
                        self.assertEqual(query.delta_t_seconds,
                                         (interval.anchor1.timestamp_ns-interval.anchor0.timestamp_ns)/1e9)

    def test_tail_is_dropped_without_promoting_an_anchor(self):
        for stride in (3, 5, 10):
            for count in range(40, 51):
                source = records(count)
                roles = budget_protocol.stride_roles([r.identity for r in source], 7, stride)
                intervals = build_anchor_intervals(source, roles)
                effective, tail = effective_records(source, intervals)
                expected_last = 8 + ((count - 1 - 8) // stride) * stride
                self.assertEqual(effective[-1].identity.candidate_index, expected_last)
                self.assertEqual(tail["trailing_excluded_count"], count - expected_last - 1)
                self.assertEqual(roles[source[8].identity.key], "anchor")
                self.assertTrue(all(roles[r.identity.key] == "anchor" for r in source[:8]))

    def test_actual_ratio_and_encoded_bytes_use_effective_population(self):
        with tempfile.TemporaryDirectory() as name:
            source = []
            for i, row in enumerate(records(40)):
                path = Path(name) / f"{i}.png"
                path.write_bytes(b"x" * (i+1))
                source.append(PrivateFrameRecord(row.identity, str(path)))
            for stride in (3, 5, 10):
                roles = budget_protocol.stride_roles([r.identity for r in source], 7, stride)
                effective, _ = effective_records(source, build_anchor_intervals(source, roles))
                result = budget_protocol.encoded_communication(effective, roles)
                anchor_indices = [i for i,r in enumerate(effective) if roles[r.identity.key] == "anchor"]
                self.assertEqual(result["actual_anchor_ratio"], len(anchor_indices)/len(effective))
                self.assertNotEqual(result["actual_anchor_ratio"], 1/stride)
                self.assertEqual(result["encoded_anchor_bytes"], sum(i+1 for i in anchor_indices))
                self.assertEqual(result["encoded_full_bytes"], sum(range(1, len(effective)+1)))
                self.assertEqual(result["encoded_byte_reduction"],
                                 1-result["encoded_anchor_bytes"]/result["encoded_full_bytes"])

    def test_scientific_settings_are_inherited_without_method_changes(self):
        for stride in (3, 5, 10):
            config = budget_protocol.budget_config(self.config, self.protocol, stride)
            for section in ("predictor", "training", "transport", "bridge", "jepa", "evaluation", "dataset"):
                self.assertEqual(config[section], self.config[section])
            self.assertEqual(config["experiment"]["seed"], self.config["experiment"]["seed"])
            self.assertEqual(config["paths"]["h1_bridge"], self.config["paths"]["h1_bridge"])
        self.assertNotIn("anchor_stride", self.config["experiment"])
        self.assertEqual(budget_protocol.OUTPUT_ROOT, repo_path("research/results/anchor-budget"))

    def test_real_stride5_exactness_without_previous_artifacts(self):
        try:
            source = load_sequence_records(self.config, "MH_01_easy")
        except FileNotFoundError:
            self.skipTest("MH01 dataset not installed")
        report = budget_protocol.backward_equivalence(source, self.protocol)
        self.assertTrue(report["all_exact"])
        b = budget_protocol.build_budget(source, 5, self.protocol["fixed_split"])
        from .run_h2 import _evaluation_population
        self.assertEqual(evaluation_population(b["records"], b["roles"], self.config),
                         _evaluation_population(b["records"], b["roles"], self.config))

    def test_frozen_split_tampering_is_rejected(self):
        protocol = copy.deepcopy(self.protocol)
        protocol["fixed_split"]["source_split_sha256"] = "changed"
        with self.assertRaisesRegex(RuntimeError, "exact-equivalence failed"):
            budget_protocol.backward_equivalence(self.records, protocol)



class VariableQueryTest(unittest.TestCase):
    def test_predictor_accepts_two_four_and_nine_queries(self):
        model = RobustTransportBlock5Predictor().eval()
        with torch.no_grad():
            for count in (2, 4, 9):
                field = torch.randn(count, 768, 2, 3)
                coverage = torch.ones(count, 1, 2, 3)
                alpha = torch.arange(1, count+1)/(count+1)
                result = model(field, torch.zeros_like(field), coverage, coverage, coverage,
                               alpha, torch.full((count,), (count+1)/10))
                # Frozen zero-residual initialization is independent of horizon.
                self.assertTrue(torch.equal(result, field))

    def test_transport_batch_preserves_query_order_and_correspondence_repeat(self):
        source = records(40)
        for stride in (3, 5, 10):
            roles = budget_protocol.stride_roles([r.identity for r in source], 7, stride)
            intervals = build_anchor_intervals(source, roles)[:2]
            robust = h2_training.RobustCorrespondenceStore({})
            for interval in intervals:
                robust.put(interval, RobustCorrespondence(*[
                    torch.full((1, 2, 2), float(interval.interval_index))
                    for _ in RobustCorrespondence.__dataclass_fields__]))
            repeated = robust.batch(intervals, torch.device("cpu"), repeat_queries=True)
            self.assertEqual(repeated.confidence_0.shape[0], 2*(stride-1))
            with patch.object(h2_training, "_field", side_effect=lambda _s,i,_t,_d:
                              torch.full((768, 2, 2), float(i.candidate_index))), \
                 patch.object(h2_training, "robust_transport_interpolation") as transport:
                _, targets, alpha, delta = h2_training._transport_batch(
                    intervals, object(), object(), torch.ones(2, 2, dtype=torch.bool), robust)
            queries = [q for interval in intervals for q in interval.hidden]
            self.assertEqual(targets[:, 0, 0, 0].tolist(), [q.identity.candidate_index for q in queries])
            self.assertTrue(torch.equal(alpha, torch.tensor([q.alpha for q in queries])))
            self.assertTrue(torch.equal(delta, torch.tensor([q.delta_t_seconds for q in queries])))
            self.assertEqual(transport.call_args.args[0].shape[0], len(queries))

    def test_online_provider_rejects_hidden_paths_at_each_stride(self):
        config = budget_protocol.load_protocol()[1]
        source = records(40)
        for stride in (3, 5, 10):
            roles = budget_protocol.stride_roles([r.identity for r in source], 7, stride)
            intervals = build_anchor_intervals(source, roles)
            effective, _ = effective_records(source, intervals)
            paths = {r.identity.key: r.rgb_path for r in effective if roles[r.identity.key] == "anchor"}
            arguments = dict(anchor_paths=paths, identities=[r.identity for r in effective],
                             intervals=intervals, transform=object(), calibration=np.ones(4),
                             config=config, temporary=Path("/tmp/unused_budget_test"), bridge=object(),
                             predictor=object(), transport_calibration={}, profiler=OnlineProfiler())
            provider = DelayedDeploymentProvider(**arguments)
            self.assertFalse(provider.usage_payload()["contains_hidden_rgb_or_path_capability"])
            self.assertNotIn("dataset", provider.config)
            self.assertNotIn("paths", provider.config)
            paths[intervals[0].hidden[0].identity.key] = "/forbidden.png"
            with self.assertRaises(PermissionError):
                DelayedDeploymentProvider(**arguments)

    def test_variable_packet_assembly_closing_anchor_and_exactly_once(self):
        config = budget_protocol.load_protocol()[1]
        source = records(40)
        for stride in (3, 5, 10):
            roles = budget_protocol.stride_roles([r.identity for r in source], 7, stride)
            intervals = build_anchor_intervals(source, roles)
            effective, _ = effective_records(source, intervals)
            paths = {r.identity.key: r.rgb_path for r in effective if roles[r.identity.key] == "anchor"}
            provider = DelayedDeploymentProvider(
                anchor_paths=paths, identities=[r.identity for r in effective], intervals=intervals,
                transform=object(), calibration=np.ones(4), config=config,
                temporary=Path("/tmp/unused_budget_test"), bridge=object(), predictor=object(),
                transport_calibration={}, profiler=OnlineProfiler())
            provider._sidecar = object()
            provider._preprocess = lambda _: (None, torch.zeros(3,16,16), torch.ones(4))
            def encode(identity, *_args):
                provider.encoded_anchor_keys.append(identity.key)
                return torch.zeros(1,768,2,2)
            def predict(interval, _right):
                self.assertIn(interval.anchor1.key, provider.available_anchor_keys)
                self.assertEqual(provider._last_anchor_key, interval.anchor0.key)
                return torch.stack([torch.full((768,2,2), float(q.identity.candidate_index))
                                    for q in interval.hidden])
            provider._encode_preprocessed = encode
            provider._predict_interval = predict
            provider._bridge_packet = lambda fields: FMapZeroContextPacket(fields[:,None,:128])
            observations = list(provider.observations())
            self.assertEqual([o.identity.key for o in observations], [r.identity.key for r in effective])
            closing = {q.identity.key: i.anchor1.timestamp_ns for i in intervals for q in i.hidden}
            for observation in observations:
                if isinstance(observation, PacketObservation):
                    self.assertEqual(observation.availability_timestamp_ns, closing[observation.identity.key])
                    self.assertEqual(observation.packet.fmap.shape, (1,1,128,2,2))
                    self.assertTrue(torch.all(observation.packet.fmap == observation.identity.candidate_index))
            usage = provider.usage_payload()
            self.assertTrue(usage["candidate_yielded_exactly_once"])
            self.assertTrue(usage["anchor_encoded_exactly_once"])
            self.assertTrue(usage["hidden_consumed_exactly_once"])

    def test_pipeline_shared_slots_follow_variable_query_count(self):
        source = records(40)
        for stride in (3,5,10):
            roles = budget_protocol.stride_roles([r.identity for r in source], 7, stride)
            interval = build_anchor_intervals(source, roles)[0]
            slots = {"left": SimpleNamespace(array=np.empty((1,768,2,3),np.float32)),
                     "right": SimpleNamespace(array=np.empty((1,768,2,3),np.float32)),
                     "output": SimpleNamespace(array=np.ones((9,768,2,3),np.float32))}
            request = Mock(return_value={"shape": [stride-1,768,2,3], "dtype": "torch.float32", "compute_ms": 1.0})
            proxy = SimpleNamespace(predictor_slots=[slots,slots], host_transfer=Mock(),
                                    workers={"predictor": object()}, _request_slots=request,
                                    transform=SimpleNamespace(token_grid_height=2, token_grid_width=3))
            left = np.full((1,768,2,3), .123456, np.float32)
            output, _, _ = CanonicalH2Pipeline._predict(proxy, left, left, interval, "test", 0)
            self.assertEqual(output.shape, (stride-1,768,2,3))
            self.assertEqual(request.call_args.args[2]["alpha"], [q.alpha for q in interval.hidden])
            self.assertEqual(request.call_args.args[2]["delta"], [q.delta_t_seconds for q in interval.hidden])
            self.assertTrue(np.array_equal(slots["left"].array, left.astype(np.float16).astype(np.float32)))

    def test_checkpoint_lineage_rejects_cross_stride_and_legacy(self):
        for stride in (3, 5, 10):
            lineage = {"protocol": "anchor_budget_fresh_predictor_v1", "anchor_stride": stride,
                       "fixed_split_definition": {"frozen": True}}
            lineage["training_lineage_sha256"] = canonical_sha256(lineage)
            training.validate_stride_lineage(lineage, lineage, stride)
            with self.assertRaisesRegex(RuntimeError, "anchor_stride"):
                training.validate_stride_lineage(lineage, lineage, stride+1)
            with self.assertRaisesRegex(RuntimeError, "lineage mismatch"):
                training.validate_stride_lineage(lineage, lineage | {"extra": 1}, stride)
        with self.assertRaisesRegex(RuntimeError, "Anchor Budget predictor lineage"):
            training.validate_stride_lineage({}, {}, 5)

    def test_empty_rpe_is_explicit_opt_in_and_ate_remains_available(self):
        timestamps = tuple(i*300_000_000 for i in range(10))
        with self.assertRaises(ValueError):
            build_rpe_pairs(timestamps)
        population = freeze_evaluation_population(timestamps, horizon_seconds=1, tolerance_ns=1_000_000,
                                                   allow_empty_rpe=True)
        self.assertEqual(population.rpe_pairs_ns, ())
        with tempfile.TemporaryDirectory() as name:
            gt = Path(name) / "gt.txt"
            xyz = np.asarray([[i, i*i, i % 3] for i in range(10)], dtype=float)
            np.savetxt(gt, np.column_stack((timestamps, xyz, np.ones(10), np.zeros((10,3)))))
            poses = np.column_stack((xyz, np.zeros((10,3)), np.ones(10)))
            result = evaluate_paired_trajectory({"timestamps_ns": timestamps, "poses": poses}, population, gt)
            self.assertAlmostEqual(result["ate_rmse_m"], 0, places=9)
            self.assertEqual(result["rpe_pair_count"], 0)
            self.assertIsNone(result["translation_rpe_rmse_m"])

    def test_horizon_groups_are_query_weighted_and_offline(self):
        metrics = ("predicted_jepa_cosine", "transport_baseline_cosine", "bridge_fmap_cosine",
                   "transport_bridge_fmap_cosine", "distance_from_previous_anchor_seconds",
                   "distance_from_closing_anchor_seconds", "alpha", "delta_t_seconds")
        rows = [{"relative_hidden_index": ordinal, **{key: value for key in metrics}}
                for ordinal, value in ((1, .2), (2, .5), (1, .6))]
        result = training.group_horizon(rows, 3)
        self.assertEqual([r["hidden_query_count"] for r in result], [2, 1])
        self.assertAlmostEqual(result[0]["predicted_jepa_cosine"], .4)
        self.assertTrue(all(r["role"] == "offline_diagnostics_only" for r in result))


class BudgetExecutionTest(unittest.TestCase):
    def test_cli_defaults_to_preparation_and_accepts_one_sequence(self):
        args = runner.parser().parse_args([])
        self.assertFalse(args.execute)
        self.assertEqual(args.strides, [3, 5, 10])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.parser().parse_args(["--sequence", "MH_03_medium"])

    def test_default_run_never_initializes_cuda_or_executes(self):
        protocol, canonical, config_path = budget_protocol.load_protocol()
        with tempfile.TemporaryDirectory() as name:
            protocol = protocol | {"output_root": name}
            with patch.object(runner, "load_protocol", return_value=(protocol, canonical, config_path)), \
                 patch.object(runner, "load_sequence_records", return_value=()), \
                 patch.object(runner, "prepare_protocol", return_value=({}, {"populations": []})), \
                 patch.object(runner, "execute_sequence") as execute, \
                 patch.object(runner, "initialize_formal_main_process") as cuda:
                result = runner.run()
            self.assertEqual(result["status"], "prepared_no_experiment_run")
            execute.assert_not_called()
            cuda.assert_not_called()

    def test_trajectory_dispatch_is_a_single_blocking_job(self):
        task = {"kind": "formal_h2_sparse"}
        with patch.object(runner, "run_sequential_trajectory_jobs", return_value=(
            [{"condition": "sparse_rgb_reference"}], {"maximum_concurrent_dpvo_instances": 1})) as jobs:
            runner._run_job(task, Path("/tmp/unused"), {"hardware": {}}, {"components": {"stage_c": {}}})
        self.assertEqual(jobs.call_args.args[0], [task])

    def test_full_once_fresh_predictors_and_sequential_sparse_ours(self):
        protocol, canonical, _ = budget_protocol.load_protocol()
        source = records()
        budgets = {s: budget_protocol.build_budget(source, s, protocol["fixed_split"]) for s in (3, 5, 10)}
        index = {"anchor_strides": [3, 5, 10], "strides": {}, "accuracy_vs_communication": [],
                 "prediction_quality_vs_horizon": []}
        events = []
        submitted = []
        def train(_records, budget, _protocol, _config, output, _temporary):
            events.append(("train", budget["anchor_stride"]))
            output.mkdir(parents=True)
            checkpoint = output / "predictor.pt"
            checkpoint.write_bytes(b"test")
            return {"checkpoint_path": checkpoint, "bridge_state": {}, "transform": None,
                    "record": {"lineage": {}, "stores_closed_before_deployment": True,
                               "horizon_resolved_quality": []}}
        def job(task, *_args):
            events.append((task["kind"], task["config"]["experiment"].get("anchor_stride")))
            submitted.append(task)
            return {"condition": task["kind"]}
        result = {"canonical_evaluation": {"ate_rmse_m": 1.0},
                  "matched_trajectory_wall_seconds": 1.0, "graph_workload": {}}
        with tempfile.TemporaryDirectory() as name, contextlib.ExitStack() as stack:
            root = Path(name)
            stack.enter_context(patch.object(torch.cuda, "device_count", return_value=3))
            stack.enter_context(patch.object(runner, "initialize_formal_main_process", return_value={"hardware": {}}))
            stack.enter_context(patch.object(runner, "cpu_numa_layout", return_value={}))
            stack.enter_context(patch.object(runner, "fixed_cpu_profile", return_value={"stage_c": {}}))
            stack.enter_context(patch.object(runner, "_run_job", side_effect=job))
            stack.enter_context(patch.object(runner, "save_trajectory", return_value=result))
            stack.enter_context(patch.object(runner, "fresh_train_budget", side_effect=train))
            stack.enter_context(patch.object(runner, "release_training_before_trajectory", return_value={}))
            stack.enter_context(patch.object(runner, "load_budget_predictor", return_value={
                "train_only_calibration": {}, "state_dict": {}, "state_dict_sha256": "test"}))
            stack.enter_context(patch.object(runner, "population_summary", side_effect=lambda b: {"anchor_stride": b["anchor_stride"]}))
            stack.enter_context(patch.object(runner, "comparison_row", side_effect=lambda p,*_: p))
            stack.enter_context(patch.object(runner, "sha256_file", return_value="test"))
            stack.enter_context(patch.object(runner, "publish_index"))
            runner.execute_sequence(source, budgets, protocol, canonical, root, root / "temporary", index)
        self.assertEqual(events, [("formal_h2_full", None), *[
            row for stride in (3,5,10) for row in (("train", stride), ("formal_h2_sparse", stride),
                                                  ("formal_h2_predicted", stride))]])
        self.assertEqual(len(submitted), 7)
        for sparse, ours in zip(submitted[1::2], submitted[2::2]):
            self.assertEqual(sparse["roles"], ours["roles"])
            self.assertEqual(sparse["records"], ours["records"])
            self.assertFalse({"oracle_store", "true_fmap_store", "groundtruth", "hidden_features"} & ours.keys())
            self.assertEqual(ours["execution"]["encoder_device"], "2")
            self.assertEqual(ours["execution"]["predictor_device"], "1")
            self.assertEqual(ours["execution"]["consumer_device"], "0")
            self.assertIn(f"models/stride_{ours['config']['experiment']['anchor_stride']}", ours["predictor_checkpoint"])


if __name__ == "__main__":
    unittest.main()
