"""CPU coverage of the exact Uniform rule and its native insertion boundary."""
import contextlib
import dataclasses
import queue
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from . import runtime as rt
from .h2_pipeline import CanonicalH2Pipeline, AnchorTask
from .oracle_packet import FMapZeroContextPacket
from .predictor import AnchorInterval, HiddenQuery
from .profiling import OnlineProfiler
from .test_h2 import frame
from .uniform_admission import AdmissionReceipt, uniform_ordinals, uniform_hidden_keys, validate_admission_trajectory


def bracket(stride=5, index=0):
    left, right = frame(index * stride), frame((index + 1) * stride)
    queries = tuple(HiddenQuery(frame(index * stride + q), q, q / stride, stride / 10)
                    for q in range(1, stride))
    return AnchorInterval(index, left, right, queries)


class UniformAdmissionTest(unittest.TestCase):
    def test_exact_quantiles_independent_of_order_and_rng(self):
        for stride, expected in ((2, ()), (3, (2,)), (5, (2, 4)), (10, (2, 4, 6, 8))):
            self.assertEqual(uniform_ordinals(stride), expected)
            interval = bracket(stride)
            wanted = {interval.hidden[i-1].identity.key for i in expected}
            with patch("random.Random", side_effect=AssertionError("no selection RNG")):
                self.assertEqual(uniform_hidden_keys([interval], stride), wanted)
                reversed_interval = dataclasses.replace(interval, hidden=interval.hidden[::-1])
                self.assertEqual(uniform_hidden_keys([reversed_interval], stride), wanted)
        for invalid in (True, 0, 1, 5.0):
            with self.assertRaises(ValueError):
                uniform_ordinals(invalid)

    def test_incomplete_duplicate_and_invalid_ordinal_brackets_fail(self):
        interval = bracket()
        bad = (dataclasses.replace(interval, hidden=interval.hidden[:-1]),
               dataclasses.replace(interval, hidden=(interval.hidden[0],) * 4),
               dataclasses.replace(interval, hidden=(dataclasses.replace(interval.hidden[0], ordinal=2), *interval.hidden[1:])))
        for value in bad:
            with self.assertRaises(ValueError):
                uniform_hidden_keys([value], 5)
        with self.assertRaises(ValueError):
            uniform_hidden_keys([interval, interval], 5)

    def test_receipt_rejects_missing_duplicate_and_out_of_order(self):
        interval = bracket()
        roles = {frame(i).key: "anchor" if i in (0, 5) else "hidden" for i in range(6)}
        receipt = AdmissionReceipt(roles, [interval], 5)
        with self.assertRaises(RuntimeError):
            receipt.finish()
        receipt.receive(frame(0))
        for identity in (frame(0), frame(2)):
            with self.assertRaises(RuntimeError):
                receipt.receive(identity)

    def test_all_generated_packets_reach_consumer_before_uniform_insertion(self):
        interval = bracket()
        roles = {frame(i).key: "anchor" if i in (0, 5) else "hidden" for i in range(6)}
        generated = []
        packets = FMapZeroContextPacket(torch.zeros(4, 1, 128, 8, 8))
        def observations():
            yield rt.AnchorRGBObservation(frame(0), torch.zeros(3, 32, 32), torch.ones(4), frame(0).timestamp_ns)
            generated.extend(q.identity.key for q in interval.hidden)
            for offset, q in enumerate(interval.hidden):
                yield rt.PacketObservation(q.identity, FMapZeroContextPacket(packets.fmap[offset:offset+1]),
                                           "hidden", frame(5).timestamp_ns)
            yield rt.AnchorRGBObservation(frame(5), torch.zeros(3, 32, 32), torch.ones(4), frame(5).timestamp_ns)
        slam = SimpleNamespace(network=SimpleNamespace(training=False), cfg=SimpleNamespace(), M=2, P=3, DIM=4,
            exp6_hidden_timestamps=set(), exp6_factor_count=0, exp6_hidden_source_factor_count=0,
            exp6_hidden_target_factor_count=0, n=0, m=0, counter=0, tlist=[], pg=SimpleNamespace(ii=torch.zeros(0)))
        draws = {}
        def track(timestamp, *args, **kwargs):
            if kwargs["kind"] == "hidden":
                self.assertEqual(len(generated), 4)
            draws[timestamp] = torch.rand(2)
            slam.tlist.append(timestamp)
            slam.counter += 1
            slam.n += 1
        slam.track_packet = Mock(side_effect=track)
        slam.terminate = lambda: (np.tile([0., 0., 0., 0., 0., 0., 1.], (slam.counter, 1)), np.asarray(slam.tlist))
        event = SimpleNamespace(record=lambda: None, synchronize=lambda: None, elapsed_time=lambda other: 0.)
        original = torch.as_tensor
        def cpu_tensor(*args, **kwargs):
            kwargs.pop("device", None)
            return original(*args, **kwargs)
        with contextlib.ExitStack() as stack:
            for name, value in (("_formal_classes", Mock(return_value=(object, object))),
                                ("_make_slam", Mock(return_value=slam)), ("_seed_everything", Mock()),
                                ("_runtime_config", Mock(return_value={})), ("_state_is_finite", Mock(return_value=True)),
                                ("extract_frontend_packet", Mock(return_value=packets))):
                stack.enter_context(patch.object(rt, name, value))
            conversion = stack.enter_context(patch.object(rt, "packet_to_native", return_value=object()))
            stack.enter_context(patch.object(torch, "as_tensor", cpu_tensor))
            for name in ("synchronize", "empty_cache", "reset_peak_memory_stats"):
                stack.enter_context(patch.object(torch.cuda, name))
            stack.enter_context(patch.object(torch.cuda, "max_memory_allocated", return_value=0))
            stack.enter_context(patch.object(torch.cuda, "Event", return_value=event))
            metrics, arrays = rt.run_deployment_observations(
                observations(), np.ones(4), {"experiment": {"seed": 1234, "post_bootstrap_anchor_interval": 5}},
                image_height=32, image_width=32, condition_name="predicted_jepa_hidden", expected_roles=roles,
                profiler=OnlineProfiler(), intervals=[interval])
        self.assertEqual(conversion.call_count, 4)
        self.assertEqual(slam.track_packet.call_count, 4)
        self.assertEqual(slam.tlist, [frame(i).timestamp_ns for i in (0, 2, 4, 5)])
        self.assertEqual(metrics["admission"]["generated_hidden_count"], 4)
        self.assertEqual(metrics["admission"]["discarded_count"], 2)
        records = [SimpleNamespace(identity=frame(i)) for i in range(6)]
        validate_admission_trajectory(arrays, records, roles, 5)
        from .observation_sampling import observation_rng_scope
        for i in (0, 2, 4, 5):
            with observation_rng_scope(frame(i), 1234):
                self.assertTrue(torch.equal(draws[frame(i).timestamp_ns], torch.rand(2)))
        corrupted = dict(arrays, admission_insert=~arrays["admission_insert"])
        with self.assertRaises(RuntimeError):
            validate_admission_trajectory(corrupted, records, roles, 5)

    def test_pipeline_yields_all_bridged_packets_through_bounded_queue(self):
        from .h2_pipeline import CanonicalH2Pipeline, AnchorTask
        from .profiling import OnlineProfiler
        pipeline = object.__new__(CanonicalH2Pipeline)
        pipeline.stop = threading.Event()
        pipeline.failures = queue.Queue()
        pipeline.consume_queue, pipeline.encode_queue = queue.Queue(2), queue.Queue(2)
        pipeline.execution = SimpleNamespace(timeout_seconds=2.)
        pipeline.profiler = OnlineProfiler()
        pipeline.threads, pipeline.waits, pipeline.timeline = [], [], []
        pipeline.available_anchor_keys, pipeline.encoded_anchor_keys = set(), []
        pipeline.yielded_candidate_keys, pipeline.consumed_hidden_keys = [], []
        pipeline.hidden_context_wait, pipeline.interval_start_compute_ms = {}, {}
        pipeline.calibration = np.ones(4)
        pipeline.anchor_frontend = Mock()
        pipeline._stage_c_h2d = lambda value, name: torch.from_numpy(value)
        pipeline._bridge_packet = Mock(side_effect=lambda value: FMapZeroContextPacket(value[:, None]))
        first = bracket(5, 0)
        tasks = [AnchorTask(first.anchor0, np.zeros((32, 32, 3)), None, 0)]
        for index in range(2):
            interval = bracket(5, index)
            task = AnchorTask(interval.anchor1, np.zeros((32, 32, 3)), interval, 0)
            task.prediction = np.zeros((4, 128, 8, 8))
            tasks.append(task)
        for task in tasks:
            task.ready.set()
        def produce():
            for task in tasks:
                pipeline._put(pipeline.consume_queue, task)
            pipeline._put(pipeline.consume_queue, None)
        pipeline._produce, pipeline._predict_tasks = produce, lambda: None
        original = torch.as_tensor
        def cpu_tensor(*args, **kwargs):
            kwargs.pop("device", None)
            return original(*args, **kwargs)
        received = []
        hidden = []
        with patch.object(torch, "as_tensor", cpu_tensor):
            for observation in pipeline.observations():
                received.append(observation.identity.key)
                if isinstance(observation, rt.PacketObservation):
                    hidden.append(observation.identity.key)
        self.assertEqual(received, [frame(i).key for i in range(11)])
        self.assertEqual(pipeline._bridge_packet.call_count, 2)
        self.assertEqual(len(hidden), 8)
        self.assertEqual(pipeline.consume_queue.unfinished_tasks, 0)
        self.assertTrue(all(task.consumed.is_set() for task in tasks))

