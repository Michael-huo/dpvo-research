"""Bounded delayed H2 pipeline; graph mutations belong exclusively to the consumer."""
from __future__ import annotations

import contextlib
import dataclasses
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .execution_runtime import (
    FormalExecution, capture_runtime,
    execution_provenance,
)
from .efficiency_profiling import TransferLedger
from .h2_deployment import DelayedDeploymentProvider
from .jepa_fmap import preprocess_full_fov_rgb
from .jepa_runtime import load_dpvo_domain
from .oracle_packet import FMapZeroContextPacket
from .profiling import distribution_ms
from .protocol import atomic_write_json
from .runtime import AnchorRGBObservation, PacketObservation
from .staged_transfer import GPUWorker, PinnedTransfer, SharedSlot


@dataclasses.dataclass
class AnchorTask:
    identity: Any
    bgr: np.ndarray
    interval: Any
    admitted_ns: int
    encoded: threading.Event = dataclasses.field(default_factory=threading.Event)
    ready: threading.Event = dataclasses.field(default_factory=threading.Event)
    frontended: threading.Event = dataclasses.field(default_factory=threading.Event)
    consumed: threading.Event = dataclasses.field(default_factory=threading.Event)
    field: Any = None
    prediction: Any = None
    decision_trace: Any = None
    encode_ms: float = 0.
    predict_ms: float = 0.
    encoded_ns: int = 0
    predicted_ns: int = 0


class CanonicalH2Pipeline(DelayedDeploymentProvider):
    def __init__(self, *, execution: FormalExecution | None = None, predictor_checkpoint: Path,
                 predictor_state_sha256: str,
                 worker_settings=None, cpu_profile=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.execution = execution or FormalExecution()
        self.predictor_checkpoint = str(predictor_checkpoint)
        self.predictor_state_sha256 = predictor_state_sha256
        self.cpu_profile = cpu_profile
        self.settings = worker_settings or capture_runtime()
        self.anchor_frontend = None
        self.stop = threading.Event(); self.failures = queue.Queue()
        self.encode_queue = queue.Queue(self.execution.queue_capacity)
        self.consume_queue = queue.Queue(self.execution.queue_capacity)
        self.workers = {}; self.slots = []; self.threads = []
        self.timeline = []; self.waits = []
        self.decision_traces = []
        self.transfer = None
        self.host_transfer = TransferLedger()
        self.worker_diagnostics = {}
        self.lifecycle_cleanup = None

    def attach_anchor_frontend(self, function):
        self.anchor_frontend = function

    def _wait(self, event, label):
        started = time.perf_counter()
        while not event.wait(.05):
            self._check()
            if time.perf_counter()-started > self.execution.timeout_seconds:
                raise TimeoutError(label)
        self._check(); self.waits.append({"name": label, "ms": (time.perf_counter()-started)*1000})

    def _check(self):
        # Polling happens on producer, predictor and consumer threads. Only
        # materialize the failure report when there is a failure: rebuilding a
        # set for every expected identity here made healthy polls O(N * seen),
        # contending for the GIL while Stage C submitted CUDA work.
        if self.failures.empty() and not self.stop.is_set():
            return
        consumed = set(self.yielded_candidate_keys)
        unconsumed = [identity.key for identity in self.identities
                      if identity.key not in consumed]
        if not self.failures.empty():
            raise RuntimeError(
                f"{self.failures.get()}\nunconsumed_identity={unconsumed}"
            )
        if self.stop.is_set():
            raise RuntimeError(f"H2 pipeline cancelled; unconsumed_identity={unconsumed}")

    def _put(self, target, value):
        started = time.perf_counter()
        while True:
            self._check()
            try: target.put(value, timeout=.05); break
            except queue.Full:
                if time.perf_counter()-started > self.execution.timeout_seconds:
                    raise TimeoutError("bounded pipeline queue stalled")
        self.waits.append({"name": "queue_put", "ms": (time.perf_counter()-started)*1000})

    def _get(self, source):
        started = time.perf_counter()
        while True:
            self._check()
            try: value = source.get(timeout=.05); break
            except queue.Empty:
                if time.perf_counter()-started > self.execution.timeout_seconds:
                    raise TimeoutError("bounded pipeline queue empty")
        self.waits.append({"name": "queue_get", "ms": (time.perf_counter()-started)*1000})
        return value

    def _guard(self, function):
        try: function()
        except BaseException as error:
            import traceback
            self.failures.put(f"{error}\n{traceback.format_exc()}")
            self.stop.set()

    @contextlib.contextmanager
    def online_session(self):
        self.temporary.mkdir(parents=True, exist_ok=True)
        worker_config = {**self.config, "worker_settings": self.settings,
                         "predictor_checkpoint": self.predictor_checkpoint,
                         "predictor_state_sha256": self.predictor_state_sha256,
                         "predictor": self.predictor_config,
                         "transport_calibration": self.transport_calibration,
                         "transform": dataclasses.asdict(self.transform),
                         "verify_transfers": self.execution.verify_transfers,
                         "decision_trace": self.execution.decision_trace}
        path = self.temporary / "worker.json"; atomic_write_json(path, worker_config)
        h, w = self.transform.token_grid_height, self.transform.token_grid_width
        count = max(len(i.hidden) for i in self.intervals)
        primary_error = None
        try:
            self.workers["encoder"] = GPUWorker(python=self.config["runtime"]["jepa_python"],
                device=self.execution.encoder_device, component="encoder", config_path=path,
                timeout=self.execution.timeout_seconds,
                cpu_profile=(self.cpu_profile or {}).get("components", {}).get("encoder"))
            self.workers["predictor"] = GPUWorker(python=sys.executable,
                device=self.execution.predictor_device, component="predictor", config_path=path,
                timeout=self.execution.timeout_seconds,
                cpu_profile=(self.cpu_profile or {}).get("components", {}).get("predictor"))
            self.jepa_worker_pid = self.workers["encoder"].process.pid
            self.encoder_slots = []
            self.predictor_slots = []
            for _ in range(2):
                enc = {"input": SharedSlot((1,3,1,self.transform.padded_height,self.transform.padded_width)),
                       "output": SharedSlot((1,h*w,768))}
                pred = {"left": SharedSlot((1,768,h,w)), "right": SharedSlot((1,768,h,w)),
                        "output": SharedSlot((count,768,h,w))}
                self.encoder_slots.append(enc); self.predictor_slots.append(pred)
                self.slots.extend([*enc.values(), *pred.values()])
            self.transfer = PinnedTransfer(verify=self.execution.verify_transfers)
            yield self
        except BaseException as error:
            primary_error = error
            raise
        finally:
            self.stop.set()
            errors = []
            worker_cleanup = {}
            for name, worker in self.workers.items():
                try:
                    worker_cleanup[name] = worker.close(
                        require_graceful=primary_error is None,
                    )
                except BaseException as error:
                    errors.append(f"worker {name}: {type(error).__name__}: {error}")
            for thread in self.threads:
                thread.join(timeout=5)
            alive = [thread.name for thread in self.threads if thread.is_alive()]
            if alive:
                errors.append(f"pipeline threads did not exit: {alive}")
            queue_cleanup = {}
            for name, value in (("encode", self.encode_queue),
                                ("consume", self.consume_queue)):
                unfinished = int(value.unfinished_tasks)
                if primary_error is not None:
                    while True:
                        try:
                            value.get_nowait(); value.task_done()
                        except queue.Empty:
                            break
                remaining = int(value.unfinished_tasks)
                queue_cleanup[name] = {
                    "unfinished_before_cleanup": unfinished,
                    "unfinished_after_cleanup": remaining,
                }
                if remaining:
                    errors.append(f"{name} queue has {remaining} unfinished tasks")
            transfer_cleanup = None
            if self.transfer is not None:
                try:
                    transfer_cleanup = self.transfer.close()
                except BaseException as error:
                    errors.append(f"pinned transfer: {type(error).__name__}: {error}")
            slot_cleanup = []
            for slot in self.slots:
                try:
                    status = slot.close(); slot_cleanup.append(status)
                    if status["was_busy"] and primary_error is None:
                        errors.append("shared slot remained busy at normal shutdown")
                except BaseException as error:
                    errors.append(f"shared slot: {type(error).__name__}: {error}")
            self.lifecycle_cleanup = {
                "workers": worker_cleanup, "threads_alive": alive,
                "queues": queue_cleanup, "transfer": transfer_cleanup,
                "shared_slots": slot_cleanup, "errors": errors,
                "complete": not errors,
            }
            if primary_error is not None:
                setattr(primary_error, "phase1_pipeline_cleanup", self.lifecycle_cleanup)
            elif errors:
                raise RuntimeError(f"H2 pipeline lifecycle cleanup failed: {errors}")

    @property
    def predictor_config(self):
        from .predictor import predictor_metadata
        # Constructor architecture comes from the already validated frozen model.
        meta = predictor_metadata(self.predictor)
        return meta | {"time_hidden_dim": meta["time_mlp"][1]}

    def prepare_online(self):
        # Warm every component before the matched timer, then reset transfer totals.
        interval = self.intervals[0]
        encoded = []
        for ordinal, identity in enumerate((interval.anchor0, interval.anchor1)):
            bgr, _ = load_dpvo_domain(self.anchor_paths[identity.key], self.calibration)
            value, _ = preprocess_full_fov_rgb(
                bgr[..., ::-1].copy(), self.transform,
            )
            field, _ = self._encode(
                value.numpy()[None], f"warmup:encoder:{ordinal}", ordinal,
            )
            encoded.append(field)
        prediction, _, _ = self._predict(
            encoded[0], encoded[1], interval, "warmup:predictor", 0,
        )
        field = self.transfer.h2d(prediction, "warmup_prediction")
        self.bridge(field.flatten(2).transpose(1, 2))
        torch.cuda.synchronize()
        for name, worker in self.workers.items():
            self.worker_diagnostics[name] = worker.request({
                "action": "barrier", "request_id": "ready",
            })
        self.transfer.close()
        self.transfer = PinnedTransfer(verify=self.execution.verify_transfers)
        self.host_transfer = TransferLedger()
        return {
            "status": "online_ready", "worker_cuda_synchronized": True,
            "execution_backend": execution_provenance(self.execution),
            "numerical_acceptance": None,
            "formal_trace_mode": "lightweight",
        }

    def flush_online(self):
        started = time.perf_counter()
        for name, worker in self.workers.items():
            self.worker_diagnostics[name] = worker.request({"action": "barrier", "request_id": "flush"})
        self.profiler.add_stage_c_cpu(
            "stage_c_queue_wait", (time.perf_counter() - started) * 1000.0,
            queue_detail="worker_flush_wait",
        )
        self.jepa_peak_online_vram_bytes = self.worker_diagnostics["encoder"]["peak_vram_bytes"]
        return {"status": "online_flushed", "worker_cuda_synchronized": True}

    def _request_slots(self, worker, slots, request):
        descriptors = {name: slot.acquire() for name, slot in slots.items()}
        try:
            result = worker.request(request | {"slots": descriptors})
            if result["generations"] != {name: d["generation"] for name,d in descriptors.items()}:
                raise RuntimeError("worker completed stale transfer generation")
            return result
        finally:
            for name, slot in slots.items():
                if slot.busy:
                    slot.release(descriptors[name]["generation"])

    def _encode(self, value, request_id, ordinal):
        slots = self.encoder_slots[ordinal % 2]
        started = time.perf_counter(); np.copyto(slots["input"].array, value)
        self.host_transfer.add("jepa_input_to_shared", byte_count=value.nbytes,
                               cpu_ms=(time.perf_counter()-started)*1000)
        response = self._request_slots(self.workers["encoder"], slots,
                                       {"action": "encode", "request_id": request_id})
        h,w = self.transform.token_grid_height,self.transform.token_grid_width
        if response["shape"] != [1, h*w, 768] or response["dtype"] != "torch.float32":
            raise RuntimeError("encoder boundary shape/dtype changed")
        started = time.perf_counter(); field = slots["output"].array.copy().transpose(0,2,1).reshape(1,768,h,w)
        self.host_transfer.add("jepa_output_from_shared", byte_count=field.nbytes,
                               cpu_ms=(time.perf_counter()-started)*1000)
        return field, response["compute_ms"]

    def _predict(self, left, right, interval, request_id, ordinal):
        slots = self.predictor_slots[ordinal % 2]
        started = time.perf_counter()
        np.copyto(slots["left"].array, left.astype(np.float16).astype(np.float32))
        np.copyto(slots["right"].array, right)
        self.host_transfer.add("predictor_endpoints_to_shared",
            byte_count=slots["left"].array.nbytes+slots["right"].array.nbytes,
            cpu_ms=(time.perf_counter()-started)*1000)
        response = self._request_slots(self.workers["predictor"], slots,
            {"action": "predict", "request_id": request_id,
             "alpha": [q.alpha for q in interval.hidden], "delta": [q.delta_t_seconds for q in interval.hidden]})
        expected_shape = [len(interval.hidden), 768,
                          self.transform.token_grid_height,
                          self.transform.token_grid_width]
        if response["shape"] != expected_shape or response["dtype"] != "torch.float32":
            raise RuntimeError("predictor boundary shape/dtype changed")
        started = time.perf_counter(); output=slots["output"].array[:len(interval.hidden)].copy()
        self.host_transfer.add("predictor_output_from_shared",byte_count=output.nbytes,
                               cpu_ms=(time.perf_counter()-started)*1000)
        return output, response["compute_ms"], response.get("decision_trace")

    def _produce(self):
        first_timestamp = self.identities[0].timestamp_ns
        epoch = time.monotonic_ns()
        anchors = [i for i in self.identities if i.key in self.anchor_paths]
        for ordinal, identity in enumerate(anchors):
            if self.execution.sensor_paced:
                target = epoch + identity.timestamp_ns-first_timestamp
                while time.monotonic_ns() < target:
                    self._check(); self.stop.wait(min(.05,(target-time.monotonic_ns())/1e9))
            started = time.perf_counter()
            bgr, _ = load_dpvo_domain(self.anchor_paths[identity.key], self.calibration)
            value, _ = preprocess_full_fov_rgb(bgr[..., ::-1].copy(), self.transform)
            task = AnchorTask(identity,bgr,self._closing.get(identity.key),time.monotonic_ns())
            self._put(self.consume_queue, task)
            task.field, task.encode_ms = self._encode(value.numpy()[None], identity.key, ordinal)
            task.encoded_ns = time.monotonic_ns(); task.encoded.set()
            self._put(self.encode_queue, task)
        self._put(self.encode_queue, None); self._put(self.consume_queue, None)

    def _predict_tasks(self):
        previous = None; ordinal = 0
        while True:
            task = self._get(self.encode_queue)
            try:
                if task is None: return
                self._wait(task.encoded,"closing_anchor_encoding")
                if task.interval is not None and self.predict_hidden:
                    if previous is None or previous.identity.key != task.interval.anchor0.key:
                        raise RuntimeError("predictor endpoint identity mismatch")
                    task.prediction, task.predict_ms, task.decision_trace = self._predict(
                        previous.field, task.field, task.interval, task.identity.key, ordinal,
                    )
                    task.predicted_ns = time.monotonic_ns()
                task.ready.set(); previous = task; ordinal += 1
            finally:
                self.encode_queue.task_done()

    def _stage_c_h2d(self, array, name):
        started = time.perf_counter()
        result = self.transfer.h2d(array, name)
        self.profiler.add_stage_c_cpu(
            "stage_c_transfer_wait", (time.perf_counter() - started) * 1000.0,
        )
        operation = self.transfer.last_operation or {}
        if operation.get("direction") == "H2D" and operation.get("name") == name:
            self.profiler.add_stage_c_cuda(
                "stage_c_transfer_wait", float(operation["cuda_ms"]),
            )
        return result

    def observations(self):
        if self.anchor_frontend is None: raise RuntimeError("native anchor frontend hook missing")
        for function in (self._produce,self._predict_tasks):
            thread = threading.Thread(target=self._guard,args=(function,),daemon=True)
            self.threads.append(thread); thread.start()
        while True:
            queue_started = time.perf_counter()
            task = self._get(self.consume_queue)
            self.profiler.add_stage_c_cpu(
                "stage_c_queue_wait", (time.perf_counter() - queue_started) * 1000.0,
                queue_detail="consume_queue_wait",
            )
            try:
                if task is None: break
                image = self._stage_c_h2d(
                    np.ascontiguousarray(task.bgr.transpose(2,0,1)), "anchor_image",
                )
                intrinsics = torch.as_tensor(self.calibration[:4],dtype=torch.float32,device="cuda")
                observation = AnchorRGBObservation(task.identity,image,intrinsics,task.identity.timestamp_ns)
                self.anchor_frontend(observation); task.frontended.set()
                prediction_started = time.perf_counter()
                self._wait(task.ready,"prediction_ready")
                prediction_wait_ms = (time.perf_counter() - prediction_started) * 1000.0
                self.profiler.add_stage_c_cpu(
                    "stage_c_queue_wait",
                    prediction_wait_ms,
                    queue_detail="prediction_ready_wait",
                )
                self.available_anchor_keys.add(task.identity.key); self.encoded_anchor_keys.append(task.identity.key)
                self.profiler.add("jepa_encoder",task.encode_ms)
                if task.prediction is not None:
                    if task.decision_trace is not None:
                        self.decision_traces.append({
                            "closing_anchor": task.identity.key,
                            "trace": task.decision_trace,
                        })
                    self.profiler.add("jepa_predictor",task.predict_ms)
                    fields = self._stage_c_h2d(
                        task.prediction, "predicted_block5",
                    )
                    packets = self._bridge_packet(fields)
                    for offset, query in enumerate(task.interval.hidden):
                        self._record_yield(query.identity); self.consumed_hidden_keys.append(query.identity.key)
                        wait = (task.identity.timestamp_ns-query.identity.timestamp_ns)/1e6
                        self.hidden_context_wait[query.identity.key] = wait
                        self.interval_start_compute_ms[query.identity.key] = task.admitted_ns
                        self.profiler.context_wait_ms.append(wait)
                        hidden_packet = FMapZeroContextPacket(
                            packets.fmap[offset:offset+1]
                        )
                        yield PacketObservation(
                            query.identity, hidden_packet, "hidden",
                            task.identity.timestamp_ns,
                        )
                self._record_yield(task.identity); yield observation
                task.consumed.set()
                self.timeline.append({"anchor": task.identity.key,"admitted_ns":task.admitted_ns,
                                      "encoded_ns":task.encoded_ns,"predicted_ns":task.predicted_ns,
                                      "consumed_ns":time.monotonic_ns()})
            finally:
                self.consume_queue.task_done()
        for thread in self.threads: thread.join(timeout=self.execution.timeout_seconds)
        alive = [thread.name for thread in self.threads if thread.is_alive()]
        if alive:
            raise RuntimeError(f"pipeline threads did not exit: {alive}")
        if self.encode_queue.unfinished_tasks or self.consume_queue.unfinished_tasks:
            raise RuntimeError("pipeline queues retain unfinished tasks")
        self._check()

    def on_tracked(self, observation, _dpvo_ms):
        if isinstance(observation,PacketObservation):
            key=observation.identity.key
            self.profiler.effective_hidden_delay_ms.append(self.hidden_context_wait[key]+
                (time.monotonic_ns()-self.interval_start_compute_ms[key])/1e6)

    def usage_payload(self):
        producer_backpressure = distribution_ms([
            row["ms"] for row in self.waits if row["name"] == "queue_put"
        ])
        if self.timeline:
            completion_ms = [
                (row["consumed_ns"] - row["admitted_ns"]) / 1e6
                for row in self.timeline
            ]
            fill_ms = (self.timeline[0]["consumed_ns"] - self.timeline[0]["admitted_ns"]) / 1e6
            drain_ms = (self.timeline[-1]["consumed_ns"] - self.timeline[-1]["admitted_ns"]) / 1e6
            steady_ms = max(
                0.0,
                (self.timeline[-1]["admitted_ns"] - self.timeline[0]["consumed_ns"]) / 1e6,
            )
            latency = {
                "count": len(completion_ms), "mean_ms": float(np.mean(completion_ms)),
                "p95_ms": float(np.quantile(completion_ms, .95)),
                "max_ms": float(max(completion_ms)),
            }
        else:
            fill_ms = steady_ms = drain_ms = 0.0
            latency = {"count": 0, "mean_ms": None, "p95_ms": None, "max_ms": None}
        payload = super().usage_payload() | {"execution_backend":execution_provenance(self.execution),
            "worker_provenance":{name:worker.provenance for name,worker in self.workers.items()},
            "workers":self.worker_diagnostics,"timeline":self.timeline,"waits":self.waits,
            "stage_c_transfer":self.transfer.payload() if self.transfer else None,
            "host_shared_memory_transfer":self.host_transfer.payload(),
            "pipeline_wall_segments": {
                "fill_ms": fill_ms, "steady_ms": steady_ms, "drain_ms": drain_ms,
                "segments_partition_pipeline_wall": True,
                "stage_compute_overlaps_and_is_not_added_to_pipeline_wall": True,
            },
            "anchor_completion_latency": latency,
            "producer_queue_backpressure": producer_backpressure,
            "producer_queue_backpressure_definition": (
                "wall in bounded queue put calls on the producer path; "
                "reported separately from Stage C consumer queue wait"
            ),
            "latency_method":"context_wait_plus_monotonic_closing_arrival_to_consumption",
            "stage_times_overlap":True,
            "lifecycle_cleanup": self.lifecycle_cleanup}
        if self.execution.decision_trace:
            payload["correspondence_decision_traces"] = self.decision_traces
        return payload
