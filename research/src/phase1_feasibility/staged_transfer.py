"""No-P2P host transport. Shared CPU memory is NOT assumed to be pinned."""
from __future__ import annotations

import hashlib
import json
import os
import queue
import shutil
import subprocess
import threading
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import torch

from .efficiency_profiling import TransferLedger
from .protocol import REPO_ROOT


class SharedSlot:
    def __init__(self, shape, dtype=np.float32):
        self.shape = tuple(shape); self.dtype = np.dtype(dtype)
        self.memory = shared_memory.SharedMemory(create=True, size=int(np.prod(shape))*self.dtype.itemsize)
        self.array = np.ndarray(self.shape, dtype=self.dtype, buffer=self.memory.buf)
        self.generation = 0; self.busy = False; self.closed = False

    def acquire(self):
        if self.closed: raise RuntimeError("shared slot is closed")
        if self.busy: raise RuntimeError("slot reused before completion")
        self.busy = True; self.generation += 1
        return {"name": self.memory.name, "shape": list(self.shape),
                "dtype": self.dtype.str, "generation": self.generation}

    def release(self, generation):
        if not self.busy or generation != self.generation: raise RuntimeError("stale slot completion")
        self.busy = False

    def close(self):
        if self.closed:
            return {"closed": True, "already_closed": True, "was_busy": False}
        was_busy = self.busy
        self.busy = False
        self.array = None
        self.memory.close()
        try:
            self.memory.unlink()
        except FileNotFoundError:
            pass
        self.closed = True
        return {"closed": True, "already_closed": False, "was_busy": was_busy}


def shared_array(descriptor):
    memory = shared_memory.SharedMemory(name=descriptor["name"])
    # Independent subprocesses do not own these slots; only the parent unlinks.
    from multiprocessing import resource_tracker
    resource_tracker.unregister(memory._name, "shared_memory")
    return memory, np.ndarray(descriptor["shape"], dtype=np.dtype(descriptor["dtype"]), buffer=memory.buf)


class PinnedTransfer:
    def __init__(self, *, verify=False):
        self.verify = verify; self.ledger = TransferLedger()
        self.stream = torch.cuda.Stream(); self.buffers = {}
        self.verifications = []; self.closed = False
        self.last_operation = None

    def close(self):
        if self.closed:
            return {"closed": True, "already_closed": True}
        self.stream.synchronize()
        self.buffers.clear()
        self.closed = True
        return {"closed": True, "already_closed": False}

    @staticmethod
    def _payload_hash(value):
        array = np.ascontiguousarray(value)
        digest = hashlib.sha256()
        digest.update(str(array.dtype).encode());digest.update(str(tuple(array.shape)).encode())
        digest.update(array.tobytes())
        return digest.hexdigest()

    def _verify(self, direction, name, before, after):
        before_hash=self._payload_hash(before);after_hash=self._payload_hash(after)
        row={"direction":direction,"name":name,"shape":list(before.shape),
             "dtype":str(before.dtype),"before_sha256":before_hash,
             "after_sha256":after_hash,"exact":before_hash==after_hash}
        self.verifications.append(row)
        if not row["exact"]: raise RuntimeError(f"{direction} payload changed: {name}")

    def payload(self):
        result=self.ledger.payload()
        result["payload_verification"]={
            "enabled":bool(self.verify),"count":len(self.verifications),
            "all_exact":bool(self.verify and self.verifications and
                             all(row["exact"] for row in self.verifications)),
            "records":list(self.verifications),
        }
        return result

    def _buffer(self, key, shape, dtype):
        spec = (key, tuple(shape), dtype)
        if spec not in self.buffers:
            self.buffers[spec] = torch.empty(shape, dtype=dtype, pin_memory=True)
        return self.buffers[spec]

    def h2d(self, array, name):
        if self.closed: raise RuntimeError("pinned transfer is closed")
        source = torch.from_numpy(array)
        buffer = self._buffer(name+"h2d", source.shape, source.dtype)
        t = time.perf_counter(); buffer.copy_(source)
        self.ledger.add(name+"_cpu_staging", byte_count=source.numel()*source.element_size(), cpu_ms=(time.perf_counter()-t)*1000)
        t = time.perf_counter()
        with torch.cuda.stream(self.stream):
            start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
            start.record(); result = buffer.to("cuda", non_blocking=True); end.record()
        wait = time.perf_counter(); end.synchronize()
        cuda_ms = float(start.elapsed_time(end))
        self.ledger.add(name+"_device_wait", cpu_ms=(time.perf_counter()-wait)*1000)
        self.ledger.add(name+"_h2d", byte_count=source.numel()*source.element_size(),
                        cpu_ms=(time.perf_counter()-t)*1000, cuda_ms=cuda_ms)
        self.last_operation = {
            "direction": "H2D", "name": name, "cuda_ms": cuda_ms,
        }
        if self.verify: self._verify("H2D",name,array,result.cpu().numpy())
        return result

    def d2h(self, value, array, name):
        if self.closed: raise RuntimeError("pinned transfer is closed")
        value = value.detach().contiguous()
        buffer = self._buffer(name+"d2h", value.shape, value.dtype)
        self.stream.wait_stream(torch.cuda.current_stream())
        t = time.perf_counter()
        with torch.cuda.stream(self.stream):
            start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
            start.record(); buffer.copy_(value, non_blocking=True); end.record()
        wait = time.perf_counter(); end.synchronize()
        self.ledger.add(name+"_pinned_buffer_wait", cpu_ms=(time.perf_counter()-wait)*1000)
        self.ledger.add(name+"_d2h", byte_count=value.numel()*value.element_size(),
                        cpu_ms=(time.perf_counter()-t)*1000, cuda_ms=start.elapsed_time(end))
        t = time.perf_counter(); np.copyto(array, buffer.numpy())
        self.ledger.add(name+"_cpu_staging", byte_count=array.nbytes, cpu_ms=(time.perf_counter()-t)*1000)
        if self.verify: self._verify("D2H",name,buffer.numpy(),array)


class GPUWorker:
    """One sequential CUDA owner; host controllers supply bounded overlap."""
    def __init__(self, *, python, device, component, config_path, timeout=120.,
                 cpu_profile=None):
        self.timeout = timeout; self.responses = queue.Queue(); self.errors = []
        env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(device)
        command = [str(python), "-m", "research.src.phase1_feasibility.pipeline_worker",
                   "--config", str(config_path), "--component", component]
        if cpu_profile is not None:
            cpus = ",".join(str(value) for value in cpu_profile["cpus"])
            env["OMP_NUM_THREADS"] = str(cpu_profile["omp_num_threads"])
            env["MKL_NUM_THREADS"] = str(cpu_profile["mkl_num_threads"])
            env["PHASE1_CPU_PROFILE"] = json.dumps(cpu_profile, sort_keys=True)
            if shutil.which("numactl") and cpu_profile.get("numa_node") is not None:
                command = [
                    "numactl", f"--physcpubind={cpus}",
                    f"--membind={int(cpu_profile['numa_node'])}", *command,
                ]
            elif shutil.which("taskset"):
                command = ["taskset", "--cpu-list", cpus, *command]
            else:
                raise RuntimeError("NUMA/CPU binding requires numactl or taskset")
        self.process = subprocess.Popen(command,
                                         cwd=REPO_ROOT, env=env, stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        def receive():
            for line in self.process.stdout:
                try: self.responses.put(json.loads(line))
                except json.JSONDecodeError: self.responses.put({"status": "error", "error": line})
            self.responses.put({"status": "error", "error": "worker exited"})
        def stderr():
            for line in self.process.stderr:
                self.errors.append(line)
                self.errors[:] = self.errors[-100:]
        self.reader = threading.Thread(target=receive, daemon=True); self.reader.start()
        self.error_reader = threading.Thread(target=stderr, daemon=True); self.error_reader.start()
        try:
            self.provenance = self.receive()
            if self.provenance.get("status") != "ready": raise RuntimeError(self.provenance)
            runtime = self.provenance.get("provenance", {})
            if (
                runtime.get("cuda_visible_devices") != str(device)
                or runtime.get("logical_cuda_device_count") != 1
                or runtime.get("current_logical_cuda_device") != 0
            ):
                raise RuntimeError(
                    f"{component} worker device binding mismatch: requested={device}, "
                    f"runtime={runtime}"
                )
            self.provenance["physical_device"] = int(device)
            self.provenance["requested_global_logical_device"] = int(device)
            self.provenance["cpu_profile"] = cpu_profile
        except BaseException as primary_error:
            try:
                cleanup = self.close(require_graceful=False)
            except BaseException as cleanup_error:
                cleanup = {
                    "closed": False,
                    "errors": [
                        f"{type(cleanup_error).__name__}: {cleanup_error}",
                    ],
                }
            setattr(primary_error, "phase1_worker_cleanup", cleanup)
            raise

    def receive(self):
        try: result = self.responses.get(timeout=self.timeout)
        except queue.Empty: raise TimeoutError("GPU worker completion timeout")
        if result.get("status") == "error":
            raise RuntimeError(f"{result}: {''.join(self.errors)}")
        return result

    def request(self, value):
        self.process.stdin.write(json.dumps(value)+"\n"); self.process.stdin.flush()
        response = self.receive()
        if response.get("request_id") != value["request_id"]:
            raise RuntimeError("worker request identity mismatch")
        return response

    def close(self, *, require_graceful=True):
        if getattr(self, "closed", False):
            return {"closed": True, "already_closed": True}
        graceful = False
        close_error = None
        errors = []
        if self.process.poll() is None and require_graceful:
            try:
                response = self.request({"action": "close", "request_id": "close"})
                graceful = response.get("status") == "closed"
                if not graceful:
                    raise RuntimeError(f"worker rejected graceful close: {response}")
                self.process.wait(timeout=5)
            except BaseException as error:
                close_error = error
        if self.process.poll() is None:
            try:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill(); self.process.wait(timeout=5)
            except BaseException as error:
                errors.append(f"process exit: {type(error).__name__}: {error}")
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except BaseException as error:
                    errors.append(f"stream close: {type(error).__name__}: {error}")
        try:
            self.reader.join(timeout=1)
            self.error_reader.join(timeout=1)
        except BaseException as error:
            errors.append(f"reader join: {type(error).__name__}: {error}")
        self.closed = True
        result = {"closed": self.process.poll() is not None, "graceful": graceful,
                  "returncode": self.process.returncode,
                  "reader_threads_stopped": not self.reader.is_alive() and not self.error_reader.is_alive(),
                  "errors": errors}
        if require_graceful and (
            close_error is not None
            or not all((
                result["closed"], result["graceful"],
                result["reader_threads_stopped"], result["returncode"] == 0,
            ))
            or errors
        ):
            raise RuntimeError(f"GPU worker lifecycle cleanup failed: {result}") from close_error
        return result
